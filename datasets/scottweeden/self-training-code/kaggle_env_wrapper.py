"""Kaggle-environments wrapper for RL training.

Provides a Gymnasium-compatible wrapper around the Kaggriculture
environment, converting observations and actions to/from tensor-friendly
formats while enforcing valid action masks.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from kaggriculture_adapter import (
    decode_action,
    encode_observation,
    encode_tiles,
    get_action_masks,
    parse_observation,
)


def _pick_legal(index: int, mask: np.ndarray) -> int:
    """Return ``index`` if legal, else a random True index (fallback PASS=0)."""
    mask = np.asarray(mask, dtype=bool)
    if 0 <= index < len(mask) and bool(mask[index]):
        return int(index)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    return int(random.choice(legal.tolist()))


class KaggleEnvWrapper(gym.Env):
    """Gymnasium Env wrapping Kaggle-environments Kaggriculture.

    Converts raw Kaggle observations into normalized tensor dictionaries
    suitable for the DQN agent. Enforces valid action masks and handles
    multi-agent observation routing.

    Parameters
    ----------
    env : kaggle_environments.Environment
        The Kaggle environment instance.
    device : str or torch.device
        Device for tensor operations.
    use_masking : bool
        If True, enforce valid action masks (default: True).
    clip_reward : bool
        If True, clip reward to [-10, 10] range (default: False).
    stack_size : int
        Number of previous observations to stack (default: 1).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        use_masking: bool = True,
        clip_reward: bool = False,
        stack_size: int = 1,
        player_id: int = 0,
        opponent_policy: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, Any]]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.env = env
        self.device = device
        self.use_masking = use_masking
        self.clip_reward = clip_reward
        self.stack_size = stack_size
        self.player_id = player_id
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode
        self._last_opponent_obs: Optional[Dict[str, torch.Tensor]] = None
        self._last_raw_obs: Optional[Dict[str, Any]] = None
        self._last_raw_opp_obs: Optional[Dict[str, Any]] = None

        self.n_hands = 6
        self.n_farmer_actions = 15
        self.n_hand_actions = 15
        self.n_market_actions = 10

        self.observation_space = spaces.Dict(
            {
                "tiles": spaces.Box(low=0, high=8, shape=(10, 10), dtype=np.int64),
                "day": spaces.Box(low=0.0, high=30.0, shape=(1,), dtype=np.float32),
                "hour": spaces.Box(low=0.0, high=72.0, shape=(1,), dtype=np.float32),
                "player_id": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "farms_p0_money": spaces.Box(
                    low=0.0, high=1e7, shape=(1,), dtype=np.float32
                ),
                "farms_p1_money": spaces.Box(
                    low=0.0, high=1e7, shape=(1,), dtype=np.float32
                ),
                "market_prices": spaces.Box(
                    low=0.0, high=1e5, shape=(5,), dtype=np.float32
                ),
                "market_inventory": spaces.Box(
                    low=0.0, high=1e6, shape=(5,), dtype=np.float32
                ),
                "seeds": spaces.Box(low=0.0, high=500.0, shape=(5,), dtype=np.float32),
                "shed": spaces.Box(low=0.0, high=1e4, shape=(5,), dtype=np.float32),
                "inventories": spaces.Box(
                    low=0.0, high=100.0, shape=(30,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Dict(
            {
                "farmer": spaces.Discrete(self.n_farmer_actions),
                "hands": spaces.MultiDiscrete(
                    [self.n_hand_actions] * self.n_hands
                ),
                "market": spaces.Discrete(self.n_market_actions),
            }
        )

        self.obs_history: List[Optional[Dict[str, torch.Tensor]]] = []
        for _ in range(stack_size - 1):
            self.obs_history.append(None)

        self.current_episode = 0
        self.total_episodes = 0
        self.current_reward = 0

    @classmethod
    def make(
        cls,
        opponent: str = "random",
        device: str = "cpu",
        use_masking: bool = True,
        clip_reward: bool = False,
        player_id: int = 0,
        debug: bool = False,
        **kwargs: Any,
    ) -> "KaggleEnvWrapper":
        """Build a wrapper around ``kaggle_environments.make("kaggriculture")``."""
        from kaggle_environments import make as kaggle_make

        env = kaggle_make("kaggriculture", debug=debug)
        # Opponent is selected per-step via opponent_policy / random; train mode
        # still needs a two-agent env. Keep the raw Environment for reset/step.
        _ = opponent  # reserved for future fixed-agent wiring
        return cls(
            env,
            device=device,
            use_masking=use_masking,
            clip_reward=clip_reward,
            player_id=player_id,
            **kwargs,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Reset the environment. Returns ``(obs, info)`` (Gymnasium API)."""
        super().reset(seed=seed)
        if seed is not None and hasattr(self.env, "seed"):
            self.env.seed(seed)

        result = self.env.reset()
        states = self._normalize_states(result)
        self._last_raw_obs = parse_observation(states[self.player_id], player_id=self.player_id)
        self._last_raw_opp_obs = parse_observation(
            states[1 - self.player_id], player_id=1 - self.player_id
        )
        self._last_opponent_obs = self._convert_observation(
            states[1 - self.player_id], player_id=1 - self.player_id
        )

        history: List[Optional[Dict[str, torch.Tensor]]] = [None] * (self.stack_size - 1)
        self.obs_history = history
        obs = self._convert_observation(
            states[self.player_id], player_id=self.player_id
        )
        self.obs_history.append(obs)
        self.current_episode += 1
        self.total_episodes += 1
        self.current_reward = 0

        return obs, {}

    def step(self, action: Dict[str, Any]) -> Tuple[
        Optional[Dict[str, torch.Tensor]], float, bool, bool, Dict[str, Any]
    ]:
        """Execute an action and return a Gymnasium 5-tuple transition."""
        if self.use_masking:
            action = self._enforce_valid_actions(action)

        opponent_action = self._opponent_action()
        actions: List[Optional[Dict[str, Any]]] = [None, None]
        actions[self.player_id] = self._decode_if_needed(action, self.player_id)
        actions[1 - self.player_id] = self._decode_if_needed(
            opponent_action, 1 - self.player_id
        )
        result = self.env.step(actions)
        states = self._normalize_states(result)
        agent_state = states[self.player_id]
        opponent_state = states[1 - self.player_id]

        reward = float(agent_state.get("reward") or 0)
        info: Dict[str, Any] = dict(agent_state.get("info") or {})

        status = agent_state.get("status")
        if status == "ACTIVE":
            terminated = False
            truncated = False
            done = False
        elif status == "DONE":
            terminated = True
            truncated = False
            done = True
        elif status == "TIMEOUT":
            terminated = False
            truncated = True
            done = True
        else:
            terminated = False
            truncated = False
            done = False

        if self.clip_reward:
            reward = max(-10.0, min(10.0, reward))

        self.current_reward += reward

        if done:
            next_obs = None
            self._last_opponent_obs = None
        else:
            next_obs = self._convert_observation(
                agent_state, player_id=self.player_id
            )
            self._last_raw_obs = parse_observation(agent_state, player_id=self.player_id)
            self._last_raw_opp_obs = parse_observation(
                opponent_state, player_id=1 - self.player_id
            )
            self._last_opponent_obs = self._convert_observation(
                opponent_state, player_id=1 - self.player_id
            )
            self.obs_history.append(next_obs)
            if len(self.obs_history) > self.stack_size:
                self.obs_history.pop(0)

        return next_obs, reward, terminated, truncated, info

    def get_action_space_info(self) -> Dict[str, int]:
        """Return action space dimensions."""
        return {
            "n_hands": self.n_hands,
            "n_farmer_actions": self.n_farmer_actions,
            "n_hand_actions": self.n_hand_actions,
            "n_market_actions": self.n_market_actions,
        }

    @staticmethod
    def _normalize_states(result: Any) -> list:
        if isinstance(result, list):
            states = result
        else:
            states = [result]
        normalized = []
        for state in states:
            if isinstance(state, dict):
                normalized.append(state)
            else:
                normalized.append({
                    "action": getattr(state, "action", None),
                    "reward": getattr(state, "reward", 0),
                    "info": getattr(state, "info", {}),
                    "observation": getattr(state, "observation", {}),
                    "status": getattr(state, "status", "ACTIVE"),
                })
        return normalized

    def _random_action(self) -> Dict[str, Any]:
        return {
            "farmer": random.randint(0, self.n_farmer_actions - 1),
            "hands": [
                random.randint(0, self.n_hand_actions - 1)
                for _ in range(self.n_hands)
            ],
            "market": random.randint(0, self.n_market_actions - 1),
        }

    def _opponent_action(self) -> Dict[str, Any]:
        if self.opponent_policy is not None and self._last_opponent_obs is not None:
            return self.opponent_policy(self._last_opponent_obs)
        return self._random_action()

    def _decode_if_needed(self, action: Dict[str, Any], player_id: int) -> Dict[str, Any]:
        """Convert integer branched action to Kaggle command lists if needed."""
        if isinstance(action.get("farmer"), list):
            return action
        if isinstance(action.get("farmer"), int):
            obs = self._last_raw_obs if player_id == self.player_id else self._last_raw_opp_obs
            if obs is None:
                obs = {"player": player_id, "farms": [], "private": {}}
            return decode_action(action, obs)
        return action

    @staticmethod
    def _encode_tiles(tiles: Any) -> np.ndarray:
        return encode_tiles(tiles)

    def _convert_observation(
        self, agent_result: Dict, player_id: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Convert raw Kaggle observation to tensor dict."""
        obs = agent_result.get("observation", agent_result)
        return encode_observation(obs, player_id, device=str(self.device))

    def _enforce_valid_actions(self, action: Dict) -> Dict:
        """Enforce valid action masks using nested ``_last_raw_obs``."""
        if not self.use_masking:
            return action

        raw = self._last_raw_obs
        if raw is None:
            return action

        masks = get_action_masks(raw)
        farmer_mask = masks["farmer_verb"]
        market_mask = masks["market"]

        out = dict(action)
        out["farmer"] = _pick_legal(int(action.get("farmer", 0)), farmer_mask)
        out["market"] = _pick_legal(int(action.get("market", 0)), market_mask)

        hands = list(action.get("hands", []))
        for i in range(len(hands)):
            hands[i] = min(int(hands[i]), self.n_hand_actions - 1)
        out["hands"] = hands
        return out

    def render(self):
        """Render the environment (passes through to Kaggle)."""
        if hasattr(self.env, "render"):
            return self.env.render(mode=self.render_mode or "human")
        return None
