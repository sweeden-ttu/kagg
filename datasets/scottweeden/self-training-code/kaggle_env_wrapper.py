"""Kaggle-environments wrapper for RL training.

Provides a Gymnasium-compatible wrapper around the Kaggriculture
environment, converting observations and actions to/from tensor-friendly
formats while enforcing valid action masks.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from kaggriculture_adapter import (
    decode_action,
    encode_observation,
    encode_tiles,
    get_action_masks,
    parse_observation,
)
from kaggriculture_rl.dqn import ActionMasker


def _pick_legal(idx: int, mask: Any) -> int:
    """Return idx if legal according to boolean mask; otherwise fallback to PASS or valid index."""
    if mask is None:
        return idx
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.size == 0:
        return idx
    if 0 <= idx < len(mask_arr) and bool(mask_arr[idx]):
        return idx
    if len(mask_arr) > 0 and bool(mask_arr[0]):
        return 0
    valid = np.where(mask_arr)[0]
    if len(valid) > 0:
        return int(valid[0])
    return 0


class KaggleEnvWrapper:
    """Gymnasium-compatible wrapper for Kaggle-environments.

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

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        use_masking: bool = True,
        clip_reward: bool = False,
        stack_size: int = 1,
        player_id: int = 0,
        opponent_policy: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, Any]]] = None,
    ):
        self.env = env
        self.device = device
        self.use_masking = use_masking
        self.clip_reward = clip_reward
        self.stack_size = stack_size
        self.player_id = player_id
        self.opponent_policy = opponent_policy
        self._last_opponent_obs: Optional[Dict[str, torch.Tensor]] = None
        self._last_raw_obs: Optional[Dict[str, Any]] = None
        self._last_raw_opp_obs: Optional[Dict[str, Any]] = None

        # Action space
        self.n_hands = 6
        self.n_farmer_actions = 15
        self.n_hand_actions = 15
        self.n_market_actions = 10

        # Stack history
        self.obs_history: List[Optional[Dict[str, torch.Tensor]]] = []
        for _ in range(stack_size - 1):
            self.obs_history.append(None)

        # Episode tracking
        self.current_episode = 0
        self.total_episodes = 0
        self.current_reward = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Reset the environment.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducibility.
        options : dict
            Additional reset options (ignored).

        Returns
        -------
        obs : dict
            Tensor-friendly observation dict.
        """
        if seed is not None:
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
        """Execute an action and return transition.

        Parameters
        ----------
        action : dict
            Action with keys "farmer", "hands", "market".

        Returns
        -------
        next_obs : dict or None
            Tensor-friendly next observation (``None`` when the episode ended).
        reward : float
            Episode reward.
        terminated : bool
            Whether the episode ended (success or failure).
        truncated : bool
            Whether the episode ended (timeout).
        info : dict
            Additional information.
        """
        # Apply masking if enabled
        if self.use_masking:
            current_obs = self.obs_history[-1] if self.obs_history else None
            action = self._enforce_valid_actions(action, current_obs)

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

        # Handle done signal
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

        # Clip reward if enabled
        if self.clip_reward:
            reward = max(-10.0, min(10.0, reward))

        self.current_reward += reward

        # Get next observation
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
            # Maintain stack size
            if len(self.obs_history) > self.stack_size:
                self.obs_history.pop(0)

        return next_obs, reward, terminated, truncated, info

    def get_action_space_info(self) -> Dict[str, int]:
        """Return action space dimensions.

        Returns
        -------
        info : dict
            Dictionary with action space dimensions.
        """
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
        """Convert raw Kaggle observation to tensor dict.

        Parameters
        ----------
        agent_result : dict
            Raw observation from Kaggle environment.
        player_id : int
            Player ID to extract observation for.

        Returns
        -------
        obs : dict
            Tensor-friendly observation dict.
        """
        obs = agent_result.get("observation", agent_result)
        farms = obs.get("farms", [])
        private = obs.get("private", {})
        market = obs.get("market", {})

        if len(farms) > player_id:
            farm = farms[player_id]
        else:
            farm = {}

        raw_tiles = farm.get("tiles", [[None] * 10 for _ in range(10)])
        tiles = self._encode_tiles(raw_tiles)
        obs_tensors = encode_observation(obs, player_id, device=str(self.device))
        return obs_tensors

    def _enforce_valid_actions(
        self, action: Dict, obs: Optional[Dict] = None
    ) -> Dict:
        """Enforce valid action masks."""
        if not self.use_masking:
            return action

        raw_obs = self._last_raw_obs
        if raw_obs is None and obs is not None:
            raw_obs = obs

        if raw_obs is None:
            return action

        try:
            masks = get_action_masks(raw_obs)
            fixed_action = copy.deepcopy(action)
            if "farmer" in fixed_action and isinstance(fixed_action["farmer"], int):
                fixed_action["farmer"] = _pick_legal(
                    fixed_action["farmer"], masks.get("farmer_verb")
                )
            if "market" in fixed_action and isinstance(fixed_action["market"], int):
                fixed_action["market"] = _pick_legal(
                    fixed_action["market"], masks.get("market")
                )
            if "hands" in fixed_action and isinstance(fixed_action["hands"], list):
                for i in range(len(fixed_action["hands"])):
                    fixed_action["hands"][i] = min(
                        max(int(fixed_action["hands"][i]), 0), self.n_hand_actions - 1
                    )
            return fixed_action
        except Exception:
            return action

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Return observation space specification."""
        return {
            "tiles": (10, 10),
            "day": (1,),
            "hour": (1,),
            "player_id": (1,),
            "farms_p0_money": (1,),
            "farms_p1_money": (1,),
            "market_prices": (5,),
            "market_inventory": (5,),
            "seeds": (5,),
            "shed": (5,),
            "inventories": (30,),
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        """Return action space specification."""
        return {
            "farmer": self.n_farmer_actions,
            "hands": [self.n_hand_actions] * self.n_hands,
            "market": self.n_market_actions,
        }

    def render(self, mode="human"):
        """Render the environment (passes through to Kaggle)."""
        return self.env.render(mode=mode)
