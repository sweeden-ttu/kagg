"""Environment wrappers for Kaggriculture self-play training.

Contains:
- KaggleCompetitiveEnv: Two-player wrapper around official kaggle-environments.
- create_competitive_env: Factory function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from kaggriculture_adapter import parse_observation


def _normalize_env_states(result: Any) -> List[Dict[str, Any]]:
    """Normalize environment step/reset result to a list of dicts."""
    if isinstance(result, list):
        return [s if isinstance(s, dict) else {
            "observation": getattr(s, "observation", {}),
            "status": getattr(s, "status", "ACTIVE"),
            "reward": getattr(s, "reward", 0),
        } for s in result]
    return [result if isinstance(result, dict) else {
        "observation": getattr(result, "observation", {}),
        "status": getattr(result, "status", "ACTIVE"),
        "reward": getattr(result, "reward", 0),
    }]


class KaggleCompetitiveEnv:
    """Two-player wrapper around official kaggle-environments.

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode (default 720 = competition standard).
    seed : int
        Random seed for the environment.
    turns_per_cycle : int
        Engine turnsPerDay (default 24 = competition parity; use 72 for
        kinematic self-play profile).
    """

    def __init__(
        self,
        max_steps: int = 720,
        seed: int = 42,
        turns_per_cycle: int = 24,
    ):
        import kaggle_environments
        self.max_steps = max_steps
        self.turns_per_cycle = int(turns_per_cycle)
        self.env = kaggle_environments.make(
            "kaggriculture",
            configuration={
                "episodeSteps": max_steps,
                "turnsPerDay": self.turns_per_cycle,
                "seed": seed,
            },
            debug=False,
        )
        self._obs: List[Dict[str, Any]] = [{}, {}]
        self._prev_money: List[float] = [0.0, 0.0]

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return player 0's observation."""
        states = _normalize_env_states(self.env.reset())
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        self._prev_money = [
            float(self._obs[0]["farms"][0].get("money", 0.0)) if self._obs[0].get("farms") else 0.0,
            float(self._obs[1]["farms"][1].get("money", 0.0)) if len(self._obs[1].get("farms", [])) > 1 else 0.0,
        ]
        return self._obs[0]

    def _get_obs(self, player: int) -> Dict[str, Any]:
        """Return parsed observation for a specific player."""
        return self._obs[player]

    def step(
        self, actions: List[Dict[str, Any]]
    ) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], List[float], bool, Dict]:
        """Execute a pair of actions and return transitions.

        Returns
        -------
        ((obs_p0, obs_p1), rewards, done, info)
        """
        states = _normalize_env_states(self.env.step(actions))
        self._obs = [
            parse_observation(states[0], player_id=0),
            parse_observation(states[1], player_id=1),
        ]
        rewards: List[float] = []
        for p in range(2):
            money = float(self._obs[p]["farms"][p].get("money", 0.0)) if self._obs[p].get("farms") else 0.0
            rewards.append((money - self._prev_money[p]) / 100.0)
            self._prev_money[p] = money
        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        return (self._obs[0], self._obs[1]), rewards, done, {}


def create_competitive_env(
    use_kaggle: bool = True,
    max_steps: int = 720,
    seed: int = 42,
    turns_per_cycle: int = 24,
):
    """Factory for creating competitive environments.

    Parameters
    ----------
    use_kaggle : bool
        If False, raises RuntimeError (offline training requires the
        official Kaggle simulator).
    """
    if not use_kaggle:
        raise RuntimeError(
            "Offline training requires the official Kaggle simulator (use_kaggle_env=True). "
            "Install kaggle-environments and attach the kaggriculture environment."
        )
    try:
        return KaggleCompetitiveEnv(
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle-environments is required for self-play training. "
            "Install with: pip install kaggle-environments"
        ) from exc
