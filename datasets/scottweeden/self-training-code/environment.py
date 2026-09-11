"""Environment wrappers for Kaggriculture self-play training.

Contains:
- KaggleCompetitiveEnv: Two-player wrapper around official kaggle-environments.
- KaggleCollaborativeEnv: Shared-score collaborative wrapper.
- create_competitive_env / create_collaborative_env: Factories.
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
    starting_money : int
        Per-farm starting bank (Experiment 3: 1500 competitive / 3000 collaborative).
    """

    challenge_mode: str = "competitive"

    def __init__(
        self,
        max_steps: int = 720,
        seed: int = 42,
        turns_per_cycle: int = 24,
        starting_money: int = 3000,
    ):
        import kaggle_environments
        self.max_steps = max_steps
        self.turns_per_cycle = int(turns_per_cycle)
        self.starting_money = int(starting_money)
        self.env = kaggle_environments.make(
            "kaggriculture",
            configuration={
                "episodeSteps": max_steps,
                "turnsPerDay": self.turns_per_cycle,
                "seed": seed,
                "startingMoney": self.starting_money,
            },
            debug=False,
        )
        self._obs: List[Dict[str, Any]] = [{}, {}]
        self._prev_money: List[float] = [0.0, 0.0]
        self._day2_money_at_start: Optional[List[float]] = None
        self._day2_engine_reward: List[float] = [0.0, 0.0]
        self._day2_refund_record: Optional[Dict[str, Any]] = None
        self._last_day: int = 0

    def transfer_bank(self, from_player: int, to_player: int, amount: float) -> bool:
        """Move ``amount`` coins from one farm bank to the other (experiment charity).

        Kaggriculture has no native donate op; this mutates the live engine farms
        and refreshes parsed observations so the transfer is publicly visible.
        """
        amount = float(amount)
        if amount <= 0 or from_player == to_player:
            return False
        if from_player not in (0, 1) or to_player not in (0, 1):
            return False

        # Prefer mutating the official engine state so later steps keep the transfer.
        engine_farms = None
        states = None
        try:
            states = getattr(self.env, "state", None)
            if states and len(states) > 0:
                obs0 = states[0].observation if hasattr(states[0], "observation") else states[0].get("observation")
                if obs0 is not None:
                    engine_farms = obs0["farms"] if isinstance(obs0, dict) else getattr(obs0, "farms", None)
        except (TypeError, KeyError, AttributeError):
            engine_farms = None

        if engine_farms is not None and len(engine_farms) > max(from_player, to_player):
            src = float(engine_farms[from_player].get("money", 0.0) or 0.0)
            if src < amount:
                return False
            engine_farms[from_player]["money"] = src - amount
            engine_farms[to_player]["money"] = float(
                engine_farms[to_player].get("money", 0.0) or 0.0
            ) + amount
            # Mirror into every seated observation's farms list when present.
            for st in states:
                o = st.observation if hasattr(st, "observation") else st.get("observation", {})
                farms = o["farms"] if isinstance(o, dict) else getattr(o, "farms", None)
                if farms is None:
                    continue
                farms[from_player]["money"] = engine_farms[from_player]["money"]
                farms[to_player]["money"] = engine_farms[to_player]["money"]

        # Always update our cached parsed observations (public money on both seats).
        for pid in (0, 1):
            farms = self._obs[pid].get("farms", []) or []
            if len(farms) <= max(from_player, to_player):
                continue
            src = float(farms[from_player].get("money", 0.0) or 0.0)
            if src < amount and engine_farms is None:
                return False
            if engine_farms is None:
                farms[from_player]["money"] = src - amount
                farms[to_player]["money"] = float(farms[to_player].get("money", 0.0) or 0.0) + amount
            else:
                farms[from_player]["money"] = float(engine_farms[from_player]["money"])
                farms[to_player]["money"] = float(engine_farms[to_player]["money"])

        self._sync_prev_money()
        return True

    def credit_bank(self, player: int, amount: float) -> bool:
        """Add coins to one farm bank (day-2 collaborative refund)."""
        amount = float(amount)
        if amount <= 0 or player not in (0, 1):
            return False
        engine_farms = None
        states = None
        try:
            states = getattr(self.env, "state", None)
            if states and len(states) > 0:
                obs0 = states[0].observation if hasattr(states[0], "observation") else states[0].get("observation")
                if obs0 is not None:
                    engine_farms = obs0["farms"] if isinstance(obs0, dict) else getattr(obs0, "farms", None)
        except (TypeError, KeyError, AttributeError):
            engine_farms = None

        if engine_farms is not None and len(engine_farms) > player:
            engine_farms[player]["money"] = float(
                engine_farms[player].get("money", 0.0) or 0.0
            ) + amount
            for st in states:
                o = st.observation if hasattr(st, "observation") else st.get("observation", {})
                farms = o["farms"] if isinstance(o, dict) else getattr(o, "farms", None)
                if farms is None:
                    continue
                farms[player]["money"] = engine_farms[player]["money"]

        for pid in (0, 1):
            farms = self._obs[pid].get("farms", []) or []
            if len(farms) <= player:
                continue
            if engine_farms is None:
                farms[player]["money"] = float(farms[player].get("money", 0.0) or 0.0) + amount
            else:
                farms[player]["money"] = float(engine_farms[player]["money"])

        self._sync_prev_money()
        return True

    def _sync_prev_money(self) -> None:
        self._prev_money = [
            float((self._obs[0].get("farms") or [{}])[0].get("money", 0.0) or 0.0)
            if len(self._obs[0].get("farms") or []) > 0
            else 0.0,
            float((self._obs[1].get("farms") or [{}, {}])[1].get("money", 0.0) or 0.0)
            if len(self._obs[1].get("farms") or []) > 1
            else 0.0,
        ]

    def _annotate_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        obs = dict(obs)
        obs["challenge_mode"] = self.challenge_mode
        obs["protocol_starting_money"] = self.starting_money
        obs["challenge_starting_money"] = self.starting_money
        return obs

    def reset(self) -> Dict[str, Any]:
        """Reset environment and return player 0's observation."""
        states = _normalize_env_states(self.env.reset())
        self._obs = [
            self._annotate_obs(parse_observation(states[0], player_id=0)),
            self._annotate_obs(parse_observation(states[1], player_id=1)),
        ]
        self._sync_prev_money()
        self._day2_money_at_start = None
        self._day2_engine_reward = [0.0, 0.0]
        self._day2_refund_record = None
        self._last_day = 0
        return self._obs[0]

    def _get_obs(self, player: int) -> Dict[str, Any]:
        """Return parsed observation for a specific player."""
        return self._obs[player]

    def _seat_money(self, seat: int) -> float:
        farms = self._obs[0].get("farms", []) or []
        if len(farms) > seat:
            return float(farms[seat].get("money", 0.0) or 0.0)
        return 0.0

    def apply_day2_kaggle_refund(self) -> Dict[str, Any]:
        """Credit each seat the day-2 net the engine attributed (collaborative).

        Uses accumulated engine reward during day 2 when available; otherwise
        falls back to the observed day-2 money delta (no silent no-op).
        """
        if self._day2_refund_record is not None:
            return dict(self._day2_refund_record)

        deltas = [0.0, 0.0]
        if self._day2_money_at_start is not None:
            for seat in (0, 1):
                deltas[seat] = self._seat_money(seat) - float(self._day2_money_at_start[seat])

        grants = [0.0, 0.0]
        for seat in (0, 1):
            engine_r = float(self._day2_engine_reward[seat])
            if abs(engine_r) > 1e-9:
                # Engine rewards in this wrapper are money_delta/100.
                grants[seat] = max(0.0, engine_r * 100.0)
            else:
                # Fallback: refund positive day-2 net (Kaggle-attributed earnings).
                grants[seat] = max(0.0, deltas[seat])
            if grants[seat] > 0:
                self.credit_bank(seat, grants[seat])

        record = {
            "ok": True,
            "challenge_mode": self.challenge_mode,
            "day2_money_at_start": list(self._day2_money_at_start or [0.0, 0.0]),
            "day2_money_deltas": deltas,
            "day2_engine_reward": list(self._day2_engine_reward),
            "grants": grants,
            "money_after": [self._seat_money(0), self._seat_money(1)],
        }
        self._day2_refund_record = record
        return record

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
            self._annotate_obs(parse_observation(states[0], player_id=0)),
            self._annotate_obs(parse_observation(states[1], player_id=1)),
        ]
        rewards: List[float] = []
        for p in range(2):
            farms = self._obs[p].get("farms", [])
            money = float(farms[p].get("money", 0.0)) if len(farms) > p else 0.0
            rewards.append((money - self._prev_money[p]) / 100.0)
            self._prev_money[p] = money

        day = int(self._obs[0].get("day", 0) or 0)
        if day == 2 and self._day2_money_at_start is None:
            self._day2_money_at_start = [self._seat_money(0), self._seat_money(1)]
        if day == 2:
            for p in range(2):
                self._day2_engine_reward[p] += float(rewards[p])
                raw = states[p].get("reward", None) if p < len(states) else None
                if raw is not None:
                    try:
                        self._day2_engine_reward[p] = max(
                            self._day2_engine_reward[p], float(raw)
                        )
                    except (TypeError, ValueError):
                        pass
        self._last_day = day

        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        info: Dict[str, Any] = {"challenge_mode": self.challenge_mode, "day": day}
        return (self._obs[0], self._obs[1]), rewards, done, info


class KaggleCollaborativeEnv(KaggleCompetitiveEnv):
    """Collaborative challenge: joint score is the sum of both farm banks."""

    challenge_mode: str = "collaborative"

    def step(
        self, actions: List[Dict[str, Any]]
    ) -> Tuple[Tuple[Dict[str, Any], Dict[str, Any]], List[float], bool, Dict]:
        (obs0, obs1), rewards, done, info = super().step(actions)
        joint = float(rewards[0]) + float(rewards[1])
        # Shared team reward; alliance weighting applied by the suite when sides set.
        alliance = info.get("alliance")
        if alliance and isinstance(alliance, dict):
            r_side = int(alliance.get("reasoning_side", 0))
            q_side = int(alliance.get("questioning_side", 1 - r_side))
            weighted = [
                joint * (0.6 if seat == r_side else 0.4)
                for seat in (0, 1)
            ]
            # Keep both agents oriented to the joint objective.
            rewards = [joint, joint]
            info["alliance_weighted"] = weighted
            info["reasoning_side"] = r_side
            info["questioning_side"] = q_side
        else:
            rewards = [joint, joint]
        info["joint_reward"] = joint
        info["team_money"] = self._seat_money(0) + self._seat_money(1)
        return (obs0, obs1), rewards, done, info


def create_competitive_env(
    use_kaggle: bool = True,
    max_steps: int = 720,
    seed: int = 42,
    turns_per_cycle: int = 24,
    starting_money: int = 3000,
):
    """Factory for competitive environments.

    Training defaults to 3000. Experiment 3 competitive ablations pass 1500.
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
            starting_money=starting_money,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle-environments is required for self-play training. "
            "Install with: pip install kaggle-environments"
        ) from exc


def create_collaborative_env(
    use_kaggle: bool = True,
    max_steps: int = 720,
    seed: int = 42,
    turns_per_cycle: int = 24,
    starting_money: int = 3000,
):
    """Factory for collaborative environments (Experiment 3: start 3000)."""
    if not use_kaggle:
        raise RuntimeError(
            "Offline training requires the official Kaggle simulator (use_kaggle_env=True). "
            "Install kaggle-environments and attach the kaggriculture environment."
        )
    try:
        return KaggleCollaborativeEnv(
            max_steps=max_steps,
            seed=seed,
            turns_per_cycle=turns_per_cycle,
            starting_money=starting_money,
        )
    except ImportError as exc:
        raise ImportError(
            "kaggle-environments is required for self-play training. "
            "Install with: pip install kaggle-environments"
        ) from exc
