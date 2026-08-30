"""Tests for KaggleEnvWrapper Gymnasium API and nested-obs masking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from kaggle_env_wrapper import KaggleEnvWrapper, _pick_legal
from kaggriculture_adapter import MARKET_ACTIONS, encode_observation, get_action_masks


class _MiniKaggleEnv:
    """Minimal two-agent env with nested Kaggriculture-shaped observations."""

    def __init__(self):
        self._step = 0
        self._obs = self._make_obs(money=100.0)

    def _make_obs(self, money: float) -> Dict[str, Any]:
        farm = {
            "money": money,
            "farmer": [0, 0],
            "tiles": [["EMPTY"] * 10 for _ in range(10)],
            "unlocked_quadrants": [[0, 0]],
            "hands": [],
        }
        return {
            "day": 1,
            "hour": 0,
            "player": 0,
            "farms": [farm, {**farm, "money": 3000.0, "farmer": [9, 9]}],
            "private": {"seeds": {}, "shed": {}, "inventories": []},
            "market": {"prices": {"FERTILIZER": 100}, "inventory": {}},
            "town": {},
        }

    def reset(self):
        self._step = 0
        self._obs = self._make_obs(money=100.0)
        return [
            {"observation": {**self._obs, "player": 0}, "reward": 0, "status": "ACTIVE", "info": {}},
            {"observation": {**self._obs, "player": 1}, "reward": 0, "status": "ACTIVE", "info": {}},
        ]

    def seed(self, seed: Optional[int] = None):
        return [seed]

    def step(self, actions: List[Any]):
        self._step += 1
        done = self._step >= 2
        status = "DONE" if done else "ACTIVE"
        return [
            {
                "observation": {**self._obs, "player": 0},
                "reward": 1.0,
                "status": status,
                "info": {},
            },
            {
                "observation": {**self._obs, "player": 1},
                "reward": 0.0,
                "status": status,
                "info": {},
            },
        ]

    def render(self, mode="human"):
        return None


def test_reset_returns_obs_info_tuple():
    env = KaggleEnvWrapper(_MiniKaggleEnv(), device="cpu", use_masking=True)
    out = env.reset()
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert "tiles" in obs
    assert isinstance(info, dict)


def test_enforce_masks_rejects_buy_land_when_broke():
    env = KaggleEnvWrapper(_MiniKaggleEnv(), device="cpu", use_masking=True)
    env.reset()
    assert env._last_raw_obs is not None
    masks = get_action_masks(env._last_raw_obs)
    assert not masks["market"][MARKET_ACTIONS["BUY_LAND"]]
    action = {
        "farmer": 0,
        "hands": [0] * 6,
        "market": MARKET_ACTIONS["BUY_LAND"],
    }
    fixed = env._enforce_valid_actions(action)
    assert fixed["market"] != MARKET_ACTIONS["BUY_LAND"] or masks["market"][fixed["market"]]
    assert masks["market"][fixed["market"]]


def test_pick_legal_fallback_to_pass():
    mask = np.zeros(10, dtype=bool)
    mask[0] = True
    assert _pick_legal(6, mask) == 0
