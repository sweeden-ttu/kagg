"""Tests for encode_observation and DuelingDoubleDQNBranching.forward."""

from __future__ import annotations

import torch

from kaggriculture_adapter import encode_observation
from kaggriculture_rl.dqn import (
    DuelingDoubleDQNBranching,
    KaggricultureFeatureExtractor,
)


REQUIRED_KEYS = (
    "tiles",
    "day",
    "hour",
    "player_id",
    "farms_p0_money",
    "farms_p1_money",
    "market_prices",
    "market_inventory",
    "seeds",
    "shed",
    "inventories",
)


def _raw_obs():
    return {
        "day": 2,
        "hour": 5,
        "farms": [
            {
                "money": 3000.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [0, 0],
            },
            {
                "money": 2800.0,
                "tiles": [[None] * 10 for _ in range(10)],
                "farmer": [9, 9],
            },
        ],
        "private": {
            "seeds": {"WHEAT": 2},
            "shed": {"CARROT": 1},
            "inventories": [{} for _ in range(6)],
        },
        "market": {
            "prices": {
                "WHEAT": 10,
                "CARROT": 20,
                "TOMATO": 30,
                "STRAWBERRY": 40,
                "MELON": 50,
            },
            "inventory": {"WHEAT": 100},
        },
    }


def test_encode_observation_keys_and_shapes():
    tensors = encode_observation(_raw_obs(), player_id=0, device="cpu")
    for key in REQUIRED_KEYS:
        assert key in tensors, key
    assert tensors["tiles"].shape == (10, 10)
    assert tensors["day"].shape == (1,)
    assert tensors["market_prices"].shape == (5,)
    assert tensors["inventories"].shape == (30,)
    assert float(tensors["farms_p0_money"][0]) == 3000.0


def test_dueling_forward_shapes_no_nameerror():
    """Regression: advantage_hands must not reference undefined ``adv``."""
    model = DuelingDoubleDQNBranching(
        feature_extractor=KaggricultureFeatureExtractor(),
    )
    model.eval()
    tensors = encode_observation(_raw_obs(), player_id=0, device="cpu")
    batch = {k: v.unsqueeze(0) if v.dim() >= 1 else v for k, v in tensors.items()}
    batch["tiles"] = batch["tiles"].long()
    out = model(batch)
    assert out["farmer_q"].shape == (1, 15)
    assert len(out["hand_q"]) == 6
    assert out["hand_q"][0].shape == (1, 15)
    assert out["market_q"].shape == (1, 10)
    assert out["value"].shape == (1, 1)
    assert torch.isfinite(out["farmer_q"]).all()
