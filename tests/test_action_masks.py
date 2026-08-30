"""Unit tests for action masks and ActionMasker delegation."""

from __future__ import annotations

import numpy as np

from kaggriculture_adapter import (
    FARMER_ACTIONS,
    LAND_PRICES,
    MARKET_ACTIONS,
    get_action_masks,
    plant_is_harvestable,
    plant_is_mature,
)
from kaggriculture_rl.dqn import ActionMasker


def _base_obs(*, player: int = 0, money: float = 0.0, unlocked=None, shed=None, seeds=None, fert_price=100.0):
    farm0 = {
        "money": money if player == 0 else 50.0,
        "farmer": [1, 1],
        "tiles": [["EMPTY"] * 10 for _ in range(10)],
        "unlocked_quadrants": unlocked if unlocked is not None else [[0, 0]],
    }
    farm1 = {
        "money": money if player == 1 else 50.0,
        "farmer": [8, 8],
        "tiles": [["EMPTY"] * 10 for _ in range(10)],
        "unlocked_quadrants": [[0, 0]],
    }
    # Place farmer on WEED for DIG legality
    farm0["tiles"][1][1] = {"kind": "WEED"}
    return {
        "player": player,
        "day": 1,
        "hour": 0,
        "farms": [farm0, farm1],
        "private": {
            "seeds": seeds or {},
            "shed": shed or {},
        },
        "market": {"prices": {"FERTILIZER": fert_price}},
        "town": {},
    }


def test_buy_seed_and_product_and_land_when_affordable():
    obs = _base_obs(money=5000.0, unlocked=[[0, 0]], shed={}, seeds={})
    masks = get_action_masks(obs)
    m = masks["market"]
    assert m[MARKET_ACTIONS["PASS"]]
    assert m[MARKET_ACTIONS["BUY_SEED"]]
    assert m[MARKET_ACTIONS["BUY_PRODUCT"]]
    assert m[MARKET_ACTIONS["BUY_ANIMAL"]]
    assert m[MARKET_ACTIONS["HIRE"]]
    assert m[MARKET_ACTIONS["BUY_LAND"]]
    # OTHER padding stays masked
    assert not m[7] and not m[8] and not m[9]


def test_buy_land_blocked_when_broke():
    obs = _base_obs(money=50.0, unlocked=[[0, 0]], fert_price=100.0)
    m = get_action_masks(obs)["market"]
    assert not m[MARKET_ACTIONS["BUY_LAND"]]
    assert m[MARKET_ACTIONS["BUY_SEED"]]  # cheapest seed is 2–10
    assert not m[MARKET_ACTIONS["BUY_PRODUCT"]]


def test_buy_land_uses_next_land_price():
    # Already bought first expansion slot → next price is LAND_PRICES[1]
    obs = _base_obs(
        money=LAND_PRICES[1] - 1,
        unlocked=[[0, 0], [0, 1]],
    )
    assert not get_action_masks(obs)["market"][MARKET_ACTIONS["BUY_LAND"]]
    obs["farms"][0]["money"] = LAND_PRICES[1]
    assert get_action_masks(obs)["market"][MARKET_ACTIONS["BUY_LAND"]]


def test_sell_when_shed_has_crop():
    obs = _base_obs(money=0.0, shed={"WHEAT": 3})
    m = get_action_masks(obs)["market"]
    assert m[MARKET_ACTIONS["SELL"]]
    assert not m[MARKET_ACTIONS["BUY_LAND"]]


def test_action_masker_delegates_and_uses_player_farm():
    # Player 1 has money; player 0 does not — mask must follow observation.player
    obs = _base_obs(player=1, money=5000.0, unlocked=[[0, 0], [1, 0], [0, 1]])
    # Make farm0 poor explicitly
    obs["farms"][0]["money"] = 0.0
    obs["farms"][1]["money"] = 5000.0
    m = ActionMasker.get_valid_market_actions(obs)
    assert m[MARKET_ACTIONS["BUY_LAND"]]
    farmer = ActionMasker.get_valid_farmer_actions(obs)
    assert farmer[0]  # PASS
    assert farmer.shape == (15,)


def test_harvest_masked_until_first_yield_day():
    obs = _base_obs(money=3000.0, seeds={"WHEAT": 1})
    wheat = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "yield_units": 1,
    }
    obs["farms"][0]["tiles"][1][1] = wheat
    obs["day"] = 1
    farmer = get_action_masks(obs)["farmer_verb"]
    assert farmer[FARMER_ACTIONS["WATER"]]
    assert not farmer[FARMER_ACTIONS["HARVEST"]]
    assert not plant_is_harvestable(wheat, 1)
    obs["day"] = 2
    farmer = get_action_masks(obs)["farmer_verb"]
    assert farmer[FARMER_ACTIONS["HARVEST"]]
    assert plant_is_harvestable(wheat, 2)
    assert not plant_is_mature(wheat, 2)  # yield_units=1 < good-enough 3
    wheat["yield_units"] = 3
    assert plant_is_mature(wheat, 2)  # good-enough yield
    wheat["yield_units"] = 1
    assert plant_is_mature(wheat, 4)  # max_yield_day
