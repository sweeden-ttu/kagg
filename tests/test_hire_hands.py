"""Daily HIRE gating and hired-hand farm decode."""

from __future__ import annotations

import numpy as np
import torch

from kaggriculture_adapter import (
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    daily_hire_orders_wanted,
    decode_hand_verb,
    decode_market_verb,
    get_action_masks,
    hire_cost_today,
    select_hand_farm_verbs,
)
from kaggriculture_path_b_rebuild import prefer_farm_invest_actions


def _obs(*, hour=0, money=3000.0, hires_today=0, hands=None, shed=None, seeds=None, tiles=None):
    farm0 = {
        "money": money,
        "farmer": [4, 4],
        "tiles": tiles if tiles is not None else [[None] * 10 for _ in range(10)],
        "unlocked_quadrants": [[0, 0]],
        "hands": hands if hands is not None else [],
        "hires_today": hires_today,
    }
    return {
        "player": 0,
        "day": 2,
        "hour": hour,
        "farms": [farm0, {"money": 3000.0, "farmer": [8, 8], "tiles": [[None] * 10 for _ in range(10)]}],
        "private": {"seeds": seeds or {"WHEAT": 2}, "shed": shed or {}},
        "market": {"prices": {"FERTILIZER": 100}},
    }


def test_hire_fib_and_morning_cap():
    assert hire_cost_today(0) == 1
    assert hire_cost_today(1) == 1
    assert hire_cost_today(2) == 2
    assert hire_cost_today(3) == 3
    assert daily_hire_orders_wanted(_obs(hour=0, hires_today=0, hands=[])) == 4
    assert daily_hire_orders_wanted(_obs(hour=5, hires_today=0, hands=[])) == 0
    assert daily_hire_orders_wanted(_obs(hour=0, hires_today=4, hands=[[1, 1]] * 4)) == 0
    assert daily_hire_orders_wanted(_obs(hour=0, money=20.0)) == 0


def test_hire_mask_allows_cheap_fib():
    m = get_action_masks(_obs(money=50.0))["market"]
    assert m[MARKET_ACTIONS["HIRE"]]


def test_decode_hand_waters_and_harvests():
    assert decode_hand_verb(FARMER_ACTIONS["WATER"]) == ["WATER"]
    assert decode_hand_verb(FARMER_ACTIONS["HARVEST"]) == ["HARVEST"]
    assert decode_hand_verb(FARMER_ACTIONS["PLANT"], 0, _obs()) == ["PLANT", "WHEAT"]


def test_select_hand_waters_standing_plant():
    tiles = [[None] * 10 for _ in range(10)]
    tiles[3][3] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "yield_units": 1,
    }
    obs = _obs(hands=[[3, 3]], tiles=tiles)
    verbs = select_hand_farm_verbs(obs)
    assert verbs[0] == FARMER_ACTIONS["WATER"]


def test_prefer_farm_boosts_morning_hire():
    q = torch.zeros(1, 10, 10)
    mask = np.ones(10, dtype=bool)
    obs = _obs(hour=0, shed={})
    _, market = prefer_farm_invest_actions(
        torch.zeros(1, 15),
        np.ones(15, dtype=bool),
        q,
        mask,
        observation=obs,
    )
    hire = MARKET_ACTIONS["HIRE"]
    assert market[0, 0, hire] > market[0, 0, MARKET_ACTIONS["PASS"]]
    assert market[0, 3, hire] > market[0, 3, MARKET_ACTIONS["BUY_ANIMAL"]]
    assert market[0, 4, hire] < market[0, 0, hire]


def test_buy_seed_batch_while_expanding():
    orders = decode_market_verb(MARKET_ACTIONS["BUY_SEED"], _obs(seeds={"WHEAT": 0}))
    assert orders and orders[0][0] == "BUY_SEED"
    assert orders[0][2] >= 2
