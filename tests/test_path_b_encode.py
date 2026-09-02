"""Tests for the Path B observation encoder (18 spatial channels + 55 numerics)."""

from __future__ import annotations

import numpy as np

from kaggriculture_adapter import PATH_B_TILE_CHANNELS, encode_path_b_observation


def _obs(**overrides):
    farm = {
        "money": 3000.0,
        "farmer": [1, 1],
        "tiles": [["EMPTY"] * 10 for _ in range(10)],
        "hands": [[0, 0], [4, 4]],
        "unlocked_quadrants": ["NW"],
    }
    obs = {
        "player": 0,
        "day": 5,
        "hour": 7,
        "farms": [farm, {"money": 2800.0, "farmer": [8, 8], "tiles": [["EMPTY"] * 10 for _ in range(10)]}],
        "private": {
            "seeds": {"WHEAT": 2},
            "shed": {"CARROT": 1, "EGG": 3, "FERTILIZER": 4},
            "inventories": [{} for _ in range(6)],
        },
        "market": {"prices": {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250},
                   "inventory": {"WHEAT": 500}},
        "town": {"unlocked_shops": ["BAKERY"]},
    }
    obs.update(overrides)
    return obs


def test_shape_and_numeric_length():
    enc = encode_path_b_observation(_obs(), player_id=0)
    assert enc["tiles"].shape == (PATH_B_TILE_CHANNELS, 10, 10)
    assert enc["numeric"].shape == (55,)
    assert enc["tiles"].dtype == np.float32


def test_spatial_channels_are_set():
    obs = _obs()
    farm0 = obs["farms"][0]
    # Farmer position (1,1) → channel 0
    farm0["tiles"][1][1] = {"kind": "WEED"}
    # Plant at (3,3)
    farm0["tiles"][3][3] = {
        "kind": "PLANT", "crop": "CARROT", "planted_day": 2,
        "watered_today": True, "yield_units": 3, "fertilized_until_day": 6,
    }
    # Animal structure at (7,7)
    farm0["tiles"][7][7] = {
        "kind": "PASTURE", "animal": "COW", "fed_today": True,
        "cared_today": False, "fertilizer_available": True, "yield_units": 2,
    }
    farm0["tiles"][2][2] = "LOCKED"

    t = encode_path_b_observation(obs, player_id=0)["tiles"]
    # Channel indices per the documented layout.
    assert t[0, 1, 1] == 1.0       # my farmer
    assert t[1, 8, 8] == 1.0       # opponent farmer
    assert t[9, 1, 1] == 1.0       # WEED
    assert t[10, 2, 2] == 1.0      # LOCKED
    assert t[4, 3, 3] == 1.0       # CARROT (channel 4)
    assert t[2, 3, 3] == 1.0       # watered_today
    assert abs(t[8, 3, 3] - 3.0 / 6.0) < 1e-6  # yield normalized by 6
    assert t[17, 3, 3] == 1.0      # fertilized (fertilized_until_day >= day)
    assert t[12, 7, 7] == 1.0      # PASTURE structure
    assert t[13, 7, 7] == 1.0      # animal present
    assert t[14, 7, 7] == 1.0      # fed_today
    assert t[15, 7, 7] == 0.0      # not cared
    assert t[16, 7, 7] == 1.0      # fertilizer_available
    assert abs(t[8, 7, 7] - 2.0 / 6.0) < 1e-6  # animal yield normalized


def test_day_hour_scaled_to_season():
    enc = encode_path_b_observation(_obs(day=15, hour=12), player_id=0)
    n = enc["numeric"]
    assert abs(n[0] - 15.0 / 30.0) < 1e-6   # day / 30
    assert abs(n[1] - 12.0 / 24.0) < 1e-6   # hour / 24


def test_hand_positions_and_animal_shed_in_numerics():
    enc = encode_path_b_observation(_obs(), player_id=0)
    n = enc["numeric"]
    # Numeric layout tail: [32]=my hand count, [33]=opp hand count,
    # [34..45]=6 hands × 2 normalized coords, [46..49]=EGG/MILK/WOOL/FERT shed.
    assert abs(n[34] - 0.0) < 1e-6 and abs(n[35] - 0.0) < 1e-6          # hand0 (0,0)
    assert abs(n[36] - 0.4) < 1e-6 and abs(n[37] - 0.4) < 1e-6          # hand1 (4,4)
    assert n[46] > 0.0                                                  # EGG shed count
    assert n[49] > 0.0                                                  # FERTILIZER shed count