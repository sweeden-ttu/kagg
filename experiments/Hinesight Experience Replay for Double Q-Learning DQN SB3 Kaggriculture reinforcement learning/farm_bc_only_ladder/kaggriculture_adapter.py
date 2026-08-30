"""Single source of truth for Kaggriculture observation/action contracts.

Maps between official Kaggle command-list actions and branched integer indices
used by DQN / Path B trainers. Shared by dataset_loader, env wrapper, train
export, eval_policy, and self-play.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Canonical constants (match dataset_loader + official episodes) ──────────

CROPS: List[str] = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]

# Kinematic feedback season (optional via ``--turns-per-cycle 72``): 3×4×6.
# Default self-play uses competition ``turnsPerDay=24`` for ladder parity.
KINEMATIC_PHASE_A: int = 3
KINEMATIC_PHASE_B: int = 4
KINEMATIC_PHASE_C: int = 6
TURNS_PER_CYCLE: int = KINEMATIC_PHASE_A * KINEMATIC_PHASE_B * KINEMATIC_PHASE_C  # 72
CYCLES_PER_EPISODE: int = 10
EPISODE_STEPS: int = TURNS_PER_CYCLE * CYCLES_PER_EPISODE  # 720
# Competition / reference-ladder / historical episode tapes (Kaggle default).
COMPETITION_TURNS_PER_DAY: int = 24

FARMER_ACTIONS: Dict[str, int] = {
    "PASS": 0,
    "DIG": 1,
    "WATER": 2,
    "PLANT": 3,
    "HARVEST": 4,
    "NORTH": 5,
    "SOUTH": 6,
    "WEST": 7,
    "EAST": 8,
    "DROP": 9,
    "PICKUP": 10,
    "BUILD_COOP": 11,
    "BUILD_PASTURE": 12,
    "BUY_ANIMAL": 13,
    "OTHER": 14,
}

# Legacy encode aliases from episode JSON
_FARMER_ENCODE_ALIASES = {
    "MOVE_NORTH": 5,
    "MOVE_SOUTH": 6,
    "MOVE_WEST": 7,
    "MOVE_EAST": 8,
}

FARMER_INDEX_TO_VERB: List[str] = [
    "PASS", "DIG", "WATER", "PLANT", "HARVEST",
    "NORTH", "SOUTH", "WEST", "EAST",
    "DROP", "PICKUP", "BUILD_COOP", "BUILD_PASTURE", "BUY_ANIMAL", "OTHER",
]

MARKET_ACTIONS: Dict[str, int] = {
    "PASS": 0,
    "BUY_SEED": 1,
    "BUY_PRODUCT": 2,
    "BUY_ANIMAL": 3,
    "SELL": 4,
    "HIRE": 5,
    "BUY_LAND": 6,
}

MARKET_INDEX_TO_VERB: List[str] = [
    "PASS", "BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE", "BUY_LAND",
    "OTHER", "OTHER", "OTHER",
]

NUM_FARMER_ACTIONS = 15
NUM_HAND_ACTIONS = 15
NUM_HANDS = 6
NUM_MARKET_ACTIONS = 10

TILE_CLASS: Dict[str, int] = {
    "EMPTY": 0,
    "LOCKED": 1,
    "WEED": 2,
    "WHEAT": 3,
    "CARROT": 4,
    "TOMATO": 5,
    "STRAWBERRY": 6,
    "MELON": 7,
    "OTHER": 8,
}

CROP_TO_TILE_CLASS = {crop: TILE_CLASS[crop] for crop in CROPS}

# Engine seed prices (kaggriculture.CROPS[*]["seed"]).
SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
LAND_PRICES = [1000, 2000, 4000]
# Engine: cost = farmHandCostMult * fib(hires_today), fib(0)=1,1,2,3,5,...
# Four morning hires cost 1+1+2+3 = 7. Eight cost 54 (through fib 21).
HIRE_FIB_COSTS = (1, 1, 2, 3, 5, 8, 13, 21)
HIRE_COST = 1
# Neural net still has NUM_HANDS=6 heads; rule-based crew can hire up to 8
# (Hana). Hands reset each dawn, so this is a daily re-hire target.
CREW_HAND_CAP = 8
DAILY_HIRE_TARGET = 8
HANDS_BY_DAY: Tuple[Tuple[int, int], ...] = ((0, 4), (4, 8))
MAX_HIRE_ORDERS_PER_TURN = 5
HIRE_MAX_UNIT_COST = 21
HIRE_HOUR_LIMIT = 2
HIRE_CASH_RESERVE = 80
# One extra quadrant (NE @ $1000) only after the wheat engine is printing cash.
LAND_BUY_TARGET = 1
LAND_CASH_BUFFER = 500
LAND_MIN_PLANTS = 8
LAND_MIN_DAY = 5
LAND_MIN_MONEY = 2400
TARGET_WHEAT_PLANTS = 16
TARGET_PLANTS_PER_QUADRANT = 18
# After NE unlock: wheat-led staples (carrot for faster cycles; no tomato yet).
STAPLE_CROPS: Tuple[str, ...] = ("WHEAT", "CARROT")
STAPLE_SHARE = {"WHEAT": 0.7, "CARROT": 0.3}
SEED_BUY_BATCH = 8
SELL_CHUNK = 30
MAX_SELL_ORDERS = 4
# Season gates (match homestead reference: invest → plant → liquidate).
INVEST_UNTIL_DAY = 22
PLANT_UNTIL_DAY = 25
LIQUIDATE_FROM_DAY = 28
FILL_RATIO_TARGET = 0.9
ANIMAL_MIN_COST = 400
SHED_CAP = 100
DEFAULT_FERTILIZER_PRICE = 100
_HAND_VERBS = frozenset({
    "PASS", "DIG", "WATER", "PLANT", "HARVEST",
    "NORTH", "SOUTH", "WEST", "EAST", "DROP", "PICKUP",
})

# Engine growth table (kaggriculture.py CROPS). Wheat is planted with
# yield_units=1 immediately, but HARVEST is a no-op until first_yield_day.
CROP_GROWTH: Dict[str, Dict[str, Any]] = {
    "WHEAT": {"first_yield_day": 2, "max_yield_day": 4, "ongoing": False, "max_yield": 6},
    "CARROT": {"first_yield_day": 2, "max_yield_day": 3, "ongoing": False, "max_yield": 4},
    "TOMATO": {"first_yield_day": 8, "max_yield_day": 8, "ongoing": True, "max_yield": 4},
    "STRAWBERRY": {"first_yield_day": 10, "max_yield_day": 10, "ongoing": True, "max_yield": 4},
    "MELON": {"first_yield_day": 10, "max_yield_day": 12, "ongoing": False, "max_yield": 6},
}


def _farm_mapping(farm: Any) -> Dict[str, Any]:
    if farm is None:
        return {}
    if isinstance(farm, dict):
        return farm
    try:
        return dict(farm)
    except (TypeError, ValueError):
        return {}


def hire_cost_today(hires_today: int) -> int:
    """Next HIRE coin cost (engine ``_hire_cost`` with mult=1)."""
    n = max(0, int(hires_today))
    if n < len(HIRE_FIB_COSTS):
        return int(HIRE_FIB_COSTS[n])
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return int(a)


def farm_labor_state(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Hour, cash, and crew size for daily HIRE gating."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    opp = _farm_mapping(farms[1 - player] if len(farms) > (1 - player) else {})
    hands = farm.get("hands") or []
    return {
        "hour": int(observation.get("hour", 0) or 0),
        "day": int(observation.get("day", 0) or 0),
        "money": float(farm.get("money", 0.0) or 0.0),
        "opp_money": float(opp.get("money", 0.0) or 0.0),
        "hires_today": int(farm.get("hires_today", 0) or 0),
        "n_hands": len(hands) if isinstance(hands, list) else 0,
    }


def should_scale_farm(observation: Dict[str, Any]) -> bool:
    """Hire 8 / buy NE only when opponent is homestead-tier (land or bank).

    Finn/Walter/Rosa stay on the 4-hand wheat loop. Scaling trips when the
    opponent has bought land (Hana+) or is clearly ahead late.
    """
    labor = farm_labor_state(observation)
    day = int(labor["day"])
    mine = float(labor["money"])
    opp = float(labor.get("opp_money") or 0.0)
    if day < 4:
        return False
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    opp_farm = _farm_mapping(farms[1 - player] if len(farms) > (1 - player) else {})
    opp_unlocked = opp_farm.get("unlocked_quadrants") or []
    opp_bought = max(0, len(opp_unlocked) - 1) if isinstance(opp_unlocked, list) else 0
    # Hana buys NE; Rosa never does. Mirror land + crew when they expand.
    if opp_bought >= 1 and day >= 4:
        return True
    if opp >= mine + 2000.0 and day >= 8:
        return True
    if mine >= 11000.0 and day >= 12:
        return True
    return False


def daily_hire_target_for_day(day: int, observation: Optional[Dict[str, Any]] = None) -> int:
    """Crew size for this calendar day (4 unless scale gate → 8)."""
    if observation is not None and not should_scale_farm(observation):
        return 4
    target = int(HANDS_BY_DAY[0][1]) if HANDS_BY_DAY else int(DAILY_HIRE_TARGET)
    for from_day, count in HANDS_BY_DAY:
        if int(day) >= int(from_day):
            target = int(count)
    if observation is None:
        return min(int(target), 4)  # conservative without obs
    return min(int(target), int(CREW_HAND_CAP))


def unlocked_quadrant_count(observation: Dict[str, Any]) -> int:
    """How many farm quadrants are unlocked (NW counts as 1)."""
    labor_player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[labor_player] if len(farms) > labor_player else {})
    unlocked = farm.get("unlocked_quadrants") or []
    return max(1, len(unlocked) if isinstance(unlocked, list) else 1)


def target_plant_count(observation: Dict[str, Any]) -> int:
    """Plant cap grows with land once a real crew can work the extra tiles."""
    n_quad = unlocked_quadrant_count(observation)
    labor = farm_labor_state(observation)
    crew = max(int(labor["n_hands"]), int(labor["hires_today"]))
    base = int(TARGET_WHEAT_PLANTS)
    # Even on one quadrant, a full crew can push past 16 watered plants.
    if crew >= 6:
        base = max(base, 20)
    if n_quad <= 1:
        return base
    return max(base, int(TARGET_PLANTS_PER_QUADRANT) * int(n_quad))


def land_buy_wanted(
    observation: Dict[str, Any],
    *,
    target: int = LAND_BUY_TARGET,
    buffer: int = LAND_CASH_BUFFER,
    cash_reserve: int = HIRE_CASH_RESERVE,
) -> bool:
    """True when one extra quadrant is affordable without starving hire/seed."""
    labor = farm_labor_state(observation)
    bought = unlocked_quadrant_count(observation) - 1
    if bought >= int(target) or bought >= len(LAND_PRICES):
        return False
    if int(labor["day"]) < int(LAND_MIN_DAY):
        return False
    if not season_phase(observation)["investing"]:
        return False
    if not should_scale_farm(observation):
        return False
    census = farm_plant_census(observation)
    if int(census["plants"]) < int(LAND_MIN_PLANTS):
        return False
    # Need a working crew before expanding the field.
    if int(labor["n_hands"]) < 4 and int(labor["hires_today"]) < 4:
        return False
    price = int(LAND_PRICES[bought])
    need = max(
        float(LAND_MIN_MONEY),
        float(price) + float(buffer) + float(cash_reserve),
    )
    return float(labor["money"]) >= need


def daily_hire_orders_wanted(
    observation: Dict[str, Any],
    *,
    target: Optional[int] = None,
    hour_limit: int = HIRE_HOUR_LIMIT,
    cash_reserve: int = HIRE_CASH_RESERVE,
    max_per_turn: int = MAX_HIRE_ORDERS_PER_TURN,
) -> int:
    """How many HIRE orders to emit this turn (early hours; fib + reserve)."""
    labor = farm_labor_state(observation)
    if int(labor["hour"]) > int(hour_limit):
        return 0
    already = int(labor["hires_today"])
    have = int(labor["n_hands"])
    if target is None:
        cap = daily_hire_target_for_day(int(labor["day"]), observation)
    else:
        cap = int(target)
    cap = min(int(cap), int(CREW_HAND_CAP))
    todo = max(0, min(cap - already, cap - have, int(CREW_HAND_CAP) - have))
    todo = min(todo, int(max_per_turn))
    money = float(labor["money"])
    while todo > 0:
        cost_sum = sum(hire_cost_today(already + i) for i in range(todo))
        last_cost = hire_cost_today(already + todo - 1)
        if last_cost <= int(HIRE_MAX_UNIT_COST) and money >= cost_sum + float(cash_reserve):
            return todo
        todo -= 1
    return 0


def plant_is_harvestable(tile: Any, day: int) -> bool:
    """True when the engine will actually collect yield (not a HARVEST no-op)."""
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    if int(tile.get("yield_units", 0) or 0) <= 0:
        return False
    crop = str(tile.get("crop") or "WHEAT")
    info = CROP_GROWTH.get(crop) or CROP_GROWTH["WHEAT"]
    planted = int(tile.get("planted_day", 0) or 0)
    return int(day) - planted >= int(info["first_yield_day"])


def plant_is_mature(tile: Any, day: int) -> bool:
    """True when a one-shot crop has finished most of its watering window.

    Wait for near-cap yield (or max-day age) instead of cutting wheat at 3/6.
    """
    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
        return False
    crop = str(tile.get("crop") or "WHEAT")
    info = CROP_GROWTH.get(crop) or CROP_GROWTH["WHEAT"]
    planted = int(tile.get("planted_day", 0) or 0)
    age = int(day) - planted
    if info.get("ongoing"):
        return plant_is_harvestable(tile, day)
    units = int(tile.get("yield_units", 0) or 0)
    cap = int(info.get("max_yield") or 0)
    max_day = int(info["max_yield_day"])
    if cap and units >= cap:
        return True
    # Near-cap is good enough once watering has done real work.
    if cap and units >= max(4, cap - 1):
        return True
    return age >= max_day


def farm_plant_census(observation: Dict[str, Any]) -> Dict[str, int]:
    """Count plants, seeds, and standing-tile farm state for decode heuristics."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    private = observation.get("private") or {}
    seeds = private.get("seeds") or {}
    shed = private.get("shed") or {}
    day = int(observation.get("day", 0) or 0)
    tiles = farm.get("tiles") or []
    farmer_pos = farm.get("farmer") or [0, 0]
    fx = int(farmer_pos[0]) if len(farmer_pos) >= 1 else 0
    fy = int(farmer_pos[1]) if len(farmer_pos) >= 2 else 0
    plants = 0
    harvestable = 0
    mature = 0
    standing = None
    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        standing = tiles[fy][fx]
    for row in tiles:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                plants += 1
                if plant_is_harvestable(tile, day):
                    harvestable += 1
                if plant_is_mature(tile, day):
                    mature += 1
    seed_count = 0
    if isinstance(seeds, dict):
        seed_count = int(sum(int(v or 0) for v in seeds.values()))
    shed_count = 0
    if isinstance(shed, dict):
        shed_count = int(sum(int(v or 0) for v in shed.values()))
    standing_plant = isinstance(standing, dict) and standing.get("kind") == "PLANT"
    standing_empty = standing is None
    watered = bool(standing_plant and standing.get("watered_today", False))
    return {
        "plants": plants,
        "harvestable": harvestable,
        "mature": mature,
        "seed_count": seed_count,
        "shed_count": shed_count,
        "standing_plant": int(standing_plant),
        "standing_empty": int(standing_empty),
        "standing_watered": int(watered),
        "standing_harvestable": int(plant_is_harvestable(standing, day)),
        "standing_mature": int(plant_is_mature(standing, day)),
        "day": day,
        "farmer_x": fx,
        "farmer_y": fy,
    }


def empty_neighbor_move_indices(
    observation: Dict[str, Any],
    pos: Optional[Tuple[int, int]] = None,
) -> Tuple[int, ...]:
    """Move verbs that step onto an unlocked empty tile (plant target)."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    tiles = farm.get("tiles") or []
    if pos is None:
        farmer_pos = farm.get("farmer") or [0, 0]
        fx = int(farmer_pos[0]) if len(farmer_pos) >= 1 else 0
        fy = int(farmer_pos[1]) if len(farmer_pos) >= 2 else 0
    else:
        fx, fy = int(pos[0]), int(pos[1])
    hits = []
    for dx, dy, idx in ((0, -1, 5), (0, 1, 6), (-1, 0, 7), (1, 0, 8)):
        nx, ny = fx + dx, fy + dy
        if ny < 0 or nx < 0 or ny >= len(tiles) or nx >= 10:
            continue
        if nx >= len(tiles[ny]):
            continue
        tile = tiles[ny][nx]
        if tile is None:
            hits.append(idx)
    return tuple(hits)


def _shed_access_xy(tiles: Any) -> Tuple[int, int]:
    """Inner-corner shed tile that is not LOCKED (engine NWSE order)."""
    for x, y in ((4, 4), (5, 4), (4, 5), (5, 5)):
        if y < len(tiles) and x < len(tiles[y]) and tiles[y][x] != "LOCKED":
            return (x, y)
    return (4, 4)


def nearest_empty_plant_tile(
    observation: Dict[str, Any],
    pos: Optional[Tuple[int, int]] = None,
    claimed: Optional[set] = None,
) -> Optional[Tuple[int, int]]:
    """Nearest unlocked empty tile, shed-proximal first (Hana fill order)."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    tiles = farm.get("tiles") or []
    if pos is None:
        farmer_pos = farm.get("farmer") or [0, 0]
        fx = int(farmer_pos[0]) if len(farmer_pos) >= 1 else 0
        fy = int(farmer_pos[1]) if len(farmer_pos) >= 2 else 0
    else:
        fx, fy = int(pos[0]), int(pos[1])
    skip = claimed or set()
    sx, sy = _shed_access_xy(tiles)
    candidates = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile is not None or (x, y) in skip:
                continue
            # Prefer tiles near the shed, then near the worker.
            shed_d = abs(x - sx) + abs(y - sy)
            self_d = abs(x - fx) + abs(y - fy)
            candidates.append((shed_d, self_d, y, x))
    if not candidates:
        return None
    candidates.sort()
    _, _, ty, tx = candidates[0]
    return (tx, ty)


def season_phase(observation: Dict[str, Any]) -> Dict[str, bool]:
    """Invest / plant / liquidate gates for late-season cash."""
    day = int(observation.get("day", 0) or 0)
    return {
        "investing": day <= int(INVEST_UNTIL_DAY),
        "planting": day <= int(PLANT_UNTIL_DAY),
        "liquidating": day >= int(LIQUIDATE_FROM_DAY),
    }


def _step_toward_move_index(fx: int, fy: int, tx: int, ty: int) -> Optional[int]:
    """One Manhattan step, horizontal first (same as the reference ladder)."""
    if fx != tx:
        return FARMER_ACTIONS["EAST"] if tx > fx else FARMER_ACTIONS["WEST"]
    if fy != ty:
        return FARMER_ACTIONS["SOUTH"] if ty > fy else FARMER_ACTIONS["NORTH"]
    return None


def farm_tour_move_index(
    observation: Dict[str, Any],
    pos: Optional[Tuple[int, int]] = None,
    claimed: Optional[set] = None,
) -> Optional[int]:
    """Step toward the nearest unwatered plant, else nearest mature crop."""
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    tiles = farm.get("tiles") or []
    if pos is None:
        farmer_pos = farm.get("farmer") or [0, 0]
        fx = int(farmer_pos[0]) if len(farmer_pos) >= 1 else 0
        fy = int(farmer_pos[1]) if len(farmer_pos) >= 2 else 0
    else:
        fx, fy = int(pos[0]), int(pos[1])
    day = int(observation.get("day", 0) or 0)
    skip = claimed or set()
    unwatered = []
    mature = []
    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                continue
            if x == fx and y == fy:
                continue
            if (x, y) in skip:
                continue
            if not tile.get("watered_today", False):
                unwatered.append((abs(x - fx) + abs(y - fy), y, x))
            elif plant_is_mature(tile, day) and plant_is_harvestable(tile, day):
                mature.append((abs(x - fx) + abs(y - fy), y, x))
    if unwatered:
        unwatered.sort()
        _, ty, tx = unwatered[0]
        return _step_toward_move_index(fx, fy, tx, ty)
    if mature:
        mature.sort()
        _, ty, tx = mature[0]
        return _step_toward_move_index(fx, fy, tx, ty)
    return None


def _hand_xy(hand: Any) -> Tuple[int, int]:
    if isinstance(hand, (list, tuple)) and len(hand) >= 2:
        return int(hand[0]), int(hand[1])
    return 0, 0


def select_hand_farm_verbs(
    observation: Dict[str, Any],
    *,
    target_plants: Optional[int] = None,
) -> List[int]:
    """WATER / HARVEST / DIG / PLANT / tour MOVE for each hired hand."""
    # Size to active crew (up to CREW_HAND_CAP); pad to NUM_HANDS for BC tensors.
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    tiles = farm.get("tiles") or []
    hands = farm.get("hands") or []
    n_out = max(int(NUM_HANDS), min(int(CREW_HAND_CAP), len(hands) if isinstance(hands, list) else 0))
    verbs = [FARMER_ACTIONS["PASS"]] * n_out
    if not isinstance(hands, list) or not hands:
        return verbs
    day = int(observation.get("day", 0) or 0)
    phase = season_phase(observation)
    census = farm_plant_census(observation)
    seed_count = int(census["seed_count"])
    plant_count = int(census["plants"])
    plant_cap = int(target_plants) if target_plants is not None else target_plant_count(observation)
    # Late season: stop expanding; liquidate standing crops.
    allow_plant = bool(phase["planting"]) and seed_count > 0 and plant_count < plant_cap
    underfilled = plant_count < int(plant_cap * float(FILL_RATIO_TARGET))
    need_fill = allow_plant and underfilled
    n_hands_live = min(len(hands), int(CREW_HAND_CAP))
    # Reserve roughly half the crew for watering/harvest once some plants exist.
    fill_slots = n_hands_live
    if plant_count >= 8 and need_fill:
        fill_slots = max(2, n_hands_live // 2)
    claimed: set = set()
    farmer_pos = farm.get("farmer") or [0, 0]
    if len(farmer_pos) >= 2:
        claimed.add((int(farmer_pos[0]), int(farmer_pos[1])))

    for i, hand in enumerate(hands):
        if i >= int(CREW_HAND_CAP):
            break
        if i >= len(verbs):
            verbs.append(FARMER_ACTIONS["PASS"])
        hx, hy = _hand_xy(hand)
        standing = None
        if 0 <= hy < len(tiles) and 0 <= hx < len(tiles[hy]):
            standing = tiles[hy][hx]
        prefer_fill = need_fill and i < fill_slots
        if isinstance(standing, dict) and standing.get("kind") == "PLANT":
            if not standing.get("watered_today", False) and not phase["liquidating"]:
                verbs[i] = FARMER_ACTIONS["WATER"]
                claimed.add((hx, hy))
                continue
            ready = plant_is_harvestable(standing, day) and (
                phase["liquidating"] or plant_is_mature(standing, day)
            )
            if ready:
                verbs[i] = FARMER_ACTIONS["HARVEST"]
                claimed.add((hx, hy))
                continue
            # Immature watered plant: leave it; fill empty land when under target.
            if prefer_fill:
                empty = nearest_empty_plant_tile(
                    observation, pos=(hx, hy), claimed=claimed
                )
                if empty is not None:
                    claimed.add(empty)
                    step = _step_toward_move_index(hx, hy, empty[0], empty[1])
                    if step is not None:
                        verbs[i] = step
                        continue
        if isinstance(standing, dict) and standing.get("kind") == "WEED":
            verbs[i] = FARMER_ACTIONS["DIG"]
            continue
        if standing is None and allow_plant and (prefer_fill or not underfilled or i < fill_slots):
            verbs[i] = FARMER_ACTIONS["PLANT"]
            plant_count += 1
            seed_count -= 1
            if seed_count <= 0 or plant_count >= plant_cap:
                allow_plant = False
                need_fill = False
                underfilled = False
            continue
        # Under-filled board: dedicated fill hands walk to shed-proximal empties.
        if prefer_fill and allow_plant:
            empty = nearest_empty_plant_tile(
                observation, pos=(hx, hy), claimed=claimed
            )
            if empty is not None:
                claimed.add(empty)
                if empty == (hx, hy):
                    verbs[i] = FARMER_ACTIONS["PLANT"]
                    plant_count += 1
                    seed_count -= 1
                else:
                    step = _step_toward_move_index(hx, hy, empty[0], empty[1])
                    if step is not None:
                        verbs[i] = step
                if seed_count <= 0 or plant_count >= plant_cap:
                    allow_plant = False
                    need_fill = False
                    underfilled = False
                continue
        target = None
        unwatered = []
        mature = []
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                if (x, y) in claimed or (x == hx and y == hy):
                    continue
                dist = abs(x - hx) + abs(y - hy)
                if phase["liquidating"]:
                    if plant_is_harvestable(tile, day):
                        mature.append((dist, y, x))
                    continue
                if not tile.get("watered_today", False):
                    unwatered.append((dist, y, x))
                elif plant_is_mature(tile, day) and plant_is_harvestable(tile, day):
                    mature.append((dist, y, x))
        if unwatered:
            unwatered.sort()
            _, ty, tx = unwatered[0]
            target = (tx, ty)
        elif mature:
            mature.sort()
            _, ty, tx = mature[0]
            target = (tx, ty)
        if target is not None:
            claimed.add(target)
            step = _step_toward_move_index(hx, hy, target[0], target[1])
            if step is not None:
                verbs[i] = step
            continue
        expand = empty_neighbor_move_indices(observation, pos=(hx, hy))
        if expand and allow_plant:
            verbs[i] = expand[0]
    return verbs


def _normalize_action_op(raw_op: Any) -> str:
    if not isinstance(raw_op, str):
        return "PASS"
    if raw_op.startswith("PLANT"):
        return "PLANT"
    if raw_op.startswith("PICKUP"):
        return "PICKUP"
    if raw_op.startswith("DROP"):
        return "DROP"
    return raw_op


def _branch_action(raw_branch: Any, default: str = "PASS") -> int:
    if isinstance(raw_branch, list) and raw_branch:
        op = _normalize_action_op(raw_branch[0])
    elif isinstance(raw_branch, str):
        op = _normalize_action_op(raw_branch)
    else:
        op = default
    if op in FARMER_ACTIONS:
        return FARMER_ACTIONS[op]
    return _FARMER_ENCODE_ALIASES.get(op, FARMER_ACTIONS["OTHER"])


def encode_tiles(raw_tiles: Any) -> np.ndarray:
    """Encode official tile grid to (10, 10) int64 class map."""
    grid = np.zeros((10, 10), dtype=np.int64)
    if not raw_tiles:
        return grid
    for y in range(min(10, len(raw_tiles))):
        row = raw_tiles[y]
        for x in range(min(10, len(row))):
            tile = row[x]
            if tile is None:
                grid[y, x] = TILE_CLASS["EMPTY"]
            elif tile == "LOCKED":
                grid[y, x] = TILE_CLASS["LOCKED"]
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "WEED":
                    grid[y, x] = TILE_CLASS["WEED"]
                elif kind == "PLANT":
                    crop = tile.get("crop", "")
                    grid[y, x] = CROP_TO_TILE_CLASS.get(crop, TILE_CLASS["OTHER"])
                else:
                    grid[y, x] = TILE_CLASS["OTHER"]
            else:
                grid[y, x] = TILE_CLASS["OTHER"]
    return grid


def observation_step_index(
    obs: Dict[str, Any],
    turns_per_cycle: int = COMPETITION_TURNS_PER_DAY,
) -> int:
    """Simulation step index (0–719 for a 720-step season).

    Defaults to competition ``turnsPerDay=24`` so historical episode tapes and
    reference-ladder agents stay aligned. Pass ``TURNS_PER_CYCLE`` (72) for the
    kinematic self-play profile.
    """
    if "step" in obs and obs["step"] is not None:
        return int(obs["step"])
    day = int(obs.get("day", 1) or 1)
    hour = int(obs.get("hour", 0) or 0)
    return max(0, (day - 1) * int(turns_per_cycle) + hour)


def parse_observation(agent_result: Dict[str, Any], player_id: Optional[int] = None) -> Dict[str, Any]:
    """Extract nested observation dict from a Kaggle agent state."""
    obs = agent_result.get("observation", agent_result)
    if player_id is None:
        pid = agent_result.get("id", obs.get("player", 0))
        if isinstance(pid, str) and pid.startswith("p"):
            try:
                player_id = int(pid[1:])
            except ValueError:
                player_id = 0
        else:
            player_id = int(pid) if pid is not None else 0
    parsed = {
        "player": player_id,
        "day": obs.get("day", 0),
        "hour": obs.get("hour", 0),
        "step": observation_step_index(obs),
        "farms": obs.get("farms", []),
        "market": obs.get("market", {}),
        "private": obs.get("private", {}),
        "town": obs.get("town", {}),
    }
    return parsed


def encode_observation(
    observation: Dict[str, Any],
    player_id: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Convert raw Kaggle observation JSON to tensor dict for main DQN."""
    farms = observation.get("farms", []) or []
    private = observation.get("private", {}) or {}
    market = observation.get("market", {}) or {}
    farm = farms[player_id] if len(farms) > player_id else {}

    tiles = encode_tiles(farm.get("tiles", []))

    prices = market.get("prices", {})
    if isinstance(prices, dict):
        market_prices = [float(prices.get(crop, 0.0)) for crop in CROPS]
    elif isinstance(prices, list):
        market_prices = (list(prices[:5]) + [0.0] * 5)[:5]
    else:
        market_prices = [0.0] * 5

    inventory = market.get("inventory", {})
    if isinstance(inventory, dict):
        market_inventory = [float(inventory.get(crop, 0.0)) for crop in CROPS]
    else:
        market_inventory = [0.0] * 5

    seeds = private.get("seeds", {}) or {}
    seed_values = [float(seeds.get(crop, 0.0)) for crop in CROPS]

    shed = private.get("shed", {}) or {}
    shed_values = [float(shed.get(crop, 0.0)) for crop in CROPS]

    inv_list: List[float] = []
    inventories = private.get("inventories", []) or []
    for hand in inventories[:NUM_HANDS]:
        if isinstance(hand, dict):
            inv_list.extend(float(hand.get(crop, 0.0)) for crop in CROPS)
        else:
            inv_list.extend([0.0] * len(CROPS))
    inv_list.extend([0.0] * (30 - len(inv_list)))

    opp_money = float(farms[1 - player_id].get("money", 0.0)) if len(farms) > 1 else 0.0

    return {
        "tiles": torch.tensor(tiles, device=device, dtype=torch.long),
        "day": torch.tensor([float(observation.get("day", 0))], device=device),
        "hour": torch.tensor([float(observation.get("hour", 0))], device=device),
        "player_id": torch.tensor([float(player_id)], device=device),
        "farms_p0_money": torch.tensor([float(farm.get("money", 0.0))], device=device),
        "farms_p1_money": torch.tensor([opp_money], device=device),
        "market_prices": torch.tensor(market_prices, device=device, dtype=torch.float32),
        "market_inventory": torch.tensor(market_inventory, device=device, dtype=torch.float32),
        "seeds": torch.tensor(seed_values, device=device, dtype=torch.float32),
        "shed": torch.tensor(shed_values, device=device, dtype=torch.float32),
        "inventories": torch.tensor(inv_list[:30], device=device, dtype=torch.float32),
    }


def encode_path_b_observation(
    observation: Dict[str, Any],
    player_id: int,
) -> Dict[str, np.ndarray]:
    """Convert official observation to Path B float tiles (9,10,10) + numeric(55)."""
    farms = observation.get("farms", []) or []
    private = observation.get("private", {}) or {}
    market = observation.get("market", {}) or {}
    my_farm = farms[player_id] if len(farms) > player_id else {}
    opp_farm = farms[1 - player_id] if len(farms) > 1 - player_id else {}

    tiles_grid = np.zeros((9, 10, 10), dtype=np.float32)
    my_farmer = my_farm.get("farmer", [0, 0])
    opp_farmer = opp_farm.get("farmer", [0, 0])

    if len(my_farmer) >= 2:
        fx, fy = int(my_farmer[0]), int(my_farmer[1])
        if 0 <= fy < 10 and 0 <= fx < 10:
            tiles_grid[0, fy, fx] = 1.0
    if len(opp_farmer) >= 2:
        fx, fy = int(opp_farmer[0]), int(opp_farmer[1])
        if 0 <= fy < 10 and 0 <= fx < 10:
            tiles_grid[1, fy, fx] = 1.0

    raw_tiles = my_farm.get("tiles", [])
    for y in range(min(10, len(raw_tiles))):
        row = raw_tiles[y]
        for x in range(min(10, len(row))):
            tile = row[x]
            if isinstance(tile, dict):
                if tile.get("watered_today"):
                    tiles_grid[2, y, x] = 1.0
                kind = tile.get("kind")
                if kind == "PLANT":
                    crop = tile.get("crop")
                    if crop in CROPS:
                        tiles_grid[3 + CROPS.index(crop), y, x] = 1.0
                    progress = tile.get("yield_units", 0)
                    if isinstance(progress, (int, float)):
                        tiles_grid[8, y, x] = min(1.0, float(progress))

    numeric: List[float] = []
    numeric.append(float(observation.get("day", 0)) / float(CYCLES_PER_EPISODE))
    numeric.append(float(observation.get("hour", 0)) / float(TURNS_PER_CYCLE))
    my_money = float(my_farm.get("money", 0.0))
    opp_money = float(opp_farm.get("money", 0.0))
    numeric.extend([
        np.tanh(my_money / 1000.0),
        np.tanh(opp_money / 1000.0),
        np.tanh((my_money - opp_money) / 1000.0),
    ])
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    for crop in CROPS:
        numeric.append(float(seeds.get(crop, 0)) / 100.0)
    for crop in CROPS:
        numeric.append(float(shed.get(crop, 0)) / 100.0)
    prices = market.get("prices", {})
    inv = market.get("inventory", {})
    for crop in CROPS:
        price = prices.get(crop, 10.0) if isinstance(prices, dict) else 10.0
        qty = inv.get(crop, 0.0) if isinstance(inv, dict) else 0.0
        numeric.append(float(price) / 100.0)
        numeric.append(float(qty) / 10000.0)
    shop_types = ["BAKERY", "PIZZA", "GROCERY", "BREWERY", "MILL", "JUICE_BAR", "SALAD_BAR"]
    unlocked = observation.get("town", {}).get("unlocked_shops", []) or []
    for shop in shop_types:
        numeric.append(1.0 if shop in unlocked else 0.0)
    numeric.append(len(my_farm.get("hands", [])) / 6.0)
    numeric.append(len(opp_farm.get("hands", [])) / 6.0)
    while len(numeric) < 55:
        numeric.append(0.0)

    return {"tiles": tiles_grid, "numeric": np.array(numeric[:55], dtype=np.float32)}


def encode_action(action_raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map Kaggle command-list action to branched integer indices."""
    farmer_act = _branch_action(action_raw.get("farmer", ["PASS"]))

    hands_act: List[int] = []
    hands = action_raw.get("hands", []) or []
    for hand_idx in range(NUM_HANDS):
        if hand_idx < len(hands):
            hands_act.append(_branch_action(hands[hand_idx]))
        else:
            hands_act.append(FARMER_ACTIONS["PASS"])

    market_orders = action_raw.get("market", []) or []
    if market_orders and isinstance(market_orders[0], list) and market_orders[0]:
        market_op = market_orders[0][0]
    else:
        market_op = "PASS"
    market_act = MARKET_ACTIONS.get(market_op, 0)

    return {"farmer": farmer_act, "hands": hands_act, "market": market_act}


def encode_path_b_action(
    action_raw: Dict[str, Any],
    max_market_orders: int = 10,
) -> Dict[str, Any]:
    """Map Kaggle command-list action to Path B hierarchical indices."""
    farmer_raw = action_raw.get("farmer", ["PASS"]) or ["PASS"]
    verb_idx = _branch_action(farmer_raw)

    crop_idx = 0
    if isinstance(farmer_raw, list) and farmer_raw:
        op = _normalize_action_op(farmer_raw[0])
        if op == "PLANT" and len(farmer_raw) > 1:
            crop_name = farmer_raw[1]
            crop_idx = CROPS.index(crop_name) if crop_name in CROPS else 0

    hands_indices: List[int] = []
    hands = action_raw.get("hands", []) or []
    for hand_idx in range(NUM_HANDS):
        if hand_idx < len(hands):
            hands_indices.append(_branch_action(hands[hand_idx]))
        else:
            hands_indices.append(FARMER_ACTIONS["PASS"])

    market_indices = [MARKET_ACTIONS["PASS"]] * max_market_orders
    market_orders = action_raw.get("market", []) or []
    for i, order in enumerate(market_orders[:max_market_orders]):
        if isinstance(order, list) and order:
            op = order[0]
            market_indices[i] = MARKET_ACTIONS.get(op, MARKET_ACTIONS["PASS"])
        else:
            market_indices[i] = MARKET_ACTIONS["PASS"]

    return {
        "verb": verb_idx,
        "crop": crop_idx,
        "hands": np.array(hands_indices, dtype=np.int64),
        "market": np.array(market_indices, dtype=np.int64),
    }


def _crop_plant_counts(observation: Dict[str, Any]) -> Dict[str, int]:
    """Standing plants per crop (for staple mix balancing)."""
    counts = {c: 0 for c in CROPS}
    player = int(observation.get("player", 0) or 0)
    farms = observation.get("farms", []) or []
    farm = _farm_mapping(farms[player] if len(farms) > player else {})
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = str(tile.get("crop") or "")
                if crop in counts:
                    counts[crop] += 1
    return counts


def _best_plant_crop(observation: Dict[str, Any]) -> str:
    seeds = observation.get("private", {}).get("seeds", {}) or {}
    # Spend wheat seeds first while under-filling land; carrot only once dense.
    if seeds.get("WHEAT", 0) > 0:
        mix_on = unlocked_quadrant_count(observation) > 1
        census = farm_plant_census(observation)
        plant_cap = target_plant_count(observation)
        underfilled = int(census["plants"]) < int(plant_cap * float(FILL_RATIO_TARGET))
        if mix_on and not underfilled and seeds.get("CARROT", 0) > 0:
            counts = _crop_plant_counts(observation)
            total = sum(counts.get(c, 0) for c in STAPLE_CROPS) + 1e-6
            carrot_share = float(counts.get("CARROT", 0)) / total
            if carrot_share + 0.05 < float(STAPLE_SHARE.get("CARROT", 0.3)):
                return "CARROT"
        return "WHEAT"
    for crop in STAPLE_CROPS:
        if seeds.get(crop, 0) > 0:
            return crop
    for crop in CROPS:
        if seeds.get(crop, 0) > 0:
            return crop
    return "WHEAT"


def _best_sell_crop(observation: Dict[str, Any]) -> Optional[str]:
    shed = observation.get("private", {}).get("shed", {}) or {}
    # Sell higher-value staples first when present.
    for crop in ("TOMATO", "CARROT", "WHEAT", "STRAWBERRY", "MELON"):
        if shed.get(crop, 0) > 0:
            return crop
    for crop in CROPS:
        if shed.get(crop, 0) > 0:
            return crop
    return None


def _sell_quantity(observation: Dict[str, Any], crop: str) -> int:
    shed = observation.get("private", {}).get("shed", {}) or {}
    have = int(shed.get(crop, 0) or 0)
    return max(1, min(int(SELL_CHUNK), have))


def _best_buy_seed_crop(observation: Dict[str, Any]) -> Optional[str]:
    if not season_phase(observation)["planting"]:
        return None
    if not season_phase(observation)["investing"] and unlocked_quadrant_count(observation) <= 1:
        # Single-quadrant late season: still top up wheat lightly while planting.
        pass
    money = 0.0
    farms = observation.get("farms", [])
    player = observation.get("player", 0)
    if len(farms) > player:
        money = float(farms[player].get("money", 0.0))
    mix_on = unlocked_quadrant_count(observation) > 1 and int(
        observation.get("day", 0) or 0
    ) >= 6
    seeds = observation.get("private", {}).get("seeds", {}) or {}
    # Prefer wheat fill until NE is densely planted; then allow carrot share.
    census = farm_plant_census(observation)
    plant_cap = target_plant_count(observation)
    underfilled = int(census["plants"]) < int(plant_cap * float(FILL_RATIO_TARGET))
    if mix_on and not underfilled:
        counts = _crop_plant_counts(observation)
        seed_counts = {
            c: int((seeds or {}).get(c, 0) or 0) for c in STAPLE_CROPS
        }
        total = sum(counts.get(c, 0) for c in STAPLE_CROPS) + 1e-6
        best = None
        best_gap = -1.0
        for crop in STAPLE_CROPS:
            cost = int(SEED_COSTS.get(crop, 999))
            if money < cost:
                continue
            if seed_counts.get(crop, 0) >= 8:
                continue
            share = float(STAPLE_SHARE.get(crop, 0.0))
            gap = share - (float(counts.get(crop, 0)) / total)
            if gap > best_gap:
                best_gap = gap
                best = crop
        if best is not None:
            return best
    # Default / underfill: wheat-only expansion (Finn/Walter/Rosa + NE fill).
    if money >= SEED_COSTS.get("WHEAT", 10) and int((seeds or {}).get("WHEAT", 0) or 0) < 12:
        return "WHEAT"
    for crop in CROPS:
        if money >= SEED_COSTS.get(crop, 999):
            return crop
    return None


def _buy_seed_quantity(observation: Dict[str, Any], crop: str) -> int:
    """Buy a staple batch while expanding; never drain the hire/land reserve."""
    if not season_phase(observation)["planting"]:
        return 1
    census = farm_plant_census(observation)
    labor = farm_labor_state(observation)
    plant_cap = target_plant_count(observation)
    gap = int(plant_cap) - int(census["plants"]) - int(census["seed_count"])
    if gap <= 0:
        return 1
    unit = int(SEED_COSTS.get(crop, 10) or 10)
    # Hold land buffer only when we are actually about to expand.
    reserve = float(HIRE_CASH_RESERVE)
    bought = unlocked_quadrant_count(observation) - 1
    if (
        bought < int(LAND_BUY_TARGET)
        and bought < len(LAND_PRICES)
        and should_scale_farm(observation)
        and season_phase(observation)["investing"]
    ):
        reserve += float(LAND_CASH_BUFFER)
    spendable = max(0.0, float(labor["money"]) - reserve)
    afford = int(spendable // max(unit, 1))
    batch = int(SEED_BUY_BATCH)
    if unlocked_quadrant_count(observation) > 1 and should_scale_farm(observation):
        batch = max(batch, 10)
    return max(1, min(batch, gap, afford if afford > 0 else 1))


def sell_orders_wanted(observation: Dict[str, Any]) -> int:
    """How many SELL market slots to boost this turn."""
    census = farm_plant_census(observation)
    shed = int(census["shed_count"])
    if shed <= 0:
        return 0
    phase = season_phase(observation)
    if phase["liquidating"]:
        # Dump everything before the final bell.
        return min(int(MAX_SELL_ORDERS), max(1, (shed + int(SELL_CHUNK) - 1) // int(SELL_CHUNK)))
    slots = 1
    if shed >= 30:
        slots = 2
    if shed >= 60:
        slots = 3
    if shed >= 90 or (should_scale_farm(observation) and shed >= 40):
        slots = max(slots, 3)
    if should_scale_farm(observation) and shed >= 20:
        slots = max(slots, 2)
    return min(int(MAX_SELL_ORDERS), slots)


def decode_farmer_verb(verb_idx: int, crop_idx: int, observation: Dict[str, Any]) -> List[Any]:
    """Decode farmer verb index (+ optional crop) to Kaggle command list."""
    verb = FARMER_INDEX_TO_VERB[min(max(verb_idx, 0), NUM_FARMER_ACTIONS - 1)]
    if verb == "PLANT":
        crop = CROPS[min(max(crop_idx, 0), len(CROPS) - 1)]
        if crop not in CROPS or observation.get("private", {}).get("seeds", {}).get(crop, 0) <= 0:
            crop = _best_plant_crop(observation)
        return ["PLANT", crop]
    return [verb]


def decode_hand_verb(
    hand_idx: int,
    crop_idx: int = 0,
    observation: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Decode a hired hand the same way as the farmer (WATER/HARVEST/MOVE/PLANT)."""
    verb = FARMER_INDEX_TO_VERB[min(max(hand_idx, 0), NUM_HAND_ACTIONS - 1)]
    if verb not in _HAND_VERBS:
        return ["PASS"]
    if verb == "PLANT":
        obs = observation or {}
        crop = CROPS[min(max(int(crop_idx), 0), len(CROPS) - 1)]
        if obs.get("private", {}).get("seeds", {}).get(crop, 0) <= 0:
            crop = _best_plant_crop(obs)
        if not crop:
            return ["PASS"]
        return ["PLANT", crop]
    return [verb]


def decode_market_verb(market_idx: int, observation: Dict[str, Any]) -> List[Any]:
    """Decode single market index to one order or empty list."""
    idx = min(max(market_idx, 0), NUM_MARKET_ACTIONS - 1)
    verb = MARKET_INDEX_TO_VERB[idx]
    if verb == "PASS":
        return []
    if verb == "BUY_SEED":
        crop = _best_buy_seed_crop(observation)
        if crop:
            n = _buy_seed_quantity(observation, crop)
            return [["BUY_SEED", crop, n]]
        return []
    if verb == "SELL":
        crop = _best_sell_crop(observation)
        if crop:
            n = _sell_quantity(observation, crop)
            return [["SELL", crop, n]]
        return []
    if verb == "HIRE":
        return [["HIRE"]]
    if verb == "BUY_ANIMAL":
        return [["BUY_ANIMAL", "COW", 1]]
    if verb == "BUY_LAND":
        return [["BUY_LAND"]]
    if verb == "BUY_PRODUCT":
        return [["BUY_PRODUCT", "FERTILIZER", 1]]
    return []


def decode_action(
    action_indices: Dict[str, Any],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode branched integer action to official Kaggle command dict."""
    crop_idx = int(action_indices.get("crop", action_indices.get("action_crop", 0)))
    farmer = decode_farmer_verb(
        int(action_indices["farmer"]),
        crop_idx,
        observation,
    )

    hands_raw = action_indices.get("hands", [])
    active_hands = []
    if observation.get("farms"):
        player = observation.get("player", 0)
        farm = observation["farms"][player] if len(observation["farms"]) > player else {}
        active_hands = farm.get("hands", []) or []

    hands_out: List[List[Any]] = []
    for i in range(len(active_hands)):
        h_idx = int(hands_raw[i]) if i < len(hands_raw) else 0
        hands_out.append(decode_hand_verb(h_idx, crop_idx, observation))

    market_indices = action_indices.get("market")
    market_orders: List[List[Any]] = []
    if isinstance(market_indices, (list, np.ndarray)):
        for m_idx in market_indices:
            orders = decode_market_verb(int(m_idx), observation)
            if not orders:
                break
            market_orders.extend(orders)
    else:
        market_orders = decode_market_verb(int(market_indices or 0), observation)

    return {"farmer": farmer, "hands": hands_out, "market": market_orders}


def decode_path_b_action(
    verb_idx: int,
    crop_idx: int,
    hands_indices: List[int],
    market_indices: List[int],
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode Path B hierarchical indices via shared decode logic."""
    return decode_action(
        {
            "farmer": verb_idx,
            "crop": crop_idx,
            "hands": hands_indices,
            "market": market_indices,
        },
        observation,
    )


def get_action_masks(observation: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build action masks from nested observation using canonical indices."""
    player = observation.get("player", 0)
    farms = observation.get("farms", []) or []
    my_farm = farms[player] if len(farms) > player else {}
    private = observation.get("private", {}) or {}
    money = float(my_farm.get("money", 0.0))
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}

    farmer_mask = np.zeros(NUM_FARMER_ACTIONS, dtype=bool)
    farmer_mask[0] = True  # PASS

    tiles = my_farm.get("tiles", [])
    farmer_pos = my_farm.get("farmer", [0, 0]) or [0, 0]
    if len(farmer_pos) >= 2:
        fx, fy = int(farmer_pos[0]), int(farmer_pos[1])
    else:
        fx, fy = 0, 0

    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        tile = tiles[fy][fx]
        if isinstance(tile, dict):
            if tile.get("kind") == "WEED":
                farmer_mask[1] = True  # DIG
            if tile.get("kind") == "PLANT":
                if not tile.get("watered_today", False):
                    farmer_mask[2] = True  # WATER
                # Wheat/carrot start with yield_units=1; the engine still
                # no-ops HARVEST until first_yield_day. Mask that gap so
                # decode can water / expand instead of spamming HARVEST.
                if plant_is_harvestable(tile, int(observation.get("day", 0) or 0)):
                    farmer_mask[4] = True  # HARVEST
        elif tile is None and any(seeds.get(c, 0) > 0 for c in CROPS):
            # Engine only accepts PLANT when the standing tile is empty (None).
            farmer_mask[3] = True  # PLANT

    for dx, dy, idx in [(0, -1, 5), (0, 1, 6), (-1, 0, 7), (1, 0, 8)]:
        nx, ny = fx + dx, fy + dy
        if 0 <= nx < 10 and 0 <= ny < 10:
            if ny < len(tiles) and nx < len(tiles[ny]) and tiles[ny][nx] != "LOCKED":
                farmer_mask[idx] = True

    crop_mask = np.zeros(len(CROPS), dtype=bool)
    for i, crop in enumerate(CROPS):
        if seeds.get(crop, 0) > 0:
            crop_mask[i] = True

    market = observation.get("market", {}) or {}
    prices = market.get("prices", {}) or {}
    fert_price = float(prices.get("FERTILIZER", DEFAULT_FERTILIZER_PRICE) or DEFAULT_FERTILIZER_PRICE)

    market_mask = np.zeros(NUM_MARKET_ACTIONS, dtype=bool)
    market_mask[0] = True  # PASS
    for crop in CROPS:
        if money >= SEED_COSTS.get(crop, 999) and seeds.get(crop, 0) < SHED_CAP:
            market_mask[1] = True  # BUY_SEED
            break
    # Decoder emits BUY_PRODUCT FERTILIZER; indices 7–9 stay OTHER padding (False).
    if money >= fert_price and shed.get("FERTILIZER", 0) < SHED_CAP:
        market_mask[2] = True  # BUY_PRODUCT
    if any(shed.get(c, 0) > 0 for c in CROPS):
        market_mask[4] = True  # SELL
    if money >= ANIMAL_MIN_COST:
        market_mask[3] = True  # BUY_ANIMAL
    if money >= hire_cost_today(int(my_farm.get("hires_today", 0) or 0)):
        market_mask[5] = True  # HIRE
    unlocked = my_farm.get("unlocked_quadrants") or []
    bought = max(0, len(unlocked) - 1)
    if bought < len(LAND_PRICES) and money >= LAND_PRICES[bought]:
        market_mask[6] = True  # BUY_LAND

    return {
        "farmer_verb": farmer_mask,
        "crop_parameter": crop_mask,
        "market": market_mask,
    }


def final_bank_coins(observation: Dict[str, Any], player_id: int) -> float:
    """Return final bank coins for rubric win/loss comparison."""
    farms = observation.get("farms", []) or []
    if len(farms) > player_id:
        return float(farms[player_id].get("money", 0.0))
    return 0.0


def compare_episode_outcome(
    obs_p0: Dict[str, Any],
    obs_p1: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Return (p0_wins, p1_wins, ties) for rubric-aligned outcome."""
    m0 = final_bank_coins(obs_p0, 0)
    m1 = final_bank_coins(obs_p1, 1)
    if m0 > m1:
        return 1, 0, 0
    if m1 > m0:
        return 0, 1, 0
    return 0, 0, 1


def mlx_is_available() -> bool:
    """True when Apple MLX Metal backend is importable and available."""
    if importlib.util.find_spec("mlx.core") is None:
        return False
    try:
        mx = importlib.import_module("mlx.core")
        metal = getattr(mx, "metal", None)
        return bool(metal and metal.is_available())
    except Exception:
        return False


def resolve_training_device(device_name: str = "auto") -> torch.device:
    """Resolve PyTorch device: cuda → mps → cpu. ``mlx`` maps to mps on Apple Silicon."""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "mlx":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def gpu_backend_diagnostics() -> Dict[str, Any]:
    """Summarize CUDA, MLX, and MPS availability for notebooks and logging."""
    diag: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "mlx_available": mlx_is_available(),
        "resolved_device": str(resolve_training_device("auto")),
    }
    if diag["cuda_available"]:
        diag["cuda_device"] = torch.cuda.get_device_name(0)
    if diag["mlx_available"]:
        try:
            mx = importlib.import_module("mlx.core")
            diag["mlx_default_device"] = str(mx.default_device())
        except Exception as exc:
            diag["mlx_error"] = str(exc)
    return diag


def any_accelerator_available() -> bool:
    """True when CUDA, PyTorch MPS, or MLX Metal is available."""
    return (
        torch.cuda.is_available()
        or torch.backends.mps.is_available()
        or mlx_is_available()
    )
