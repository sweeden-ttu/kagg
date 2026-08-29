"""Rubric-aligned policy evaluation: win rate vs frozen baseline.

Compares final bank coins per Rubric.md (win / loss / tie; margin irrelevant).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kaggriculture_adapter import (
    CROPS,
    FARMER_ACTIONS,
    MARKET_ACTIONS,
    NUM_HANDS,
    _best_buy_seed_crop,
    _best_plant_crop,
    _best_sell_crop,
    compare_episode_outcome,
    decode_action,
    decode_path_b_action,
    parse_observation,
)

logger = logging.getLogger(__name__)

KAGG_ROOT = Path(__file__).resolve().parent
DEFAULT_OPPONENTS_DIR = KAGG_ROOT / "opponents"
DEFAULT_OPPONENT_MANIFEST = KAGG_ROOT / "datasets" / "reference" / "agents_manifest.csv"


def load_kaggle_agent_policy(agent_path: Path | str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Load a Kaggle submission module that exposes ``agent(obs)``."""
    path = Path(agent_path)
    module_name = f"kag_opp_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load opponent agent: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "agent"):
        raise AttributeError(f"{path} has no agent(obs) entry point")
    return module.agent


def load_experiment_agent_policy(
    exp_dir: Path | str,
    *,
    repo_root: Optional[Path | str] = None,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Load experiments/<name>/agent.py (Agent class or agent() function)."""
    root = Path(exp_dir)
    agent_path = root / "agent.py"
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing {agent_path}")

    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parent
    for path in (root, repo):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    module_name = f"kag_exp_agent_{root.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "Agent"):
        instance = module.Agent()
        return lambda obs: instance.act(obs)
    if hasattr(module, "agent"):
        return module.agent
    raise AttributeError(f"{agent_path} has no Agent class or agent() function")


def discover_reference_opponents(
    opponents_dir: Optional[Path | str] = None,
) -> List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:
    """Return (slug, policy) pairs for the reference ladder, ordered by tier."""
    root = Path(opponents_dir) if opponents_dir else DEFAULT_OPPONENTS_DIR
    manifest = DEFAULT_OPPONENT_MANIFEST if DEFAULT_OPPONENT_MANIFEST.exists() else root / "agents_manifest.csv"
    entries: List[Tuple[str, int, Path]] = []

    if manifest.exists():
        with open(manifest, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                path = root / row["file"]
                if path.exists():
                    entries.append((row["agent_slug"], int(row["tier"]), path))
        entries.sort(key=lambda item: item[1])
    else:
        for path in sorted(root.glob("*.py")):
            entries.append((path.stem, 0, path))

    opponents: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = []
    for slug, _, path in entries:
        opponents.append((slug, load_kaggle_agent_policy(path)))
    return opponents


def _make_kaggle_env(max_steps: int = 720, seed: int = 42):
    import kaggle_environments

    return kaggle_environments.make(
        "kaggriculture",
        configuration={"episodeSteps": max_steps, "seed": seed},
        debug=False,
    )


def _normalize_states(result: Any) -> list:
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


def run_head_to_head_episode(
    env: Any,
    policy_p0: Callable[[Dict[str, Any]], Dict[str, Any]],
    policy_p1: Callable[[Dict[str, Any]], Dict[str, Any]],
    max_steps: int = 720,
) -> Dict[str, Any]:
    """Run one 2-player episode; policies receive nested observation dicts."""
    states = _normalize_states(env.reset())
    obs_p0 = parse_observation(states[0], player_id=0)
    obs_p1 = parse_observation(states[1], player_id=1)
    steps = 0
    done = False

    while not done and steps < max_steps:
        act_p0 = policy_p0(obs_p0)
        act_p1 = policy_p1(obs_p1)
        states = _normalize_states(env.step([act_p0, act_p1]))
        obs_p0 = parse_observation(states[0], player_id=0)
        obs_p1 = parse_observation(states[1], player_id=1)
        status = states[0].get("status", "ACTIVE")
        done = status in ("DONE", "TIMEOUT", "INVALID")
        steps += 1

    w0, w1, ties = compare_episode_outcome(obs_p0, obs_p1)
    return {
        "p0_win": w0,
        "p1_win": w1,
        "tie": ties,
        "steps": steps,
        "p0_money": obs_p0["farms"][0]["money"] if obs_p0.get("farms") else 0,
        "p1_money": obs_p1["farms"][1]["money"] if len(obs_p1.get("farms", [])) > 1 else 0,
    }


def evaluate_win_rate(
    challenger_policy: Callable[[Dict[str, Any]], Dict[str, Any]],
    baseline_policy: Callable[[Dict[str, Any]], Dict[str, Any]],
    n_episodes: int = 20,
    max_steps: int = 720,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate challenger vs baseline using rubric win/loss/tie rule."""
    wins = losses = ties = 0
    episode_details: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        try:
            env = _make_kaggle_env(max_steps=max_steps, seed=base_seed + ep)
            result = run_head_to_head_episode(
                env, challenger_policy, baseline_policy, max_steps=max_steps
            )
            if result["p0_win"]:
                wins += 1
            elif result["p1_win"]:
                losses += 1
            else:
                ties += 1
            episode_details.append(result)
        except Exception as exc:
            logger.warning("Episode %d failed: %s", ep, exc)
            continue

    n = max(len(episode_details), 1)
    return {
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "tie_rate": ties / n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_episodes": len(episode_details),
        "max_steps": max_steps,
        "episodes": episode_details,
    }


def heuristic_baseline_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based farming opponent: harvest, water, weed, plant, sell, buy seeds."""
    player = obs.get("player", 0)
    farms = obs.get("farms", []) or []
    my_farm = farms[player] if len(farms) > player else {}
    tiles = my_farm.get("tiles", []) or []
    farmer_pos = my_farm.get("farmer", [0, 0])
    fx = int(farmer_pos[0]) if farmer_pos else 0
    fy = int(farmer_pos[1]) if len(farmer_pos) > 1 else 0
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}) or {}

    verb_idx = FARMER_ACTIONS["PASS"]
    crop_idx = 0

    if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]):
        tile = tiles[fy][fx]
        if isinstance(tile, dict):
            kind = tile.get("kind", "")
            if kind == "PLANT":
                if tile.get("yield_units", 0) > 0:
                    verb_idx = FARMER_ACTIONS["HARVEST"]
                elif not tile.get("watered_today", False):
                    verb_idx = FARMER_ACTIONS["WATER"]
            elif kind == "WEED":
                verb_idx = FARMER_ACTIONS["DIG"]
            elif kind not in ("LOCKED",) and any(seeds.get(c, 0) > 0 for c in CROPS):
                verb_idx = FARMER_ACTIONS["PLANT"]
                crop_idx = CROPS.index(_best_plant_crop(obs))

    if verb_idx == FARMER_ACTIONS["PASS"] and any(seeds.get(c, 0) > 0 for c in CROPS):
        verb_idx = FARMER_ACTIONS["PLANT"]
        crop_idx = CROPS.index(_best_plant_crop(obs))

    if verb_idx == FARMER_ACTIONS["PASS"]:
        for dx, dy, move_idx in ((0, -1, 5), (0, 1, 6), (-1, 0, 7), (1, 0, 8)):
            nx, ny = fx + dx, fy + dy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                continue
            if ny >= len(tiles) or nx >= len(tiles[ny]):
                continue
            neighbor = tiles[ny][nx]
            if neighbor == "LOCKED":
                continue
            if isinstance(neighbor, dict) and neighbor.get("kind") in ("PLANT", "WEED"):
                verb_idx = move_idx
                break
            if neighbor in ("EMPTY", "") or (
                isinstance(neighbor, dict) and neighbor.get("kind") not in ("LOCKED",)
            ):
                if any(seeds.get(c, 0) > 0 for c in CROPS):
                    verb_idx = move_idx
                    break

    market_indices = [MARKET_ACTIONS["PASS"]] * 10
    if _best_sell_crop(obs):
        market_indices[0] = MARKET_ACTIONS["SELL"]
    elif sum(int(seeds.get(c, 0)) for c in CROPS) < 5 and _best_buy_seed_crop(obs):
        market_indices[0] = MARKET_ACTIONS["BUY_SEED"]

    hands_indices = [0] * NUM_HANDS
    return decode_path_b_action(verb_idx, crop_idx, hands_indices, market_indices, obs)


def random_baseline_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Random valid-ish baseline for smoke evaluation."""
    import random

    return decode_action(
        {
            "farmer": random.randint(0, 14),
            "crop": random.randint(0, 4),
            "hands": [random.randint(0, 14) for _ in range(NUM_HANDS)],
            "market": random.randint(0, 9),
        },
        obs,
    )


def build_opponent_matrix(
    opponents_dir: Optional[Path | str] = None,
    *,
    include_baselines: bool = False,
) -> List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:
    """All reference opponents, optionally plus random/heuristic baselines."""
    matrix = discover_reference_opponents(opponents_dir)
    if include_baselines:
        matrix = [
            ("random", random_baseline_policy),
            ("heuristic", heuristic_baseline_policy),
            *matrix,
        ]
    return matrix


def save_eval_report(stats: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in stats.items() if k != "episodes"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = evaluate_win_rate(random_baseline_policy, random_baseline_policy, n_episodes=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "episodes"}, indent=2))
