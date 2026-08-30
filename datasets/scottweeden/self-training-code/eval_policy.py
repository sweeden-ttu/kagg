"""Rubric-aligned policy evaluation: win rate vs frozen baseline.

Compares final bank coins per Rubric.md (win / loss / tie; margin irrelevant).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from kaggriculture_adapter import compare_episode_outcome, decode_action, parse_observation

logger = logging.getLogger(__name__)


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


def random_baseline_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Random valid-ish baseline for smoke evaluation."""
    import random
    from kaggriculture_adapter import NUM_HANDS

    return decode_action(
        {
            "farmer": random.randint(0, 14),
            "crop": random.randint(0, 4),
            "hands": [random.randint(0, 14) for _ in range(NUM_HANDS)],
            "market": random.randint(0, 9),
        },
        obs,
    )


def save_eval_report(stats: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {k: v for k, v in stats.items() if k != "episodes"}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = evaluate_win_rate(random_baseline_policy, random_baseline_policy, n_episodes=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "episodes"}, indent=2))
