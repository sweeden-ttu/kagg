"""Rubric-aligned policy evaluation: win rate vs frozen baseline.

Compares final bank coins per Rubric.md (win / loss / tie; margin irrelevant).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from kaggriculture_adapter import compare_episode_outcome, decode_action, parse_observation

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    """Walk up from this module until ``opponents/`` or repo ``eval.py`` is found."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "opponents").is_dir() and (candidate / "eval.py").exists():
            return candidate
        if (candidate / "opponents").is_dir():
            return candidate
    return here.parents[3] if len(here.parents) > 3 else here.parent


DEFAULT_OPPONENTS_DIR = _repo_root() / "opponents"
DEFAULT_OPPONENT_MANIFEST = _repo_root() / "datasets" / "reference" / "agents_manifest.csv"


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


def resolve_opponents_dir(
    explicit: Optional[str | Path] = None,
    *,
    code_src: Optional[str | Path] = None,
) -> Optional[Path]:
    """Locate the reference ladder ``opponents/`` directory."""
    if explicit:
        path = Path(explicit)
        return path.resolve() if path.is_dir() else None

    candidates: List[Path] = []
    if code_src:
        candidates.append(Path(code_src) / "opponents")
    candidates.extend(
        [
            Path("/kaggle/input/opponents"),
            Path("/kaggle/input/kaggriculture-reference-agents"),
            Path("/kaggle/input/datasets/raykkretzschmar/kaggriculture-reference-agents"),
            _repo_root() / "opponents",
            Path("~/kagg/opponents").expanduser(),
        ]
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.py")):
            return candidate.resolve()
    return None


def count_reference_opponent_files(opponents_dir: Path | str) -> int:
    """Count ladder agent modules under ``opponents_dir`` (manifest or ``*.py``)."""
    root = Path(opponents_dir)
    manifest = (
        DEFAULT_OPPONENT_MANIFEST
        if DEFAULT_OPPONENT_MANIFEST.exists()
        else root / "agents_manifest.csv"
    )
    if manifest.exists():
        with open(manifest, encoding="utf-8") as fh:
            return sum(1 for row in csv.DictReader(fh) if (root / row["file"]).exists())
    return len(list(root.glob("*.py")))


def discover_reference_opponents(
    opponents_dir: Optional[Path | str] = None,
) -> List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]]:
    """Return (slug, policy) pairs for the reference ladder, ordered by tier."""
    root = Path(opponents_dir) if opponents_dir else resolve_opponents_dir(code_src=code_src)
    if root is None:
        raise FileNotFoundError("Reference opponents directory not found")
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

    return [(slug, load_kaggle_agent_policy(path)) for slug, _, path in entries]


def evaluate_ladder(
    challenger_policy: Callable[[Dict[str, Any]], Dict[str, Any]],
    *,
    opponents_dir: Optional[str | Path] = None,
    code_src: Optional[str | Path] = None,
    n_episodes: int = 10,
    max_steps: int = 720,
    base_seed: int = 42,
    win_rate_target: float = 0.5,
    turns_per_day: int = 24,
) -> Dict[str, Any]:
    """Head-to-head eval vs every reference ladder opponent.

    Defaults ``turns_per_day=24`` (competition parity) because reference agents
    hard-code a 24-turn day. Kinematic self-play (72) is a different profile.
    """
    opponents = discover_reference_opponents(opponents_dir or resolve_opponents_dir(code_src=code_src))
    results: Dict[str, Any] = {}
    beats_all = True
    wins_total = losses_total = ties_total = 0
    opponents_cleared = 0
    for slug, opp_fn in opponents:
        stats = evaluate_win_rate(
            challenger_policy,
            opp_fn,
            n_episodes=n_episodes,
            max_steps=max_steps,
            base_seed=base_seed,
            turns_per_day=turns_per_day,
        )
        summary = {k: v for k, v in stats.items() if k != "episodes"}
        if stats["episodes"]:
            summary["avg_p0_money"] = sum(e["p0_money"] for e in stats["episodes"]) / len(stats["episodes"])
            summary["avg_p1_money"] = sum(e["p1_money"] for e in stats["episodes"]) / len(stats["episodes"])
        summary["cleared"] = stats["win_rate"] >= win_rate_target
        beats_all &= summary["cleared"]
        if summary["cleared"]:
            opponents_cleared += 1
        wins_total += int(stats.get("wins", 0))
        losses_total += int(stats.get("losses", 0))
        ties_total += int(stats.get("ties", 0))
        results[slug] = summary
    n_opponents = len(opponents)
    n_episodes_total = n_opponents * n_episodes
    return {
        "opponents_dir": str(
            resolve_opponents_dir(opponents_dir, code_src=code_src) or opponents_dir or ""
        ),
        "n_episodes_per_opponent": n_episodes,
        "n_opponents": n_opponents,
        "n_episodes_total": n_episodes_total,
        "wins_total": wins_total,
        "losses_total": losses_total,
        "ties_total": ties_total,
        "opponents_cleared": opponents_cleared,
        "win_rate_target": win_rate_target,
        "beats_all_opponents": beats_all,
        "results": results,
    }


def _make_kaggle_env(max_steps: int = 720, seed: int = 42, turns_per_day: int = 24):
    import kaggle_environments

    return kaggle_environments.make(
        "kaggriculture",
        configuration={
            "episodeSteps": max_steps,
            "turnsPerDay": int(turns_per_day),
            "seed": seed,
        },
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
    turns_per_day: int = 24,
) -> Dict[str, Any]:
    """Evaluate challenger vs baseline using rubric win/loss/tie rule."""
    wins = losses = ties = 0
    episode_details: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        try:
            env = _make_kaggle_env(
                max_steps=max_steps,
                seed=base_seed + ep,
                turns_per_day=turns_per_day,
            )
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
        "turns_per_day": turns_per_day,
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
