#!/usr/bin/env python3
"""Evaluate every experiments/*/agent.py vs the full reference opponent ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import (
    build_opponent_matrix,
    evaluate_win_rate,
    load_experiment_agent_policy,
)


def discover_experiments(experiments_root: Path) -> List[Path]:
    if not experiments_root.is_dir():
        return []
    return sorted(
        p for p in experiments_root.iterdir()
        if p.is_dir() and (p / "agent.py").exists()
    )


def load_config(exp_dir: Path) -> Dict[str, Any]:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as fh:
        return json.load(fh)


def summarize_stats(
    experiment: str,
    opponent: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    episodes = stats.get("episodes") or []
    avg_p0 = avg_p1 = None
    if episodes:
        avg_p0 = sum(e["p0_money"] for e in episodes) / len(episodes)
        avg_p1 = sum(e["p1_money"] for e in episodes) / len(episodes)
    return {
        "experiment": experiment,
        "opponent": opponent,
        "win_rate": stats["win_rate"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "ties": stats["ties"],
        "n_episodes": stats["n_episodes"],
        "avg_p0_money": avg_p0,
        "avg_p1_money": avg_p1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-eval experiment agents vs all reference opponents"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=ROOT / "datasets" / "experiments",
    )
    parser.add_argument(
        "--extra-agent-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional agent bundle (e.g. .tmp/published-eval)",
    )
    parser.add_argument(
        "--opponents-dir",
        type=Path,
        default=ROOT / "opponents",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also eval vs random and heuristic (not part of reference ladder)",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "eval_summary.json",
    )
    args = parser.parse_args()

    opponents = build_opponent_matrix(
        args.opponents_dir,
        include_baselines=args.include_baselines,
    )
    if not opponents:
        print(f"No opponents found under {args.opponents_dir}")
        sys.exit(1)

    agent_dirs: List[tuple[str, Path]] = []
    for exp_dir in discover_experiments(args.experiments_dir):
        agent_dirs.append((exp_dir.name, exp_dir))
    for extra in args.extra_agent_dir:
        if (extra / "agent.py").exists():
            agent_dirs.append((extra.name, extra))

    if not agent_dirs:
        print(f"No agent.py found under {args.experiments_dir}")
        sys.exit(1)

    results: List[Dict[str, Any]] = []
    print(
        f"=== Batch eval: {len(agent_dirs)} agent(s) vs {len(opponents)} opponent(s), "
        f"{args.n_episodes} eps each ===\n"
    )

    for name, exp_dir in agent_dirs:
        cfg = load_config(exp_dir)
        print(f"--- {name} ---")
        if cfg.get("last_completed_episode") is not None:
            print(f"  last_completed_episode={cfg['last_completed_episode']}")
        try:
            policy = load_experiment_agent_policy(exp_dir, repo_root=ROOT)
        except Exception as exc:
            print(f"  SKIP load failed: {exc}\n")
            results.append({"experiment": name, "error": str(exc)})
            continue

        for opp_name, opp_fn in opponents:
            stats = evaluate_win_rate(
                policy,
                opp_fn,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                base_seed=args.seed,
            )
            row = summarize_stats(name, opp_name, stats)
            results.append(row)
            print(
                f"  vs {opp_name:16s}: "
                f"{row['wins']}/{row['n_episodes']} ({row['win_rate']:.0%})  "
                f"money {row['avg_p0_money']:.0f} vs {row['avg_p1_money']:.0f}"
            )
        print()

    payload = {
        "n_episodes": args.n_episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "opponents": [name for name, _ in opponents],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
