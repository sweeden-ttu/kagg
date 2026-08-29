#!/usr/bin/env python3
"""Evaluate one agent against the full reference opponent ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval import (
    build_opponent_matrix,
    evaluate_win_rate,
    load_experiment_agent_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair one challenger vs all Kaggriculture reference opponents"
    )
    parser.add_argument(
        "--agent-dir",
        type=Path,
        required=True,
        help="Directory containing agent.py (+ models/model.pth)",
    )
    parser.add_argument(
        "--opponents-dir",
        type=Path,
        default=ROOT / "opponents",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also eval vs random and heuristic",
    )
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: <agent-dir>/opponent_eval.json)",
    )
    args = parser.parse_args()

    opponents = build_opponent_matrix(
        args.opponents_dir,
        include_baselines=args.include_baselines,
    )
    policy = load_experiment_agent_policy(args.agent_dir, repo_root=ROOT)

    print(f"=== {args.agent_dir.name} vs {len(opponents)} opponent(s) ===\n")
    results = {}
    for opp_name, opp_fn in opponents:
        stats = evaluate_win_rate(
            policy,
            opp_fn,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            base_seed=args.seed,
        )
        results[opp_name] = {k: v for k, v in stats.items() if k != "episodes"}
        avg_p0 = avg_p1 = 0.0
        if stats["episodes"]:
            avg_p0 = sum(e["p0_money"] for e in stats["episodes"]) / len(stats["episodes"])
            avg_p1 = sum(e["p1_money"] for e in stats["episodes"]) / len(stats["episodes"])
        print(
            f"vs {opp_name:16s}: "
            f"{stats['wins']}/{stats['n_episodes']} ({stats['win_rate']:.0%})  "
            f"money {avg_p0:.0f} vs {avg_p1:.0f}"
        )

    out = args.output or (args.agent_dir / "opponent_eval.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "agent_dir": str(args.agent_dir),
                "n_episodes": args.n_episodes,
                "opponents": [name for name, _ in opponents],
                "results": results,
            },
            fh,
            indent=2,
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
