#!/usr/bin/env python3
"""Evaluate published Kaggriculture champion vs full reference opponent ladder."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_AGENT_DIR = ROOT / "datasets" / "published-champion"


def ensure_published_agent(agent_dir: Path) -> None:
    """Download agent bundle from code dataset if missing locally."""
    if (agent_dir / "agent.py").exists() and (agent_dir / "models" / "model.pth").exists():
        return
    agent_dir.mkdir(parents=True, exist_ok=True)
    for rel in (
        "training_artifacts/agent.py",
        "training_artifacts/models/model.pth",
        "training_artifacts/config.json",
    ):
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "scottweeden/kaggriculture-self-training-code",
                "-f", rel, "-p", str(agent_dir), "-q", "--unzip",
            ],
            check=True,
        )
    if (agent_dir / "agent.py").exists():
        return
    # Kaggle CLI may flatten paths
    for name in ("agent.py", "model.pth", "config.json"):
        src = agent_dir / name
        if name == "model.pth" and src.exists():
            dest = agent_dir / "models" / "model.pth"
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, default=DEFAULT_AGENT_DIR)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()

    ensure_published_agent(args.agent_dir)

    config_path = args.agent_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        print(
            f"Published champion: ep {cfg.get('last_completed_episode')}, "
            f"{len(cfg.get('bootstrapped_dates', []))} bootstrap days"
        )

    cmd = [
        sys.executable,
        str(ROOT / "pair_match.py"),
        "--agent-dir", str(args.agent_dir),
        "--n-episodes", str(args.n_episodes),
        "--output", str(ROOT / "datasets" / "published_eval.json"),
    ]
    if args.include_baselines:
        cmd.append("--include-baselines")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
