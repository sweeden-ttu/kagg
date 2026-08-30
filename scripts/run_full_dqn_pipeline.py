#!/usr/bin/env python
"""Full self-play run: bootstrap + BC + 16×720 episodes + competition ladder.

Default self-play uses turnsPerDay=24 (competition / opponents ladder parity).
Pass turns_per_cycle=72 for the optional kinematic 3×4×6 profile.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "datasets/scottweeden/self-training-code"
EXP = ROOT / (
    "experiments/Hinesight Experience Replay for Double Q-Learning DQN SB3 "
    "Kaggriculture reinforcement learning"
)
sys.path.insert(0, str(CODE_SRC))

from kaggriculture_self_play_training import train_self_play  # noqa: E402
from visualize import update_experiment_plots  # noqa: E402

if __name__ == "__main__":
    train_self_play(
        experiment_dir=str(EXP),
        code_src=str(CODE_SRC),
        use_kaggle_env=True,
        seed=42,
        device_name="auto",
        total_episodes=16,
        learning_start_episodes=2,
        batch_size=32,
        checkpoint_interval=5,
        turns_per_cycle=24,
        max_episode_steps=720,
        n_eval_episodes=10,
        bootstrap_episodes=None,
        bootstrap_transitions=None,
        data_dir=str(ROOT / "working/kaggle_episodes"),
        metadata_path=str(ROOT / "working/kaggle_episodes/metadata.json"),
        download_bootstrap=False,
        bc_epochs=15,
        bc_batch_size=64,
        bc_steps_per_epoch=200,
        buffer_capacity=50_000,
        bootstrap_top_per_day=20,
        bootstrap_passes=1,
        bc_epochs_per_pass=2,
        verbose=True,
        bootstrap_mode="daily_incremental",
        bootstrap_days_per_run=1,
        publish_code_dataset=False,
        opponents_dir=str(ROOT / "opponents"),
        ladder_eval_episodes=10,
        ladder_win_rate_target=0.75,
        min_self_play_episodes=0,
        resume=None,
    )
    summary = update_experiment_plots(EXP)
    print(
        f"Plots updated: {len(summary.get('plots_written') or [])} → "
        f"{summary.get('plots_dir')}"
    )
