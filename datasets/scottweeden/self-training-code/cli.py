"""CLI entry point for Kaggriculture self-play training.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides main() — the argparse entry point that delegates to train_self_play().
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from train_orchestrator import train_self_play
from checkpoints import resolve_resume_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggriculture hierarchical DQN self-play training (Path B rebuild)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/self_play",
        help="Experiment output directory (default: experiments/self_play)",
    )
    parser.add_argument("--total-episodes", type=int, default=15)
    parser.add_argument("--learning-start-episodes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps", "mlx"],
        default="auto",
    )
    parser.add_argument(
        "--use-kaggle-env",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the official Kaggle simulator (default: on). Pass --no-use-kaggle-env to refuse.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument(
        "--turns-per-cycle",
        type=int,
        default=24,
        help=(
            "Engine turnsPerDay for self-play. Default 24 = competition / ladder parity "
            "(30×24=720). Optional kinematic profile: 72 (3×4×6, 10 cycles)."
        ),
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help=(
            "Episodes per league opponent when --ladder-eval-episodes is 0. "
            "Never runs a random baseline; requires opponents/."
        ),
    )
    parser.add_argument(
        "--bootstrap-episodes",
        type=int,
        default=None,
        help="Cap episode JSONs during bootstrap (omit=all catalog; 0=skip bootstrap)",
    )
    parser.add_argument(
        "--bootstrap-transitions",
        type=int,
        default=50_000,
        help="Maximum expert transitions to load during bootstrap",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="working/kaggle_episodes",
        help="Directory containing episodes/*.json for bootstrap",
    )
    parser.add_argument(
        "--download-bootstrap",
        action="store_true",
        help="Download Kaggle episodes dataset before bootstrap",
    )
    parser.add_argument(
        "--bootstrap-passes",
        type=int,
        default=1,
        help="Shuffle-fill-BC passes over corpus (>1 enables streaming bootstrap)",
    )
    parser.add_argument(
        "--bc-epochs-per-pass",
        type=int,
        default=2,
        help="BC stream epochs per bootstrap day/pass (default: 2)",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=15,
        help="Behavioral cloning pretrain epochs on bootstrapped buffer (0=skip)",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=64,
        help="Batch size for BC pretrain",
    )
    parser.add_argument(
        "--bc-steps-per-epoch",
        type=int,
        default=None,
        help="Cap BC gradient steps per epoch (default: buffer // batch_size)",
    )
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=10_000,
        help="Replay buffer capacity",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Merged metadata.json for score-ranked bootstrap ordering",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume training from an experiment directory or checkpoint file. "
            "Prefers checkpoints/training_state_latest.pt (full state). "
            "--total-episodes is the cumulative target episode count."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging for bootstrap, BC, and self-play steps",
    )
    parser.add_argument(
        "--code-src",
        type=str,
        default=None,
        help=(
            "Read-only code dataset root for kaggriculture_adapter.py "
            "(default: /kaggle/input/datasets/scottweeden/kaggriculture-self-training-code)"
        ),
    )
    parser.add_argument(
        "--bootstrap-mode",
        type=str,
        choices=["streaming", "daily_incremental"],
        default="daily_incremental",
        help=(
            "Bootstrap strategy: daily_incremental (next chronological unseen days, "
            "persisted in bootstrap_state.json) or streaming (one calendar day per pass)"
        ),
    )
    parser.add_argument(
        "--bootstrap-days-per-run",
        type=int,
        default=3,
        help="Days to bootstrap per run when --bootstrap-mode daily_incremental",
    )
    parser.add_argument(
        "--publish-code-dataset",
        action="store_true",
        help="After bootstrapping new days, publish model artifacts to code dataset via Kaggle CLI",
    )
    parser.add_argument(
        "--opponents-dir",
        type=str,
        default=None,
        help="Reference ladder opponents/ directory for post-training ladder eval",
    )
    parser.add_argument(
        "--ladder-eval-episodes",
        type=int,
        default=0,
        help=(
            "Head-to-head episodes per reference opponent after training. "
            "0 uses --n-eval-episodes (does not skip league eval)"
        ),
    )
    parser.add_argument(
        "--ladder-win-rate-target",
        type=float,
        default=0.75,
        help="Win rate threshold counted as clearing an opponent in ladder eval",
    )
    parser.add_argument(
        "--min-self-play-episodes",
        type=int,
        default=0,
        help="When resuming past total_episodes, run at least this many additional self-play episodes",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if args.resume:
        exp_root, _, _ = resolve_resume_path(args.resume)
        if Path(args.experiment_dir).resolve() != exp_root.resolve():
            if args.experiment_dir != "experiments/self_play":
                logging.warning(
                    "Both --experiment-dir and --resume given; using resume directory %s",
                    exp_root,
                )
            experiment_dir = str(exp_root)

    train_self_play(
        total_episodes=args.total_episodes,
        learning_start_episodes=args.learning_start_episodes,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        experiment_dir=experiment_dir,
        seed=args.seed,
        device_name=args.device,
        use_kaggle_env=args.use_kaggle_env,
        max_episode_steps=args.max_episode_steps,
        turns_per_cycle=args.turns_per_cycle,
        n_eval_episodes=args.n_eval_episodes,
        resume=args.resume,
        bootstrap_episodes=args.bootstrap_episodes,
        bootstrap_transitions=args.bootstrap_transitions,
        data_dir=args.data_dir,
        download_bootstrap=args.download_bootstrap,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_steps_per_epoch=args.bc_steps_per_epoch,
        buffer_capacity=args.buffer_capacity,
        metadata_path=args.metadata_path,
        bootstrap_passes=args.bootstrap_passes,
        bc_epochs_per_pass=args.bc_epochs_per_pass,
        verbose=args.verbose,
        code_src=args.code_src,
        bootstrap_mode=args.bootstrap_mode,
        bootstrap_days_per_run=args.bootstrap_days_per_run,
        publish_code_dataset=args.publish_code_dataset,
        opponents_dir=args.opponents_dir,
        ladder_eval_episodes=args.ladder_eval_episodes,
        ladder_win_rate_target=args.ladder_win_rate_target,
        min_self_play_episodes=args.min_self_play_episodes,
    )


if __name__ == "__main__":
    main()
