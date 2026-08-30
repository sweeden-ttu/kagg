"""CLI entry for the legacy branched DQN stack.

Canonical training is Path B::

    python -m kaggriculture_self_play_training

This module only smoke-trains ``kaggriculture_rl.dqn_sb3.DQN`` against
``KaggleEnvWrapper``.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy branched DQN smoke trainer. "
            "Prefer Path B: python -m kaggriculture_self_play_training"
        )
    )
    parser.add_argument(
        "--learn-steps",
        type=int,
        default=8,
        help="Number of DQN.learn timesteps (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    args = parser.parse_args(argv)

    print(
        "Note: Path B (kaggriculture_self_play_training.train_self_play) "
        "is the canonical trainer. This entry point only exercises the "
        "legacy branched DQN + KaggleEnvWrapper stack."
    )

    from kaggle_env_wrapper import KaggleEnvWrapper
    from kaggriculture_rl.dqn_sb3 import DQN

    env = KaggleEnvWrapper.make(device=args.device, use_masking=True)
    model = DQN(
        "KaggricultureCNN",
        env,
        device=args.device,
        learning_starts=0,
        buffer_size=10_000,
        batch_size=4,
        train_freq=1,
        verbose=1,
        learning_rate=1e-4,
    )
    model.learn(total_timesteps=max(1, args.learn_steps), log_interval=1)
    print(f"Completed {args.learn_steps} learn steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
