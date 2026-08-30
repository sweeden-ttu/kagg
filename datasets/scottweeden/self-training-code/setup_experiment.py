"""Experiment directory setup.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides setup_experiment_dirs().
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def setup_experiment_dirs(experiment_dir: Path) -> Dict[str, Path]:
    """Create experiment output layout matching train.py."""
    experiment_dir = Path(experiment_dir)
    subdirs = {
        "root": experiment_dir,
        "models": experiment_dir / "models",
        "checkpoints": experiment_dir / "checkpoints",
        "logs": experiment_dir / "logs",
        "metrics": experiment_dir / "metrics",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs
