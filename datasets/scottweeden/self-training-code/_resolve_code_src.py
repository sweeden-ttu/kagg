"""Resolve code dataset source for Kaggriculture training.

Extracted from the monolithic kaggriculture_self_play_training.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def _resolve_code_src(explicit: Optional[str] = None) -> Path:
    """Locate read-only Kaggle code dataset (adapter modules)."""
    if explicit:
        return Path(explicit)
    datasets_root = (
        Path("/kaggle/input/datasets")
        if Path("/kaggle/input/datasets").exists()
        else Path("~/kagg/datasets").expanduser()
    )
    for candidate in (
        Path("/kaggle/input/datasets/scottweeden/kaggriculture-self-training-code"),
        Path("/kaggle/input/kaggriculture-self-training-code"),
        datasets_root / "scottweeden" / "self-training-code",
        datasets_root / "scottweeden" / "kaggriculture-self-training-code",
        Path("~/kagg/datasets/scottweeden/self-training-code").expanduser(),
        Path("~/kagg/datasets/scottweeden/kaggriculture-self-training-code").expanduser(),
        Path(__file__).resolve().parent,
    ):
        if (candidate / "kaggriculture_adapter.py").exists():
            return candidate.resolve()
    return Path(__file__).resolve().parent
