#!/usr/bin/env python3
"""CLI wrapper to run 6-month 8-opponent QKD model training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reasoning_vs_questioning"))
sys.path.insert(0, str(ROOT))

from train_qkd_model_6months_8opponents import main

if __name__ == "__main__":
    main()
