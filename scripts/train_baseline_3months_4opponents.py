#!/usr/bin/env python3
"""CLI wrapper to run 3-month 4-opponent baseline training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reasoning_vs_questioning"))
sys.path.insert(0, str(ROOT))

from train_baseline_3months_4opponents import main

if __name__ == "__main__":
    main()
