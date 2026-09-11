#!/usr/bin/env python3
"""Repo-root wrapper for the primary BC → PPO + HER training path.

Hierarchical DQN Path B (train_self_play / train_tier*_champion) is ablation-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP = ROOT / "kaggle-mcp-server"
sys.path.insert(0, str(MCP))

from kagg_rl.train_primary import main  # noqa: E402

if __name__ == "__main__":
    main()
