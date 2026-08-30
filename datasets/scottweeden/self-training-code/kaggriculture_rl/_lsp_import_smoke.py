"""Smoke imports for basedpyright / Cursor LSP (legacy branched DQN notebook)."""
from __future__ import annotations

import torch.nn.functional as F
import torch.optim as optim

from dataset_loader import parse_kaggriculture_episode
from kaggle_env_wrapper import KaggleEnvWrapper
from kaggriculture_rl.dqn import (
    DuelingDoubleDQNBranching,
    KaggricultureFeatureExtractor,
    ReplayBuffer,
)
from kaggriculture_rl.dqn_sb3 import DQN

__all__ = [
    "F",
    "optim",
    "parse_kaggriculture_episode",
    "KaggleEnvWrapper",
    "DuelingDoubleDQNBranching",
    "KaggricultureFeatureExtractor",
    "ReplayBuffer",
    "DQN",
]
