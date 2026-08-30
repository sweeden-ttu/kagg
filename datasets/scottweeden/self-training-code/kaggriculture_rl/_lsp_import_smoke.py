"""Smoke imports for basedpyright / Cursor LSP (matches ImitationLearning.ipynb)."""
from __future__ import annotations

import torch.optim as optim
from gymnasium import spaces
from kaggle_environments import make
from stable_baselines3.common.buffers import ReplayBuffer

__all__ = ["optim", "spaces", "make", "ReplayBuffer"]
