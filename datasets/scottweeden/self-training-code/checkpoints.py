"""Checkpoint and resume logic for Kaggriculture self-play training.

Extracted from the monolithic kaggriculture_self_play_training.py.
Provides all checkpoint/resume/load functions.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import PrioritizedReplayBuffer
from agent_coordinator import SelfPlayCoordinator
from kaggriculture_path_b_rebuild import HierarchicalDoubleDQNLearner


TRAINING_STATE_FILENAME = "training_state_latest.pt"
_CHECKPOINT_EP_PATTERN = re.compile(r"checkpoint_ep_(\d+)\.pt$")
_GZIP_MAGIC = b"\x1f\x8b"


def _open_state_file(path: Path, mode: str = "rb"):
    """Open a checkpoint file, transparently wrapping gzip-compressed files.

    torch.save/torch.load work on file-like objects, so both plain and
    gzip'd checkpoints are supported regardless of file extension.
    """
    fh = open(path, mode)
    if mode == "rb":
        magic = fh.read(2)
        fh.seek(0)
        if magic == _GZIP_MAGIC:
            fh.close()
            return gzip.GzipFile(path, "rb")
    return fh


def _load_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _training_state_path(experiment_root: Path) -> Path:
    return experiment_root / "checkpoints" / TRAINING_STATE_FILENAME


def _episode_from_checkpoint_name(path: Path) -> int:
    match = _CHECKPOINT_EP_PATTERN.match(path.name)
    if match:
        return int(match.group(1))
    return 0


def _load_episode_metrics(experiment_root: Path) -> List[Dict[str, float]]:
    metrics_path = experiment_root / "metrics" / "episode_metrics.json"
    if not metrics_path.exists():
        return []
    try:
        with open(metrics_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return list(data.get("episodes", []))
    except (json.JSONDecodeError, OSError):
        return []


def resolve_resume_path(resume: str) -> Tuple[Path, Path, str]:
    """Resolve --resume to (experiment_dir, checkpoint_file, kind).

    kind is ``full`` for training_state_latest.pt or ``weights`` for weight-only files.
    """
    path = Path(resume).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path not found: {path}")

    if path.is_dir():
        exp_dir = path
        state_file = _training_state_path(exp_dir)
        if state_file.exists():
            return exp_dir, state_file, "full"
        weight_files = sorted(
            exp_dir.glob("checkpoints/checkpoint_ep_*.pt"),
            key=_episode_from_checkpoint_name,
        )
        if weight_files:
            return exp_dir, weight_files[-1], "weights"
        model_file = exp_dir / "models" / "model.pth"
        if model_file.exists():
            return exp_dir, model_file, "weights"
        raise FileNotFoundError(
            f"No resume checkpoint in {exp_dir}. Expected "
            f"checkpoints/{TRAINING_STATE_FILENAME} or checkpoints/checkpoint_ep_*.pt"
        )

    if path.name == TRAINING_STATE_FILENAME:
        exp_dir = path.parent.parent
        return exp_dir, path, "full"
    if _CHECKPOINT_EP_PATTERN.match(path.name):
        exp_dir = path.parent.parent
        return exp_dir, path, "weights"
    if path.name == "model.pth":
        exp_dir = path.parent.parent
        return exp_dir, path, "weights"

    raise ValueError(
        f"Unsupported resume file: {path}. Use an experiment directory, "
        f"{TRAINING_STATE_FILENAME}, checkpoint_ep_N.pt, or model.pth"
    )


def save_training_state(
    path: Path,
    *,
    last_completed_episode: int,
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    coordinator: SelfPlayCoordinator,
    episode_metrics: List[Dict[str, float]],
    config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng_state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    payload = {
        "version": 1,
        "last_completed_episode": last_completed_episode,
        "online_net": online_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "buffer": buffer.state_dict(),
        "opponent_pool": list(coordinator.opponent_pool),
        "opponent_select_i": int(coordinator._opponent_select_i),
        "current_tier_idx": getattr(coordinator, "current_tier_idx", 0),
        "tier_history": getattr(coordinator, "tier_history", {}),
        "episode_metrics": episode_metrics,
        "config": config,
        "rng_state": rng_state,
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(payload, path)


def _coerce_cpu_byte_rng_state(state: Any) -> torch.Tensor:
    """Restore torch RNG state saved across devices / PyTorch versions."""
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().to(dtype=torch.uint8)
    return torch.as_tensor(state, dtype=torch.uint8)


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if not rng_state:
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(_coerce_cpu_byte_rng_state(rng_state["torch"]))
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            cuda_states = [
                _coerce_cpu_byte_rng_state(s) for s in rng_state["torch_cuda"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
    except (TypeError, ValueError) as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("RNG state restore skipped (non-fatal): %s", exc)


def load_training_state(
    path: Path,
    device: torch.device,
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: optim.Optimizer,
    buffer: PrioritizedReplayBuffer,
    coordinator: SelfPlayCoordinator,
) -> Tuple[int, List[Dict[str, float]], Dict[str, Any]]:
    """Load full training state; returns (last_completed_episode, metrics, config)."""
    with _open_state_file(path) as fh:
        payload = torch.load(fh, map_location=device, weights_only=False)
    online_net.load_state_dict(payload["online_net"])
    target_net.load_state_dict(payload["target_net"])
    optimizer.load_state_dict(payload["optimizer"])
    buffer.load_state_dict(payload["buffer"])
    coordinator.restore_opponent_pool(payload.get("opponent_pool"))
    coordinator._opponent_select_i = int(payload.get("opponent_select_i", 0))
    if "current_tier_idx" in payload:
        coordinator.current_tier_idx = int(payload["current_tier_idx"])
    if "tier_history" in payload and isinstance(payload["tier_history"], dict):
        coordinator.tier_history = payload["tier_history"]
    _restore_rng_state(payload.get("rng_state", {}))

    return (
        int(payload["last_completed_episode"]),
        list(payload.get("episode_metrics", [])),
        dict(payload.get("config", {})),
    )


def load_weights_checkpoint(
    path: Path,
    device: torch.device,
    online_net: nn.Module,
    target_net: nn.Module,
    learner: HierarchicalDoubleDQNLearner,
    coordinator: SelfPlayCoordinator,
    experiment_root: Path,
) -> Tuple[int, List[Dict[str, float]]]:
    """Load weights-only checkpoint (legacy or model.pth)."""
    online_net.load_state_dict(_load_state_dict(path, device))
    target_net.load_state_dict(online_net.state_dict())
    learner.target.load_state_dict(online_net.state_dict())
    coordinator.restore_opponent_pool()

    last_episode = _episode_from_checkpoint_name(path)
    if last_episode == 0 and path.name == "model.pth":
        metrics = _load_episode_metrics(experiment_root)
        if metrics:
            last_episode = int(metrics[-1].get("episode", len(metrics)))
    return last_episode, _load_episode_metrics(experiment_root)
