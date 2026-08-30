"""Dual-partition Prioritized Experience Replay buffer with index-safe sampling.

FIX for §2.4 index corruption: The original tagged-index scheme used
np.array of tuples which can produce shape mismatches when one partition
is empty. We fix this by using explicit partitioned updates — each
partition's indices are returned separately and updated independently,
eliminating any possibility of cross-partition corruption.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

SOURCE_BOOTSTRAP = "bootstrap"
SOURCE_SELFPLAY = "selfplay"

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """Dual-partition PER buffer: 50% past-gameplay bootstrap, 50% self-play.

    Each partition is an independent circular buffer with its own priority array.
    Sampling always draws from both partitions proportionally.

    Index-safety fix (v2): Instead of flattening tagged indices into a single
    np.array of tuples (which can produce shape mismatches when one partition
    is empty), we build tagged indices *after* sampling each partition
    independently. This ensures neither partition's indices ever contaminate
    the other's priority array.
    """

    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        bootstrap_fraction: float = 0.5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_capacity = max(1, int(capacity * bootstrap_fraction))
        self.selfplay_capacity = max(1, capacity - self.bootstrap_capacity)
        self._init_partition(SOURCE_BOOTSTRAP)
        self._init_partition(SOURCE_SELFPLAY)

    def _init_partition(self, source: str) -> None:
        """Initialize a partition's list buffer, circular position, and priority array."""
        if source == SOURCE_BOOTSTRAP:
            self.bootstrap_buffer: List[tuple] = []
            self.bootstrap_pos: int = 0
            self.bootstrap_priorities: np.ndarray = np.zeros(self.bootstrap_capacity, dtype=np.float32)
        else:
            self.selfplay_buffer: List[tuple] = []
            self.selfplay_pos: int = 0
            self.selfplay_priorities: np.ndarray = np.zeros(self.selfplay_capacity, dtype=np.float32)

    def _partition(self, source: str):
        """Return (buffer_list, position_attr_name, priorities_array, capacity_int)."""
        if source == SOURCE_BOOTSTRAP:
            return (
                self.bootstrap_buffer,
                "bootstrap_pos",
                self.bootstrap_priorities,
                self.bootstrap_capacity,
            )
        return (
            self.selfplay_buffer,
            "selfplay_pos",
            self.selfplay_priorities,
            self.selfplay_capacity,
        )

    def push(
        self,
        tiles: np.ndarray,
        numeric: np.ndarray,
        action_verb: int,
        action_crop: int,
        action_hands: np.ndarray,
        action_market: np.ndarray,
        reward: float,
        next_tiles: np.ndarray,
        next_numeric: np.ndarray,
        done: bool,
        source: str = SOURCE_SELFPLAY,
    ) -> None:
        """Push a transition into the specified partition."""
        if source not in (SOURCE_BOOTSTRAP, SOURCE_SELFPLAY):
            raise ValueError(
                f"source must be {SOURCE_BOOTSTRAP!r} or {SOURCE_SELFPLAY!r}, got {source!r}"
            )

        buf, pos_attr, prios, cap = self._partition(source)
        transition = (
            tiles, numeric, action_verb, action_crop, action_hands,
            action_market, reward, next_tiles, next_numeric, done,
        )

        # Determine effective priority (use max of existing or 1.0)
        current_prios = prios[: len(buf)]
        max_prio = float(current_prios.max()) if len(current_prios) > 0 else 1.0

        if len(buf) < cap:
            buf.append(transition)
            slot = len(buf) - 1
        else:
            pos = getattr(self, pos_attr)
            buf[pos] = transition
            slot = pos
            setattr(self, pos_attr, (pos + 1) % cap)

        prios[slot] = max_prio

    def _sample_from_partition(
        self,
        source: str,
        batch_size: int,
        beta: float,
    ) -> Tuple[List[tuple], List[int], np.ndarray]:
        """Sample from a single partition.

        Returns (samples, local_indices, importance_weights).
        local_indices are plain ints — not tagged — so update_priorities
        can address each partition independently.
        """
        buf, _, prios, _ = self._partition(source)
        if not buf or batch_size <= 0:
            return [], [], np.array([], dtype=np.float32)

        n = min(batch_size, len(buf))
        prios_slice = prios[: len(buf)].astype(np.float64)
        probs = prios_slice ** self.alpha
        probs += 1e-10  # prevent zero-probability transitions
        probs /= probs.sum()

        local_indices = np.random.choice(len(buf), n, p=probs)
        samples = [buf[int(i)] for i in local_indices]
        weights = (len(buf) * probs[local_indices]) ** (-beta)
        weights /= weights.max()
        return samples, list(local_indices), weights.astype(np.float32)

    def _batch_from_samples(
        self, samples: List[tuple], weights: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Collate a list of transitions into a tensor batch dict."""
        tiles_b, numeric_b, act_v_b, act_c_b, act_h_b, act_m_b, r_b, n_tiles_b, n_num_b, d_b = zip(*samples)
        return {
            "tiles": torch.as_tensor(np.array(tiles_b), dtype=torch.float32),
            "numeric": torch.as_tensor(np.array(numeric_b), dtype=torch.float32),
            "action_verb": torch.as_tensor(act_v_b, dtype=torch.long),
            "action_crop": torch.as_tensor(act_c_b, dtype=torch.long),
            "action_hands": torch.as_tensor(np.array(act_h_b), dtype=torch.long),
            "action_market": torch.as_tensor(np.array(act_m_b), dtype=torch.long),
            "reward": torch.as_tensor(r_b, dtype=torch.float32),
            "next_tiles": torch.as_tensor(np.array(n_tiles_b), dtype=torch.float32),
            "next_numeric": torch.as_tensor(np.array(n_num_b), dtype=torch.float32),
            "done": torch.as_tensor(d_b, dtype=torch.float32),
            "weights": torch.as_tensor(weights, dtype=torch.float32),
        }

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample a batch, splitting evenly between bootstrap and self-play.

        Returns (batch_dict, indices_for_update, importance_weights).

        The indices_for_update is an object-array of (source_flag, local_idx)
        pairs. Each partition is sampled *independently* and tagged only after
        sampling, so neither partition's indices ever contaminate the other's
        priority array — this is the core fix for §2.4.
        """
        n_bootstrap = batch_size // 2
        n_selfplay = batch_size - n_bootstrap

        b_samples, b_indices, b_weights = self._sample_from_partition(
            SOURCE_BOOTSTRAP, n_bootstrap, beta
        )
        s_samples, s_indices, s_weights = self._sample_from_partition(
            SOURCE_SELFPLAY, n_selfplay, beta
        )

        if not b_samples and not s_samples:
            return {}, np.array([]), np.array([])

        # Combine samples
        samples = b_samples + s_samples
        combined_weights = np.concatenate([b_weights, s_weights]) if b_weights.size > 0 and s_weights.size > 0 else (b_weights if b_weights.size > 0 else s_weights)
        batch = self._batch_from_samples(samples, combined_weights)

        # Build tagged indices — each partition gets only its own source flag
        # This avoids the shape-mismatch bug when one partition is empty.
        tagged: List[tuple] = []
        for i in b_indices:
            tagged.append((0, i))
        for i in s_indices:
            tagged.append((1, i))

        indices_arr = np.empty(len(tagged), dtype=object)
        for idx_val, (sf, li) in enumerate(tagged):
            indices_arr[idx_val] = (sf, li)

        return batch, indices_arr, combined_weights

    def sample_uniform(
        self, batch_size: int, source: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """Uniform sample for BC — defaults to bootstrap partition."""
        src = source or SOURCE_BOOTSTRAP
        buf, _, _, _ = self._partition(src)
        if not buf:
            return {}
        n = min(batch_size, len(buf))
        indices = np.random.choice(len(buf), n, replace=False)
        samples = [buf[int(i)] for i in indices]
        weights = np.ones(n, dtype=np.float32)
        return self._batch_from_samples(samples, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions.

        FIX: Each index is (source_flag, local_idx). We look up only the
        corresponding partition's priority array, eliminating cross-talk.
        We process transitions one-by-one instead of batch-assigning to avoid
        any array-shape confusion.
        """
        indices = np.asarray(indices)
        priorities = np.asarray(priorities).reshape(-1)

        if indices.size == 0 or priorities.size == 0:
            return

        for idx_val in range(min(indices.size, priorities.size)):
            item = indices[idx_val]
            prio = priorities[idx_val]
            try:
                source_flag, local_idx = int(item[0]), int(item[1])
            except (TypeError, IndexError, ValueError):
                continue

            source = SOURCE_BOOTSTRAP if source_flag == 0 else SOURCE_SELFPLAY
            _, _, prios, _ = self._partition(source)
            if 0 <= local_idx < len(prios):
                prios[local_idx] = max(float(prio), 1e-6)

    @property
    def bootstrap_size(self) -> int:
        return len(self.bootstrap_buffer)

    @property
    def selfplay_size(self) -> int:
        return len(self.selfplay_buffer)

    def __len__(self) -> int:
        return len(self.bootstrap_buffer) + len(self.selfplay_buffer)

    def clear(self, source: Optional[str] = None) -> None:
        """Clear one or both partitions."""
        if source is None:
            self._init_partition(SOURCE_BOOTSTRAP)
            self._init_partition(SOURCE_SELFPLAY)
        elif source == SOURCE_BOOTSTRAP:
            self._init_partition(SOURCE_BOOTSTRAP)
        elif source == SOURCE_SELFPLAY:
            self._init_partition(SOURCE_SELFPLAY)
        else:
            raise ValueError(f"unknown source: {source!r}")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "bootstrap_fraction": self.bootstrap_fraction,
            "bootstrap_capacity": self.bootstrap_capacity,
            "selfplay_capacity": self.selfplay_capacity,
            "bootstrap_pos": self.bootstrap_pos,
            "selfplay_pos": self.selfplay_pos,
            "bootstrap_buffer": self.bootstrap_buffer,
            "selfplay_buffer": self.selfplay_buffer,
            "bootstrap_priorities": self.bootstrap_priorities[: len(self.bootstrap_buffer)].copy(),
            "selfplay_priorities": self.selfplay_priorities[: len(self.selfplay_buffer)].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore buffer state (supports legacy single-buffer checkpoints)."""
        if "bootstrap_buffer" in state:
            self.capacity = int(state["capacity"])
            self.alpha = float(state["alpha"])
            self.bootstrap_fraction = float(state.get("bootstrap_fraction", 0.5))
            self.bootstrap_capacity = int(state.get("bootstrap_capacity", max(1, self.capacity // 2)))
            self.selfplay_capacity = int(state.get("selfplay_capacity", self.capacity - self.bootstrap_capacity))
            self.bootstrap_pos = int(state.get("bootstrap_pos", 0))
            self.selfplay_pos = int(state.get("selfplay_pos", 0))
            self.bootstrap_buffer = list(state.get("bootstrap_buffer", []))
            self.selfplay_buffer = list(state.get("selfplay_buffer", []))
            self.bootstrap_priorities = np.zeros((self.bootstrap_capacity,), dtype=np.float32)
            self.selfplay_priorities = np.zeros((self.selfplay_capacity,), dtype=np.float32)
            bp = state.get("bootstrap_priorities")
            sp = state.get("selfplay_priorities")
            if bp is not None and len(self.bootstrap_buffer):
                self.bootstrap_priorities[: len(self.bootstrap_buffer)] = np.asarray(bp, dtype=np.float32)
            if sp is not None and len(self.selfplay_buffer):
                self.selfplay_priorities[: len(self.selfplay_buffer)] = np.asarray(sp, dtype=np.float32)
            return

        # Legacy single-buffer checkpoint → assign to bootstrap partition
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.bootstrap_fraction = 0.5
        self.bootstrap_capacity = max(1, self.capacity // 2)
        self.selfplay_capacity = max(1, self.capacity - self.bootstrap_capacity)
        legacy = list(state.get("buffer", []))
        self.bootstrap_buffer = legacy[: self.bootstrap_capacity]
        self.selfplay_buffer = legacy[self.bootstrap_capacity : self.bootstrap_capacity + self.selfplay_capacity]
        self.bootstrap_pos = len(self.bootstrap_buffer) % self.bootstrap_capacity
        self.selfplay_pos = len(self.selfplay_buffer) % self.selfplay_capacity
        self.bootstrap_priorities = np.zeros((self.bootstrap_capacity,), dtype=np.float32)
        self.selfplay_priorities = np.zeros((self.selfplay_capacity,), dtype=np.float32)
        legacy_p = state.get("priorities")
        if legacy_p is not None:
            legacy_p = np.asarray(legacy_p, dtype=np.float32)
            n_b = min(len(legacy_p), len(self.bootstrap_buffer))
            n_s = min(max(0, len(legacy_p) - n_b), len(self.selfplay_buffer))
            if n_b:
                self.bootstrap_priorities[:n_b] = legacy_p[:n_b]
            if n_s:
                self.selfplay_priorities[:n_s] = legacy_p[n_b : n_b + n_s]
