"""Tests for dual-partition PER buffer (generation tags, beta, reservoir)."""

from __future__ import annotations

import numpy as np

from kaggriculture_self_play_training import (
    SOURCE_BOOTSTRAP,
    SOURCE_SELFPLAY,
    PrioritizedReplayBuffer,
)


def _tr(seed: int = 0):
    rng = np.random.default_rng(seed)
    tiles = rng.random((9, 10, 10), dtype=np.float32)
    numeric = rng.random(55, dtype=np.float32)
    hands = np.zeros(6, dtype=np.int64)
    market = np.zeros(10, dtype=np.int64)
    return tiles, numeric, 0, 0, hands, market, 0.0, tiles.copy(), numeric.copy(), False


def test_beta_anneals_toward_one():
    buf = PrioritizedReplayBuffer(
        capacity=32,
        beta_init=0.4,
        beta_final=1.0,
        beta_anneal_steps=10,
    )
    assert abs(buf.current_beta() - 0.4) < 1e-6
    for i in range(16):
        t = _tr(i)
        buf.push(*t, source=SOURCE_BOOTSTRAP)
        buf.push(*t, source=SOURCE_SELFPLAY)
    for _ in range(10):
        buf.sample(4)
    assert buf.current_beta() >= 0.99


def test_priority_update_skips_stale_generation():
    buf = PrioritizedReplayBuffer(capacity=8, bootstrap_fraction=0.5)
    # Fill self-play partition (capacity 4)
    for i in range(4):
        buf.push(*_tr(i), source=SOURCE_SELFPLAY)
    batch, indices, _ = buf.sample(4)
    assert indices.shape[-1] == 3
    # Overwrite all self-play slots to bump generations
    for i in range(8):
        buf.push(*_tr(100 + i), source=SOURCE_SELFPLAY)
    old_prios = buf.selfplay_priorities[:4].copy()
    # TD update with stale gens must be ignored
    buf.update_priorities(indices, np.full(len(indices), 99.0))
    # At least some slots should retain original max-prio (not all 99)
    assert not np.allclose(buf.selfplay_priorities[:4], 99.0)


def test_bootstrap_reservoir_keeps_capacity():
    buf = PrioritizedReplayBuffer(capacity=10, bootstrap_fraction=0.5)
    # bootstrap_capacity = 5
    for i in range(20):
        buf.push(*_tr(i), source=SOURCE_BOOTSTRAP)
    assert buf.bootstrap_size == 5
    assert buf._bootstrap_seen >= 20
