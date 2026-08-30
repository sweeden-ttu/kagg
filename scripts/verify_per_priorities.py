#!/usr/bin/env python
"""Assert Dual-partition PER reweights priorities from TD-error-like values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_SRC = Path(__file__).resolve().parents[1] / "datasets/scottweeden/self-training-code"
sys.path.insert(0, str(CODE_SRC))

from kaggriculture_self_play_training import (  # noqa: E402
    SOURCE_BOOTSTRAP,
    SOURCE_SELFPLAY,
    PrioritizedReplayBuffer,
)


def _dummy_transition(reward: float):
    tiles = np.zeros((4, 4, 2), dtype=np.float32)
    numeric = np.zeros(8, dtype=np.float32)
    hands = np.zeros(6, dtype=np.int64)
    market = np.zeros(10, dtype=np.int64)
    return dict(
        tiles=tiles,
        numeric=numeric,
        action_verb=0,
        action_crop=0,
        action_hands=hands,
        action_market=market,
        reward=reward,
        next_tiles=tiles.copy(),
        next_numeric=numeric.copy(),
        done=False,
    )


def main() -> int:
    buf = PrioritizedReplayBuffer(capacity=32, alpha=0.6, bootstrap_fraction=0.5)
    for i in range(8):
        buf.push(**_dummy_transition(float(i)), source=SOURCE_BOOTSTRAP)
        buf.push(**_dummy_transition(float(i + 10)), source=SOURCE_SELFPLAY)

    before_b = buf.bootstrap_priorities[: buf.bootstrap_size].copy()
    before_s = buf.selfplay_priorities[: buf.selfplay_size].copy()
    assert np.allclose(before_b, before_b[0]), "bootstrap priorities should start uniform"
    assert np.allclose(before_s, before_s[0]), "self-play priorities should start uniform"

    batch, indices, _weights = buf.sample(batch_size=8, beta=0.4)
    assert batch and len(indices) == 8, "expected mixed PER sample of size 8"

    td_errors = np.linspace(0.1, 9.0, num=len(indices), dtype=np.float32)
    buf.update_priorities(indices, td_errors)

    after_b = buf.bootstrap_priorities[: buf.bootstrap_size]
    after_s = buf.selfplay_priorities[: buf.selfplay_size]
    changed = (not np.allclose(before_b, after_b)) or (not np.allclose(before_s, after_s))
    if not changed:
        print("FAIL: priorities unchanged after TD-error update_priorities")
        print(f"  indices={indices}")
        print(f"  bootstrap {before_b} -> {after_b}")
        print(f"  selfplay  {before_s} -> {after_s}")
        return 1

    print("PASS: Dual-partition PER priorities changed after TD-error reweighting")
    print(f"  sampled indices (source_flag, local_idx)=\n{indices}")
    print(f"  td_errors={td_errors}")
    print(f"  bootstrap priorities: {before_b} -> {after_b}")
    print(f"  selfplay priorities:  {before_s} -> {after_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
