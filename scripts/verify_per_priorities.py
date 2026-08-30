#!/usr/bin/env python
"""Assert Dual-partition PER reweights priorities from TD-error-like values."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

CODE_SRC = Path(__file__).resolve().parents[1] / "datasets/scottweeden/self-training-code"
sys.path.insert(0, str(CODE_SRC))

from kaggriculture_adapter import (  # noqa: E402
    NUM_HANDS,
    NUM_MARKET_ACTIONS,
    encode_path_b_action,
    encode_path_b_observation,
)
from kaggriculture_self_play_training import (  # noqa: E402
    SOURCE_BOOTSTRAP,
    SOURCE_SELFPLAY,
    PrioritizedReplayBuffer,
    create_competitive_env,
)

PASS_ACTION: Dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}


def _path_b_transition(obs: Dict[str, Any], next_obs: Dict[str, Any], reward: float) -> Dict[str, Any]:
    encoded = encode_path_b_observation(obs, player_id=int(obs.get("player", 0)))
    encoded_next = encode_path_b_observation(
        next_obs, player_id=int(next_obs.get("player", 0))
    )
    action = encode_path_b_action(PASS_ACTION)
    tiles = encoded["tiles"]
    numeric = encoded["numeric"]
    if tiles.shape != (9, 10, 10):
        raise ValueError(f"expected Path B tiles (9, 10, 10), got {tiles.shape}")
    if numeric.shape != (55,):
        raise ValueError(f"expected Path B numeric (55,), got {numeric.shape}")
    if action["hands"].shape != (NUM_HANDS,):
        raise ValueError(f"expected hands ({NUM_HANDS},), got {action['hands'].shape}")
    if action["market"].shape != (NUM_MARKET_ACTIONS,):
        raise ValueError(
            f"expected market ({NUM_MARKET_ACTIONS},), got {action['market'].shape}"
        )
    return dict(
        tiles=tiles,
        numeric=numeric,
        action_verb=int(action["verb"]),
        action_crop=int(action["crop"]),
        action_hands=action["hands"],
        action_market=action["market"],
        reward=reward,
        next_tiles=encoded_next["tiles"],
        next_numeric=encoded_next["numeric"],
        done=False,
    )


def main() -> int:
    env = create_competitive_env(use_kaggle=True, max_steps=720, seed=42, turns_per_cycle=24)
    obs_p0 = env.reset()
    (next_p0, _), _, done, _ = env.step([PASS_ACTION, PASS_ACTION])
    if done:
        print("FAIL: competition env ended on the first PASS step")
        return 1

    buf = PrioritizedReplayBuffer(capacity=32, alpha=0.6, bootstrap_fraction=0.5)
    for i in range(8):
        buf.push(
            **_path_b_transition(obs_p0, next_p0, float(i)),
            source=SOURCE_BOOTSTRAP,
        )
        buf.push(
            **_path_b_transition(obs_p0, next_p0, float(i + 10)),
            source=SOURCE_SELFPLAY,
        )

    before_b = buf.bootstrap_priorities[: buf.bootstrap_size].copy()
    before_s = buf.selfplay_priorities[: buf.selfplay_size].copy()
    assert np.allclose(before_b, before_b[0]), "bootstrap priorities should start uniform"
    assert np.allclose(before_s, before_s[0]), "self-play priorities should start uniform"

    batch, indices, _weights = buf.sample(batch_size=8, beta=0.4)
    assert batch and len(indices) == 8, "expected mixed PER sample of size 8"
    assert batch["tiles"].shape == (8, 9, 10, 10), batch["tiles"].shape
    assert batch["numeric"].shape == (8, 55), batch["numeric"].shape

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
