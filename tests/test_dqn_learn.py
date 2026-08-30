"""Short DQN.learn() loop against a Gymnasium-compatible env."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from kaggriculture_adapter import encode_observation
from kaggriculture_rl.dqn_sb3 import DQN


class BranchedToyEnv(gym.Env):
    """Fully implemented tiny env returning encode_observation-shaped tensors."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 4):
        super().__init__()
        self.max_steps = max_steps
        self._t = 0
        self.action_space = spaces.Dict(
            {
                "farmer": spaces.Discrete(15),
                "hands": spaces.MultiDiscrete([15] * 6),
                "market": spaces.Discrete(10),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "tiles": spaces.Box(0, 8, (10, 10), dtype=np.int64),
            }
        )

    def _obs(self) -> Dict[str, torch.Tensor]:
        raw = {
            "day": 1,
            "hour": self._t,
            "farms": [
                {
                    "money": 3000.0,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [0, 0],
                },
                {
                    "money": 3000.0,
                    "tiles": [[None] * 10 for _ in range(10)],
                    "farmer": [9, 9],
                },
            ],
            "private": {
                "seeds": {},
                "shed": {},
                "inventories": [{} for _ in range(6)],
            },
            "market": {"prices": {}, "inventory": {}},
        }
        return encode_observation(raw, player_id=0, device="cpu")

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], float, bool, bool, dict]:
        self._t += 1
        done = self._t >= self.max_steps
        if done:
            return None, 0.0, True, False, {}
        return self._obs(), 0.1, False, False, {}


def test_dqn_learn_few_steps():
    env = BranchedToyEnv(max_steps=3)
    model = DQN(
        "KaggricultureCNN",
        env,
        device="cpu",
        learning_starts=0,
        buffer_size=128,
        batch_size=4,
        train_freq=1,
        verbose=0,
        learning_rate=1e-4,
    )
    model.learn(total_timesteps=6, log_interval=100)
    assert model.learner.step_count >= 6
