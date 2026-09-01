"""Dueling Double DQN with Action Branching for the Kaggriculture simulation.

This module provides the core RL components for training an off-policy agent
that navigates a multi-branch action space:
    - Farmer actions    (15 discrete actions)
    - Hand actions      (6 hands × 15 discrete actions each)
    - Market actions    (10 discrete actions)

Rather than a flat action space of 15 × 15^6 × 10 ≈ 2.9 × 10^12, the network
uses **action branching** with 122 Q-value outputs, combined through a
**Dueling** architecture that separates state value V(s) from action advantage
A(s, a) to stabilize learning.

Double Q-learning prevents overestimation bias by using the online network to
select the best action and the target network to evaluate it.

Usage
-----
    model = DuelingDoubleDQNBranching(observation_space, features_dim=512)
    target = DuelingDoubleDQNBranching(observation_space, features_dim=512)
    target.load_state_dict(model.state_dict())  # hard copy

    learner = DoubleDQNLearner(
        online_network=model,
        target_network=target,
        replay_buffer=ReplayBuffer(capacity=1_000_000, use_priority=True),
        gamma=0.995,
        batch_size=64,
        use_soft_update=True,
        tau=0.001,
    )
"""

from __future__ import annotations

import collections
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kaggriculture_adapter import CYCLES_PER_EPISODE, TURNS_PER_CYCLE


# ─────────────────────────────────────────────────────────────
# 1. Feature Extractor
# ─────────────────────────────────────────────────────────────

class KaggricultureFeatureExtractor(nn.Module):
    """Shared feature extractor for observation encoding.

    Combines:
      - CNN branch: one-hot tile grid → spatial patterns
      - MLP branch: flattened numeric features (market, inventory,
        temporal, private state) → relational features
      - Fusion: shared latent vector for all Q-heads
    """

    def __init__(
        self,
        tile_types: int = 9,
        board_size: int = 10,
        numeric_dim: int = 55,
        hidden_dim: int = 256,
        features_dim: int = 512,
    ):
        super().__init__()
        self.board_size = board_size

        # ── CNN Branch: Tile Grid ──
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(tile_types, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # board_size → board_size//2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # board_size//2 → board_size//4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),  # fixed output
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, tile_types, board_size, board_size)
            self.grid_out_size = self.grid_cnn(sample).shape[1]

        # ── MLP Branch: Numeric Features ──
        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True), nn.Dropout(0.1),
        )

        # ── Fusion ──
        combined = self.grid_out_size + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined, features_dim),
            nn.LayerNorm(features_dim), nn.ReLU(inplace=True),
            nn.Linear(features_dim, features_dim),
            nn.LayerNorm(features_dim), nn.ReLU(inplace=True),
        )

    def forward(self, observations: Dict[str, Any]) -> torch.Tensor:
        """Extract shared latent vector from observation dict.

        Args:
            observations: Dict with at least:
                - "tiles": (B, H, W) tile IDs
                - "day", "hour", "player_id"
                - "farms.p0.money", "farms.p1.money"
                - "market.prices", "market.inventory"
                - "private.seeds", "private.shed", "private.inventories"

        Returns:
            Latent vector of shape (B, features_dim)
        """
        observations = self._ensure_batch(observations)

        # ── CNN Branch ──
        grid = observations["tiles"]  # (B, H, W)
        grid_onehot = F.one_hot(grid.long(), num_classes=9).permute(0, 3, 1, 2).float()
        grid_features = self.grid_cnn(grid_onehot)

        # ── MLP Branch ──
        parts: List[torch.Tensor] = []
        parts.append(observations["day"].float() / float(max(CYCLES_PER_EPISODE - 1, 1)))
        parts.append(observations["hour"].float() / float(TURNS_PER_CYCLE))
        parts.append(observations["player_id"].float())
        parts.append(observations["farms_p0_money"].float() / 10_000.0)
        parts.append(observations["farms_p1_money"].float() / 10_000.0)
        parts.append(observations["market_prices"].float() / 500.0)
        parts.append(observations["market_inventory"].float() / 10_000.0)
        parts.append(observations["seeds"].float() / 500.0)
        parts.append(observations["shed"].float() / 1_000.0)
        parts.append(observations["inventories"].float() / 100.0)
        # Flatten inventories from (B, N, D) → (B, N*D) if needed
        if observations["inventories"].dim() == 3:
            parts[-1] = parts[-1].view(parts[-1].shape[0], -1)

        numeric = torch.cat(parts, dim=-1)
        numeric_features = self.numeric_mlp(numeric)

        # ── Fusion ──
        combined = torch.cat([grid_features, numeric_features], dim=-1)
        return self.fusion(combined)

    @staticmethod
    def _ensure_batch(observations: Dict[str, Any]) -> Dict[str, Any]:
        """Add batch dimension when a single observation is passed."""
        def _to_tensor(value: Any) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value)

        tiles = _to_tensor(observations["tiles"])
        if tiles.dim() == 2:
            batched: Dict[str, Any] = {}
            for key, value in observations.items():
                t = _to_tensor(value)
                batched[key] = t.unsqueeze(0) if t.dim() >= 1 else t
            return batched

        return {key: _to_tensor(value) for key, value in observations.items()}


# ─────────────────────────────────────────────────────────────
# 2. Dueling Double DQN with Action Branching
# ─────────────────────────────────────────────────────────────

class BranchingQOutput(TypedDict):
    farmer_q: torch.Tensor
    hand_q: List[torch.Tensor]
    market_q: torch.Tensor
    value: torch.Tensor


class DuelingDoubleDQNBranching(nn.Module):
    """Dueling Double DQN with separate action heads.

    Architecture:
    ─────────────
    Observation → Feature Extractor → Shared Layers → Dueling Split

    Dueling Split:
    ├── V(s) Stream → Value Head (scalar per state)
    └── A(s,·) Stream → Branch Q-Heads:
        ├── Farmer Branch: 15 Q-values
        ├── Hand_0 Branch: 15 Q-values
        ├── Hand_1 Branch: 15 Q-values
        ├── ...
        ├── Hand_5 Branch: 15 Q-values
        └── Market Branch: 10 Q-values

    Q(s, a) = V(s) + [A(s, a) - (1/|A|) Σ_a' A(s, a')]

    For branched actions the decomposition sums over branches:
        Q = V(s) + Σ_branch [A_branch(s, a_branch) - mean(A_branch(s,·))]

    Total Q-value outputs: 1 (V) + 15 (farmer) + 6×15 (hands) + 10 (market)
                          = 122  (vs. 2.9×10^12 for a flat encoding)
    """

    def __init__(
        self,
        feature_extractor: KaggricultureFeatureExtractor,
        features_dim: int = 512,
        n_farmer_actions: int = 15,
        n_hand_actions: int = 15,
        n_hands: int = 6,
        n_market_actions: int = 10,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.n_hands = n_hands
        self.n_farmer_actions = n_farmer_actions
        self.n_hand_actions = n_hand_actions
        self.n_market_actions = n_market_actions

        # ── Shared Dense Layers (post feature-extractor) ──
        self.shared_layers = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True),
        )

        # ── Dueling: Value Stream ──
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # ── Dueling: Advantage Streams ──
        self.advantage_farmer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_farmer_actions),
        )

        self.advantage_hands = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, n_hand_actions),
            )
            for _ in range(n_hands)
        ])

        self.advantage_market = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_market_actions),
        )

    # ── Forward Pass ───────────────────────────────────────────

    def forward(self, observations: Dict[str, Any]) -> BranchingQOutput:
        """Forward pass returning Q-values for every branch.

        Returns a dict:
            {
                "farmer_q":     (B, 15),  Q-values for farmer actions
                "hand_q":       list of (B, 15),  one per hand
                "market_q":     (B, 10),  Q-values for market actions
                "value":        (B, 1),   State value V(s)
            }
        """
        latent = self.feature_extractor(observations)
        shared = self.shared_layers(latent)

        # Value stream
        value = self.value_stream(shared)  # (B, 1)

        # Advantage streams
        adv_farmer = self.advantage_farmer(shared)    # (B, 15)
        adv_hands = [self.advantage_hands[i](shared)
                     for i in range(self.n_hands)]    # list of (B, 15)
        adv_market = self.advantage_market(shared)     # (B, 10)

        # Dueling aggregation: Q = V + (A - mean(A))
        farmer_q = value + (adv_farmer - adv_farmer.mean(dim=1, keepdim=True))
        hand_qs = [
            value + (adv - adv.mean(dim=1, keepdim=True))
            for adv in adv_hands
        ]
        market_q = value + (adv_market - adv_market.mean(dim=1, keepdim=True))

        return {
            "farmer_q": farmer_q,
            "hand_q": hand_qs,
            "market_q": market_q,
            "value": value,
        }

    # ── Inference ──────────────────────────────────────────────

    def get_action(
        self,
        observations: Dict[str, Any],
        epsilon: float = 0.0,
    ) -> Dict[str, Any]:
        """Select action using ε-greedy policy.

        Args:
            observations: Observation dict (same format as forward).
            epsilon: Exploration probability.

        Returns:
            Action dict:
                {"farmer": int, "hands": List[int], "market": int}
        """
        if np.random.random() < epsilon:
            return self._random_action()

        with torch.no_grad():
            was_training = self.training
            self.eval()
            try:
                q = self.forward(observations)
            finally:
                self.train(was_training)

            # Select best farmer action
            farmer_action = q["farmer_q"].argmax(dim=1).item()

            # Select best action per hand
            hand_actions = [
                q["hand_q"][i].argmax(dim=1).item()
                for i in range(self.n_hands)
            ]

            # Select best market action
            market_action = q["market_q"].argmax(dim=1).item()

        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market_action,
        }

    def _random_action(self) -> Dict[str, Any]:
        """Return a random valid action."""
        return {
            "farmer": random.randint(0, self.n_farmer_actions - 1),
            "hands": [random.randint(0, self.n_hand_actions - 1)
                      for _ in range(self.n_hands)],
            "market": random.randint(0, self.n_market_actions - 1),
        }

    def get_q_values(
        self,
        observations: Dict[str, Any],
    ) -> BranchingQOutput:
        """Return raw Q-values (for debugging / logging)."""
        return self.forward(observations)

    # ── Serialization ──────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save model to file."""
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
    ) -> "DuelingDoubleDQNBranching":
        """Load model from file (creates default structure)."""
        model = cls(
            feature_extractor=KaggricultureFeatureExtractor(),
            features_dim=512,
        )
        model.load_state_dict(torch.load(
            Path(path), map_location=device, weights_only=True
        ))
        return model


# ─────────────────────────────────────────────────────────────
# 3. Replay Buffer
# ─────────────────────────────────────────────────────────────

_STATE_SPECS: Dict[str, Tuple[Tuple[int, ...], np.dtype[Any]]] = {
    "tiles": ((10, 10), np.dtype(np.int64)),
    "day": ((1,), np.dtype(np.float32)),
    "hour": ((1,), np.dtype(np.float32)),
    "player_id": ((1,), np.dtype(np.float32)),
    "farms_p0_money": ((1,), np.dtype(np.float32)),
    "farms_p1_money": ((1,), np.dtype(np.float32)),
    "market_prices": ((5,), np.dtype(np.float32)),
    "market_inventory": ((5,), np.dtype(np.float32)),
    "seeds": ((5,), np.dtype(np.float32)),
    "shed": ((5,), np.dtype(np.float32)),
    "inventories": ((30,), np.dtype(np.float32)),
}


def _state_to_numpy(state: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Convert observation dict tensors to numpy for replay storage."""
    arrays: Dict[str, np.ndarray] = {}
    for key, (shape, dtype) in _STATE_SPECS.items():
        value = state.get(key, 0)
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy().astype(dtype, copy=False)
        else:
            arr = np.asarray(value, dtype=dtype)
        arr = np.reshape(arr, shape)
        arrays[key] = arr
    return arrays


class ReplayBuffer:
    """Circular FIFO replay buffer for DQN training.

    Supports optional Prioritized Experience Replay (PER) where
    transitions with high |TD_error| are sampled more frequently.

    Args:
        capacity: Maximum number of transitions to store.
        use_priority: If True, use Prioritized Experience Replay.
        alpha: Priority exponent (0 = uniform, 1 = fully prioritized).
        beta_init: Initial beta for importance sampling correction.
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        use_priority: bool = False,
        alpha: float = 0.6,
        beta_init: float = 0.4,
    ):
        self.capacity = capacity
        self.use_priority = use_priority
        self.alpha = alpha
        self.beta = beta_init
        self.beta_init = beta_init
        self.beta_max = 1.0
        self.beta_anneal_steps = 500_000

        # Action buffers — branched format
        self._action_farmer = np.zeros(capacity, dtype=np.int64)
        self._action_hands = [
            np.zeros(capacity, dtype=np.int64) for _ in range(6)
        ]
        self._action_market = np.zeros(capacity, dtype=np.int64)

        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.bool_)

        self._state: Dict[str, np.ndarray] = {
            key: np.zeros((capacity, *shape), dtype=dtype)
            for key, (shape, dtype) in _STATE_SPECS.items()
        }
        self._next_state: Dict[str, np.ndarray] = {
            key: np.zeros((capacity, *shape), dtype=dtype)
            for key, (shape, dtype) in _STATE_SPECS.items()
        }

        if use_priority:
            self._priority = np.zeros(capacity, dtype=np.float32)
            self._max_priority = 1.0

        self._size = 0
        self._position = 0
        self._step_count = 0

    # ── Storage ────────────────────────────────────────────────

    def store(
        self,
        state: Dict[str, torch.Tensor],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Store a single transition."""
        state_np = _state_to_numpy(state)
        next_state_np = _state_to_numpy(next_state if next_state is not None else state)
        for key, arr in state_np.items():
            self._state[key][self._position] = arr
        for key, arr in next_state_np.items():
            self._next_state[key][self._position] = arr

        self._action_farmer[self._position] = action["farmer"]
        for h in range(6):
            self._action_hands[h][self._position] = action["hands"][h]
        self._action_market[self._position] = action["market"]
        self._reward[self._position] = reward
        self._done[self._position] = done

        if self.use_priority:
            p = priority if priority is not None else self._max_priority
            self._priority[self._position] = p
            self._max_priority = max(self._max_priority, p)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._step_count += 1

    # ── Sampling ───────────────────────────────────────────────

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a random (or prioritized) batch.

        Returns a dict with keys:
            state, next_state, action_farmer, action_hands,
            action_market, reward, done, weights, indices
        """
        if self.use_priority:
            indices, weights = self._sample_prioritized(batch_size)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        return {
            "state": {key: self._state[key][indices] for key in self._state},
            "next_state": {
                key: self._next_state[key][indices] for key in self._next_state
            },
            "action_farmer": self._action_farmer[indices],
            "action_hands": [b[indices] for b in self._action_hands],
            "action_market": self._action_market[indices],
            "reward": self._reward[indices],
            "done": self._done[indices],
            "weights": torch.FloatTensor(weights),
            "indices": indices,
        }

    # ── Priority Management ────────────────────────────────────

    def update_priorities(self, indices: np.ndarray,
                          priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions."""
        priorities = np.asarray(priorities, dtype=np.float32) + 1e-8
        self._priority[indices] = priorities
        self._max_priority = float(max(self._max_priority, priorities.max()))

    def update_beta(self) -> None:
        """Anneal beta toward 1.0."""
        self.beta = min(
            self.beta_max,
            self.beta_init + (self.beta_max - self.beta_init)
            * (self._step_count / self.beta_anneal_steps),
        )

    def _sample_prioritized(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample transitions proportional to priority."""
        priorities = self._priority[:self._size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self._size, size=batch_size, p=probabilities)
        weights = (self._size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return indices, weights

    # ── Utility ────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size >= self.capacity


# ─────────────────────────────────────────────────────────────
# 4. Double DQN Learner
# ─────────────────────────────────────────────────────────────

class DoubleDQNLearner:
    """Double DQN training loop with Experience Replay.

    Implements the full training loop:
      1. ε-greedy action selection
      2. Transition storage in replay buffer
      3. Periodic training with Double Q-learning
      4. Target network updates (hard or soft)
      5. Priority updates (if using PER)

    Double Q-learning prevents overestimation bias:
        target = r + γ · Q_target(s', argmax_a Q_online(s', a))

    This contrasts with standard DQN which uses:
        target = r + γ · max_a Q_target(s', a)
    where the same network's max is used for both selection and
    evaluation, leading to systematic overestimation.
    """

    def __init__(
        self,
        online_network: DuelingDoubleDQNBranching,
        target_network: DuelingDoubleDQNBranching,
        replay_buffer: ReplayBuffer,
        gamma: float = 0.995,
        batch_size: int = 64,
        huber_delta: float = 1.0,
        learning_starts: int = 50_000,
        train_frequency: int = 4,
        target_update_frequency: int = 10_000,
        use_soft_update: bool = True,
        tau: float = 0.001,
        epsilon_init: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay_steps: int = 2_000_000,
        epsilon_decay_method: str = "linear",
        max_grad_norm: float = 0.5,
    ):
        self.online = online_network
        self.target = target_network
        self.buffer = replay_buffer
        self.gamma = gamma
        self.batch_size = batch_size
        self.huber_delta = huber_delta
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.use_soft_update = use_soft_update
        self.tau = tau

        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay_method = epsilon_decay_method
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(
            online_network.parameters(),
            lr=1e-4, betas=(0.9, 0.999), eps=1e-8
        )
        self.step_count = 0
        self.loss_history: List[float] = []

    # ── Epsilon Schedule ─────────────────────────────────────

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.step_count / self.epsilon_decay_steps)

        if self.epsilon_decay_method == "linear":
            return max(self.epsilon_final,
                       self.epsilon_init * (1.0 - progress))
        elif self.epsilon_decay_method == "exponential":
            return max(self.epsilon_final,
                       self.epsilon_init * (0.999 ** self.step_count))
        elif self.epsilon_decay_method == "cosine":
            return self.epsilon_final + 0.5 * (
                self.epsilon_init - self.epsilon_final
            ) * (1.0 + np.cos(progress * np.pi))
        else:
            return max(self.epsilon_final,
                       self.epsilon_init * (1.0 - progress))

    # ── Interaction ───────────────────────────────────────────

    def store_transition(
        self,
        state: Dict[str, torch.Tensor],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, torch.Tensor],
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.store(state, action, reward, next_state, done)

    def act_and_train(
        self,
        env,
        observation: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], bool, Optional[Dict[str, float]]]:
        """One step of interaction + training.

        1. Select ε-greedy action
        2. Step environment
        3. Store transition
        4. Train if conditions met
        5. Update target network if needed

        Args:
            env: Gymnasium environment with step(obs) → (next_obs, reward, done, info)
            observation: Current observation dict

        Returns:
            ``(next_obs_or_none, done, result)`` where ``result`` is a
            training dict with ``loss`` and ``td_error``, or ``None`` if
            training was skipped.
        """
        action = self.online.get_action(observation, epsilon=self.epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        self.store_transition(
            observation, action, reward,
            next_obs if next_obs is not None else observation,
            done,
        )

        self.step_count += 1

        result = None
        if (self.step_count >= self.learning_starts
                and self.step_count % self.train_frequency == 0
                and len(self.buffer) >= self.batch_size):
            batch = self.buffer.sample(self.batch_size)
            result = self.train_step(batch)
            self.loss_history.append(result["loss"])

            # Update priorities for PER
            if self.buffer.use_priority:
                td_errors = self.get_td_error(batch)
                self.buffer.update_priorities(
                    batch["indices"], td_errors.detach().cpu().numpy()
                )
                self.buffer.update_beta()

        return next_obs if not done else None, done, result

    def act(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Inference only: select action with ε=0."""
        return self.online.get_action(observation, epsilon=0.0)

    # ── Training ──────────────────────────────────────────────

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train one step on a batch of transitions.

        Uses Huber loss on the Double DQN TD target.
        """
        device = next(self.online.parameters()).device
        s = {
            key: torch.as_tensor(arr, device=device).float()
            for key, arr in batch["state"].items()
        }
        s_next = {
            key: torch.as_tensor(arr, device=device).float()
            for key, arr in batch["next_state"].items()
        }
        a_farmer = torch.LongTensor(batch["action_farmer"])
        a_hands = [torch.LongTensor(h) for h in batch["action_hands"]]
        a_market = torch.LongTensor(batch["action_market"])
        reward = torch.FloatTensor(batch["reward"])
        done = torch.FloatTensor(batch["done"])

        # ── Online network: select best next actions ──
        q_next_online = self.online(s_next)
        best_farmer = q_next_online["farmer_q"].argmax(dim=1)
        best_hands = [
            q_next_online["hand_q"][i].argmax(dim=1)
            for i in range(self.online.n_hands)
        ]
        best_market = q_next_online["market_q"].argmax(dim=1)

        # ── Target network: evaluate selected actions ──
        q_next_target = self.target(s_next)

        target_farmer = q_next_target["farmer_q"][
            torch.arange(len(a_farmer)), best_farmer
        ].unsqueeze(1)  # (B, 1)

        target_hands = torch.stack([
            q_next_target["hand_q"][i][
                torch.arange(len(a_hands[i])), best_hands[i]
            ].unsqueeze(1)
            for i in range(self.online.n_hands)
        ]).sum(dim=0)  # sum across hands → (B, 1)

        target_market = q_next_target["market_q"][
            torch.arange(len(a_market)), best_market
        ].unsqueeze(1)  # (B, 1)

        # Double DQN target: r + γ · Q_target(s', argmax Q_online(s',·))
        td_target = reward.unsqueeze(1) + self.gamma * (
            target_farmer + target_hands + target_market
        ) * (1.0 - done.unsqueeze(1))

        # ── Online Q-values ──
        q_online = self.online(s)

        q_farmer = q_online["farmer_q"][
            torch.arange(len(a_farmer)), a_farmer
        ]  # (B,)
        q_hands = sum(
            q_online["hand_q"][i][
                torch.arange(len(a_hands[i])), a_hands[i]
            ]
            for i in range(self.online.n_hands)
        )  # (B,)
        q_market = q_online["market_q"][
            torch.arange(len(a_market)), a_market
        ]  # (B,)
        q_total = q_farmer + q_hands + q_market  # (B,)

        # Huber loss with importance sampling weights
        td = td_target.squeeze(1) - q_total  # (B,)
        huber_loss = F.smooth_l1_loss(q_total, td.detach(), reduction='none')

        weights = batch.get("weights", torch.ones_like(huber_loss))
        loss = (huber_loss * weights).mean()

        # Backward + optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), max_norm=self.max_grad_norm
        )
        self.optimizer.step()

        # ── Target network update ──
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            if self.use_soft_update:
                self._soft_update_target()
            else:
                self._hard_update_target()

        return {
            "loss": loss.item(),
            "td_error": td.abs().mean().item(),
        }

    def get_td_error(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute |TD_error| for each transition in a batch.

        Used to update priorities in PER.
        """
        device = next(self.online.parameters()).device
        s = {
            key: torch.as_tensor(arr, device=device).float()
            for key, arr in batch["state"].items()
        }
        s_next = {
            key: torch.as_tensor(arr, device=device).float()
            for key, arr in batch["next_state"].items()
        }
        a_farmer = torch.LongTensor(batch["action_farmer"]).to(device)
        a_hands = [torch.LongTensor(h).to(device) for h in batch["action_hands"]]
        a_market = torch.LongTensor(batch["action_market"]).to(device)
        reward = torch.FloatTensor(batch["reward"]).to(device)
        done = torch.FloatTensor(batch["done"]).to(device)

        with torch.no_grad():
            q_next_online = self.online(s_next)
            best_farmer = q_next_online["farmer_q"].argmax(dim=1)
            best_hands = [
                q_next_online["hand_q"][i].argmax(dim=1)
                for i in range(self.online.n_hands)
            ]
            best_market = q_next_online["market_q"].argmax(dim=1)

            q_next_target = self.target(s_next)

            target_farmer = q_next_target["farmer_q"][
                torch.arange(len(a_farmer), device=device), best_farmer
            ].unsqueeze(1)
            target_hands = torch.stack([
                q_next_target["hand_q"][i][
                    torch.arange(len(a_hands[i]), device=device), best_hands[i]
                ].unsqueeze(1)
                for i in range(self.online.n_hands)
            ]).sum(dim=0)
            target_market = q_next_target["market_q"][
                torch.arange(len(a_market), device=device), best_market
            ].unsqueeze(1)

            td_target = reward.unsqueeze(1) + self.gamma * (
                target_farmer + target_hands + target_market
            ) * (1.0 - done.unsqueeze(1))

            q_online = self.online(s)
            q_farmer = q_online["farmer_q"][
                torch.arange(len(a_farmer), device=device), a_farmer
            ]
            q_hands = sum(
                q_online["hand_q"][i][
                    torch.arange(len(a_hands[i]), device=device), a_hands[i]
                ]
                for i in range(self.online.n_hands)
            )
            q_market = q_online["market_q"][
                torch.arange(len(a_market), device=device), a_market
            ]
            q_total = q_farmer + q_hands + q_market

        return (td_target.squeeze(1) - q_total).abs()

    # ── Target Updates ────────────────────────────────────────

    def _soft_update_target(self) -> None:
        """Soft update: θ_target ← τ·θ_online + (1-τ)·θ_target."""
        for target_param, online_param in zip(
            self.target.parameters(), self.online.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data
                + (1 - self.tau) * target_param.data
            )

    def _hard_update_target(self) -> None:
        """Hard update: copy online weights to target network."""
        self.target.load_state_dict(self.online.state_dict())

    # ── Serialization ─────────────────────────────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save full training state."""
        torch.save({
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
            "loss_history": self.loss_history,
        }, Path(path))

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        online_network: DuelingDoubleDQNBranching,
        target_network: DuelingDoubleDQNBranching,
        replay_buffer: ReplayBuffer,
        device: str = "cpu",
    ) -> "DoubleDQNLearner":
        """Load full training state from checkpoint."""
        data = torch.load(Path(path), map_location=device, weights_only=False)

        learner = cls(
            online_network=online_network,
            target_network=target_network,
            replay_buffer=replay_buffer,
        )
        learner.online.load_state_dict(data["online_state_dict"])
        learner.target.load_state_dict(data["target_state_dict"])
        learner.optimizer.load_state_dict(data["optimizer_state_dict"])
        learner.step_count = data["step_count"]
        learner.epsilon = data["epsilon"]
        learner.loss_history = data["loss_history"]

        return learner


# ─────────────────────────────────────────────────────────────
# 5. Action Masking (for valid-action enforcement)
# ─────────────────────────────────────────────────────────────

class ActionMasker:
    """Generate valid-action masks based on current game state.

    Masks prevent the agent from selecting illegal actions,
    which is especially important when the action space is
    branched and each branch may have different constraints.
    """

    @staticmethod
    def get_valid_farmer_actions(obs: Dict) -> np.ndarray:
        """Return boolean mask for farmer actions (length 15).

        Index mapping:
            0=PASS, 1=DIG, 2=WATER, 3=PLANT, 4=HARVEST,
            5=NORTH, 6=SOUTH, 7=WEST, 8=EAST,
            9=DROP, 10=PICKUP, 11=BUILD_COOP, 12=BUILD_PASTURE,
            13=BUY_ANIMAL, 14=OTHER
        """
        mask = np.zeros(15, dtype=bool)
        mask[0] = True  # PASS always valid

        farm = obs["farms"][0]
        private = obs["private"]
        x, y = int(farm["farmer"][0]), int(farm["farmer"][1])
        tile = farm["tiles"][y][x]

        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            mask[1] = True  # DIG
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                mask[4] = True  # HARVEST
            if not tile.get("watered_today", False):
                mask[2] = True  # WATER
            mask[3] = True  # PLANT (if space available)

        # Movement actions
        moves = [(0, -1, 5, "NORTH"), (0, 1, 6, "SOUTH"),
                 (-1, 0, 7, "WEST"), (1, 0, 8, "EAST")]
        for dx, dy, idx, _ in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < 10 and 0 <= ny < 10
                    and farm["tiles"][ny][nx] != "LOCKED"):
                mask[idx] = True

        return mask

    @staticmethod
    def get_valid_market_actions(obs: Dict) -> np.ndarray:
        """Return boolean mask for market actions (length 10).

        Index mapping:
            0=PASS, 1=BUY_SEED, 2=BUY_PRODUCT, 3=BUY_ANIMAL,
            4=SELL, 5=HIRE, 6=BUY_LAND, 7-9=OTHER
        """
        mask = np.zeros(10, dtype=bool)
        mask[0] = True  # PASS always valid

        money = obs["farms"][0]["money"]
        seeds = obs["private"]["seeds"]

        # Buy seeds
        seed_costs = {"WHEAT": 10, "CARROT": 8, "TOMATO": 5,
                      "STRAWBERRY": 3, "MELON": 2}
        for seed, cost in seed_costs.items():
            if money >= cost and seeds.get(seed, 0) < 100:
                mask[1] = True

        # Buy animals
        if money >= 400:
            mask[3] = True  # BUY_ANIMAL
            mask[5] = True  # HIRE

        # Sell
        shed = obs["private"].get("shed", {})
        if any(v > 0 for v in shed.values()):
            mask[4] = True  # SELL

        return mask

    @staticmethod
    def apply_mask_to_logits(
        logits: torch.Tensor,
        mask: np.ndarray,
    ) -> torch.Tensor:
        """Set logits for invalid actions to -infinity before softmax."""
        logits = logits.clone()
        logits[~mask.astype(bool)] = -1e9
        return logits
