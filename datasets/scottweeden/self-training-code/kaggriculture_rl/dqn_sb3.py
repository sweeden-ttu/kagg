"""Stable-Baselines3-style DQN wrapper for Kaggriculture.

**Canonical training path:** ``kaggriculture_self_play_training.train_self_play``
(Path B hierarchical DQN + bootstrap + self-play + ladder eval). This module wraps
the legacy flat-branch ``kaggriculture_rl.dqn`` stack for SB3-compatible
experiments and is optional — not used by the main training notebook.

Inspired by stable-baselines3 DQN (https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
but adapted for the multi-branch action space of Kaggriculture:
    - Farmer actions    (15 discrete actions)
    - Hand actions      (6 hands × 15 discrete actions each)
    - Market actions    (10 discrete actions)

Usage (mimicking SB3 API):
    model = DQN(
        "KaggricultureCNN",
        env,
        policy_kwargs=dict(features_dim=512),
        learning_rate=1e-4,
        buffer_size=1_000_000,
        learning_starts=50_000,
        batch_size=64,
        tau=0.001,
        gamma=0.995,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        max_grad_norm=0.5,
        use_sde=False,
        tensorboard_log="./tb_logs/",
    )
    model.learn(total_timesteps=10_000_000, progress_bar=True)
    action, _states = model.predict(observation, deterministic=True)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from kaggriculture_rl.dqn import (
    DuelingDoubleDQNBranching,
    DoubleDQNLearner,
    KaggricultureFeatureExtractor,
    ReplayBuffer,
)
from kaggriculture_adapter import resolve_training_device


class DQN:
    """Dueling Double DQN with Action Branching — SB3-style interface.

    This class wraps `DuelingDoubleDQNBranching` and `DoubleDQNLearner` in a
    clean API that mirrors stable-baselines3's `DQN` class:

        model = DQN(policy, env, **kwargs)
        model.learn(total_timesteps=...)
        action, _ = model.predict(obs, deterministic=True)
        model.save("path/to/model")
        model = DQN.load("path/to/model")

    Parameters
    ----------
    policy : str
        Currently only ``"KaggricultureCNN"`` is supported.
    env : gymnasium.Env
        Kaggriculture environment (must support ``step(action)`` and
        ``reset()`` returning observation dicts).
    device : str or torch.device
        Device to train on (``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``).
    learning_rate : float
        Adam learning rate (default: 1e-4).
    buffer_size : int
        Maximum number of transitions to store in replay buffer
        (default: 1_000_000).
    max_buffer_size : int or None
        Deprecated alias for ``buffer_size``.
    gamma : float
        Discount factor (default: 0.995).
    target_update_interval : int
        Hard-update interval in environment steps (default: 10_000).
    train_freq : int
        Train every N environment steps. Pass a negative number to train
        every N *episode* steps (not supported).
    gradient_steps : int
        Number of gradient steps per training event. Only 1 is supported
        for off-policy DQN (default: 1).
    batch_size : int
        Mini-batch size for replay buffer sampling (default: 64).
    learning_starts : int
        Steps before the agent starts training (default: 50_000).
    repeat_training : int
        Deprecated; use ``train_freq`` instead.
    use_cpu_parallel : bool
        Deprecated.
    create_eval_env : bool
        Deprecated.
    eval_freq : int
        Deprecated.
    n_eval_episodes : int
        Deprecated.
    eval_log_path : str
        Deprecated.
    stats_wrapper : object
        Deprecated.
    seed : int or None
        Random seed for reproducibility (default: None).
    verbose : int
        Verbosity level (0=silent, 1=print logs, 2=tensorboard).
    tensorboard_log : str or None
        Path for TensorBoard logs (default: None).
    policy_kwargs : dict
        Keyword arguments for the neural network:
            - features_dim: int (default: 512)
            - hidden_dim: int (default: 256)
            - n_hands: int (default: 6)
            - n_farmer_actions: int (default: 15)
            - n_hand_actions: int (default: 15)
            - n_market_actions: int (default: 10)
    exploration_fraction : float
        Fraction of total timesteps for epsilon decay
        (default: 0.15).
    exploration_initial_eps : float
        Initial epsilon for exploration (default: 1.0).
    exploration_final_eps : float
        Final epsilon after decay (default: 0.05).
    exploration_decay_steps : int or None
        Override steps for epsilon decay. If None, uses
        ``exploration_fraction * total_timesteps`` at ``learn()`` time.
    use_priority_replay : bool
        If True, use Prioritized Experience Replay (default: False).
    priority_replay_alpha : float
        Alpha exponent for PER (default: 0.6).
    priority_replay_beta_init : float
        Initial beta for importance sampling in PER (default: 0.4).
    priority_replay_beta_final : float
        Final beta for importance sampling in PER (default: 1.0).
    priority_replay_anneal_steps : int
        Steps to anneal beta from init to final (default: 500_000).
    use_soft_update : bool
        If True, use soft target updates (Polyak average). Otherwise
        hard-update (default: True).
    tau : float
        Polyak averaging coefficient for soft updates (default: 0.001).
    max_grad_norm : float
        Gradient clipping norm (default: 0.5).
    n_cpu_envs : int
        Deprecated.
    optimizer_class : torch.optim class
        Optimizer class to use (default: ``torch.optim.Adam``).
    optimizer_kwargs : dict
        Additional kwargs for the optimizer (default: empty dict).
    optimize_memory_usage : bool
        Deprecated.
    enable_extra_checks : bool
        Deprecated.
    supported_action_types : tuple
        Tuple of supported action types.

    Attributes
    ----------
    learning_rate : float
    gamma : float
    batch_size : int
    buffer_size : int
    learning_starts : int
    train_freq : int
    target_update_interval : int
    exploration_fraction : float
    exploration_initial_eps : float
    exploration_final_eps : float
    verbose : int
    device : torch.device
    _tensorboard_writer : SummaryWriter or None
    """

    supported_policy_types = ("KaggricultureCNN",)
    supported_action_types = ("multi_discrete", "multi_categorical", "dict")

    def __init__(
        self,
        policy: str,
        env: Any = None,
        device: Union[str, torch.device] = "auto",
        # DQN-specific
        learning_rate: float = 1e-4,
        buffer_size: int = 1_000_000,
        max_buffer_size: Optional[int] = None,
        gamma: float = 0.995,
        target_update_interval: int = 10_000,
        train_freq: int = 4,
        gradient_steps: int = -1,
        batch_size: int = 64,
        learning_starts: int = 50_000,
        repeat_training: int = 0,
        use_cpu_parallel: bool = False,
        create_eval_env: bool = False,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        eval_log_path: Optional[str] = None,
        stats_wrapper: Any = None,
        # SE
        seed: Optional[int] = None,
        # TE
        verbose: int = 0,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict] = None,
        exploration_fraction: float = 0.15,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        exploration_decay_steps: Optional[int] = None,
        # PER
        use_priority_replay: bool = False,
        priority_replay_alpha: float = 0.6,
        priority_replay_beta_init: float = 0.4,
        priority_replay_beta_final: float = 1.0,
        priority_replay_anneal_steps: int = 500_000,
        # Target update
        use_soft_update: bool = True,
        tau: float = 0.001,
        # Gradient clipping
        max_grad_norm: float = 0.5,
        # Deprecated
        n_cpu_envs: int = 1,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = None,
        optimize_memory_usage: bool = False,
        enable_extra_checks: bool = False,
    ):
        # ── Deprecation warnings ──────────────────────────────────
        if max_buffer_size is not None:
            buffer_size = max_buffer_size
        if gradient_steps is not None and gradient_steps != -1:
            raise NotImplementedError(
                "gradient_steps > 0 is not yet supported for DQN. "
                "Set to -1 to disable (use train_freq instead)."
            )
        if repeat_training != 0:
            raise NotImplementedError(
                "repeat_training is deprecated. Use train_freq instead."
            )
        if use_cpu_parallel:
            raise NotImplementedError("use_cpu_parallel is not supported.")
        if create_eval_env:
            raise NotImplementedError("create_eval_env is deprecated.")
        if eval_freq != 10_000 or n_eval_episodes != 5:
            raise NotImplementedError("eval_freq/n_eval_episodes are deprecated.")
        if eval_log_path is not None:
            raise NotImplementedError("eval_log_path is deprecated.")
        if stats_wrapper is not None:
            raise NotImplementedError("stats_wrapper is deprecated.")
        if use_cpu_parallel:
            raise NotImplementedError("use_cpu_parallel is deprecated.")
        if optimize_memory_usage:
            raise NotImplementedError("optimize_memory_usage is deprecated.")
        if enable_extra_checks:
            raise NotImplementedError("enable_extra_checks is deprecated.")

        # ── Validate policy ───────────────────────────────────────
        if policy not in self.supported_policy_types:
            raise ValueError(
                f"Policy {policy} not supported. "
                f"Use one of {self.supported_policy_types}."
            )

        # ── Resolve device ────────────────────────────────────────
        self.device = resolve_training_device(str(device))
        self.env = env

        # ── Store hyperparameters (SB3-style) ─────────────────────
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_decay_steps = exploration_decay_steps
        self.verbose = verbose
        self.seed = seed

        # ── PER parameters ────────────────────────────────────────
        self.use_priority_replay = use_priority_replay
        self.priority_replay_alpha = priority_replay_alpha
        self.priority_replay_beta_init = priority_replay_beta_init
        self.priority_replay_beta_final = priority_replay_beta_final
        self.priority_replay_anneal_steps = priority_replay_anneal_steps

        # ── Target update ─────────────────────────────────────────
        self.use_soft_update = use_soft_update
        self.tau = tau

        # ── Gradient clipping ─────────────────────────────────────
        self.max_grad_norm = max_grad_norm

        # ── Optimizer ─────────────────────────────────────────────
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}

        # ── TensorBoard ───────────────────────────────────────────
        self._tensorboard_writer = None
        if tensorboard_log is not None and verbose >= 1:
            self._tensorboard_writer = SummaryWriter(tensorboard_log)

        # ── Set seed ──────────────────────────────────────────────
        if seed is not None:
            self._set_seed(seed)

        # ── Initialize model components ───────────────────────────
        self.policy_kwargs = policy_kwargs or {}
        self._init_model(env)

    # ────────────────────────────────────────────────────────────
    #  Initialization
    # ────────────────────────────────────────────────────────────

    def _init_model(self, env: Any = None):
        """Build the neural network, replay buffer, and learner."""
        # Feature extractor
        self.feature_extractor = KaggricultureFeatureExtractor(
            tile_types=9,
            board_size=10,
            numeric_dim=55,
            hidden_dim=self.policy_kwargs.get("hidden_dim", 256),
            features_dim=self.policy_kwargs.get("features_dim", 512),
        ).to(self.device)

        # Dueling network
        self.online_network = DuelingDoubleDQNBranching(
            feature_extractor=self.feature_extractor,
            features_dim=self.policy_kwargs.get("features_dim", 512),
            n_farmer_actions=self.policy_kwargs.get("n_farmer_actions", 15),
            n_hand_actions=self.policy_kwargs.get("n_hand_actions", 15),
            n_hands=self.policy_kwargs.get("n_hands", 6),
            n_market_actions=self.policy_kwargs.get("n_market_actions", 10),
            hidden_dim=self.policy_kwargs.get("hidden_dim", 256),
        ).to(self.device)

        # Target network (hard-synced copy)
        self.target_network = DuelingDoubleDQNBranching(
            feature_extractor=self.feature_extractor,
            features_dim=self.policy_kwargs.get("features_dim", 512),
            n_farmer_actions=self.policy_kwargs.get("n_farmer_actions", 15),
            n_hand_actions=self.policy_kwargs.get("n_hand_actions", 15),
            n_hands=self.policy_kwargs.get("n_hands", 6),
            n_market_actions=self.policy_kwargs.get("n_market_actions", 10),
            hidden_dim=self.policy_kwargs.get("hidden_dim", 256),
        ).to(self.device)
        self._hard_sync_target()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.buffer_size,
            use_priority=self.use_priority_replay,
            alpha=self.priority_replay_alpha,
            beta_init=self.priority_replay_beta_init,
        )

        # Learner
        self.learner = DoubleDQNLearner(
            online_network=self.online_network,
            target_network=self.target_network,
            replay_buffer=self.replay_buffer,
            gamma=self.gamma,
            batch_size=self.batch_size,
            learning_starts=self.learning_starts,
            train_frequency=self.train_freq,
            target_update_frequency=self.target_update_interval,
            use_soft_update=self.use_soft_update,
            tau=self.tau,
            epsilon_init=self.exploration_initial_eps,
            epsilon_final=self.exploration_final_eps,
            epsilon_decay_steps=(
                self.exploration_decay_steps
                if self.exploration_decay_steps is not None
                else 2_000_000
            ),
            max_grad_norm=self.max_grad_norm,
        )
        self.learner.optimizer = self.optimizer_class(
            self.online_network.parameters(),
            lr=self.learning_rate,
            **self.optimizer_kwargs,
        )

    def _hard_sync_target(self):
        """Hard-sync target network from online network."""
        self.target_network.load_state_dict(self.online_network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    # ────────────────────────────────────────────────────────────
    #  Learn
    # ────────────────────────────────────────────────────────────

    def learn(
        self,
        total_timesteps: int,
        callback: Any = None,
        log_interval: int = 100,
        eval_env: Any = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "DQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = False,
        progress_bar: bool = False,
    ) -> "DQN":
        """Train the model.

        Parameters
        ----------
        total_timesteps : int
            Total number of training timesteps.
        callback : callable or None
            Callback function called at the end of each training step.
            The callback receives ``self`` as its first argument.
        log_interval : int
            Log training metrics every N steps (default: 100).
        eval_env : Env or None
            Environment for periodic evaluation.
        eval_freq : int
            Evaluate every N steps. Use -1 to disable.
        n_eval_episodes : int
            Number of episodes to evaluate over.
        tb_log_name : str
            TensorBoard log name.
        eval_log_path : str or None
            Path to save evaluation metrics.
        reset_num_timesteps : bool
            If True, reset timesteps to 0 (for continued training).
        progress_bar : bool
            If True, show a progress bar.

        Returns
        -------
        self : DQN
            The trained model instance.
        """
        existing_timesteps = self.learner.step_count
        if reset_num_timesteps:
            # We don't reset step_count but track offset
            pass

        # Determine epsilon decay steps from exploration fraction
        if self.exploration_decay_steps is None:
            self.learner.epsilon_decay_steps = (
                int(self.exploration_fraction * total_timesteps)
            )
        else:
            self.learner.epsilon_decay_steps = self.exploration_decay_steps

        # Progress bar setup
        progress_iter = None
        if progress_bar:
            try:
                from tqdm import tqdm
                progress_iter = tqdm(total=total_timesteps)
            except ImportError:
                progress_iter = None

        # Training loop
        start_time = time.time()
        self._train_log_interval = log_interval
        self._train_tb_log_name = tb_log_name

        if self.env is None:
            raise ValueError("DQN.learn() requires an environment passed to DQN(...)")

        _r = self.env.reset()
        obs = _r[0] if isinstance(_r, tuple) else _r
        for timestep in range(total_timesteps):
            if obs is None:
                _r = self.env.reset()
                obs = _r[0] if isinstance(_r, tuple) else _r

            obs, done, result = self.learner.act_and_train(self.env, obs)

            # ── Logging ───────────────────────────────────────
            if result is not None and timestep % log_interval == 0:
                self._log_metrics(result, timestep, start_time)

            # ── Callback ──────────────────────────────────────
            if callback is not None:
                callback(self)

            # ── Evaluation ────────────────────────────────────
            if eval_env is not None and eval_freq > 0 and timestep > 0 and (
                timestep + existing_timesteps
            ) % eval_freq == 0:
                eval_stats = self._evaluate_policy(
                    eval_env, n_eval_episodes
                )
                self._log_eval(eval_stats, timestep)
                if eval_log_path:
                    self._save_eval_results(eval_stats, eval_log_path)

            # ── Progress bar update ───────────────────────────
            if progress_iter is not None:
                progress_iter.update(1)
                if timestep % log_interval == 0:
                    progress_iter.set_postfix(
                        loss=f"{result['loss']:.4f}"
                        if result is not None
                        else "N/A"
                    )

        if progress_iter is not None:
            progress_iter.close()

        return self

    def _train_step(self, env: Any = None, timestep: int = 0):
        """Execute one training step (train on buffer if conditions met)."""
        if timestep % self.learner.train_frequency != 0:
            return None

        if len(self.replay_buffer) < self.learner.batch_size:
            return None

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Compute observation tensors for current step
        batch["state"] = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v).float().to(self.device))
            for k, v in batch["state"].items()
            if k in ["tiles", "day", "hour", "player_id",
                      "farms_p0_money", "farms_p1_money",
                      "market_prices", "market_inventory",
                      "seeds", "shed", "inventories"]
        }
        batch["next_state"] = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v).float().to(self.device))
            for k, v in batch["next_state"].items()
            if k in ["tiles", "day", "hour", "player_id",
                      "farms_p0_money", "farms_p1_money",
                      "market_prices", "market_inventory",
                      "seeds", "shed", "inventories"]
        }

        result = self.learner.train_step(batch)
        return result

    def _log_metrics(
        self, result: Dict[str, float], timestep: int, start_time: float
    ):
        """Log training metrics."""
        elapsed = time.time() - start_time
        steps_per_sec = timestep / elapsed if elapsed > 0 else 0

        msg = (
            f"timestep={timestep}  "
            f"loss={result['loss']:.4f}  "
            f"td_error={result['td_error']:.4f}  "
            f"epsilon={self.learner.epsilon:.4f}  "
            f"buffer_size={len(self.replay_buffer)}  "
            f"steps/sec={steps_per_sec:.0f}"
        )
        if self.verbose >= 1:
            print(msg)

        # TensorBoard
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/train/loss",
                result["loss"],
                timestep,
            )
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/train/td_error",
                result["td_error"],
                timestep,
            )
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/train/epsilon",
                self.learner.epsilon,
                timestep,
            )
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/train/buffer_size",
                len(self.replay_buffer),
                timestep,
            )

    def _evaluate_policy(
        self, env: Any, n_episodes: int = 5
    ) -> Dict[str, float]:
        """Evaluate the agent on the environment for N episodes."""
        rewards = []
        for ep in range(n_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.predict(
                    obs, deterministic=True
                )[0]
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
            rewards.append(total_reward)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }

    def _log_eval(self, stats: Dict[str, float], timestep: int):
        """Log evaluation metrics to TensorBoard."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/eval/mean_reward",
                stats["mean_reward"],
                timestep,
            )
            self._tensorboard_writer.add_scalar(
                f"{self._train_tb_log_name}/eval/std_reward",
                stats["std_reward"],
                timestep,
            )

    def _save_eval_results(
        self, stats: Dict[str, float], path: str
    ):
        """Save evaluation results to a file."""
        with open(path, "a") as f:
            f.write(
                f"timestep,mean_reward,std_reward,min_reward,max_reward\n"
            )
            f.write(
                f"{self.learner.step_count},"
                f"{stats['mean_reward']:.4f},"
                f"{stats['std_reward']:.4f},"
                f"{stats['min_reward']:.4f},"
                f"{stats['max_reward']:.4f}\n"
            )

    # ────────────────────────────────────────────────────────────
    #  Predict
    # ────────────────────────────────────────────────────────────

    def predict(
        self,
        observation: Dict[str, Any],
        state: Optional[Tuple] = None,
        deterministic: bool = True,
    ) -> Tuple[Dict[str, Any], Optional[Tuple]]:
        """Get the action for a given observation.

        Parameters
        ----------
        observation : dict
            Observation from the environment.
        state : tuple
            Previous hidden state (for RNN policies, not used here).
        deterministic : bool
            If True, use argmax selection (epsilon=0). If False,
            sample from Q-value distribution.

        Returns
        -------
        action : dict
            Action dict with keys "farmer", "hands", "market".
        state : tuple or None
            Hidden state (unchanged).
        """
        with torch.no_grad():
            if deterministic:
                action = self.online_network.get_action(
                    observation, epsilon=0.0
                )
            else:
                action = self.online_network.get_action(
                    observation, epsilon=0.5
                )
        return action, state

    # ────────────────────────────────────────────────────────────
    #  Get/Learn Rate
    # ────────────────────────────────────────────────────────────

    def get_learning_rate(self) -> float:
        """Get the current learning rate (constant for this implementation)."""
        return self.learning_rate

    def set_learning_rate(self, learning_rate: float):
        """Update the learning rate in the optimizer."""
        self.learning_rate = learning_rate
        for param_group in self.learner.optimizer.param_groups:
            param_group["lr"] = learning_rate

    # ────────────────────────────────────────────────────────────
    #  Get/Set Policy
    # ────────────────────────────────────────────────────────────

    def get_policy(self) -> Any:
        """Return the policy (DuelingDoubleDQNBranching network)."""
        return self.online_network

    def set_policy(self, policy: DuelingDoubleDQNBranching):
        """Replace the online network with a new policy."""
        self.online_network = policy.to(self.device)
        self.target_network = DuelingDoubleDQNBranching(
            feature_extractor=policy.feature_extractor,
            features_dim=512,
            n_farmer_actions=policy.n_farmer_actions,
            n_hand_actions=policy.n_hand_actions,
            n_hands=policy.n_hands,
            n_market_actions=policy.n_market_actions,
            hidden_dim=256,
        ).to(self.device)
        self._hard_sync_target()

    # ────────────────────────────────────────────────────────────
    #  Serialization
    # ────────────────────────────────────────────────────────────

    def save(
        self,
        path: str,
        exclusion: Optional[set] = None,
        exclude_from_env: bool = False,
    ) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : str
            File path to save to.
        exclusion : set or None
            Attributes to exclude from serialization.
        exclude_from_env : bool
            Deprecated.
        """
        torch.save(
            {
                "policy_dict": self.online_network.state_dict(),
                "target_dict": self.target_network.state_dict(),
                "optimizer_dict": self.learner.optimizer.state_dict(),
                "hyperparams": {
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "batch_size": self.batch_size,
                    "buffer_size": self.buffer_size,
                    "learning_starts": self.learning_starts,
                    "train_freq": self.train_freq,
                    "target_update_interval": self.target_update_interval,
                    "exploration_fraction": self.exploration_fraction,
                    "exploration_initial_eps": self.exploration_initial_eps,
                    "exploration_final_eps": self.exploration_final_eps,
                    "verbose": self.verbose,
                    "seed": self.seed,
                    "use_priority_replay": self.use_priority_replay,
                    "priority_replay_alpha": self.priority_replay_alpha,
                    "priority_replay_beta_init": self.priority_replay_beta_init,
                    "priority_replay_beta_final": self.priority_replay_beta_final,
                    "priority_replay_anneal_steps": self.priority_replay_anneal_steps,
                    "use_soft_update": self.use_soft_update,
                    "tau": self.tau,
                    "max_grad_norm": self.max_grad_norm,
                    "policy_kwargs": self.policy_kwargs,
                    "device": str(self.device),
                    "optimizer_class": self.optimizer_class,
                    "optimizer_kwargs": self.optimizer_kwargs,
                    "step_count": self.learner.step_count,
                    "loss_history": self.learner.loss_history,
                },
            },
            path,
        )
        if self.verbose >= 1:
            print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Union[str, torch.device] = "auto",
        env: Any = None,
        **kwargs,
    ) -> "DQN":
        """Load a model from disk.

        Parameters
        ----------
        path : str
            File path to load from.
        device : str or torch.device
            Device to load the model to.
        env : Env or None
            Environment (not used for loading but kept for API compatibility).
        **kwargs : dict
            Additional keyword arguments passed to ``__init__``.

        Returns
        -------
        model : DQN
            The loaded model.
        """
        data = torch.load(path, map_location=device, weights_only=False)

        hyperparams = data["hyperparams"]
        policy_kwargs = hyperparams.pop("policy_kwargs", {})
        device_str = hyperparams.pop("device", device)

        # Remove unsupported kwargs
        for k in ["verbose", "seed"]:
            if k in hyperparams:
                pass  # already handled
        for k in list(hyperparams.keys()):
            if k not in ["learning_rate", "gamma", "batch_size",
                          "buffer_size", "learning_starts", "train_freq",
                          "target_update_interval", "exploration_fraction",
                          "exploration_initial_eps", "exploration_final_eps",
                          "verbose", "seed", "use_priority_replay",
                          "priority_replay_alpha", "priority_replay_beta_init",
                          "priority_replay_beta_final",
                          "priority_replay_anneal_steps", "use_soft_update",
                          "tau", "max_grad_norm", "device",
                          "optimizer_class", "optimizer_kwargs",
                          "step_count", "loss_history",
                          "exploration_decay_steps"]:
                hyperparams.pop(k, None)

        model = cls(
            policy="KaggricultureCNN",
            env=env,
            device=device_str,
            **hyperparams,
            policy_kwargs=policy_kwargs,
        )

        model.online_network.load_state_dict(data["policy_dict"])
        model.target_network.load_state_dict(data["target_dict"])
        model.learner.optimizer.load_state_dict(data["optimizer_dict"])
        model.learner.step_count = data["hyperparams"]["step_count"]
        model.learner.loss_history = data["hyperparams"]["loss_history"]

        return model

    # ────────────────────────────────────────────────────────────
    #  Callback support
    # ────────────────────────────────────────────────────────────

    def _on_training_start(self) -> None:
        """Hook called at the start of training."""
        pass

    def _on_rollout_start(self) -> None:
        """Hook called at the start of each rollout."""
        pass

    def _on_training_end(self) -> None:
        """Hook called at the end of training."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
