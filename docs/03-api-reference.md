# Stable Baselines3 — API Reference

This document provides a comprehensive reference for the stable-baselines3 API, covering model persistence, vectorized environments, callbacks, replay buffers, policy architectures, and custom feature extraction patterns used in the Kaggriculture project.

---

## Table of Contents

- [Model Loading & Saving](#model-loading--saving)
- [Vectorized Environments](#vectorized-environments)
- [VecEnvWrappers](#vecenvwrappers)
- [Callbacks](#callbacks)
- [ReplayBuffer](#replaybuffer)
- [Policy Networks](#policy-networks)
- [Feature Extractors](#feature-extractors)
- [Custom KaggricultureFeatureExtractor](#custom-kaggriculturefeatureextractor)

---

## Model Loading & Saving

### Saving Models

```python
from stable_baselines3 import PPO, SAC, DQN
import gymnasium as gym

model = PPO("MlpPolicy", gym.make("CartPole-v1"), verbose=1)
model.learn(total_timesteps=10000)

# Save the model (includes model weights and environment info)
model.save("ppo_cartpole")

# For cloud/storage, save to directory
model.save("./checkpoints/ppo_cartpole_v1")
```

### Loading Models

```python
from stable_baselines3 import PPO
import gymnasium as gym

# Load model — the environment info is stored in the .zip file
model = PPO.load("ppo_cartpole", device="cpu")  # or "cuda" for GPU

# Load with custom environment (if env config changed)
env = gym.make("CartPole-v1")
model.set_env(env)

# Load from bytes (useful for APIs/cloud)
import io
with open("ppo_cartpole.zip", "rb") as f:
    model = PPO.load(f, device="cuda")
```

### Key `Model` Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `model.save(path)` | Save model to file/directory | None |
| `PPO.load(path, ...)` | Load a saved model | Model instance |
| `model.predict(obs, ...)` | Get action from observation | (action, state) |
| `model.predict(obs, deterministic=True)` | Deterministic prediction | (action, state) |
| `model.policy(obs)` | Forward pass through policy | Policy output |
| `model.policy.features_extractor(obs)` | Get extracted features | Feature tensor |

### Predict API

```python
import numpy as np

obs = env.reset()
action, _states = model.predict(obs, deterministic=False)

# For vectorized environments
obs = np.array([env.reset()[0] for _ in range(4)])
action, _states = model.predict(obs, deterministic=True)
```

---

## Vectorized Environments

### VecEnv — The Abstraction

VecEnv provides a uniform interface for running multiple environments in parallel. It abstracts away the details of synchronization, subprocessing, and frame stacking.

```python
from stable_baselines3.common.vec_env import (
    VecEnv,
    VecEnvWrapper,
    DummyVecEnv,
    SubprocVecEnv,
)

# DummyVecEnv — runs environments in the same process
env = DummyVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])

# SubprocVecEnv — runs environments in separate subprocesses
# Better for environments that use threads or have complex state
env = SubprocVecEnv([lambda: gym.make("CartPole-v1") for _ in range(4)])
```

### Creating Vectorized Environments

```python
from stable_baselines3.common.env_util import make_vec_env, make_atari_env

# Simple wrapper
env = make_vec_env("CartPole-v1", n_envs=8)

# With SubprocVecEnv (faster for heavy environments)
env = make_vec_env("CartPole-v1", n_envs=8, vec_env_cls=SubprocVecEnv)

# With custom seed
env = make_vec_env("CartPole-v1", n_envs=4, seed=42)

# Atari environments (with frame stacking and other wrappers)
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=8, seed=0)

# Custom environment factories
def make_env(rank):
    def _init():
        env = gym.make("MyCustomEnv-v0")
        return env
    return _init

env = make_vec_env(make_env, n_envs=4, vec_env_cls=SubprocVecEnv)
```

### VecEnv Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `step_wait()` | Step all envs, wait for results | (obs, rewards, dones, infos) |
| `step_async()` | Step all envs asynchronously | None |
| `reset()` | Reset all environments | observations |
| `render(mode="rgb_array")` | Render environments | image array |
| `observation_space` | Observation space | gym.spaces.Space |
| `action_space` | Action space | gym.spaces.Space |

---

## VecEnvWrappers

VecEnvWrappers modify observations, actions, or rewards across all vectorized environments.

### Common Wrappers

```python
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecTransposeImage,
    VecNormalize,
    VecReshapeObs,
)

# Frame Stacking — stack N consecutive frames (for visual observation)
# Example: Stack 4 frames for Atari-style input
env = VecFrameStack(env, n_stack=4)

# Observation: shape (batch, H, W, C×4) or (batch, H×4, W, C)
# Useful when the RL algorithm expects sequential visual input

# Transpose Image — change image channel dimension order
env = VecTransposeImage(env)
# Converts (batch, H, W, C) → (batch, C, H, W)
# Required for PyTorch CNNPolicy which expects channels-first

# VecNormalize — online normalization of observations and rewards
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
# Saves/loads normalization statistics
env.save("./vec_normalize")
env = VecNormalize.load("./vec_normalize", env)
```

### Building a Preprocessing Pipeline

```python
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    VecNormalize,
)
import gymnasium as gym

def make_vec_env(observation_space_type="image"):
    """Build a complete vectorized environment pipeline."""

    def make_env():
        return gym.make("BreakoutNoFrameskip-v4")

    # 1. Create parallel environments
    if observation_space_type == "image":
        env_cls = SubprocVecEnv  # Subprocess for image environments
    else:
        env_cls = DummyVecEnv

    env = make_vec_env("BreakoutNoFrameskip-v4", n_envs=8, vec_env_cls=env_cls)

    # 2. Apply wrappers
    if observation_space_type == "image":
        env = VecFrameStack(env, n_stack=4)  # Stack 4 frames
        env = VecTransposeImage(env)          # HWC → CHW for PyTorch
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    return env

env = make_vec_env("image")
```

---

## Callbacks

Callbacks are hooks that run at various points during training. They are the primary mechanism for monitoring, checkpointing, and evaluating training progress.

### Base Callback Pattern

```python
from stable_baselines3.common.callbacks import BaseCallback

class MyCustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Called after every environment step
        if self.n_calls % 1000 == 0:
            mean_reward = self.model.ep_info_buffer[-1]["r"]
            print(f"Step {self.n_calls}: Mean episode reward = {mean_reward:.2f}")
        return True  # Continue training

    def _on_training_end(self) -> None:
        # Called when training ends
        self.model.save("final_model")
```

### Common Callbacks

#### CheckpointCallback

Save the model periodically during training:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=10000,              # Save every 10,000 steps
    save_path="./logs/",          # Directory for checkpoints
    name_prefix="ppo_cartpole",   # Prefix for checkpoint files
    save_replay_buffer=False,     # Save replay buffer (off-policy only)
    save_vecnormalize=False,      # Save VecNormalize stats
)

# Results:
# ./logs/ppo_cartpole_10000_steps.zip
# ./logs/ppo_cartpole_20000_steps.zip
# ...
```

#### EvalCallback

Evaluate the model periodically against a target environment:

```python
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

eval_env = gym.make("CartPole-v1")

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model/",
    log_path="./logs/eval/",
    eval_freq=5000,                # Evaluate every 5,000 steps
    n_eval_episodes=10,            # Number of episodes per evaluation
    deterministic=True,            # Use deterministic policy
    render=False,                  # Don't render (too slow)
    callback_after_eval=None,      # Optional callback after eval
)

# EvalCallback automatically:
# - Saves the best model based on mean reward
# - Logs episode rewards to TensorBoard
# - Can automatically save checkpoints when a new best is found
```

#### TensorBoardCallback

Log training metrics to TensorBoard:

```python
from stable_baselines3.common.callbacks import TensorBoardCallback

tb_callback = TensorBoardCallback(
    log_path="./logs/tensorboard/",
    reset_num_timesteps=False,     # Continue from previous logging
)

# Use with learn():
model.learn(total_timesteps=50000, callback=[eval_callback, tb_callback])

# Then view in browser:
# tensorboard --logdir ./logs/tensorboard/
```

### Combining Callbacks

```python
model.learn(
    total_timesteps=100000,
    callback=[
        CheckpointCallback(save_freq=10000, save_path="./logs/"),
        EvalCallback(eval_env, eval_freq=5000, best_model_save_path="./logs/best/"),
        TensorBoardCallback(log_path="./logs/tb/"),
        MyCustomCallback(),
    ],
)
```

### Full Kaggriculture Training Pipeline Callbacks

```python
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    TensorBoardCallback,
    ProgressBarCallback,
)

# Create all callbacks
callbacks = CallbackList([
    CheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints/",
        name_prefix="dqn_kaggriculture",
    ),
    EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best/",
        eval_freq=5000,
        n_eval_episodes=20,
        deterministic=True,
    ),
    TensorBoardCallback(
        log_path="./logs/tb/",
    ),
    ProgressBarCallback(),  # Shows a progress bar during training
])

model.learn(total_timesteps=500000, callback=callbacks)
```

---

## ReplayBuffer

Replay buffers store transition tuples (state, action, reward, next_state, done) for off-policy learning.

### Base ReplayBuffer

```python
from stable_baselines3.common.buffers import ReplayBuffer

buffer = ReplayBuffer(
    buffer_size=100000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device="cpu",  # or "cuda"
    optimize_memory_usage=False,
)

# Add transitions
for _ in range(1000):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(
        obs=np.array(env.obs),
        next_obs=np.array(next_obs),
        action=action,
        reward=reward,
        terminal=terminated or truncated,
    )

# Sample a batch
observations, actions, rewards, next_observations, dones = buffer.sample(batch_size=64)
```

### Prioritized Experience Replay (PER)

```python
from stable_baselines3.common.buffers import PrioritizedReplayBuffer

per_buffer = PrioritizedReplayBuffer(
    buffer_size=100000,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device="cpu",
    alpha=0.6,    # How much to prioritize (0 = uniform, 1 = fully prioritized)
    beta=0.4,     # Importance sampling exponent (increases during training)
)

# Add works the same as ReplayBuffer
buffer.add(...)

# Sample with priority weighting
observations, actions, rewards, next_observations, dones, weights, indices = buffer.sample(batch_size=64)
# weights are used to weight the TD loss for each sample
```

### DictReplayBuffer (for Dict observation spaces)

```python
from stable_baselines3.common.buffers import DictReplayBuffer

buffer = DictReplayBuffer(
    buffer_size=100000,
    observation_space=dict_observation_space,
    action_space=env.action_space,
)
```

### PrioritizedDictReplayBuffer

```python
from stable_baselines3.common.buffers import PrioritizedDictReplayBuffer

buffer = PrioritizedDictReplayBuffer(
    buffer_size=100000,
    observation_space=dict_observation_space,
    action_space=env.action_space,
    alpha=0.6,
    beta=0.4,
)
```

---

## Policy Networks

### MlpPolicy

For tabular/low-dimensional observations (Box, Discrete, MultiDiscrete spaces):

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    "CartPole-v1",
    verbose=1,
    policy_kwargs=dict(
        net_arch=[
            dict(pi=[256, 256], vf=[256, 256])
        ],  # Actor and Critic networks
        activation_fn=torch.nn.ReLU,
    ),
)
```

### CNNPolicy

For pixel observations (typically image-like Box spaces):

```python
from stable_baselines3 import PPO

model = PPO(
    "CNNPolicy",
    "BreakoutNoFrameskip-v4",
    verbose=1,
    policy_kwargs=dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
    ),
)
```

### Custom Policy Network Architecture

```python
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Define a custom network architecture
policy_kwargs = dict(
    net_arch=[
        dict(pi=[512, 256, 128], vf=[512, 256, 128])
    ],  # Separate actor and critic networks
    activation_fn=nn.GELU,
    features_extractor_class=CustomFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=256),
)
```

---

## Feature Extractors

The features extractor preprocesses raw observations into a fixed-dimensional representation used by both the policy and value networks.

### BaseFeaturesExtractor

All custom feature extractors inherit from `BaseFeaturesExtractor`:

```python
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor combining CNN and MLP layers."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Define your feature extraction layers here
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute the output size of the CNN
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(
                    observation_space.sample()[None]
                ).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
```

### Using Pre-built Extractors

```python
from stable_baselines3.common.torch_layers import (
    MultiInputFeaturesExtractor,  # For dict/multi-modal observations
    FlattenExtract,                  # Simple flatten (default for MlpPolicy)
)

# Multi-input feature extractor (e.g., combining image + vector observations)
from stable_baselines3.common.torch_layers import MultiInputFeaturesExtractor

policy_kwargs = dict(
    features_extractor_class=MultiInputFeaturesExtractor,
    features_extractor_kwargs=dict(
        features_dim=256,
    ),
)
```

---

## Custom KaggricultureFeatureExtractor

The Kaggriculture project uses a sophisticated feature extractor designed for a branching action space with multi-stage crop decision making. This extractor processes environmental observations (soil properties, weather, crop data) into a fixed-dimensional representation.

### Architecture

```
Observation Vector (n_features)
         │
    ┌────┴────┐
    │  Layer 1 │  (batch norm + ReLU)
    │  256 → 128│
    └────┬────┘
         │
    ┌────┴────┐
    │  Layer 2 │  (batch norm + GELU)
    │  128 → 64 │
    └────┬────┘
         │
    ┌────┴────┐
    │  Layer 3 │  (batch norm + ReLU)
    │  64 → 32 │
    └────┬────┘
         │
         ▼
  Fixed Dim Features (32)
         │
    ┌────┴────┐        ┌───────────────┐
    │  Actor   │───────▶│  Action Branch│
    │  Network │        │  (Stage 1 + 2)│
    └──────────┘        └───────────────┘
         │
    ┌────┴────┐        ┌───────────────┐
    │  Critic  │───────▶│  Value Network│
    │  Network │        └───────────────┘
    └──────────┘
```

### Implementation

```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KaggricultureFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the Kaggriculture branching action space problem.

    Processes environmental observations (soil, weather, crop, field data)
    into a fixed-dimensional representation for the Dueling DQN policy.

    Args:
        observation_space: Gymnasium observation space
        features_dim: Dimension of output features (default: 64)
    """

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)

        n_input = observation_space.shape[0]

        self.network = nn.Sequential(
            # Layer 1: Batch normalization stabilizes training
            nn.Linear(n_input, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Layer 2: GELU provides smoother gradients
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Layer 3: Further compression
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Output layer
            nn.Linear(32, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor."""
        return self.network(observations)


# Usage with Dueling DQN
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import PrioritizedReplayBuffer

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=dict(
        features_extractor_class=KaggricultureFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[32],  # Dueling DQN single-value branch size
    ),
    replay_buffer_class=PrioritizedReplayBuffer,
    replay_buffer_kwargs=dict(alpha=0.6, beta=0.4),
    use_dueling=True,
    use_double_dqn=True,
    verbose=1,
)
```

### Branching Action Space Wrapper

```python
from stable_baselines3.common.envs import ActionSpaceWrapper
from gymnasium import spaces


class BranchingActionWrapper(gym.Wrapper):
    """
    Wraps an environment to use a branching action space.

    The branching space decomposes a large discrete action space
    into multiple stages:
      Stage 1: Choose crop type (0-4, 5 options)
      Stage 2: Choose treatment (0-199, 200 options per crop)

    Effective action space: 5 × 200 = 1000 discrete actions
    """

    def __init__(self, env):
        super().__init__(env)
        n_groups = 5       # Number of crop groups
        n_actions_per_group = 200  # Actions per group
        self.n_groups = n_groups
        self.n_actions_per_group = n_actions_per_group

        self.action_space = spaces.MultiDiscrete(
            [n_groups, n_actions_per_group]
        )

    def step(self, action):
        # Flatten branching action to original action
        flat_action = action[0] * self.n_actions_per_group + action[1]
        return self.env.step(flat_action)
```

---

## Related Documentation

- [Overview](01-overview.md) — Core concepts and architecture
- [Algorithm Reference](02-algorithms.md) — Algorithm details and comparisons
- [Training Guide](04-training-guide.md) — Production training patterns
- [RL + CV Integration](../integration/01-rl-cv-integration.md) — Combining with keras-retinanet
