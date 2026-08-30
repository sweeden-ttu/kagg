# Stable Baselines3 — Overview

## Introduction

**Stable Baselines3 (SB3)** is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the third major version of the popular Stable Baselines library, rebuilt from the ground up to leverage modern PyTorch capabilities and improve code quality, performance, and usability.

SB3 aims to lower the barrier to entry for experimenting with reinforcement learning (RL) by providing clean, modular, and well-tested implementations of state-of-the-art algorithms.

### Key Features

| Feature | Description |
|---------|-------------|
| **PyTorch Backend** | Built natively on PyTorch for flexibility and performance |
| **Modern RL Algorithms** | PPO, SAC, TD3, DQN, A2C, DDPG, and their variants |
| **Gymnasium Support** | Full compatibility with the Gymnasium API (successor to OpenAI Gym) |
| **Vectorized Environments** | Built-in support for parallel environment stepping |
| **Callbacks System** | Extensible hooks for monitoring, checkpointing, and evaluation |
| **TensorBoard Integration** | Built-in training metrics logging |
| **Modular Design** | Clean separation of policy, environment, and training components |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      SB3 Architecture                            │
│                                                                  │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │  Environments │    │  VecEnv       │    │  Training Loop   │  │
│  │  (Gymnasium)  │───▶│  Wrappers     │───▶│  (on_policy     │  │
│  └──────────────┘    └───────────────┘    │   /off_policy)   │  │
│                                           └────────┬─────────┘  │
│                                                      │           │
│                                           ┌──────────▼─────────┐ │
│                                           │    Model (Agent)   │ │
│                                           │                    │ │
│  ┌──────────────┐    ┌───────────────┐   │  ┌──────────────┐  │ │
│  │  Callbacks   │◀───│  ReplayBuffer │   │  │  PolicyNet   │  │ │
│  │  & Logging   │    └───────────────┘   │  │  (Actor)     │  │ │
│  └──────────────┘                        │  └──────────────┘  │ │
│                                          │  ┌──────────────┐  │ │
│                                          │  │  ValueNet    │  │ │
│                                          │  │  (Critic)    │  │ │
│                                          │  └──────────────┘  │ │
│                                          └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Environments
SB3 works with any environment implementing the [Gymnasium](https://gymnasium.farama.org/) API:

```python
import gymnasium as gym

# Create a standard environment
env = gym.make("CartPole-v1")

# Create a vectorized environment for parallel training
from stable_baselines3.common.env_util import make_vec_env
vec_env = make_vec_env("CartPole-v1", n_envs=4)
```

The Gymnasium API requires:
- `reset()`: Reset environment to initial state, returns observation and info
- `step(action)`: Take action, returns (observation, reward, terminated, truncated, info)
- `observation_space`: `gym.spaces.Space` defining valid observations
- `action_space`: `gym.spaces.Space` defining valid actions

#### 2. Vectorized Environments (VecEnv)
VecEnv wraps multiple environments to enable parallel data collection:

```python
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

# Subprocess-based parallel environments
vec_env = make_vec_env("MountainCar-v0", n_envs=8, vec_env_cls=SubprocVecEnv)

# Frame stacking wrapper (for visual observations)
vec_env = VecFrameStack(vec_env, n_stack=4)
```

#### 3. Models (Agents)
Each algorithm is a class that wraps the policy network and training loop:

```
Model
  ├── Policy (Actor)
  │     ├── features_extractor (preprocesses observations)
  │     ├── net_arch (neural network architecture)
  │     └── action_distribution (maps values to actions)
  ├── Value Network (Critic)
  ├── ReplayBuffer (for off-policy algorithms)
  └── optimizer (learning rate, etc.)
```

#### 4. The `learn()` Interface
All models share a unified training interface:

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
model.learn(
    total_timesteps=10000,
    log_interval=1,
    callback=my_callback,       # Optional callback for monitoring
    progress_bar=True,          # Optional progress bar
)
```

---

## Actor-Critic Architecture

Many SB3 algorithms use the **Actor-Critic** paradigm, which combines policy gradient methods (Actor) with value function estimation (Critic).

### The Actor-Critic Framework

```
                    ┌─────────────┐
  Observation (s) ─▶│  Features   │
                    │  Extractor  │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐       ┌────────────────┐
     │  Actor (Policy) │       │  Critic (Value)│
     │  π(a|s; θ)     │       │  V(s; φ)      │
     └────────┬───────┘       └────────┬───────┘
              │                        │
              ▼                        ▼
         Action (a)             Value Estimate
                                   │
                                   ▼
                         ┌─────────────────┐
                         │   Advantage     │
                         │   A(s,a) =      │
                         │   Q(s,a) - V(s) │
                         └─────────────────┘
```

- **Actor**: Learns the policy — the probability distribution over actions given states
- **Critic**: Learns the value function — how good a state (or state-action pair) is
- **Advantage**: Tells us whether an action is better or worse than average

### Policy Networks

SB3 uses different network architectures depending on the observation space:

| Network Type | Observation Space | Use Case |
|-------------|-------------------|----------|
| `MlpPolicy` | `Box`, `Discrete`, `MultiDiscrete` | Tabular/low-dimensional observations |
| `CNNPolicy` | `Box` with image-like shape | Pixel observations (e.g., Atari) |
| Custom | Any | Extract custom features from complex observations |

### Value Networks

For algorithms that use a value function (PPO, SAC, A2C), the critic estimates:

- **State Value V(s)**: Expected return from state s
- **State-Action Value Q(s, a)**: Expected return from taking action a in state s

---

## The Learn Interface

All SB3 models expose the `learn()` method:

```python
model.learn(
    total_timesteps: int,           # Number of training steps
    reset_num_timesteps: bool = True,  # Reset step counter
    tb_log_name: str = None,        # TensorBoard log name
    progress_bar: bool = False,     # Show progress bar
    callback: BaseCallback = None,  # Callbacks to run
    log_interval: int = None,       # Logging interval
)
```

### Complete Training Example

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Create environments
train_env = make_vec_env("LunarLander-v2", n_envs=8)
eval_env = gym.make("LunarLander-v2")

# 2. Create model
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,
)

# 3. Set up callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix="ppo_lunarlander",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best/",
    log_path="./logs/eval/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
)

# 4. Train
model.learn(
    total_timesteps=100000,
    callback=[checkpoint_callback, eval_callback],
)

# 5. Evaluate
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# 6. Save the trained model
model.save("ppo_lunarlander_final")
```

---

## Algorithm Selection Guide

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Discrete actions, sample-efficient | SAC, PPO |
| Continuous control | SAC, TD3, PPO |
| Fast training, less sample efficient | PPO |
| Best sample efficiency (continuous) | SAC |
| Best sample efficiency (discrete) | DQN with Prioritized Replay |
| Multi-agent or complex rewards | PPO |

> **PPO** is the default recommendation for most use cases due to its balance of sample efficiency, stability, and ease of tuning. See [02-algorithms.md](02-algorithms.md) for detailed algorithm comparisons.

---

## Related Documentation

- [Algorithm Reference](02-algorithms.md) — Detailed breakdown of all algorithms
- [API Reference](03-api-reference.md) — Complete API documentation
- [Training Guide](04-training-guide.md) — Production training patterns
- [RL + CV Integration](../integration/01-rl-cv-integration.md) — Combining with keras-retinanet

---

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
