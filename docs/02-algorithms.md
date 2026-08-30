# Algorithm Reference

**Kaggriculture Path B trains hierarchical Dueling Double DQN** (`HierarchicalDQNBranching` + `HierarchicalDoubleDQNLearner` in `kaggriculture_path_b_rebuild.py`). It is not PPO, SAC, or stock SB3 `DQN`. The SB3 sections below are algorithm theory and optional `kaggriculture_rl.dqn_sb3` context.

Path B eval is `eval_policy.evaluate_ladder` vs `opponents/`, not SB3 `evaluate_policy`.

---

## Table of Contents

| Algorithm | Type | Action Space | Off-Policy | Paper |
|-----------|------|-------------|------------|-------|
| DQN | Value-based | Discrete | Yes | Mnih et al. 2015 |
| Double DQN | Value-based | Discrete | Yes | Hasselt et al. 2016 |
| Dueling DQN | Value-based | Discrete | Yes | Wang et al. 2016 |
| PPO | Policy-based | Both | No | Schulman et al. 2017 |
| A2C | Policy-based | Both | No | Mnih et al. 2016 |
| SAC | Policy-based | Continuous | Yes | Haarnoja et al. 2018 |
| TD3 | Policy-based | Continuous | Yes | Fujimoto et al. 2018 |
| DDPG | Policy-based | Continuous | Yes | Lillicrap et al. 2015 |

---

## 1. Deep Q-Network (DQN)

### Description
DQN learns a state-action value function Q(s, a) that predicts the expected cumulative discounted reward for taking action a in state s and following the optimal policy thereafter. It uses two key innovations: **Experience Replay** and a **Target Network**.

### Mathematical Formulation

The loss function is the mean squared error between the Q-value and the TD target:

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q(s, a; \theta) - y \right)^2 \right]$$

where the TD target is:
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

- $\mathcal{D}$ is the experience replay buffer
- $\theta^-$ are the target network parameters (delayed copy of $\theta$)
- $\gamma$ is the discount factor

### SB3 Implementation

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    "CartPole-v1",
    learning_rate=1e-3,
    buffer_size=100000,
    batch_size=64,
    epsilon_init=1.0,        # Initial exploration rate
    epsilon_final=0.05,      # Final exploration rate
    epsilon_decay_steps=100000,
    train_freq=4,            # Steps between each training update
    target_update_interval=1000,
    verbose=1,
)
```

### Key Hyperparameters

| Hyperparameter | Default | Description |
|---------------|---------|-------------|
| `learning_rate` | 1e-4 | Learning rate for the optimizer |
| `buffer_size` | 100000 | Size of the replay buffer |
| `batch_size` | 64 | Mini-batch size for training |
| `gamma` | 0.99 | Discount factor |
| `learning_starts` | 100 | Steps before learning begins |
| `train_freq` | 4 | Training frequency (every N steps) |
| `target_update_interval` | 1000 | Target network update interval |
| `epsilon_init` | 1.0 | Initial ε for ε-greedy exploration |
| `epsilon_final` | 0.05 | Final ε for ε-greedy exploration |
| `epsilon_decay_steps` | 100000 | Steps over which ε decays |
| `optimize_memory_usage` | False | Memory-efficient replay buffer |
| `gradient_steps` | 1 | Number of gradient steps per update |

### When to Use DQN
- ✅ Discrete action spaces
- ✅ When you need a baseline implementation
- ✅ When sample efficiency is not critical
- ❌ Continuous action spaces
- ❌ When you need maximum sample efficiency

---

## 2. Double DQN

### Description
Double DQN addresses the **overestimation bias** of standard DQN, where Q-values are systematically overestimated due to the max operator applied to both action selection and evaluation in the TD target:

$$y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$$

### The Fix

Double DQN decouples action selection from evaluation:

$$y = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)$$

- The **online network** ($\theta$) selects the best action
- The **target network** ($\theta^-$) evaluates that action
- This reduces the maximization bias significantly

### SB3 Implementation

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    "CartPole-v1",
    use_double_dqn=True,  # Enable Double DQN
    verbose=1,
)
```

### When to Use Double DQN
- ✅ Discrete action spaces
- ✅ When standard DQN converges too slowly or oscillates
- ✅ When you observe high variance in Q-value estimates
- ⚡ Drop-in replacement for DQN — use this instead of DQN by default

---

## 3. Dueling DQN

### Description
Dueling DQN modifies the network architecture to separately estimate the **state value** V(s) and the **advantage** A(s, a) of each action, then combines them to derive the Q-values:

$$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')\right)$$

The mean subtraction ensures that the advantage is centered around zero, so the value function remains identifiable.

### Architecture Diagram

```
                    Observation (s)
                           │
                    ┌──────┴──────┐
                    │  Shared     │
                    │  Layers     │
                    └──────┬──────┘
                   ┌───────┴───────┐
                   ▼               ▼
            ┌────────────┐  ┌────────────┐
            │  Value Head │  │ Adv. Head   │
            │   V(s)      │  │ A(s, a)    │
            └──────┬─────┘  └──────┬─────┘
                   │               │
                   └───────┬───────┘
                           ▼
                    Q(s, a) = V(s) + A(s, a) - mean(A)
```

### SB3 Implementation

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    "CartPole-v1",
    use_dueling=True,  # Enable Dueling DQN
    verbose=1,
)
```

### When to Use Dueling DQN
- ✅ Discrete action spaces
- ✅ When the state value matters more than individual action values
- ✅ Environments with many similar actions (some clearly bad, others similar)
- ✅ The Kaggriculture project uses this pattern

---

## 4. Proximal Policy Optimization (PPO)

### Description
PPO is an **on-policy** algorithm that uses a clipped surrogate objective to limit the size of policy updates. This ensures stable training while maintaining high sample efficiency. PPO is the recommended default algorithm for most problems.

### The Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right) \right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the estimated advantage at time t
- $\epsilon$ is the clip parameter (typically 0.2)

```
Policy Ratio r(θ)
  │
  │         Clip region
  │         ┌───────┐
  │         │       │
0.8─────────┤       ├─────────1.2  ← ε = 0.2
  │         │       │
  │         └───────┘
  │
  └──────────┼───────────────────
             │
          Clipped objective
          ensures that updates
          stay within a trust
          region around the
          old policy
```

### Key Features
- **Entropy Bonus**: Encourages exploration by adding entropy to the objective
- **Generalized Advantage Estimation (GAE)**: Low-variance advantage estimation
- **Multiple Epochs**: Multiple passes over collected data for data efficiency

### SB3 Implementation

```python
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    "Pendulum-v1",
    verbose=1,
    n_steps=2048,        # Steps per collection cycle
    batch_size=64,        # Mini-batch size
    n_epochs=10,          # Epochs per update
    gamma=0.99,
    learning_rate=3e-4,
    clip_range=0.2,       # Clip parameter ε
    ent_coef=0.0,         # Entropy coefficient
    vf_coef=0.5,          # Value function coefficient
    max_grad_norm=0.5,    # Gradient clipping norm
    gae_lambda=0.95,      # GAE lambda
)
```

### Key Hyperparameters

| Hyperparameter | Default | Description |
|---------------|---------|-------------|
| `n_steps` | 2048 | Steps to collect per environment per update |
| `batch_size` | 64 | Mini-batch size |
| `n_epochs` | 10 | Number of epoch iterations per update |
| `learning_rate` | 3e-4 | Initial learning rate |
| `clip_range` | 0.2 | Clip parameter ε |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `ent_coef` | 0.0 | Entropy loss coefficient |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping norm |
| `normalize_advantage` | True | Normalize advantages during training |

### When to Use PPO
- ✅ Generic SB3 experiments — **not** Kaggriculture Path B
- ✅ Both discrete and continuous action spaces
- ⚠️ Requires more samples than off-policy methods
- ❌ Do not swap Path B self-play for `PPO.learn` + gym `evaluate_policy`

---

## 5. Advantage Actor-Critic (A2C)

### Description
A2C is a synchronous, deterministic variant of Advantage Actor-Critic. It runs multiple parallel actors that collect trajectories independently, then aggregates gradients before updating the shared policy and value networks.

### How It Differs from PPO
- **No clipping objective** — uses plain policy gradient
- **Synchronous updates** — all workers update simultaneously
- **Generally requires more tuning** than PPO
- **Often faster per sample** due to synchronous parallelism

```python
from stable_baselines3 import A2C

model = A2C(
    "MlpPolicy",
    "CartPole-v1",
    verbose=1,
    n_steps=5,            # Steps per worker per update
    gamma=0.99,
    learning_rate=7e-4,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    n_devices=1,           # Number of parallel workers
)
```

### When to Use A2C
- ✅ When you have many parallel environments
- ✅ When computational speed matters more than sample efficiency
- ⚠️ Less robust than PPO — requires more careful tuning

---

## 6. Soft Actor-Critic (SAC)

### Description
SAC is an **off-policy** algorithm that maximizes both expected reward **and** entropy, encouraging exploration. It is currently the most sample-efficient algorithm for continuous control problems.

### Maximum Entropy RL

The standard RL objective maximizes expected return:
$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T r(s_t, a_t) \right]$$

SAC adds an entropy term:
$$\max_\theta \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right) \right]$$

where $\mathcal{H}(\pi(\cdot|s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|s_t)]$

### Key Features
- **Autoregulating Temperature**: The temperature parameter α is automatically tuned to maintain target entropy
- **Two Q-Networks**: Clipped double Q-learning reduces overestimation
- **Off-Policy**: Can reuse data from previous episodes
- **Continuous Actions**: Native support for continuous action spaces

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    "Pendulum-v1",
    verbose=1,
    buffer_size=int(1e6),
    batch_size=256,
    ent_coef="auto",       # Auto-tune temperature
    learning_rate=3e-4,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(
        log_std_init=-0.5,
        net_arch=[256, 256],
    ),
)
```

### Key Hyperparameters

| Hyperparameter | Default | Description |
|---------------|---------|-------------|
| `buffer_size` | 1e6 | Replay buffer size |
| `batch_size` | 256 | Mini-batch size |
| `ent_coef` | "auto" | Temperature coefficient ("auto" tunes automatically) |
| `learning_rate` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `train_freq` | 1 | Training frequency |
| `gradient_steps` | 1 | Gradient steps per update |
| `policy_kwargs.net_arch` | [256, 256] | Network architecture |

### When to Use SAC
- ✅ **Continuous action spaces** — best sample efficiency
- ✅ When you need maximum sample efficiency
- ✅ When the environment is expensive to evaluate
- ✅ Robotics, control systems, game AI
- ❌ Discrete action spaces — use DQN/PPO instead

---

## 7. Twin Delayed DDPG (TD3)

### Description
TD3 improves upon DDPG with three key tricks:
1. **Clipped Double Q-Learning**: Use two Q-networks and take the minimum to reduce overestimation bias
2. **Delayed Policy Updates**: Update the policy less frequently than the Q-networks
3. **Target Policy Smoothing**: Add noise to target actions to smooth the Q-function and prevent exploitation of Q-function errors

### The Three Tricks

```
Trick 1: Clipped Double Q
┌──────────┐  ┌──────────┐
│  Q₁      │  │  Q₂      │
│  (target)│  │  (target)│
└────┬─────┘  └────┬─────┘
     │              │
     ▼              ▼
  target₁     target₂
     │              │
     └──────┬───────┘
            ▼
         min(target₁, target₂)  ← Smaller Q-value

Trick 2: Delayed Policy Update
Policy update every N steps
Q-network update every step
(Usually N = 2)

Trick 3: Target Policy Smoothing
target_action = π(s') + noise(ε)
Noise smooths the Q-function
Prevents exploitation of sharp peaks
```

### SB3 Implementation

```python
from stable_baselines3 import TD3

model = TD3(
    "MlpPolicy",
    "Pendulum-v1",
    verbose=1,
    buffer_size=int(1e6),
    batch_size=100,
    learning_rate=1e-3,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_delay=2,       # Delay between policy updates
    target_policy_noise=0.2,  # Smoothing noise
    target_noise_clip=0.5,     # Noise clipping
)
```

### Key Hyperparameters

| Hyperparameter | Default | Description |
|---------------|---------|-------------|
| `policy_delay` | 2 | Steps between policy updates |
| `target_policy_noise` | 0.2 | Smoothing noise std dev |
| `target_noise_clip` | 0.5 | Noise clipping range |
| `buffer_size` | 1e6 | Replay buffer size |
| `batch_size` | 100 | Mini-batch size |
| `learning_rate` | 1e-3 | Learning rate |

### When to Use TD3
- ✅ Continuous action spaces
- ✅ When you need better stability than DDPG
- ✅ When DDPG training is unstable
- ⚡ Generally outperforms DDPG with no extra tuning cost

---

## 8. Deep Deterministic Policy Gradient (DDPG)

### Description
DDPG extends DQN to continuous action spaces using a deterministic policy (outputs a single action, not a distribution). It uses an **actor-critic** architecture with deterministic policy gradient.

### Mathematical Formulation

The deterministic policy gradient theorem states:

$$\nabla_\theta J(\mu) = \mathbb{E}_{s \sim \rho^\beta} \left[ \nabla_\theta \mu(s|\theta) \cdot \nabla_a Q(s, a|\theta^Q)|_{a=\mu(s)} \right]$$

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Actor (Policy Network)                          │
│  π(s) → continuous action a                     │
│  ┌──────────────────────────────────┐            │
│  │  State → Dense → Tanh → Action   │            │
│  └──────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  Critic (Q-Network)                              │
│  Q(s, a) → state-action value                   │
│  ┌──────────────────────────────────┐            │
│  │  State, Action → Dense → Value   │            │
│  └──────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

```python
from stable_baselines3 import DDPG

model = DDPG(
    "MlpPolicy",
    "Pendulum-v1",
    verbose=1,
    buffer_size=int(1e6),
    batch_size=256,
    ent_coef="auto",  # Not used by DDPG but required by SAC
    learning_rate=1e-3,
)
```

### When to Use DDPG
- ✅ Continuous action spaces (historical baseline)
- ❌ Prefer TD3 or SAC for new projects — they are strictly better

---

## Path B: Hierarchical Dueling Double DQN (Kaggriculture)

This is what `train_self_play` trains. Observations go through `KaggricultureFeatureExtractor` (CNN on a 10×10 tile grid + MLP on 55 numeric features → 512-d latent). The policy is `HierarchicalDQNBranching`:

```
Observation dict (tiles + numerics)
        │
        ▼
KaggricultureFeatureExtractor  →  latent (B, 512)
        │
        ▼
Shared dense (BatchNorm + ReLU)
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
V(s)                           Advantage branches
                               ├── farmer verb (15)
                               ├── crop parameter (5), conditioned on verb
                               ├── hands: 6 independent heads × 15
                               └── market: GRU, up to 10 sequential orders
```

Double Q-learning (same as §2): the **online** net selects next actions; the **target** net evaluates them. Dueling aggregation is per branch: \(Q_b = V + A_b - \mathrm{mean}(A_b)\).

A flat Discrete over the same space is intractable (\(15 \times 15^6 \times 10\)). Branching keeps one Q-head family per decision.

### Training loop (not SB3 `DQN.learn`)

1. **Bootstrap / BC** — `path_b_bootstrap.run_bc_pretrain_over_episode_files` (stream epochs = `bc_epochs_per_pass`, default 2), then optional buffer BC (`bc_epochs=15` when `bootstrap_passes ≤ 1`).
2. **Self-play** — `KaggleCompetitiveEnv` (`use_kaggle_env=True`), PER buffer 50% expert / 50% online.
3. **Ladder** — `evaluate_ladder` at `turns_per_day=24`.

### Knobs used in Path B

| Knob | Typical | Where |
|------|---------|--------|
| `batch_size` | 32 | `train_self_play` |
| `buffer_capacity` | 10_000 default; 50k–600k in notebook presets | PER |
| `bc_epochs_per_pass` | 2 | Stream BC per day |
| `bc_epochs` | 15 | Buffer BC after bootstrap |
| `max_episode_steps` | 720 | 30 × 24 competition day |
| `turns_per_cycle` | 24 | Must match reference agents for ladder |
| `learning_start_episodes` | 2 | Self-play SGD starts after this many episodes |

The optional `kaggriculture_rl.dqn.DuelingDoubleDQNBranching` stack is the **legacy flat-branch** net (farmer 15 + 6×15 hands + market 10, 122 Q-outputs). Path B’s hierarchical net is the one `train_self_play` instantiates.

---

## Algorithm Comparison Summary

| Feature | DQN | Double DQN | Dueling DQN | PPO | A2C | SAC | TD3 | DDPG |
|---------|-----|-----------|-------------|-----|-----|-----|-----|------|
| **Action Type** | Discrete | Discrete | Discrete | Any | Any | Continuous | Continuous | Continuous |
| **Policy Type** | Off-policy | Off-policy | Off-policy | On-policy | On-policy | Off-policy | Off-policy | Off-policy |
| **Sample Efficient** | ❌ | ❌ | ⚠️ | ⚠️ | ❌ | ✅ | ⚠️ | ❌ |
| **Training Stable** | ⚠️ | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ❌ |
| **Easy to Tune** | ✅ | ✅ | ✅ | ✅ | ❌ | ⚠️ | ⚠️ | ❌ |
| **Parallel Training** | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ |

---

## Related Documentation

- [Overview](01-overview.md) — Core concepts and architecture
- [API Reference](03-api-reference.md) — Complete API documentation
- [Training Guide](04-training-guide.md) — Production training patterns
- [RL + CV Integration](../integration/01-rl-cv-integration.md) — Combining with keras-retinanet

---

## References

- Mnih et al., "Human-level control through deep reinforcement learning" (Nature 2015)
- Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
- Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)
- Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
