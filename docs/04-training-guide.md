# Stable Baselines3 — Training Guide

This document covers training best practices for stable-baselines3, including hyperparameter tuning strategies, monitoring with TensorBoard, distributed training, checkpointing, deployment patterns, and the bootstrap-from-dataset technique used in the Kaggriculture training pipeline.

**Kaggriculture Path B is the required training path.** Use `kaggriculture_self_play_training.train_self_play` with daily-incremental episode-JSON bootstrap, then league eval via `eval_policy.evaluate_ladder`. The PPO / SAC / Optuna / FastAPI snippets below are generic SB3 reference — they are not how this repo trains or scores win rate. See [Bootstrap-from-Dataset Pattern](#bootstrap-from-dataset-pattern).

---

## Table of Contents

- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Monitoring with TensorBoard](#monitoring-with-tensorboard)
- [Distributed Training](#distributed-training)
- [Checkpointing & Resuming](#checkpointing--resuming)
- [Deployment & Export](#deployment--export)
- [Bootstrap-from-Dataset Pattern](#bootstrap-from-dataset-pattern)
- [Common Training Issues](#common-training-issues)

---

## Hyperparameter Tuning

### The SB3 Hyperparameter Hierarchy

Not all hyperparameters are equally important. Focus on these in order:

```
Critical (tune first)
├── Learning rate
├── Network architecture (net_arch)
├── Batch size
└── n_steps / buffer_size

Important (tune second)
├── γ (gamma) — discount factor
├── ε (clip range for PPO)
├── GAE lambda
└── Entropy coefficient

Fine-tuning (tune last)
├── vf_coef
├── max_grad_norm
└── train_freq
```

### PPO Hyperparameter Guide

```python
from stable_baselines3 import PPO

# ── PPO Default (good starting point) ──────────────────────────
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=2048,       # 2048 steps per environment per update
    batch_size=64,       # 64 samples per mini-batch
    n_epochs=10,         # 10 epochs per update cycle
    learning_rate=3e-4,  # 0.0003
    clip_range=0.2,      # 0.2 clip epsilon
    gamma=0.99,          # 0.99 discount
    gae_lambda=0.95,     # 0.95 GAE
    ent_coef=0.0,        # No entropy bonus (or 0.01 for more exploration)
    vf_coef=0.5,         # 0.5 value loss weight
    max_grad_norm=0.5,   # 0.5 gradient clipping
)

# ── Tuned for CartPole (short episodes) ────────────────────────
model = PPO(
    "MlpPolicy", env,
    n_steps=512,          # Fewer steps for short episodes
    batch_size=64,
    n_epochs=20,          # More epochs for small batches
    learning_rate=5e-4,   # Slightly higher LR
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.01,        # Small entropy bonus for exploration
)

# ── Tuned for Hard Environments ────────────────────────────────
model = PPO(
    "MlpPolicy", env,
    n_steps=4096,         # More steps per update (better gradient signal)
    batch_size=256,       # Larger batch for stability
    n_epochs=5,           # Fewer epochs to prevent overfitting
    learning_rate=1e-4,   # Lower LR for stability
    clip_range=0.1,       # Tighter clipping for conservative updates
    gamma=0.995,          # Slightly higher discount for long-horizon
    gae_lambda=0.98,      # Higher GAE lambda for lower bias
    ent_coef=0.005,       # Small entropy bonus
    vf_coef=0.25,         # Lower value loss weight
    max_grad_norm=0.3,    # Tighter gradient clipping
)
```

### SAC Hyperparameter Guide

```python
from stable_baselines3 import SAC

# ── SAC Default ─────────────────────────────────────────────────
model = SAC(
    "MlpPolicy", env,
    verbose=1,
    buffer_size=int(1e6),
    batch_size=256,
    ent_coef="auto",      # Automatic temperature tuning
    learning_rate=3e-4,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs=dict(
        net_arch=[256, 256],
        log_std_init=-0.5,
    ),
)

# ── Tuned for Continuous Control ────────────────────────────────
model = SAC(
    "MlpPolicy", env,
    buffer_size=int(1e6),
    batch_size=512,               # Larger batch for continuous control
    ent_coef="auto",
    learning_rate=7e-4,           # Higher LR for policy
    gamma=0.99,
    train_freq=4,                 # Train every 4 steps
    gradient_steps=4,             # 4 gradient steps per update
    policy_kwargs=dict(
        net_arch=[512, 256, 128],  # Deeper network
        log_std_init=-2.0,         # Initial std dev = exp(-2) ≈ 0.135
    ),
)
```

### Path B hyperparameter guide (required)

Tune these on `train_self_play`, not SB3 `DQN(...)`:

```
Critical
├── bc_epochs_per_pass (stream BC per day; default 2)
├── bc_epochs (buffer BC; default 15; skipped if bootstrap_passes > 1)
├── buffer_capacity + batch_size
└── learning rate inside HierarchicalDoubleDQNLearner

Important
├── bootstrap_days_per_run / bootstrap_mode
├── turns_per_cycle = 24 (ladder parity)
└── max_episode_steps = 720

Eval (not gym reward)
└── n_eval_episodes / ladder_eval_episodes vs opponents/
```

```python
from kaggriculture_self_play_training import train_self_play

train_self_play(
    use_kaggle_env=True,
    bootstrap_mode="daily_incremental",
    bootstrap_episodes=None,
    metadata_path="working/kaggle_episodes/metadata.json",
    data_dir="working/kaggle_episodes",
    bc_epochs_per_pass=2,
    bc_epochs=15,
    batch_size=32,
    buffer_capacity=50_000,
    max_episode_steps=720,
    turns_per_cycle=24,
    n_eval_episodes=10,
)
```

### SB3 DQN (legacy wrapper only)

`kaggriculture_rl.dqn_sb3.DQN` is optional and unused by Path B. Do not treat the following as the training entry point.

### Using Optuna for Automated Tuning

```python
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym


def objective(trial):
    """Optuna objective function for hyperparameter tuning."""

    # Define the search space
    hparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_steps": trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024]),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "clip_range": trial.suggest_float("clip_range", 0.05, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "net_arch": trial.suggest_categorical("net_arch", [
            [64, 64],
            [128, 128],
            [256],
            [256, 128],
            [128, 64, 64],
        ]),
    }

    # Create environment
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])

    # Create and train model
    model = PPO("MlpPolicy", env, verbose=0, **hparams)
    model.learn(total_timesteps=50000)

    # Evaluate
    rewards = []
    for _ in range(10):
        obs = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)

    return sum(rewards) / len(rewards)


# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
```

---

## Monitoring with TensorBoard

### Setting Up TensorBoard

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import TensorBoardCallback

model = PPO("MlpPolicy", "Pendulum-v1", verbose=1)

tb_callback = TensorBoardCallback(
    log_path="./logs/tensorboard/",
    reset_num_timesteps=False,  # Continue logging across runs
)

model.learn(total_timesteps=100000, callback=tb_callback)
```

### Viewing Training Metrics

```bash
# Start TensorBoard server
tensorboard --logdir ./logs/tensorboard/

# Open in browser
# Go to http://localhost:6006
```

### Key Metrics to Monitor

```
Policy Loss
├── entropy_loss        — Policy entropy (monitor exploration)
├── policy_loss         — Policy gradient loss
└── value_loss          — Critic value loss

Environment
├── ep_rew_mean         — Mean episode reward ← PRIMARY METRIC
├── ep_rew_std          — Std of episode reward
├── ep_len_mean         — Mean episode length
└── time_fps            — Steps per second

Advantage / GAE
├── approx_kl           — Approximate KL divergence (should stay < 0.01-0.02)
└── clip_fraction       — Fraction of samples clipped (monitor constraint usage)
```

### Interpreting TensorBoard Charts

```
Good Training Pattern:
  ep_rew_mean → ↑ steadily, with occasional dips during evaluation
  policy_loss → decreases over time (fluctuating is normal)
  value_loss  → decreases over time
  approx_kl   → stays low (0.001 - 0.01 range)
  entropy_loss → decreases or stays stable (not increasing!)

Bad Training Pattern (needs attention):
  ep_rew_mean → flat or decreasing
  approx_kl   → spikes > 0.1 (policy updates too large)
  entropy_loss → increasing (policy is becoming random)
  value_loss  → very high (critic is not learning well)
```

---

## Distributed Training

### Vectorized Environments for Parallel Data Collection

The primary way to scale data collection in SB3 is through vectorized environments:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 8 parallel environments (each in a subprocess)
env = make_vec_env(
    "LunarLander-v2",
    n_envs=8,
    vec_env_cls=SubprocVecEnv,  # Separate processes
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### Scaling Up

```python
# For environments that can handle threading
from stable_baselines3.common.vec_env import VecSubprocFrameStack, VecFrameStack

env = make_vec_env("MyEnv-v0", n_envs=32, vec_env_cls=SubprocVecEnv)
env = VecFrameStack(env, n_stack=4)

# Effective: 32 environments × 4 stacks = 128 frames of data per step
```

### Multi-GPU Training

```python
import torch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model on GPU
model = PPO(
    "MlpPolicy",
    "Pendulum-v1",
    device=device,
    verbose=1,
)

model.learn(total_timesteps=100000)

# Check GPU usage
# nvidia-smi
```

### Ray RLlib Integration (Advanced)

```python
# For distributed training across multiple machines,
# consider using Ray RLlib with SB3 policies:

from ray.rllib.algorithms.ppo import PPO as RayPPO
from ray.rllib.env.wrappers.sb3 import SB3EnvWrapper

# See Ray RLlib documentation for full setup
# Ray scales to hundreds of workers across clusters
```

---

## Checkpointing & Resuming

### Path B (required)

`train_self_play` writes `checkpoints/training_state_latest.pt` (full state) and `config.json` (`last_completed_episode`). `--total-episodes` is cumulative.

```bash
python kaggriculture_self_play_training.py \
  --resume experiments/my_run \
  --total-episodes 40 \
  --min-self-play-episodes 5
```

`daily_incremental` still runs bootstrap on resume (new calendar days). Other modes skip bootstrap on a full resume when the buffer is already populated.

### SB3 Automatic Checkpointing (not Path B)

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=50000,                  # Save every 50k steps
    save_path="./checkpoints/",
    name_prefix="ppo_lunarlander",
    save_replay_buffer=True,          # For off-policy algorithms
    save_vecnormalize=True,           # Save normalization stats
)

model.learn(total_timesteps=500000, callback=checkpoint_callback)
```

### Resuming Training

```python
from stable_baselines3 import PPO

# Load the last checkpoint
model = PPO.load("./checkpoints/ppo_lunarlander_last", verbose=1)

# Continue training from where you left off
model.learn(total_timesteps=500000)  # Adds to existing step count

# To reset the step counter:
model.learn(total_timesteps=500000, reset_num_timesteps=True)
```

### Manual Checkpoint Management

```python
import os
from stable_baselines3.common.callbacks import BaseCallback


class SmartCheckpointCallback(BaseCallback):
    """
    Saves checkpoints with size management.
    Keeps only the last N checkpoints and the best model.
    """

    def __init__(self, save_freq=10000, save_path="./checkpoints/", max_checkpoints=5):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.max_checkpoints = max_checkpoints

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"model_{self.n_calls}_steps",
            )
            self.model.save(path)

            # Clean old checkpoints
            checkpoints = sorted(
                [f for f in os.listdir(self.save_path) if f.startswith("model_")],
                key=lambda x: int(x.split("_")[-2]),  # Sort by step number
            )
            while len(checkpoints) > self.max_checkpoints:
                os.rmdir(os.path.join(self.save_path, checkpoints.pop(0)))

        return True
```

### Resuming from a Dataset (Bootstrap)

See the [Bootstrap-from-Dataset Pattern](#bootstrap-from-dataset-pattern) section for loading pre-collected data into the replay buffer before training.

---

## Deployment & Export

### Saving for Inference

```python
from stable_baselines3 import PPO
import torch

model = PPO.load("best_model", device="cpu")

# ── Option 1: SB3 ZIP format (recommended for SB3 environments) ──
model.save("deploy_model")
# Load with: model = PPO.load("deploy_model")

# ── Option 2: TorchScript (for high-performance inference) ────
# Convert policy to TorchScript
traced_policy = torch.jit.trace(
    model.policy.forward,
    next(iter(model.policy.observation_space.spaces.values()).sample()[None])
    if hasattr(model.policy.observation_space, 'spaces')
    else model.policy.observation_space.sample()[None]
)
traced_policy.save("policy_torchscript.pt")

# ── Option 3: ONNX export ──────────────────────────────────────
# For cross-platform deployment
import onnx

sample_input = model.policy.observation_space.sample()[None]
torch.onnx.export(
    model.policy,
    sample_input,
    "policy.onnx",
    input_names=["observation"],
    output_names=["action"],
    dynamic_axes={"observation": {0: "batch_size"}},
)

# ── Option 4: Custom inference script ──────────────────────────
import numpy as np

def predict_action(model, observation, deterministic=True):
    """Simple inference function for deployment."""
    action, _ = model.predict(observation, deterministic=deterministic)
    return action
```

### Serving with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import PPO

app = FastAPI()
model = PPO.load("deploy_model", device="cpu")


class PredictionRequest(BaseModel):
    observation: list[float]
    deterministic: bool = True


class PredictionResponse(BaseModel):
    action: int
    probability: float = None


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    obs = np.array(req.observation)
    action, _ = model.predict(obs, deterministic=req.deterministic)
    return PredictionResponse(action=int(action))


# Run: uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy_model.zip .
COPY api.py .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Bootstrap-from-Dataset Pattern

The Kaggriculture training pipeline uses a **bootstrap-from-dataset** pattern to initialize training with real data from a dataset. This dramatically speeds up convergence by providing the agent with meaningful initial behavior, rather than starting from random exploration.

Path B does **not** load a CSV into an SB3 `ReplayBuffer`. It streams Kaggle episode JSONs through behavioral cloning, seeds the hierarchical replay buffer, then runs self-play. The entry point is `kaggriculture_self_play_training.train_self_play`; the loaders live in `path_b_bootstrap.py`.

### The Problem

Standard RL starts with **random exploration**:
```
Step 0-1000:    Random actions → most transitions are noise
Step 1000-10000: Gradually learns from experience
Step 10000+:    Converges to a reasonable policy
```

With bootstrap, we start with **informed behavior**:
```
Day stream:     BC over expert episode JSONs (log: BC stream epoch N/M)
Buffer BC:      Extra epochs on the seeded replay buffer (when bootstrap_passes ≤ 1)
Self-play:      DQN on mixed expert + online transitions
Ladder:         Win rate vs opponents/ — not a single-agent gym eval
```

### How Path B bootstraps

1. Resolve episode files from `metadata.json` + `working/kaggle_episodes/` (or the Kaggle dataset).
2. Stream transitions through `run_bc_pretrain_over_episode_files` — one shuffle through that day's JSONs per stream epoch. `--bc-epochs-per-pass` (default **2**) is this loop; `epoch 1/2 step 8200` means ~8200 optimizer steps in the first pass, not a tiny update.
3. Seed the replay buffer from the same files (`seed_buffer_from_episode_files`).
4. Record progress in `metrics/bootstrap_state.json` (`bootstrapped_dates`, `total_transitions`, `runs`) so later runs skip days already done.
5. If `bc_epochs > 0` and `bootstrap_passes ≤ 1`, run `run_bc_pretrain` on the buffer (default **15** epochs). Streaming with `bootstrap_passes > 1` skips this stage and uses per-pass stream epochs only.
6. Self-play DQN, then league eval via `eval_policy.evaluate_ladder`.

### Modes

| `bootstrap_mode` | When | Behavior |
|------------------|------|----------|
| `daily_incremental` | Notebook / typical local runs | Next `bootstrap_days_per_run` chronological days not in `bootstrap_state.json`; stream BC each day (`bc_epochs_per_pass` epochs/day) |
| streaming (`bootstrap_passes > 1`) | Multi-pass corpus | One full calendar day per pass via `stream_bootstrap_bc_pretrain` |
| buffer fill | `bootstrap_passes ≤ 1` and mode is not daily | `bootstrap_path_b_replay_buffer` then buffer BC (`bc_epochs`) |

### Knobs

```text
--bootstrap-mode daily_incremental
--bootstrap-days-per-run 3
--bc-epochs-per-pass 2    # stream epochs per day (BC stream epoch N/M)
--bc-epochs 15            # buffer BC after bootstrap; skipped if bootstrap_passes > 1
--bc-steps-per-epoch      # cap SGD steps per stream/buffer epoch (None = all)
--metadata-path working/kaggle_episodes/metadata.json
```

```python
from kaggriculture_self_play_training import train_self_play

train_self_play(
    use_kaggle_env=True,
    bootstrap_mode="daily_incremental",
    bootstrap_episodes=None,
    metadata_path="working/kaggle_episodes/metadata.json",
    data_dir="working/kaggle_episodes",
    bootstrap_days_per_run=3,
    bc_epochs_per_pass=2,
    bc_epochs=15,
    bootstrap_passes=1,
)
```

> **Kaggriculture note:** Do **not** use `stable_baselines3.common.evaluation.evaluate_policy` against a single-agent gym env as the win-rate signal. That env typically pairs you with a random (or heuristic) opponent and is *not* competition-aligned. Post-training win rate is the reference ladder under `opponents/` via `eval_policy.evaluate_ladder` (`metrics/ladder_eval.json` and `metrics/win_rate_eval.json`).

### Dataset on disk

Bootstrap reads **Kaggle episode JSON** plus a catalog, not flat `(s, a, r, s')` CSV rows:

```text
working/kaggle_episodes/
  metadata.json          # episode index (dates, scores, paths)
  episodes/*.json        # raw kaggle_environments logs
experiment/metrics/
  bootstrap_state.json   # days already streamed
  bc_pretrain.json       # stream + buffer BC losses
```

---

## Common Training Issues

### Path B

| Symptom | Check |
|---------|--------|
| `BC stream epoch 1/1` | `bc_epochs_per_pass` is 1; default is **2** |
| Bootstrap skipped | `bootstrap_episodes=0` turns bootstrap off; use `None` |
| `use_kaggle_env` RuntimeError | Official simulator required (`True` / omit `--no-use-kaggle-env`) |
| Ladder money ~3000 vs 3000 | Pass-spawn deadlock or ε=1 wipe; see `TODO.md` — still use `evaluate_ladder`, not gym reward |
| Ladder unfair vs reference agents | `turns_per_cycle` must be **24** |
| Resume did no new self-play | Raise `--total-episodes` or `--min-self-play-episodes` |

### Issue 1: Training is Unstable (High Variance) — SB3/PPO reference

```
Symptoms:
- Episode reward oscillates widely
- TensorBoard shows spiky ep_rew_mean

Solutions:
1. Reduce learning rate
2. Increase batch_size
3. Increase n_steps (more data per update)
4. Reduce n_epochs (fewer passes over data)
5. Increase gae_lambda (closer to 1.0 for lower variance)

model = PPO("MlpPolicy", env,
    learning_rate=1e-4,    # ↓ from 3e-4
    batch_size=256,        # ↑ from 64
    n_steps=4096,          # ↑ from 2048
    n_epochs=5,            # ↓ from 10
    gae_lambda=0.98,       # ↑ from 0.95
)
```

### Issue 2: Policy Collapse (Entropy Decreasing to 0)

```
Symptoms:
- Entropy loss drops to near 0
- Policy consistently picks same action
- No improvement in episode reward

Solutions:
1. Increase entropy coefficient
2. Reduce learning rate
3. Add noise to exploration

model = PPO("MlpPolicy", env,
    ent_coef=0.02,         # ↑ from 0.0
    learning_rate=1e-4,    # ↓ from 3e-4
)
```

### Issue 3: Slow Training

```
Solutions:
1. Use SubprocVecEnv instead of DummyVecEnv
2. Increase n_envs (more parallel workers)
3. Decrease n_epochs (fewer passes over data)
4. Use GPU (device="cuda")
5. Use optimize_memory_usage for DQN

env = make_vec_env("MyEnv", n_envs=16, vec_env_cls=SubprocVecEnv)
model = DQN("MlpPolicy", env, optimize_memory_usage=True)
model = PPO("MlpPolicy", env, n_epochs=3)  # Fewer epochs
```

### Issue 4: Value Network Not Learning

```
Symptoms:
- Value loss stays high
- Advantage estimates are noisy

Solutions:
1. Increase vf_coef (weight of value loss)
2. Use separate value network architecture
3. Normalize advantages

model = PPO("MlpPolicy", env,
    vf_coef=0.7,           # ↑ from 0.5
    policy_kwargs=dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256, 128])
    ),
    normalize_advantage=True,
)
```

---

## Related Documentation

- [Overview](01-overview.md) — Core concepts and architecture
- [Algorithm Reference](02-algorithms.md) — Algorithm details and comparisons
- [API Reference](03-api-reference.md) — Complete API documentation
- [basedpyright LSP](basedpyright-lsp.md) — Typecheck the `kagg` conda env
- [RL + CV Integration](../integration/01-rl-cv-integration.md) — Combining with keras-retinanet
- [Setup Guide](../setup/01-installation.md) — Installation and environment setup
