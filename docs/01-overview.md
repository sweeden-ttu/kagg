# Kaggriculture training — Overview

**Path B is the required training path.** Self-play uses the official Kaggle simulator, hierarchical Dueling Double DQN, episode-JSON bootstrap + behavioral cloning, then league eval against `opponents/`. Do not treat Stable-Baselines3 `PPO.learn()` / `evaluate_policy` as how this repo trains or scores win rate.

The optional `kaggriculture_rl.dqn_sb3.DQN` wrapper mimics an SB3-style API around the **legacy flat-branch** `kaggriculture_rl.dqn` stack. The notebook and `train_self_play` do **not** use it.

---

## Pipeline

```
Kaggle episode JSONs          Official kaggle-environments
(metadata.json + episodes/)   (turnsPerDay=24, 720 steps)
            │                              │
            ▼                              ▼
   path_b_bootstrap.py              KaggleCompetitiveEnv
   stream BC + seed buffer                 │
            │                              │
            └──────────┬───────────────────┘
                       ▼
         train_self_play (hierarchical DDQN)
                       │
                       ▼
         eval_policy.evaluate_ladder
         metrics/ladder_eval.json
         metrics/win_rate_eval.json
```

| Stage | Module | Role |
|-------|--------|------|
| Catalog | `episode_catalog.py`, `dataset_loader.py` | Resolve dated episode JSONs |
| Bootstrap / BC | `path_b_bootstrap.py` | Stream expert transitions, seed PER buffer |
| Train | `kaggriculture_self_play_training.train_self_play` | Hierarchical DDQN self-play |
| Net | `kaggriculture_path_b_rebuild.HierarchicalDQNBranching` | Farmer / crop / 6 hands / market GRU |
| Features | `kaggriculture_rl.dqn.KaggricultureFeatureExtractor` | CNN tiles + MLP numerics → 512-d latent |
| Eval | `eval_policy.evaluate_ladder` | Head-to-head vs `opponents/` |

---

## Observation and action

The extractor (`KaggricultureFeatureExtractor`) expects a dict, not a flat vector:

- **Tiles:** `(B, 10, 10)` tile IDs → one-hot CNN
- **Numerics (55-d after concat):** day, hour, player_id, both farms' money, market prices/inventory, seeds, shed, inventories

The Path B policy (`HierarchicalDQNBranching`) outputs **branched** Q-heads, not one flat Discrete:

| Branch | Outputs | Notes |
|--------|---------|--------|
| Farmer verb | 15 | Primary farm action |
| Crop parameter | 5 | Conditioned on the verb |
| Hands | 6 × 15 | One head per hand |
| Market | up to 10 orders | Autoregressive GRU decoder |

A flat encoding of the same space is on the order of \(15 \times 15^6 \times 10\) — branching keeps ~122 Q-outputs on the legacy stack, plus the hierarchical crop/market heads on Path B.

---

## The learn path (Path B)

```python
from kaggriculture_self_play_training import train_self_play
from eval_policy import evaluate_ladder

train_self_play(
    use_kaggle_env=True,
    bootstrap_mode="daily_incremental",
    bootstrap_episodes=None,  # all catalog days; 0 skips bootstrap
    metadata_path="working/kaggle_episodes/metadata.json",
    data_dir="working/kaggle_episodes",
    bootstrap_days_per_run=3,
    bc_epochs_per_pass=2,
    bc_epochs=15,
    bootstrap_passes=1,
    opponents_dir="opponents",
    n_eval_episodes=10,
)
```

CLI defaults match that call (`--no-use-kaggle-env` to refuse the official simulator). Resume from `checkpoints/training_state_latest.pt` via `--resume <experiment_dir>`.

Win rate is **not** mean gym reward:

```python
from eval_policy import evaluate_ladder, win_rate_eval_from_ladder

ladder = evaluate_ladder(
    challenger_policy,
    opponents_dir="opponents",
    n_episodes=10,
    max_steps=720,
    turns_per_day=24,  # competition / reference-agent parity
    win_rate_target=0.75,
)
summary = win_rate_eval_from_ladder(ladder)
# writes conceptually: metrics/ladder_eval.json + metrics/win_rate_eval.json
```

`stable_baselines3.common.evaluation.evaluate_policy` against a single-agent gym env pairs you with a random or heuristic opponent. That is **not** competition-aligned.

---

## Dueling Double Q (what Path B actually trains)

Path B is **value-based, off-policy**, not PPO actor-critic:

- **Double Q:** online net selects \(\arg\max_a Q_{\text{online}}(s', a)\); target net evaluates it
- **Dueling:** \(Q = V(s) + \sum_{\text{branch}} [A_b - \mathrm{mean}(A_b)]\)
- **PER:** `PrioritizedReplayBuffer` in `kaggriculture_self_play_training.py` — 50% bootstrap / 50% self-play partitions
- **ε-greedy** during self-play; BC pretrain is supervised on expert actions

PPO / SAC / A2C remain useful as *algorithm theory*. They are not the Kaggriculture trainer. See [02-algorithms.md](02-algorithms.md).

---

## Related documentation

- [Algorithm Reference](02-algorithms.md) — Double/Dueling DQN and Path B heads
- [API Reference](03-api-reference.md) — `train_self_play`, bootstrap, ladder
- [Training Guide](04-training-guide.md) — knobs, bootstrap-from-dataset, resume
- [basedpyright LSP](basedpyright-lsp.md) — typecheck the `kagg` conda env

---

## References

- [kaggle-environments](https://github.com/Kaggle/kaggle-environments)
- [Gymnasium](https://gymnasium.farama.org/) (wrappers only; self-play is the Kaggle engine)
- Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) (optional `dqn_sb3` wrapper only)
