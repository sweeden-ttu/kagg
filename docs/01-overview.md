# Kaggriculture training — Overview

**Primary stack: BC → PPO (multi-discrete) + HER milestones** in
`kaggle-mcp-server/kagg_rl` (`python -m kagg_rl.train_primary`).
Action space is branched discrete (farmer / market heads), not continuous —
ignore A2C / DDPG / SAC / TD3 for the main farm policy.

**Path B hierarchical Dueling Double DQN + PER** (`train_self_play`,
`scripts/train_tier*_champion.py`) is retained as an **ablation / offline
baseline**, not the submission ceiling.

League scoring still uses `eval_policy.evaluate_ladder` vs `opponents/`.
Do not treat stock SB3 `PPO.learn()` / gym `evaluate_policy` as the ladder.

---

## Pipeline (primary)

```
Top-agent episode histories          Official kaggle-environments
(until ~1 week before close)         (turnsPerDay=24, 720 steps)
            │
            ▼
   kagg_rl.il.train_bc (behavioral cloning)
            │
            ▼
   kagg_rl.ppo.train / train_primary
   (clipped PPO + HER milestone bonuses)
            │
            ▼
   eval_policy.evaluate_ladder
```

| Stage | Module | Role |
|-------|--------|------|
| Primary BC | `kaggle-mcp-server/kagg_rl/il/` | Clone top-agent seats |
| Primary online | `kagg_rl.ppo` + `kagg_rl.her` | PPO fine-tune + sparse milestone HER |
| Entrypoint | `kagg_rl.train_primary` / `scripts/train_ppo_her_primary.py` | BC → PPO+HER |
| Ablation (Path B) | `train_self_play` / hierarchical DDQN | Controlled DQN baseline |
| Eval | `eval_policy.evaluate_ladder` | Head-to-head vs `opponents/` |

### Ablation pipeline (Path B DQN)

```
Kaggle episode JSONs  →  path_b_bootstrap BC + PER  →  hierarchical DDQN self-play  →  ladder
```

---

## Observation and action

Primary IL/PPO features (`kagg_rl.il.features`) are a compact float vector (time,
money, tile summary, market, seeds/shed). The policy is **multi-discrete** heads
(farmer op/item, market op/item) plus soft qty regression — see
`kagg_rl.action_space`.

Path B ablation still uses CNN tiles + hierarchical Q-heads:

| Branch | Outputs | Notes |
|--------|---------|--------|
| Farmer verb | 15 | Primary farm action |
| Crop parameter | 5 | Conditioned on the verb |
| Hands | 6 × 15 | One head per hand |
| Market | up to 10 orders | Autoregressive GRU decoder |

---

## The learn path (primary)

```bash
cd kaggle-mcp-server
python -m kagg_rl.train_primary --dry-run --updates 5 \
  --out checkpoints/ppo_her_primary.pt
# With histories:
# python -m kagg_rl.train_primary --episodes-dir ... --min-reward 100000 --winners-only
```

Or from repo root: `python scripts/train_ppo_her_primary.py --dry-run`.

### Ablation learn path (Path B DQN)

```python
from kaggriculture_self_play_training import train_self_play
from eval_policy import evaluate_ladder

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
    opponents_dir="opponents",
    n_eval_episodes=10,
)
```

Win rate is **not** mean gym reward:

```python
from eval_policy import evaluate_ladder, win_rate_eval_from_ladder

ladder = evaluate_ladder(
    challenger_policy,
    opponents_dir="opponents",
    n_episodes=10,
    max_steps=720,
    turns_per_day=24,
    win_rate_target=0.75,
)
summary = win_rate_eval_from_ladder(ladder)
```

`stable_baselines3.common.evaluation.evaluate_policy` against a single-agent gym env is **not** competition-aligned.

---

## Primary vs ablation algorithms

- **Primary:** on-policy **PPO** (clipped) after BC; **HER** densifies sparse season-end / milestone cash goals (`kagg_rl.her`).
- **Ablation Path B:** value-based hierarchical Dueling Double DQN + PER + ε-greedy.
- **Ignore for main farm policy:** A2C, DDPG, SAC, TD3.

See [02-algorithms.md](02-algorithms.md).

---

## Related documentation

- [Algorithm Reference](02-algorithms.md) — PPO+HER primary; Path B DQN ablation
- [API Reference](03-api-reference.md) — primary `kagg_rl` + Path B APIs
- [Training Guide](04-training-guide.md) — knobs, bootstrap-from-dataset, resume
- [kagg_rl README](../kaggle-mcp-server/kagg_rl/README.md) — BC → PPO+HER CLI
- [basedpyright LSP](basedpyright-lsp.md) — typecheck the `kagg` conda env

---

## References

- [kaggle-environments](https://github.com/Kaggle/kaggle-environments)
- [Gymnasium](https://gymnasium.farama.org/) (wrappers only; self-play is the Kaggle engine)
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- Andrychowicz et al., "Hindsight Experience Replay" (2017)
- Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016) — Path B ablation
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016) — Path B ablation
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) (optional reference only)

