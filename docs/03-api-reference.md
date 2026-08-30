# Kaggriculture Path B — API Reference

APIs the training notebook and scripts actually call. Stock SB3 `PPO.save` / VecEnv / `evaluate_policy` are **not** the Path B interface. The optional `kaggriculture_rl.dqn_sb3.DQN` class is a legacy SB3-shaped wrapper around flat-branch `kaggriculture_rl.dqn` and is unused by `train_self_play`.

---

## Table of Contents

- [`train_self_play`](#train_self_play)
- [Bootstrap](#bootstrap)
- [Networks](#networks)
- [`evaluate_ladder`](#evaluate_ladder)
- [Checkpoints](#checkpoints)
- [Experiment layout](#experiment-layout)

---

## `train_self_play`

**Module:** `kaggriculture_self_play_training`

Coordinates catalog bootstrap, BC, hierarchical DDQN self-play, checkpoints, and league eval.

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
    opponents_dir="opponents",
    n_eval_episodes=10,
    max_episode_steps=720,
    turns_per_cycle=24,
    resume=None,  # or experiment dir / training_state_latest.pt
)
```

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `use_kaggle_env` | `True` | Official simulator required. `--no-use-kaggle-env` raises. |
| `bootstrap_mode` | `daily_incremental` | Next chronological unseen days, or `streaming` when `bootstrap_passes > 1` |
| `bootstrap_episodes` | `None` | Cap episode JSONs; `None` = catalog; `0` = skip bootstrap |
| `bc_epochs_per_pass` | `2` | Stream BC epochs per day (`BC stream epoch N/M`) |
| `bc_epochs` | `15` | Buffer BC after bootstrap; skipped if `bootstrap_passes > 1` |
| `bootstrap_passes` | `1` | `> 1` enables streaming day-per-pass |
| `bootstrap_days_per_run` | `3` | Days per `daily_incremental` run |
| `data_dir` | `working/kaggle_episodes` | Episode JSON root |
| `n_eval_episodes` | `10` | Ladder episodes per opponent when `ladder_eval_episodes` is 0 |
| `ladder_eval_episodes` | `0` | If `> 0`, overrides `n_eval_episodes`; `0` does **not** skip the ladder |
| `turns_per_cycle` | `24` | Must stay 24 for reference-agent ladder parity |
| `resume` | `None` | Experiment dir or `checkpoints/training_state_latest.pt` |

CLI mirrors these (`python -m kaggriculture_self_play_training` from the code dataset path).

`create_competitive_env(use_kaggle=True)` builds `KaggleCompetitiveEnv`. Offline / gym-only training is rejected.

---

## Bootstrap

**Module:** `path_b_bootstrap`

| Function | Role |
|----------|------|
| `run_bc_pretrain_over_episode_files` | Stream episode JSONs through BC (no buffer cap). Returns `(epoch_losses, transition_count)`. |
| `seed_buffer_from_episode_files` | Load transitions into the PER buffer |
| `incremental_daily_bootstrap_bc` | Next `days_per_run` dates not in `bootstrap_state.json` |
| `stream_bootstrap_bc_pretrain` | One calendar day per pass when `bootstrap_passes > 1` |
| `run_bc_pretrain` | BC on an already-filled buffer |
| `load_bootstrap_state` / `save_bootstrap_state` | `metrics/bootstrap_state.json` |
| `merge_bootstrap_state_from_code_dataset` | Resume days already published to the code dataset |

State keys: `bootstrapped_dates`, `runs`, `total_transitions`.

On-disk catalog (not CSV rows):

```text
working/kaggle_episodes/metadata.json
working/kaggle_episodes/episodes/*.json
```

---

## Networks

### `KaggricultureFeatureExtractor`

**Module:** `kaggriculture_rl.dqn`

CNN on one-hot tiles `(B, 9, 10, 10)` + MLP on 55 numeric features, fused to `features_dim=512`.

Required observation keys: `tiles`, `day`, `hour`, `player_id`, `farms_p0_money`, `farms_p1_money`, `market_prices`, `market_inventory`, `seeds`, `shed`, `inventories`.

### `HierarchicalDQNBranching`

**Module:** `kaggriculture_path_b_rebuild`

Path B policy. Forward returns a dict of Q tensors:

- `farmer_verb`, `crop_parameter`, `hands` (list), `market` (sequence), plus value

`HierarchicalActionMasker` applies legal-action masks before argmax.

`HierarchicalDoubleDQNLearner` owns online/target nets, Adam, Double-Q TD, and `compute_bc_loss` for bootstrap.

### Legacy flat-branch (`kaggriculture_rl.dqn`)

`DuelingDoubleDQNBranching` + `DoubleDQNLearner` + `ReplayBuffer`. Q dict is `BranchingQOutput`: `farmer_q`, `hand_q` (list of 6 tensors), `market_q`, `value`. Used by `dqn_sb3.DQN`, not by `train_self_play`.

### `PrioritizedReplayBuffer`

**Module:** `kaggriculture_self_play_training`

Dual-partition PER (`alpha=0.6`): `bootstrap_fraction=0.5` expert vs self-play. `sample_uniform` is required for buffer BC.

---

## `evaluate_ladder`

**Module:** `eval_policy`

```python
from eval_policy import evaluate_ladder, win_rate_eval_from_ladder

ladder = evaluate_ladder(
    challenger_policy,          # obs, cfg=None → kaggle action dict
    opponents_dir="opponents",
    n_episodes=10,
    max_steps=720,
    turns_per_day=24,
    win_rate_target=0.75,
)
summary = win_rate_eval_from_ladder(ladder)
```

`train_self_play` writes `metrics/ladder_eval.json` and `metrics/win_rate_eval.json`.

Do **not** use `stable_baselines3.common.evaluation.evaluate_policy` as the win-rate signal.

---

## Checkpoints

Path B does not write SB3 `.zip` files.

| Path | Contents |
|------|----------|
| `checkpoints/training_state_latest.pt` | Preferred full resume (nets, optimizer, buffer, episode) |
| `checkpoints/*.pt` | Periodic snapshots (`checkpoint_interval` episodes) |
| `config.json` | Run knobs + `last_completed_episode` |

```text
--resume experiments/my_run
# or
--resume experiments/my_run/checkpoints/training_state_latest.pt
```

`--total-episodes` is a **cumulative** target. If resume already meets it, `min_self_play_episodes` extends the target.

---

## Experiment layout

```text
<experiment_dir>/
  config.json
  checkpoints/training_state_latest.pt
  models/
  metrics/
    bootstrap_state.json
    bc_pretrain.json
    episode_metrics.json
    ladder_eval.json
    win_rate_eval.json
  logs/
  plots/
```

`visualize.update_experiment_plots` reads `metrics/` and writes `plots/`.

---

## Related documentation

- [Overview](01-overview.md)
- [Algorithm Reference](02-algorithms.md)
- [Training Guide](04-training-guide.md)
- [basedpyright LSP](basedpyright-lsp.md)
