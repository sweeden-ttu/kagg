# Kaggriculture — TODO

## Full DQN replay pipeline — next run

- [x] Scale to higher episode count (15+) and 720-step seasons
- [x] Enable non-zero BC epochs for behavioral cloning pretrain
- [x] Verify PER-driven priority updates (TD-error-based reweighting)
- [x] Full training run: bootstrap + BC pretrain + self-play + ladder eval
- [ ] Pass ladder eval: win rate ≥ 50% vs every opponent including tier 0

**Ladder status (2026-08-30, multi-wheat decode heuristics, 10 ep, 720×24):** **2/10** cleared.
Finn **PASS** (win 100%, p0 **$7300** vs $3000). Walter **PASS** (win 100%, p0 **$6789** vs $6211).
Other 8 **FAIL** — next bottleneck **rotation_rosa** (p0 ~$6.2k vs ~$11.1k; Rosa hires 4 hands for ~7 coins/day + 3-crop rotation). Decode still soft-penalizes HIRE; hands never hired (0 hand actions). Self-play still wipes BC. Experiment: `farm_bc_only_ladder`.

---

# Critical Code Analysis: Kaggriculture Self-Training RL System

> **Status (2026-08-30):** Priority-table items below were addressed in code:
> - Path B extractor renamed to `PathBFeatureExtractor` (legacy alias kept); docs distinguish `dqn.py` dict/one-hot vs Path B channels.
> - PER: generation-tagged indices, bootstrap reservoir (no circular expert wipe), beta annealing.
> - Reward: stake amplifies leads only (not deficits).
> - Epsilon after BC: `0.35 → 0.05` (was `0.12 → 0.03`).
> - BC CE weights softened (PASS/farm less extreme).
> - Agent export resolves multiple `model.pth` locations.
> - Wrapper masking uses nested `_last_raw_obs` (earlier fix). Monolith split of `self_play_training.py` remains out of scope.

## 1. Architecture Overview

The system implements a **Dueling Double DQN with hierarchical action branching** for the Kaggle Kaggriculture competition. It has two parallel model variants:

| Component | `dqn.py` (Branching DQN) | `kaggriculture_path_b_rebuild.py` (Path B) |
|---|---|---|
| Observation encoding | CNN on one-hot tiles + MLP on 55 numeric features (from tensor dict) | CNN on 9-channel spatial grid + MLP on 55 numeric features (from numpy arrays) |
| Action space | Flat branching: 15 + 6×15 + 10 = 122 outputs | Hierarchical: 15 verbs × 5 crops + 6×15 hands + 10×10 sequential market |
| Market modeling | Single 10-action head | **Autoregressive GRU** decoder for up to 10 sequential market orders |
| Dueling | V(s) + Σ(A_branch − mean(A_branch)) | Same, but with conditioned crop head |
| Reward | Simple bank delta / 100 | Competitive shaping with kinematic invest/liquidate schedule |
| Bootstrap | `dataset_loader.py` — loads into flat ReplayBuffer | `path_b_bootstrap.py` — loads into PER dual-partition buffer |

The self-play coordinator (`kaggriculture_self_play_training.py`, **1928 lines**) is the orchestrator, wiring together bootstrap → behavioral cloning → self-play training → ladder evaluation.

---

## 2. Critical Issues

### 2.1 🚨 Model Mismatch: `dqn.py` and `path_b_rebuild.py` Use Different Observation Formats

**`dqn.py`** (`KaggricultureFeatureExtractor.forward()`) expects a **dict of PyTorch tensors** with keys like `"tiles"` (B×H×W long), `"day"`, `"hour"`, `"player_id"`, etc. It performs one-hot encoding internally (`F.one_hot(grid.long(), num_classes=9)`).

**`path_b_rebuild.py`** (`KaggricultureFeatureExtractor.forward()`) expects **raw numpy arrays** — `tiles` as (B, 9, 10, 10) float32 (already channel-encoded) and `numeric` as (B, 55) float32. It does **not** do one-hot encoding.

Both modules have identically-named classes but incompatible signatures. The self-play trainer imports from `path_b_rebuild.py`, so the `dqn.py` variant is **dead code** that would fail if used. If someone switches imports, the training loop would crash silently or produce garbage gradients.

**Severity: HIGH** — potential for silent correctness bugs if the wrong module is imported.

### 2.2 🚨 Reward Shaping Has a Sign Bug That Can Reverse the Signal

In `CompetitiveRewardShaper.shape_reward()` (path_b_rebuild.py, ~line 510):

```python
stake = max(1.0, (my_money + opp_money) / self.stake_reference)
competitive_delta = stake * (my_money - opp_money) / self.margin_scale
```

This scales the money difference by a **stake factor** ≥ 1. When the player is ahead, this *amplifies* the reward. But when `my_money < opp_money`, the term is **negative and amplified**, creating a stronger penalty for being behind — which could push the agent into **desperation behavior** (risky bets, ignoring farming) rather than gradual improvement. The `mix_bonus` term compounds this by rewarding actions based on a kinematic schedule that may conflict with actual game state.

**Combined effect**: the agent may oscillate between over-investing (when behind) and over-conservation (when ahead), rather than following a stable farming loop.

**Severity: MEDIUM** — may cause unstable training dynamics.

### 2.3 🚨 Self-Play Epsilon Schedule Is Too Aggressive and Starts Too Low

```python
eps_start = 0.12 if bc_loss_history else 1.0
eps_end = 0.03
eps_decay_steps = max(1, total_episodes - learning_start_episodes)
```

When BC has been done (the common case), exploration starts at **ε=0.12** and linearly decays to **0.03** over the total self-play episodes. For a 720-step competitive environment with partial observability and adversarial opponent, this is **insufficient exploration**:

- The opponent pool rotates through checkpoints, so the state distribution shifts every ~5 episodes
- ε=0.12 means the agent plays greedily 88% of the time from episode 1, preventing discovery of strategies that counter the initial opponent pool
- With only 25 episodes in "medium" mode, each ε step is ~0.004 — effectively static at 0.12

**Severity: MEDIUM** — agent converges to local optimum early.

### 2.4 Prioritized Replay Buffer: Index Corruption in Dual-Partition Design

The `PrioritizedReplayBuffer` stores two partitions (bootstrap, selfplay) with separate circular buffers. In `_sample_partition()`:

```python
tagged_indices = np.array([(0 if source == SOURCE_BOOTSTRAP else 1, int(i)) for i in local_indices])
```

Then in `update_priorities()`:

```python
source_flag, local_idx = int(idx[0]), int(idx[1])
```

The problem: if a bootstrap partition is full and wraps around, `local_idx` indices can **collide** with earlier entries. The PER update then modifies the wrong transition. Similarly, in `sample()`, the `global_indices` variable is computed but never used — it's immediately discarded in favor of `tagged_indices`.

**Severity: HIGH** — priority updates can corrupt the buffer's training signal.

### 2.5 BC Loss Weights Encode Heuristics, Not Data-Driven Signals

```python
def _bc_farmer_verb_weights(verb):
    w = torch.full_like(verb, 1.5)
    w = torch.where(verb == "PASS", 0.15)
    for idx in _FARM_VERBS: w = torch.where(verb == idx, 5.0)
```

The weighting scheme (PASS downweighted 10×, farm actions upweighted 3.3×) is **hardcoded**, not learned from the data. The real episode corpus will already be dominated by productive actions (farming loop), so downweighting PASS aggressively may cause the BC model to **over-imitate** aggressive actions at steps where a passive "wait" would be optimal. The same applies to `_bc_market_action_weights()`.

**Severity: LOW-MEDIUM** — likely produces reasonable BC, but suboptimal imitation fidelity.

### 2.6 `_enforce_valid_actions()` in `kaggle_env_wrapper.py` Is Broken

```python
def _enforce_valid_actions(self, action, obs):
    # ...
    obs_dict = {}
    if "farms_p0_money" in obs:
        obs_dict["farms"] = [{"money": ..., "tiles": [[0]*10 for _ in range(10)], "farmer": [0,0]}]
    if "seeds" in obs:
        obs_dict["private"] = {"seeds": {}, "shed": {}}
    farmer_mask = ActionMasker.get_valid_farmer_actions(obs_dict)
    action["farmer"] = min(action["farmer"], len(farmer_mask) - 1)
```

The masks are built from a **synthetic observation** with empty tiles and zero inventory. This means every mask will be nearly all-True (or all-false for actions like DIG/WATER). The clamping `min(action["farmer"], len(farmer_mask)-1)` is effectively `min(x, 14)` — it **never actually enforces constraints**. The real masking happens via `HierarchicalActionMasker` in the self-play loop, so this wrapper method is dead code with a false sense of safety.

**Severity: LOW** — has no practical effect; masking works elsewhere.

### 2.7 Catastrophic Forgetation Between Bootstrap and Self-Play

The dual-partition buffer keeps 50% bootstrap data and 50% self-play data. As self-play fills the buffer:

1. Bootstrap transitions get **cycled out** (circular buffer)
2. The PER mechanism re-weights toward high-TD-error self-play transitions
3. The model's policy drifts away from expert demonstrations

However, the `bootstrap_fraction=0.5` is **static** — it doesn't decay. In practice, `bootstrap_capacity` = capacity × 0.5 = 100,000 for "medium" mode (200k capacity). But the bootstrap partition **still wraps around** — old expert traces are overwritten by self-play data pushed with `source="selfplay"`. The buffer name is misleading; it's not truly dual-source, it's just a circular buffer with two indices.

**Severity: MEDIUM** — the intended "expert memory" degrades faster than expected.

---

## 3. Design & Code Quality Issues

### 3.8 Massive Single Responsibility Violation: `kaggriculture_self_play_training.py`

At **1928 lines**, this file is a monolith containing:
- Environment wrapper (`KaggleCompetitiveEnv`)
- Replay buffer (`PrioritizedReplayBuffer`)
- Training coordinator (`SelfPlayCoordinator`)
- Checkpoint resume logic
- The entire training loop
- Agent export logic
- CLI argument parsing
- Ladder evaluation invocation

This makes it untestable, unmaintainable, and causes circular import issues (it imports from `path_b_bootstrap`, which imports from `training_metrics`, which may import from `eval_policy` in practice).

### 3.9 No Gradient Accumulation for Large Batch Sizes

The `HierarchicalDQNBranching` model has ~512 + 256 latent dimensions, plus a GRU for market decoding. With `batch_size=32`, the autoregressive market decoder unrolls 10 steps of GRU per sample, creating significant GPU memory pressure. There's no gradient accumulation, so larger effective batches (which help stabilize Q-learning) are impossible.

### 3.10 BatchNorm in Evaluation Mode with Batch Size 1

In the self-play loop:

```python
online_net.eval()
with torch.no_grad():
    q_out = online_net(tiles_t, numeric_t)  # batch size = 1
```

The `KaggricultureFeatureExtractor` uses `BatchNorm2d` in the CNN and `BatchNorm1d` in the MLP. During evaluation with **batch size 1**, BatchNorm uses running statistics (fine), but the `HierarchicalDoubleDQNLearner.update_target_network()` uses soft update (`tau=0.001`), meaning the target network's running stats may be stale, causing distribution shift between online and target predictions.

### 3.11 No Replay Buffer Capacity Pressure in PER Sampling

The PER sampling doesn't implement the standard beta annealing from [Schaul et al. 2016](https://arxiv.org/abs/1511.05952). `beta` is fixed at 0.4:

```python
def sample(self, batch_size: int, beta: float = 0.4):
    ...
```

This means the importance sampling correction is **under-corrected**, biasing the update toward high-priority (high-error) transitions without compensating, which can cause divergence.

### 3.12 `resolve_episode_paths_from_metadata()` Has Silent Failures

When an episode ID exists in metadata but the JSON file is missing (e.g., dataset changed, partial mount), the function silently skips it without logging:

```python
if path is not None:
    resolved.append(path)
# else: silently dropped
```

Over many runs, this can lead to **incremental data loss** that's hard to diagnose.

### 3.13 Agent Export Hardcodes Paths

```python
model_path = os.path.join(os.path.dirname(__file__), "models", "model.pth")
```

This relative path assumes the agent module runs from the same directory as the model. On Kaggle submission, this may not hold, causing `FileNotFoundError` at inference time.

### 3.14 No Validation of Observation/Action Shape Contracts

The adapter layer (`kaggriculture_adapter.py`) has multiple functions that silently handle mismatched shapes:

```python
# If prices is a list shorter than 5 crops:
market_prices = (list(prices[:5]) + [0.0] * 5)[:5]
```

This masks upstream data pipeline bugs.

---

## 4. Positive Design Decisions

### 4.15 Action Branching with Dynamic Masks

The `HierarchicalActionMasker` correctly prevents illegal actions (e.g., harvesting non-existent crops, watering watered plants). This is essential for a discrete-action RL agent — without it, wasted steps on invalid actions would severely slow learning.

### 4.16 Autoregressive Market Decoder

The GRU-based market order sequence generation (up to 10 orders per step) is a clever approach to the combinatorial market action space. A flat 10-action head can only select one order; the sequence model can compose multi-step market strategies.

### 4.17 Kinematic Invest/Liquidate Schedule

The `CompetitiveRewardShaper`'s season-progress-based target (25% invest → 75% invest over the episode) mirrors optimal play patterns for the Kaggriculture competition. This is a strong inductive bias that bootstraps early training.

### 4.18 Daily-Incremental Bootstrap

The `daily_incremental` bootstrap mode (processing one calendar day at a time, with `bootstrap_state.json` persistence) is excellent for incremental training. It avoids re-processing old data, enables resuming after crashes, and creates natural checkpoints for analysis.

### 4.19 Ladder Evaluation vs Reference Agents

The evaluation framework (`eval_policy.py`) properly evaluates against a tiered reference ladder with rubric-aligned win/loss/tie scoring — this is the **correct evaluation protocol** for a Kaggle competition agent.

### 4.20 Checkpoint/Resume Completeness

The training state save includes RNG states (Python, NumPy, Torch, CUDA), opponent pool, and episode metrics — ensuring fully deterministic resumption. This is production-grade.

---

## 5. File-by-File Assessment

| File | Lines | Quality | Key Concern |
|---|---|---|---|
| `kaggriculture_adapter.py` | 659 | Good | Two parallel observation encoders (Path B vs legacy) create maintenance burden |
| `kaggriculture_path_b_rebuild.py` | 742 | Good | Autoregressive market GRU is novel; BC loss weights are heuristic |
| `path_b_bootstrap.py` | 1221 | Good | Well-structured daily-incremental pipeline; buffer seeding logic is sound |
| `episode_catalog.py` | 563 | Good | Robust metadata merging; silent file-missing failures |
| `dataset_loader.py` | 337 | Fair | Legacy code; `dqn.py` observations incompatible with `path_b_rebuild.py` |
| `kaggle_env_wrapper.py` | 392 | Fair | Masking enforcement is broken; mostly dead code (self-play uses direct env) |
| `eval_policy.py` | 346 | Good | Clean ladder evaluation; rubric-aligned scoring |
| `training_metrics.py` | 390 | Good | Comprehensive progress tracking; schema versioning |
| `visualize.py` | 1288 | Fair | Large, but well-organized |
| `kaggriculture_self_play_training.py` | 1928 | **Poor** | Massive monolith; needs splitting |
| `dqn.py` | 1057 | Fair | Well-documented; unused dead code relative to training path |
| `kaggriculture_dataset_publish.py` | 288 | Good | Clean artifact publishing pipeline |

---

## 6. Summary of Risks by Priority

| Priority | Issue | Impact |
|---|---|---|
| **HIGH** | Observation format mismatch between `dqn.py` and `path_b_rebuild.py` | Silent correctness failure if wrong module used |
| **HIGH** | PER dual-buffer index corruption on wrap-around | Training divergence, phantom priority spikes |
| **MEDIUM** | Reward shaping can reverse gradient signal when behind | Unstable training, policy oscillation |
| **MEDIUM** | Epsilon too low for competitive self-play | Early convergence to suboptimal policy |
| **MEDIUM** | Bootstrap buffer partition wraps, losing expert data | Catastrophic forgetation |
| **MEDIUM** | Fixed beta=0.4 in PER (no annealing) | Importance sampling under-correction |
| **LOW** | Mask enforcement in wrapper is broken | No practical impact (masking works elsewhere) |
| **LOW** | BC loss weights are hardcoded heuristics | Suboptimal imitation, but likely works |
| **LOW** | Agent export hardcodes relative paths | Kaggle submission may fail |

The system is architecturally ambitious with several innovative elements (autoregressive market decoder, kinematic reward shaping, daily-incremental bootstrap), but the **PER buffer index bug** and the **observation format inconsistency** are the most critical issues that could silently corrupt training. The monolithic `self_play_training.py` file is the most urgent maintenance concern.