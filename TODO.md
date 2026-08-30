# Kaggriculture — TODO

Tracking fixes for the training notebook and pipeline. Primary target: [`kaggriculture-self-training/kaggriculture-self-training.ipynb`](kaggriculture-self-training/kaggriculture-self-training.ipynb).

---

## Critical — notebook & Kaggle parity

- [x] **Attach/bundle opponents for Kaggle + resolve from code dataset fallback** *(partial: see validation)*
  - Added `raykkretzschmar/kaggriculture-reference-agents` to `kernel-metadata.json`.
  - `eval_policy.resolve_opponents_dir` checks Kaggle reference mount, `CODE_SRC/opponents`, repo `opponents/`.
  - **Not done:** physical `opponents/` bundle inside code dataset (optional follow-up).

- [x] **Add `ladder_eval_episodes: 10` to `medium` preset**

- [x] **Pop `kaggriculture_self_play_training` from `sys.modules`**
  - Via `notebook_paths.bust_stale_modules()` in deploy/configure/preflight cells.

- [x] **Preflight `OPPONENTS_DIR.exists()` when ladder > 0**
  - §1c raises if opponents missing; prints agent count from manifest.

- [x] **Add `ladder_eval.json` to publish artifacts**

- [x] **Clear stale outputs; fix “~1 min” comment/ETA**
  - Notebook outputs cleared; dry_run `_eta` → ~15–30 minutes locally.

- [x] **Split setup cell**
  - §1a Deploy → §1b Configure → §1c Preflight.

- [x] **Resume policy — explicit fresh-run flag for dry_run**
  - `KAGGLE_FRESH_RUN=1` disables resume; `dry_run` needs `KAGGLE_RESUME=1` to resume.

---

## High — pipeline correctness

- [x] **Document training vs ladder eval** — notebook intro + README Phase 2.

- [x] **Remove duplicate ladder display** — §2b markdown-only; §3 prints ladder table.

- [x] **Deduplicate path bootstrap in §4 / §5** — shared `notebook_paths.py`.

- [x] **Remove empty notebook cell**

- [x] **Fix README directory map** — `kaggriculture-self-training/` replaces `kernel1/`/`kernel2/`.

---

## Medium — docs & hygiene

- [x] Fix notebook title typo: “Immitation” → “Imitation”
- [x] Drop unused imports in setup (removed via split + lean imports)
- [x] Add deprecation banner to `ImitationLearning.ipynb`
- [x] Note in `dqn_sb3.py` that Path B `train_self_play` is canonical

---

## Done (earlier)

- [x] Preserve `step` in `parse_observation` (tier 6–9 tape opponents)
- [x] Wire ladder eval into `train_self_play`
- [x] Improve dry_run presets (self-play learning, 720 steps, ladder eval)
- [x] Resume extension via `min_self_play_episodes`
- [x] Rename “ELO” reward shaper docs
- [x] Fix misleading import “fallbacks” message

---

## Verification checklist

Validated locally **2026-08-29** (automated + spot checks):

| Check | Status | Notes |
|-------|--------|-------|
| Code: `parse_observation` preserves/derives `step` | **PASS** | tier 6–9 tape index works |
| Code: `resolve_opponents_dir` → 10 agents locally | **PASS** | `/Users/sweeden/kagg/opponents` |
| Code: `ladder_eval.json` in publish paths | **PASS** | `kaggriculture_dataset_publish.py` |
| Code: resume env (`KAGGLE_FRESH_RUN`, `KAGGLE_RESUME`) | **PASS** | dry_run default no resume |
| Notebook: 16 cells, 0 stale outputs, §1a/1b/1c split | **PASS** | |
| Notebook: medium `ladder_eval_episodes: 10` | **PASS** | lines in MODE_PRESETS |
| Notebook: §2b markdown-only (no duplicate ladder code) | **PASS** | |
| Preflight simulation (ladder > 0, opponents exist) | **PASS** | |
| Re-run §1a→§2 full pipeline on fresh kernel | **NOT RUN** | needs Jupyter execution |
| `working/run/metrics/ladder_eval.json` on disk | **NOT RUN** | file absent until §2 completes |
| Kaggle GPU Run All + reference agents mount | **NOT RUN** | needs push + Kaggle |
| `medium` publish copies `ladder_eval.json` | **NOT RUN** | needs publish on Kaggle |

- [ ] Re-run §1a→§1c → §2 on fresh kernel; outputs match `TRAINING_CONFIG` JSON
- [ ] `metrics/ladder_eval.json` exists with 10 opponents and realistic tier 6–9 bank totals
- [ ] Kaggle kernel Run All with GPU: bootstrap + self-play + ladder (reference agents attached)
- [ ] `medium` publish includes `ladder_eval.json` in code dataset artifacts

---

## Optional follow-ups

- [ ] Bundle `opponents/` inside `scottweeden/kaggriculture-self-training-code` for offline Kaggle fallback
- [ ] Ladder curriculum: sample reference opponents during self-play (not just post-hoc eval)
- [ ] Re-run loop when ladder eval fails (previously removed §2b retrain)
