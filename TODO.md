# Kaggriculture — TODO

Tracking fixes for the training notebook and pipeline. Primary target: [`kaggriculture-self-training/kaggriculture-self-training.ipynb`](kaggriculture-self-training/kaggriculture-self-training.ipynb).

---

## Critical — notebook & Kaggle parity

- [ ] **Attach/bundle opponents for Kaggle + resolve from code dataset fallback**
  - `kernel-metadata.json` does not attach reference agents; ladder eval skips on Kaggle today.
  - Options: add `raykkretzschmar/kaggriculture-reference-agents` to dataset sources, or ship `opponents/` inside `scottweeden/kaggriculture-self-training-code`.
  - Update `OPPONENTS_DIR` resolution: `KAGGLE_INPUT/opponents` → `CODE_SRC/opponents` → `~/kagg/opponents`.

- [ ] **Add `ladder_eval_episodes: 10` to `medium` preset**
  - `dry_run` has 3; `full` has 20; `medium` currently defaults to 0 via `.get(..., 0)`.
  - §2b markdown claims ladder runs in medium/full — align preset with docs.

- [ ] **Pop `kaggriculture_self_play_training` from `sys.modules`**
  - Setup cell busts cache for `eval_policy`, `path_b_bootstrap`, etc., but not the trainer.
  - Kernel re-runs can import stale trainer while fresh code is copied to `working/`.

- [ ] **Preflight `OPPONENTS_DIR.exists()` when ladder > 0**
  - Fail fast in §1 Setup if `ladder_eval_episodes > 0` and no opponents directory.
  - Print resolved path and opponent count from `agents_manifest.csv`.

- [ ] **Add `ladder_eval.json` to publish artifacts**
  - Extend `kaggriculture_dataset_publish.DEFAULT_ARTIFACT_REL_PATHS` with `metrics/ladder_eval.json`.
  - Ensures ladder results round-trip through the code dataset on Kaggle re-runs.

- [ ] **Clear stale outputs; fix “~1 min” comment/ETA**
  - Notebook cell outputs still show old dry_run (2 episodes, `learning_start_episodes: 99`, 50 steps).
  - Update comment `# Default dry_run locally (~1 min)` and `_eta` to match 720-step + ladder runtime.

- [ ] **Split setup cell (optional but high leverage)**
  - §1 is ~400 lines: deploy, config, restore, GPU diag, local downloads, metadata merge.
  - Split into: Deploy → Configure → Index episodes → Build `TRAINING_CONFIG`.

- [ ] **Resume policy — explicit fresh-run flag for dry_run**
  - Resume triggers whenever checkpoint or code-dataset artifacts exist (`_restored` or `training_state_latest.pt`).
  - Add e.g. `KAGGLE_FRESH_RUN=1` or disable auto-resume in `dry_run` for clean local smoke tests.

---

## High — pipeline correctness

- [ ] **Document training vs ladder eval**
  - Self-play trains vs checkpoint pool; ladder is post-hoc eval only (no retrain loop on failure).
  - Update notebook intro / README Phase 2 table if curriculum vs eval-only is intentional.

- [ ] **Remove duplicate ladder display**
  - §2b and §3 both print `ladder_eval.json`; merge or make §2b markdown-only.

- [ ] **Deduplicate path bootstrap in §4 / §5**
  - Visualize and publish cells re-resolve `KAGGLE_INPUT` / `CODE_SRC`; share helpers from §1 or a small `notebook_paths.py`.

- [ ] **Remove empty notebook cell (§2b gap / cell 7)**

- [ ] **Fix README directory map**
  - Replace placeholder `kernel1/` / `kernel2/` with `kaggriculture-self-training/`.

---

## Medium — docs & hygiene

- [ ] Fix notebook title typo: “Immitation” → “Imitation”
- [ ] Drop unused imports in setup cell (`pick_next_bootstrap_days`, `ensure_episode_datasets_for_range`, `load_bootstrap_state`)
- [ ] Add deprecation banner to `kaggriculture_rl/ImitationLearning.ipynb` (superseded by Path B notebook)
- [ ] Note in `dqn_sb3.py` that Path B `train_self_play` is the canonical training path

---

## Done (recent)

- [x] Preserve `step` in `parse_observation` (tier 6–9 tape opponents)
- [x] Wire ladder eval into `train_self_play` (`opponents_dir`, `ladder_eval_episodes`, `metrics/ladder_eval.json`)
- [x] Improve dry_run presets (self-play learning, 720 steps, ladder eval)
- [x] Resume extension via `min_self_play_episodes` when already past `total_episodes`
- [x] Remove cosmetic §2b SB3 retrain loop; ladder runs from §2
- [x] Rename “ELO” reward shaper docs to relative margin shaping
- [x] Fix misleading import “fallbacks” message in `kaggriculture_self_play_training.py`

---

## Verification checklist (after critical items)

- [ ] Re-run §1 → §2 on fresh kernel; outputs match `TRAINING_CONFIG` JSON
- [ ] `metrics/ladder_eval.json` exists with 10 opponents and realistic tier 6–9 bank totals
- [ ] Kaggle kernel Run All with GPU: bootstrap + self-play + ladder (opponents mounted)
- [ ] `medium` publish includes `ladder_eval.json` in code dataset artifacts
