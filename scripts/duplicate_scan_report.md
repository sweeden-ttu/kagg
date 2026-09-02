# Duplicate / Near-Duplicate Scan Report

Generated: 2026-09-02 across `/Users/sweeden/kagg` (excluding `.git`, `__pycache__`, `.DS_Store`).

**No files were deleted.** This report lists duplicate candidates with keep/delete
recommendations based on content completeness and date. Canonical keeper for the
agent code = `submission/` (created 09-02 03:01–03:43, demand-aware + correct engine
costs + gzip checkpoint). Live training trees that import code at runtime
(`datasets/scottweeden/self-training-code`, `working/*`) should be **updated to
canonical, not deleted**.

---

## 1. Version-variant families (same logical file, multiple content versions)

### `kaggriculture_adapter.py` (17 copies, 9 distinct contents)

| Hash | Size | n | Demand+Shops | Correct costs | Where | Verdict |
|---|---|---|---|---|---|---|
| `3af858672f87` | 48,644 | 1 | **yes** | yes | `submission/` | **KEEP (canonical)** |
| `c3bce2a1c030` | 45,316 | 4 | no | yes | artifacts-backup, publish/training_artifacts, staging×2 | Update to canonical or DELETE (superseded 09-01 20:54 by 3af858) |
| `0bdf31c50949` | 53,058 | 1 | no | yes | experiments/…/farm_bc_only_ladder | DELETE (historical; superseded by canonical) |
| `d3f04666eb9c` | 29,938 | 2 | no | no (legacy) | **datasets/scottweeden/self-training-code/**, working/run | **UPDATE to canonical** (this is CODE_SRC, imported by tier trainers) |
| `4aa60ea9cc17` | 27,825 | 2 | no | no (legacy) | datasets/…/training_artifacts, working/ | UPDATE to canonical or DELETE |
| `8f29ffd41419` | 22,935 | 3 | no | no (legacy) | experiments bc_only_ladder, farm_bc_ladder, publish/ | DELETE |
| `97ea3dc58942` | 22,747 | 2 | no | no (legacy) | experiments bc_pass_fix_ladder, turns24 | DELETE |
| `a82dd253770e` | 22,756 | 1 | no | no (legacy) | experiments/…/ (root) | DELETE |
| `80778b45f169` | 19,695 | 1 | no | no (legacy) | results/run | DELETE |

### `kaggriculture_path_b_rebuild.py` (17 copies, 11 distinct contents)

| Hash | Size | n | Has `PathBFeatureExtractor` alias | Where | Verdict |
|---|---|---|---|---|---|
| `36839c276229` | 38,483 | 5 | **yes** | submission/, artifacts-backup, publish/training_artifacts, staging×2 | **KEEP (canonical)** |
| `567f91cfee57` | 39,002 | 1 | yes | experiments/…/farm_bc_only_ladder | DELETE (historical) |
| `3e7ff6be114d` | 38,427 | 2 | no (separate class) | **datasets/scottweeden/self-training-code/**, working/run | UPDATE to canonical (imports compatible; keeps separate class) |
| `ed2f3bccdc0c` | 37,566 | 2 | no | datasets/…/training_artifacts, working/ | UPDATE to canonical or DELETE |
| `8a283ebec3bd` | 30,555 | 2 | no | experiments farm_bc_ladder, publish/ | DELETE |
| `8887e174c016` | 27,004 | 1 | no | experiments turns24 | DELETE |
| `af7a7d24817a` | 26,199 | 1 | no | experiments bc_pass_fix_ladder | DELETE |
| `b4eb0094e655` | 17,789 | 2 | no | experiments (root), bc_only_ladder | DELETE |
| `df519b7a1a59` | 16,571 | 1 | no | results/run | DELETE |

### `checkpoints.py` (4 copies, 3 distinct contents)

| Hash | Size | Where | Verdict |
|---|---|---|---|
| `ecfe3761cff6` | 8,489 | **datasets/scottweeden/self-training-code/** | **KEEP (canonical, gzip-patched)** |
| `bd52af7722bb` | 7,924 | working/ | **UPDATE to canonical** (missing gzip loader) |
| `920b6134e474` | 7,536 | publish/, staging/ | UPDATE to canonical or DELETE |

### `model.pth` (13 copies, 7 distinct weights)

| Hash | Size | Where | Verdict |
|---|---|---|---|
| `bc679f79548a` | 6,907,155 | **submission/models/**, artifacts-backup, publish/training_artifacts, staging/training_artifacts | **KEEP (canonical new weights)** — dedupe: keep submission; mirrors are identical so safe to delete extras |
| `c77567a72846` | 6,907,155 | results/run/models/ | DELETE (stale build of same size class) |
| `77eb0f5e1e1c` | 6,905,683 | working/run/models/ | DELETE/UPDATE (old weights) |
| `985aec4bed52` | 6,905,683 | datasets/…/training_artifacts/models/ | DELETE/UPDATE (old weights) |
| `fe77d5ec5c7b`, `5d96c10ae891`, `b319bbe44343`, `95b9e48cb80b`, `afcdf5037cb0`, `4f33a0708ce8` | 6,905,683 | experiments ladder dirs ×6 | DELETE (historical ladder weights) |

### `training_state_latest.pt` (12 copies, varying)

| Hash | Size | Where | Verdict |
|---|---|---|---|
| `d3f64f7675e0` | 25,745,505 (gzip) | **submission/training_artifacts/checkpoints/** | **KEEP (canonical, gzip < 100 MiB)** |
| `82dd584850b9` | 1,740,120,447 | artifacts-backup, publish/training_artifacts, staging/training_artifacts | Source of gzip; KEEP one, DELETE 2 (saves ~3.5 GB), or gzip all |
| `4736d8f57f36` | 1,741,872,679 | working/run/checkpoints/ | KEEP if last live run artifacts wanted; else DELETE (~1.7 GB) |
| `339e9b95d269` | 1,739,162,303 | results/run/checkpoints/ | DELETE (superseded by 82dd5848 line) |
| `95d485aeb61c` | 145,873,251 | datasets/…/training_artifacts/checkpoints/ | DELETE/UPDATE (older, un-gzipped) |
| `45717340dadc`, `63737dc28f0b`, `7ef046154393`, `a43ab8af7ec7`, `476c8ff4bfa8`, `6b9e0655fcf4` | 13.8–451 MB | experiments ladder dirs ×6 | DELETE (historical run checkpoints) |

### `agent.py` / `main.py`

| Hash | Size | n | Where | Verdict |
|---|---|---|---|---|
| `f800c9d36913` | 4,291 | 1 | **submission/main.py** | **KEEP (canonical, gzip loader + demand)** |
| `a9dcbd38a3b5` | 1,975 | 5 | experiments (root, bc_only, bc_pass_fix, turns24), results/run | DELETE (old inference wrapper, superseded) |
| `619d6f9a733a` | 3,211 | 4 | farm_bc_only_ladder, artifacts-backup, publish/training_artifacts, staging/training_artifacts | DELETE/UPDATE (pre-gzip wrapper) |
| `ba68ce4b3995` | 2,683 | 1 | datasets/…/training_artifacts/agent.py | DELETE/UPDATE |
| `fa109b4f1389` | 2,660 | 1 | experiments farm_bc_ladder | DELETE |
| `021586ed4507` | 38,432 | 1 | working/run/agent.py | == `opponents/ledger_lena.py` (hash dup); keeper is opponents copy → DELETE working/run snapshot |

---

## 2. Notable exact-duplicate groups

- **`training_state_latest.pt`** `82dd584850b9` ×3 in working trees — see variant table.
- **checkpoint_ep_0.pt** `cf959df9621a` (6.9 MB) ×6 in experiments + working/run — all identical; KEEP one (experiments root), DELETE rest.
- **`model.pth`** `bc679f79548a` ×4 (submission + 3 working mirrors) — canonical everywhere; identical copies fine to dedupe to submission only.
- **`kaggriculture_path_b_rebuild.py`** `36839c276229` ×5 — canonical across working mirrors.
- **plots/*.png** — 7 training plots duplicated ×4–5 (datasets/training_artifacts + artifacts-backup + publish/training_artifacts + staging/training_artifacts, some in experiments). KEEP dataset/training_artifacts copies; DELETE mirror copies.
- **`bc_pretrain_loss.png`** `2f0091198139` ×5 — same as above.
- **metrics/** (`ladder_eval.json`, `win_rate_eval.json`, `bc_pretrain.json`, `episode_metrics.json`, `training_progress.json`, `bootstrap_state.json`, logs) — duplicated ×2–4 in working mirror trees. KEEP dataset/training_artifacts or artifacts-backup; DELETE mirror copies.
- **`train_tier{1-8}_champion.py`** — `scripts/` == root copies (hashes 102–109). KEEP `scripts/`; DELETE root duplicates.
- **`agent_export.py`** — datasets/ + working/ identical (`4266d3b6bd50`); publish/staging variants differ. KEEP datasets/working; DELETE/UPDATE mirrors.
- **replays `104757231(1).json` / `104758155(1).json`** — byte-identical to originals. DELETE `(1)` copies.
- **daily manifests** `results/kaggle_episodes/daily_manifests/*.csv` duplicated in `working/kaggle_episodes/daily_manifests/` (28 files, #73–#101). Pick one canonical dir (results/) and DELETE working mirrors.
- **kaggriculture-self-training/** (input_requirements.txt, kernel-metadata.json, pip_install.sh) mirrored in working/scottweeden-… . Keep originals, DELETE mirrors.
- **egg-info + misc module mirrors** (`cli.py`, `dataset_loader.py`, `visualize.py`, `training_metrics.py`, `setup_experiment.py`, `environment.py`, `episode_catalog.py`, `eval_policy.py`, `kaggle_env_wrapper.py`, `path_b_bootstrap.py`, `replay_buffer.py`, `train_loop.py`, `train_orchestrator.py`, `kaggriculture_rl/*`, dataset-metadata.json, `_resolve_code_src.py`) — duplicated across `datasets/scottweeden/self-training-code` + `working/` + `working/publish` + `working/staging`. Keep `datasets/scottweeden/self-training-code/` (CODE_SRC) + one working copy; DELETE publish/staging mirrors.

---

## 3. Near-duplicate images (regenerated training plots)

`run_win_rate_eval.png`, `run_buffer_size.png`, `run_self_play_episodes.png` and
others exist in ~same-size variants across experiments (08-30) vs working/training_artifacts
(09-01) vs results/run (09-01 13:36) — these are regenerated per-run artifacts, not user
screenshots. KEEP the newest per run-tree; DELETE older variants.

---

## Summary / recommended actions (awaiting approval)

1. **No deletions performed.** Apply only after review:
2. Update **CODE_SRC live files** to canonical wherever legacy: `datasets/scottweeden/self-training-code/kaggriculture_adapter.py` (d3f046 → 3af858), `…/kaggriculture_path_b_rebuild.py` (3e7ff6 → 36839c), `…/checkpoints.py` (already canonical), `…/training_artifacts/models/model.pth` + `training_artifacts/agent.py` → canonical.
3. Mirrors `working/run`, `working/` root, `results/` → update to canonical or delete (est. ~5+ GB reclaimable from redundant `.pt`/`.pth` copies).
4. `experiments/` ladder snapshots + old `model.pth` variants + old checkpoints → DELETE (fully superseded).
5. Root `train_tier{1-8}_champion.py`, replay `(1)` copies, working mirror CSVs/plots/egg-info → DELETE.