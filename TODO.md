# Kaggriculture — TODO

---

## Full DQN replay pipeline — next run

- [x] Scale to higher episode count (15+) and 720-step seasons
- [x] Enable non-zero BC epochs for behavioral cloning pretrain
- [x] Verify PER-driven priority updates (TD-error-based reweighting)
- [x] Full training run: bootstrap + BC pretrain + self-play + ladder eval
- [ ] Pass ladder eval: win rate ≥ 50% vs every opponent including tier 0

### Evidence (2026-08-30)

| Item | Result | Notes |
|------|--------|-------|
| BC epochs | **PASS** | CLI/default/`train_self_play` `bc_epochs=15`; notebook `medium`/`full` use 15; experiment `config.json` has 15. Smoke/dry_run still uses `bc_epochs=0` + `bc_epochs_per_pass`. Log: `BC epoch 15/15 \| avg loss: 0.24277`. |
| PER priorities | **PASS** | Bug: `update_priorities` skipped numpy `(source, idx)` rows (`isinstance(..., (list, tuple))`). Fixed. `scripts/verify_per_priorities.py` → PASS; training log: `PER: TD-error priority reweighting applied (n=32 ...)`. |
| Full pipeline | **PASS** | `scripts/run_full_dqn_pipeline.py`: reseed bootstrap → BC 15×200 → 16×720 self-play → competition ladder (`turnsPerDay=24`, 720 steps). ~30 min on MPS. Log: `logs/full_pipeline_2026-08-30.log`. |
| Ladder ≥50% all | **FAIL** | 0/10 opponents cleared. vs `fallow_finn`: win 0%, money **669 vs 3000** (after ε=1 self-play wipe). BC-only: **3000 vs 3000** ties (PASS-spawn deadlock). After `break_pass_spawn_deadlock`: moves but still ~3000 (no plant/harvest). |

### Ladder table (full pipeline, 10 ep/opponent, 720×24)

| Opponent | Win% | P0 $ | P1 $ | Cleared |
|----------|-----:|-----:|-----:|:-------:|
| fallow_finn | 0 | 669 | 3000 | no |
| wheat_walter | 0 | 752 | 6526 | no |
| rotation_rosa | 0 | 691 | 12164 | no |
| homestead_hana | 0 | 120 | 15381 | no |
| melon_mateo | 0 | 9 | 24105 | no |
| rancher_rita | 0 | 167 | 44201 | no |
| broker_bea | 0 | 3 | 44097 | no |
| ledger_lena | 0 | 3 | 43552 | no |
| slotter_silas | 0 | 3 | 42949 | no |
| closer_cleo | 0 | 4 | 17429 | no |

### Remaining blockers / next steps

1. **Policy does not farm**: after leaving spawn, agent wanders (N/S/E/W) without BUY_SEED → PLANT → WATER → HARVEST. Need BC that upweights market invest + plant/harvest (or filter PASS/move-heavy frames), and/or more self-play with low ε after BC (`eps_start=0.25` now).
2. **Do not use ε=1 after BC** — already mitigated (`eps_start=0.25` when BC ran).
3. Re-run: BC (PASS-downweighted + invest upweight) → low-ε self-play 25–50 eps × 720 → ladder 10×720×24.
4. Optional: short ladder-only smoke vs Finn after each BC to confirm money > 3000 before long self-play.

---

## Completed (cleared 2026-08-30)

All prior tasks resolved: notebook parity fixes, pipeline correctness, docs & hygiene, Kaggle/GitHub sync phases 0–4, and verification checklist items. See git history for details.
