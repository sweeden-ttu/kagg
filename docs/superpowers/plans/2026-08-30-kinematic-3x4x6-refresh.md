# Plan: Kinematic 3×4×6 refresh wiring

**Spec:** `docs/superpowers/specs/2026-08-30-kinematic-3x4x6-refresh-design.md`

## Files

| File | Change |
|------|--------|
| Experiment `config.json` | Add kinematic keys; keep `max_episode_steps=720` |
| `README.md` | Short kinematic season note |
| `kaggriculture_adapter.py` (+ experiment mirror) | Named kinematic constants; hour norm / cycle |
| `kaggriculture_self_play_training.py` | Pass `turnsPerDay` into `make()` |
| `eval_policy.py` | Optional `turns_per_day` on `_make_kaggle_env` |
| Root `eval.py` | Keep competition default 24; optional override |

## Tasks

1. Add constants + config keys  
2. Wire env `make()`  
3. README note  
4. Sanity: `3*4*6==72` and `10*72==720`
