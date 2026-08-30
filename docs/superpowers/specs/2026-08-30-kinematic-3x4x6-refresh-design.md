# Kinematic 3×4×6 end-of-cycle refresh

**Date:** 2026-08-30  
**Status:** Approved — Approach B  
**Scope:** Experiment config + docs + env `turnsPerDay` wiring for self-play

## Problem

Kaggriculture’s default season is `episodeSteps=720` with `turnsPerDay=24`, which the engine and docs present as “30 days × 24 hours.” The calendar-month framing is not a useful learning signal. What matters for RL is the **end-of-cycle refresh** (growth, weeds, shop ticks) as a kinematic feedback period.

## Decision

Keep **720** total steps. Redefine the refresh period as:

\[
72 = 3 \times 4 \times 6, \quad 720 = 10 \times 72
\]

| Constant | Value | Meaning |
|----------|------:|---------|
| `kinematic_phase_a` | 3 | Fine kinematic factor |
| `kinematic_phase_b` | 4 | Mid kinematic factor |
| `kinematic_phase_c` | 6 | Coarse kinematic factor |
| `turns_per_cycle` | 72 | Engine `turnsPerDay` (refresh every 72 steps) |
| `cycles_per_episode` | 10 | Kinematic season length |
| `max_episode_steps` | 720 | `10 × 72` |

Episode framing: **10 kinematic cycles**, not 30 calendar days.

## Env wiring

Self-play / kinematic training passes:

```python
configuration={
    "episodeSteps": max_episode_steps,  # 720
    "turnsPerDay": turns_per_cycle,     # 72
    "seed": seed,
}
```

Engine refresh still fires when `(step + 1) % turnsPerDay == 0`.

## Consequences

- **Default self-play** uses competition `turnsPerDay=24`, `episodeSteps=720` (30×24) so training matches the reference ladder under `opponents/` and historical BC tapes.
- Optional kinematic profile: pass `--turns-per-cycle 72` → engine `day` advances once per **72** steps → **10** engine-days per episode. Crop/animal tables still use day units; crops needing >10 days (e.g. melon peak at day 12) do not complete under that profile.
- **Competition / reference ladder parity** remains `turnsPerDay=24`, `episodeSteps=720` (30×24). Root `eval.py` / ladder scripts keep that default.
- Historical Kaggle episode tapes were recorded at 24; bootstrap BC must not assume 72 for tape step indices.

## Out of scope (follow-ups)

- Nested multi-scale rewards at 3 / 4 / 6
- Opponent ladder retuning for 72-step cycles (only needed if training on the optional kinematic profile)
- Changing published competition defaults

## Status note (2026-08-30)

Training defaults were switched back to **24** after ladder eval showed 0/100 wins when self-play used 72 against opponents hard-coded for 24-turn days. Kinematic 72 remains available via CLI.

## Files

- `docs/superpowers/specs/2026-08-30-kinematic-3x4x6-refresh-design.md` (this spec)
- Experiment `config.json` kinematic keys
- `README.md` short note
- `kaggriculture_self_play_training.py`, `eval_policy.py`, adapter constants
