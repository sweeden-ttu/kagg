# Experiment 3 Synthesis — Reasoning vs Questioning

**Run id:** `_full_e3`  
**Generated:** 2026-09-11T20:52:10.907091+00:00  
**Seeds:** [42, 43, 44] · **Horizon:** 720 steps (30 days × 24 turns)  
**Scoring:** `cumulative_farm_delta` (opening gift 888 travels with Agent2 identity, not land equity)  
**Acceptance:** Move 2 gate **PASSED** (33/33 checks)

---

## Executive Summary

Ten ablations ran under the Experiment 3 protocol: **Exp 1–5 competitive** (start 1500 + 888 gift) and **Exp 6–10 collaborative** (start 3000 + 888 gift + day-2 refund). Agents traded seats every 3 days (9 trades/match), acted on stated prime / Fib schedules, and were scored by cumulative farm Δ while controlling seats.

Headline results under farm-Δ scoring:
- Competitive baselines favor **Questioning** on net value generated (Exp 1–2: Q_wr = 1.0, mean QΔ ≈ 2118.3 vs RΔ ≈ 769.7).
- Memory ceiling (Exp 3, mem=30) **hurts Reasoning** (RΔ ≈ -1938.3) while Questioning stays near flat positive.
- Collaborative Exp 10 shows **strong memory sensitivity**: team farm Δ peaks at mem=10 (2888.0) and declines through mem=30 (-2613.0); std ≈ 2012.9.
- Protocol compliance is complete: schedule_ok / starting_money_ok everywhere; mean seat trades = 9.0.

---

## Competitive Findings (Exp 1–5)

| Exp | Name | R_wr | Q_wr | mean RΔ | mean QΔ | Notes |
|-----|------|------|------|---------|---------|-------|
| 1 | baseline_seats | 0.0 | 1.0 | 769.7 | 2118.3 | Seat-only, mem=10 |
| 2 | memory_floor | 0.0 | 1.0 | 769.7 | 2118.3 | Q omits determined truths |
| 3 | memory_ceiling | 0.0 | 1.0 | -1938.3 | 50.7 | Both mem=30; high-fid capital burn |
| 4 | asymmetric_memory | r10q30 R=0.0; r30q10 R=0.0 | — | r10q30 RΔ=-210.3 / QΔ=1607.3; r30q10 RΔ=-1650.0 / QΔ=-32.0 | — | Asymmetry arms |
| 5 | day_edge_questioning | H1=False H2=False | — | Q day0=30 mid R=396 Q day29=144 | — | Edge logs present; hypotheses unsupported at n=3 |

**Interpretation**
- **Determined facts vs noisy summaries (Exp 2):** Same Q_wr pattern as baseline under farm-Δ — omitting determined truths did not reverse Questioning’s lead at this seed budget.
- **Memory ceiling (Exp 3):** Extra slots raise fidelity and unlock secondary-crop spend; Reasoning’s farm Δ turns sharply negative — memory is binding, not inert.
- **Asymmetry (Exp 4):** Both arms keep Reasoning win-rate at 0 under farm-Δ; slot count alone does not restore Reasoning throughput under the Fib/prime schedule split.
- **Day-edge (Exp 5):** Stratified logs exist (day0 / mid / day29), but H1/H2 were not supported at `--n-seeds 3`.

---

## Collaborative Findings (Exp 6–10)

| Exp | Name | Team farm Δ (or arm) | Team $ (diagnostic) | Notes |
|-----|------|----------------------|---------------------|-------|
| 6 | self_talk_detection | 2888.0 | 8888.0 | force_self_talk both agents |
| 7 | hidden_in_plain_sight | 2888.0 | 8888.0 | Q omit_determined_truths |
| 8 | coin_lead_validation | agg Δ=2888.0; idle Δ=1472.0 | agg $=8888.0; idle $=7472.0 | Idle when ahead reduces team Δ |
| 9 | day2_refund_day3_sides | 2888.0 | 8888.0 | refunds=6/6; alliances=6/6 |
| 10 | win_all_memory_search | best_slots=10 | see curve | Non-flat scaling |

### Exp 10 memory scaling curve (`mean_team_farm_delta`)

| Memory slots | Team farm Δ | Team $ (terminal diagnostic) |
|-------------:|------------:|-------------------------------:|
| 10 | 2888.0 | 8888.0 |
| 15 | -1540.0 | 4460.0 |
| 20 | -1733.0 | 4267.0 |
| 25 | -2316.0 | 3684.0 |
| 30 | -2613.0 | 3387.0 |

**std(mean_team_farm_delta) ≈ 2012.9** (acceptance: > 0). Peak joint farm Δ at **mem=10**; larger banks increase fidelity-driven capital burn and lower cumulative team Δ under this policy.

**Day-2 / Day-3 (Exp 9):** 100% day-2 Kaggle refund activation and complementary alliances (`reasoning_side != questioning_side`) across all 6 matches.

---

## Protocol Compliance Verification

| Check | Result |
|-------|--------|
| Agent1 hours | `[0, 2, 3, 5, 7, 11, 13, 17, 19, 23]` |
| Agent2 hours | `[1, 4, 5, 13, 21]` |
| `schedule_ok_all` | true (all experiment summaries) |
| `starting_money_ok_all` | true; competitive 1500 / collaborative 3000 |
| `mean_seat_trades` | **9.0** (≥ 9 required) |
| `scoring_mode` | `cumulative_farm_delta` |
| Opening charity | 888 (Agent2 identity transfer) |
| Move 2 acceptance | PASSED — see `acceptance_gate.json` |

---

## Artifacts

| Path | Role |
|------|------|
| `suite_report.json` | Full suite rollup |
| `acceptance_gate.json` | Move 2 gate evaluation |
| `launch_contract.json` | Move 1 determined launch contract |
| `exp{{1..10}}/metrics.json` | Per-ablation metrics |
| `experiment3_artifacts.tar.gz` | Packaged submission archive |
| `EXPERIMENT_3_SYNTHESIS.md` | This document |

---

## Ready for Move 4

State: **artifact packaging complete** → `ready_for_move_4_final_handoff`.
