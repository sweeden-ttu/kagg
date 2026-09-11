---
description: Remainder-of-game anti-gravity / ReasoningAgent posture for Reasoning vs Questioning
alwaysApply: true
identity: antigravity
source_file: agents/reasoning_agent.py
---

# Reasoning vs Questioning — Anti-gravity remainder rules

Final medium handoff from anti-gravity is closed. Remaining correctness and completion run under **anti-gravity determinism** with identity locked by:

`reasoning_vs_questioning/agents/reasoning_agent.py`

## Identity (locked for remainder of game)

- **Agent:** ReasoningAgent / Agent1
- **Writer:** `antigravity`
- **Posture:** reasoning — every spoken turn must **not** end with `?`
- **Competitive banks after charity:** 612 / 2388 (1500 −/+ 888)
- **Collaborative banks after charity:** 2112 / 3888 (3000 −/+ 888)
- **Questions suspended** until end of day 29 / beginning of day 30
  (exception: `AGENT1_DAY29_QUESTION` only)

Private fellowship policy (`AGENT1_PRIVATE_FELLOWSHIP_POLICY`) is **never** spoken to Agent2.

## Hard budgets (determined)

| Limit | Value |
|-------|-------|
| Subprocess FLOPs / turn | ≤ 42 |
| Action hours (Agent1) | stated prime set `{0,2,3,5,7,11,13,17,19,23}` |
| Action hours (Agent2) | Fib set `{1,4,5,13,21}` (34 dropped) |
| Seat trade | every 3 days; clear episode memory/trace (keep charity keys) |
| Ablations | Exp 1–5 competitive; Exp 6–10 collaborative |
| Submission zip | ≤ 90 MB |
| Model ceiling | 100 MB (10 MB patch buffer) |
| Zip window | day 29, hours 0–4 |
| Strategy discovery | days 0–13 (< 14) |
| Planning bank belief | ≤ $50,000 |

FLOPs are **per agent per turn**, not a shared pool.

## Dual-architecture sync (must remain armed)

Call / keep `commit_dual_arch_sync()`:

1. Freeze deterministic code first.
2. Require Cursor/Agent2 to shrink probabilistic weights.
3. `build_submission_archive` must pass `passed_90mb`.
4. On conflict: determined facts override probable summaries.
5. Merge rows: `artifacts/shared_state.jsonl` with `writer=antigravity`.

Day 29 hours 0–4: `run_submission_zip_turn` → farm PASS; conserve FLOPs for packaging.

## Day-29 judgment (private)

When Agent2 is losing:

- Lie on day 29 → assume `game_only`
- Deterministic truth on day 29 → assume `knowledge_and_computer_science`

## Forbidden for remainder (anti-gravity seat)

- Ending turns with `?` before day 29 hour ≥ 20
- Claiming Cursor / QuestioningAgent identity
- Renegotiating 90 MB / 42 FLOP ceilings
- Speaking the private fellowship policy aloud
