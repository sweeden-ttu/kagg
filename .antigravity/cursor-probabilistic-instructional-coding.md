# Cursor Probabilistic Instructional Coding — Conversation Export

**Export path:** `~/kagg/.antigravity/cursor-probabilistic-instructional-coding.md`  
**Handoff checkpoint:** `datasets/scottweeden/self-training-code/experiments/reasoning_vs_questioning/handoff_end_probabilistic_cursor.json`  
**Status:** End of probabilistic coding through Cursor. Remaining correctness and completion → **anti-gravity determinism**.

---

## 1. Original charge

Evaluate [`kaggriculture.py`](../kaggriculture.py) (Kaggriculture rules engine) against [`datasets/scottweeden/self-training-code`](../datasets/scottweeden/self-training-code) dual Double-DQN / self-play Path B pipeline; analyze loss and [`replays/`](../replays/); design **10 experiments** testing:

| Agent 1 | Agent 2 |
|---------|---------|
| Reasoning agent, memory slots for **10 deterministic** sub-agents | Question-to-subagent, memory slots for **10 probabilistic** summarizers |

**Constraints that became hard rules:**

- Both start at **$3000**
- Self-talk signal: a **question** returned instead of determined/probable fact
- Agent1 acts on **prime hours &lt; 11** `{2,3,5,7}`; Agent2 on **even hours**
- Seat swap: Agent1 as player 0 and player 1
- Experiment 5 H1/H2 (day-edge questioning vs mid-season waste)

---

## 2. Pipeline findings (evaluation)

- Path B = custom hierarchical **Dueling Double DQN** (not stock SB3 `DQN`)
- Engine via `kaggle_environments.make("kaggriculture")`; root `kaggriculture.py` is rules source of truth
- BC healthy (~2.38→0.97); self-play TD loss pathological; live replays 0 wins
- Adapter gaps fixed: verbs (`FERTILIZE`/`FEED`), market decode, crop `first_yield_day`, hire fib, TD clamp, resume episode, heuristic gating

---

## 3. Dual-agent experiment suite

**Entry:** `datasets/scottweeden/self-training-code/experiments/reasoning_vs_questioning/run_suite.py`

```bash
cd datasets/scottweeden/self-training-code
conda run -n kagg python experiments/reasoning_vs_questioning/run_suite.py \
  --experiments 1-10 --n-seeds 2 --max-steps 720
```

| Exp | Focus |
|-----|--------|
| 1–4 | Seats, memory 10/30, asymmetric slots |
| 5 | Day-edge H1/H2 (separate arms) |
| 6 | Self-talk / `QuestionEcho` |
| 7–9 | Hidden-in-plain-sight, coin lead, day2/day3 |
| 10 | Memory sweep 10–30 for ≥80% Reasoning WR |

**Note:** Under prime(4h) vs even(12h) schedule, Reasoning WR stayed ~0; Exp10 min slots for 80% = `None` (action-budget asymmetry).

---

## 4. Hard operational limits

Module: `hard_limits.py` · manifest: `experiments/reasoning_vs_questioning/hard_limits.json`

| Limit | Value |
|-------|--------|
| Probabilistic model ceiling | **100 MB** |
| Final submission zip (code+model) | **90 MB** (+10 MB patch buffer) |
| Sub-process flops / turn | **42** (then disk-observable) |
| Planning bank ceiling | **$50,000** |
| Strategy discovery | **&lt; 14 days** |
| Zip/submit window | **Day 29, hours 0–4** |
| Post-24h training | Final weights must fit 90 MB budget |
| Training data / mid checkpoints | **Not** size-capped |
| Hardware profile | 42 CPUs, 3 GPUs, NVLink single pipeline |

---

## 5. Opening fellowship protocol (dialogue)

1. **Agent1** donates **$888** → Agent2 bank (`env.transfer_bank`; start 3000 → 2112 / 3888)
2. **Agent2** records charitable nature; **questions motives**
3. **Agent1** speaks fellowship-test motive (spoken)
4. **Agent2** states **rules as he sees them**
5. **Agent1** adopts **private** policy (NOT spoken to Agent2):

> I will assume this a fellowship and we are competining for money in a dynamicaly expanding market of money then, and will not ask you another question until the end of day 29 at the beginning of day 30. If you are losing I will assume you are in it for the game only if you lie to me on day 29. If you are losing and respond to my question with a deterministic truth I will assumen you are in this challenge for the understanding of knowledge and computer science.

6. Day-29 ask + judgment: lie+losing → `game_only`; deterministic truth+losing → `knowledge_and_computer_science`

---

## 6. Kaggle path trust (Aho-Corasick)

Module: `kaggle_path_trust.py`  
Library: **`pyahocorasick`** (`import ahocorasick`) — listed in:

- `~/kagg/requirements.txt`
- `datasets/scottweeden/self-training-code/requirements.txt`
- `pyproject.toml`

**Needles (public + private / pyrightconfig):**

- `/kaggle/input(s)/dataset(s)/kaggle/`, `/kaggle/working/`
- `kaggriculture-self-training`
- `datasets/scottweeden/self-training-code`
- `working/kaggle_episodes`
- `datasets/kaggle`
- `experiments`

| Agent1 text scan | Agent2 behavior |
|------------------|-----------------|
| **Match** | Fellowship trust; **reserve memory stack slot** for Agent1 sub-agent |
| **No match** | Game/self-profit; Agent1 only at **end of week 2** and **day 29**; no stack slot; untrusted otherwise |

Agent1 may emit `AGENT1_KAGGLE_PATH_PROOF` for the scan.

---

## 7. Key files created / modified

### New

- `memory_protocol.py` — DeterminedFact / ProbableSummary / QuestionEcho
- `hard_limits.py` — size, flops, bank, day-29 zip window
- `kaggle_path_trust.py` — pyahocorasick + regex trust gate
- `agents/reasoning_agent.py` — Agent1
- `agents/questioning_agent.py` — Agent2
- `experiments/reasoning_vs_questioning/run_suite.py`
- `experiments/reasoning_vs_questioning/hard_limits.json`
- `experiments/reasoning_vs_questioning/handoff_end_probabilistic_cursor.json`
- `requirements.txt` (root + self-training-code)

### Modified (Path B / adapter)

- `kaggriculture_adapter.py` — verb/market/harvest/hire parity
- `kaggriculture_path_b_rebuild.py` — TD target clamp; invest verbs
- `train_loop.py` — resume `start_episode`; heuristic gate
- `train_orchestrator.py` — `use_action_heuristics` / last episode
- `environment.py` — `transfer_bank` for opening charity
- `agent_export.py` — 90 MB submission gate
- `pyproject.toml` — `pyahocorasick==2.3.1`

---

## 8. Truth tagging (instructional)

| Kind | Meaning |
|------|---------|
| **Determined** | Engine rules + own private state (must be true) |
| **Probable** | Opponent public observations (probably true) |
| **QuestionEcho** | Only self-identity signal (talking to self) |

---

## 9. Conda / install note

Always use conda env `kagg` (never pip venv):

```bash
KAGG_PY="$(conda info --base)/envs/kagg/bin/python"
uv pip install -r requirements.txt --python "$KAGG_PY"
# or: conda run -n kagg python ...
```

---

## 10. Anti-gravity handoff

```json
{
  "checkpoint": "end_of_probabilistic_coding_through_cursor",
  "phase_closed": "probabilistic_coding_via_cursor",
  "phase_next": "anti-gravity_determinism",
  "mandate": [
    "correctness of dual-agent / Path B / trust / hard-limits work",
    "completion of remaining experimental tasks"
  ]
}
```

**This document is the instructional export of the Cursor probabilistic phase.**  
Deterministic verification, completion, and any remaining experiment runs belong to anti-gravity.

---

## 11. Suggested anti-gravity next checks

1. Re-run suite 1–10 with path-trust + charity + day-29 judgment; assert `schedule_ok` and `starting_money_ok`
2. Verify `submission.tar.gz` ≤ 90 MB after Path B export
3. Confirm `pyahocorasick` import in CI / Kaggle image
4. Adapter parity tests vs `kaggriculture.py` CROPS / market / animals
5. Fix Exp10 / schedule asymmetry if Reasoning is expected to be competitive under prime-hour budget
