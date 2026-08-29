# Kaggriculture — compact training home

**Repository:** https://github.com/sweeden-ttu/kagg

Production home for the Kaggriculture agent. Replaces the sprawling `challenges/kaggriculture/experiments/` tree with a **fixed file budget** copied from the 20 most recently touched files in the legacy folder (2026-08-29 migration).

---

## Directory map

```text
~/kagg/
├── README.md                 ← architecture, limits, inventory (this file)
│
├── eval.py                   ← rubric-aligned head-to-head eval + opponent loader
├── pair_match.py             ← CLI: one challenger vs full opponent ladder
├── run_experiments_eval.py   ← CLI: batch eval across agent bundles
├── run_published_eval.py     ← CLI: download Kaggle champion + pair_match
│
├── opponents/                ← 10 reference ladder agents (tier 0–9)
│   ├── fallow_finn.py        ← tier 0 — reward floor (~3k bank)
│   ├── wheat_walter.py       ← tier 1
│   ├── rotation_rosa.py        ← tier 2
│   ├── homestead_hana.py     ← tier 3
│   ├── melon_mateo.py        ← tier 4 — premium crop baseline
│   ├── rancher_rita.py       ← tier 5 — livestock
│   ├── broker_bea.py         ← tier 6 — meta field plan
│   ├── ledger_lena.py        ← tier 7
│   ├── slotter_silas.py      ← tier 8
│   └── closer_cleo.py        ← tier 9
│
├── datasets/
│   ├── reference/            ← opponent ladder metadata (from Kaggle reference dataset)
│   │   ├── agents_manifest.csv      ← tier order, expected bank, strategy notes
│   │   ├── baseline_league.csv      ← ladder standings
│   │   ├── head_to_head_games.csv   ← precomputed reference matchups
│   │   ├── crop_economics.csv
│   │   ├── price_curves.csv
│   │   └── season_timeline.csv
│   ├── published-champion/   ← gitignored — downloaded Kaggle model bundle
│   ├── published_eval.json   ← output from run_published_eval.py
│   └── eval_summary.json     ← output from run_experiments_eval.py
│
└── kernels/                  ← reserved for the single training notebook (not migrated yet)
    └── (empty — add train.ipynb here when training code lands)
```

### Why each directory exists

| Location | Meaning | Why it matters |
|---|---|---|
| **Repo root `.py`** | Evaluation + pairing only (4 files) | Head-to-head win rate is the promotion gate; keep eval next to the repo root so Kaggle kernels can import with one path. |
| **`opponents/`** | Frozen `agent(obs)` policies | Self-play seat 1 and all eval run against **this roster**, not random/heuristic noise. Does **not** count toward the 10-file Python cap. |
| **`datasets/reference/`** | CSV metadata for the ladder | Tier ordering, expected banks, and precomputed H2H — read-only context for eval and reporting. |
| **`datasets/` (runtime)** | Downloaded weights, eval JSON | Artifacts land here; large binaries stay gitignored under `published-champion/`. |
| **`kernels/`** | Exactly one notebook | Kaggle train + publish loop lives here once `train.py` / adapter modules are merged in. |

---

## Migrated files (top 20 by mtime from `challenges/kaggriculture/`)

| # | Source | Destination |
|---|---|---|
| 1 | `eval_policy.py` | `eval.py` |
| 2 | `scripts/pair_match.py` | `pair_match.py` |
| 3 | `scripts/run_experiments_eval.py` | `run_experiments_eval.py` |
| 4 | `scripts/run_published_eval.py` | `run_published_eval.py` |
| 5–14 | `agents/opponents/*.py` (10 agents) | `opponents/` |
| 15–20 | `agents/opponents/*.csv` (6 tables) | `datasets/reference/` |

**Python file count after migration: 14 total** (4 root + 10 opponents). Root trainable cap is **≤ 10** — room for **6 more** training modules (`adapter`, `train`, `replay`, `dqn_sb3`, `main`, …) before the ceiling.

---

## ⛔ Do not add more files

This repository is **frozen at the migrated surface area**. The legacy project grew unbounded (`experiments/*`, duplicate scripts, `.tmp/` copies) and became impossible to audit or upload to Kaggle.

**Rules (non-negotiable after this copy):**

1. **No new `.py` files** unless you **delete or merge** an existing one first (root budget ≤ 10 trainable modules).
2. **No new notebooks** — only `kernels/train.ipynb` when training lands (replaces nothing until added).
3. **No new policy/config/model paths** — one `main.py`, one `config.json`, one `model.pth`, one `bootstrap_model.pth`.
4. **No `experiments/`, `scripts/`, or `.tmp/` trees** — runtime output goes to `datasets/` or Kaggle `/kaggle/working/`.
5. **Opponent changes** — swap body of an existing `opponents/<name>.py`, do not add tier 10+ files without removing another.

Need something new? **Merge into an existing file** or keep it outside `~/kagg/`.

---

## Training architecture

### Phase 0 — Imitation replay bootstrap (last 30 days)

The replay buffer is seeded from the **last 30 calendar days** of public Kaggriculture episode JSONs before RL gradients run:

1. Pull daily episode JSONs from the Kaggle corpus.
2. Parse `(obs, action, reward, next_obs, done)` from top-ranked seats.
3. Fill PER buffer — **behavioral cloning priors**, not random rollouts.

### Phase 1 — DQN (SB3-style)

**Dueling Double DQN with action branching**, API aligned with [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html): online + target networks, hierarchical action heads (farmer / crop / hands / market).

### Phase 2 — Dual adversarial pipeline

| Seat | Role |
|---|---|
| **Player 0** | Online DQN challenger — learns from replay |
| **Player 1** | Frozen policy from `opponents/` |

Promotion = head-to-head win rate vs **every** file in `opponents/`.

```text
  Episode JSONs (30d) ──► Imitation buffer ──► BC warm-start
                                                    │
  opponents/*.py ◄──── self-play 720-step ──── DQN challenger
                                                    │
                                              main.py + model.pth
```

*Training modules (`train.py`, `adapter.py`, …) and `kernels/train.ipynb` are not in this migration — eval + opponent ladder only.*

---

## Evaluation

```bash
# One agent dir vs full ladder
python pair_match.py --agent-dir datasets/published-champion

# Download Kaggle champion + eval
python run_published_eval.py --n-episodes 20
```

Or native Kaggle:

```python
from kaggle_environments import evaluate
from pathlib import Path

for opp in sorted(Path("opponents").glob("*.py")):
    r = evaluate("kaggriculture", ["main.py", str(opp)], num_episodes=20)
    print(opp.stem, sum(1 for x in r if x[0] > x[1]), len(r))
```

Win/loss/tie: higher final bank wins ([Kaggle rubric](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation)).

---

## Hard limits (full tree)

| Artifact | Limit |
|---|---|
| **Python at repo root** | **≤ 10** (currently **4** — room for 6 training modules) |
| **`opponents/*.py`** | reference roster only — **not** counted in the 10 |
| **Notebooks** | **1** → `kernels/train.ipynb` |
| **Submission** | **1** `main.py` with `agent(obs, configuration)` |
| **Config / weights** | **1** `config.json`, **1** `model.pth`, **1** `bootstrap_model.pth` |

---

## Legacy source

Copied from: `~/kaggle-leaderboard-notebooks/challenges/kaggriculture/`  
Do not sync experiments or scripts back from legacy without explicit merge into the file budget above.
