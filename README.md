# Kaggriculture — compact training home

This folder is the **production home** for the Kaggriculture agent: training kernels, opponent roster, and published datasets. It replaces the sprawling multi-experiment layout under `challenges/kaggriculture/experiments/`.

## Layout

```text
~/kagg/
├── README.md           ← this file
├── opponents/          ← reference + ladder agents for head-to-head eval and self-play pairing
├── kernels/            ← Kaggle notebook(s) for train + publish
└── datasets/           ← code bundles, episode indexes, published artifacts
```

| Path | Purpose |
|---|---|
| `opponents/` | Frozen opponent policies (`agent(obs)` modules). Reference ladder (Fallow Finn → Broker Bea) and any pinned competitors. |
| `kernels/` | The single training notebook and nothing else. Runs bootstrap → RL → export on Kaggle. |
| `datasets/` | Read-only inputs attached to kernels: daily episode JSONs, merged metadata, published model/checkpoint exports. |

---

## Training architecture

### Phase 0 — Imitation replay bootstrap (last 30 days)

Before any gradient step from self-play, the **replay buffer is seeded from imitation**:

1. Pull the **last 30 calendar days** of public Kaggriculture episode JSONs (720-turn, 2-player duels from the Kaggle episode corpus).
2. Parse `(observation, action, reward, next_observation, done)` transitions from top-ranked seats in those games.
3. Fill a prioritized replay buffer (PER) to capacity with expert transitions — **behavioral cloning priors**, not random rollouts.

This is the bootstrap: the agent starts with “what good players already did” on recent meta, not from scratch.

### Phase 1 — DQN (SB3-style) deep reinforcement learning

The policy is a **Dueling Double DQN with action branching**, trained through an API aligned with [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html):

- **Online network** selects actions; **target network** stabilizes TD targets.
- Hierarchical action heads (farmer / crop / hands / market) instead of a flat discrete space.
- Experience replay + ε-greedy exploration after the imitation buffer reaches `learning_start`.

BC loss on the bootstrapped buffer can run for additional epochs each time new daily data is ingested.

### Phase 2 — Dual adversarial pipeline

Training is **two-player adversarial**, not single-agent vs a stationary world model:

| Seat | Role |
|---|---|
| **Player 0 (challenger)** | Online DQN — learns from shaped rewards and replay. |
| **Player 1 (opponent)** | Frozen policy from `opponents/` (reference ladder, heuristic, or historical checkpoint). |

Each episode is a full 720-step Kaggle environment match. Only challenger transitions enter the PER buffer. Checkpoints are saved on an interval; promotion is gated by head-to-head win rate vs the opponent roster in `opponents/`.

End-of-run eval is **vs every opponent in `opponents/`**, not vs random noise.

```text
  ┌─────────────────────┐     last 30 days      ┌──────────────────┐
  │ Kaggle episode JSONs│ ────────────────────► │ Imitation replay │
  └─────────────────────┘                       │ buffer (bootstrap)│
                                                └────────┬─────────┘
                                                         │ BC warm-start
                                                         ▼
  ┌─────────────────────┐   self-play matches   ┌──────────────────┐
  │ opponents/*.py      │ ◄──────────────────► │ DQN (SB3-style)  │
  └─────────────────────┘                       │ challenger       │
                                                └────────┬─────────┘
                                                         │
                                                         ▼
                                                main.py + model.pth
                                                (competition submit)
```

---

## Hard limits (this home folder)

These rules keep the tree **deployable, auditable, and Kaggle-uploadable**. Violating them is a process error — merge or delete before shipping.

| Artifact | Limit |
|---|---|
| **Python modules** (`.py` anywhere under `~/kagg/`) | **≤ 10 files total** |
| **Notebooks** (`.ipynb`) | **1** (lives in `kernels/`) |
| **Policy / submission entry** | **1** — `main.py` (must expose `agent(obs, configuration)`) |
| **Config** | **1** — `config.json` (hyperparameters, bootstrap cursor, resume pointers) |
| **Champion model** | **1** — `model.pth` (weights for submission + inference) |
| **Bootstrap checkpoint** | **1** — `bootstrap_model.pth` (post-BC / pre–self-play snapshot for resume) |

Everything else — logs, plots, traces, extra checkpoints, scratch experiments — stays **outside** `~/kagg/` or inside Kaggle `/kaggle/working/` for the duration of a run only.

### Allowed shape (example)

```text
~/kagg/
├── README.md
├── config.json
├── main.py
├── model.pth
├── bootstrap_model.pth
├── adapter.py              ┐
├── env.py                  │
├── replay.py               ├─ ≤ 10 .py total (including main.py)
├── train.py                │
├── eval.py                 │
├── dqn_sb3.py              ┘
├── opponents/              ← opponent .py files do NOT count toward the 10 (read-only roster)
├── kernels/
│   └── train.ipynb         ← the one notebook
└── datasets/
    └── …                   ← data only, no training code
```

> **Note:** Opponent policies under `opponents/` are evaluation fixtures, not part of the trainable codebase quota. If you inline opponents into the 10-file budget, you will run out of room immediately.

---

## Evaluation

Pair the champion against **all** of `opponents/`:

```python
from kaggle_environments import evaluate

for opp in sorted(Path("~/kagg/opponents").glob("*.py")):
    results = evaluate("kaggriculture", ["main.py", str(opp)], num_episodes=20)
    wins = sum(1 for r in results if r[0] > r[1])
    print(opp.stem, wins, "/", len(results))
```

Win/loss/tie follows the [Kaggriculture rubric](https://www.kaggle.com/competitions/kaggriculture/overview/evaluation): higher final bank balance wins; margin does not matter.

---

## Migration from `challenges/kaggriculture/`

The legacy repo accumulated many `experiments/*` trees, duplicate scripts, and multiple notebooks. When moving here:

1. Pick **one** champion checkpoint → `model.pth` + `bootstrap_model.pth`.
2. Collapse training code to **≤ 10** `.py` files; delete or archive the rest.
3. Move the self-training notebook → `kernels/train.ipynb` (only notebook).
4. Copy reference ladder → `opponents/`.
5. Point dataset mounts → `datasets/`.

Do not copy `experiments/`, `scripts/`, or `.tmp/` into this tree.
