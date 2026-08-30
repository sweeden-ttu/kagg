# Kaggriculture — Training home for scottweeden/kaggriculture-self-training

**Repository:** https://github.com/sweeden-ttu/kagg

Production home for the Kaggriculture agent.

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
│   ├── scottweeden/self-training-code/   ← gitignored — downloaded Kaggle model bundle
│   ├── published_eval.json   ← output from run_published_eval.py
│   └── eval_summary.json     ← output from run_experiments_eval.py
│
└── kernel1/                  ← reserved for the single training notebook (not migrated yet)
|    └── (thees contain .ipynb that are operating experiments )
├── kernel2/                  ← reserved for the single training notebook (not migrated yet)
     └── (empty — add train.ipynb here when training code lands)
```
---

## Development environment

Local work uses a **miniforge** conda env named `kagg` (Python 3.12). Packages are installed with **`uv pip`** inside that env — fast resolver, same wheels as pip.

### Stable-Baselines3 dependency stack

SB3 is **PyTorch-only** — it does **not** use TensorFlow. Install PyTorch first, then SB3.

| Layer | Packages | Role |
|---|---|---|
| **Deep learning** | `torch`, `torchvision`, `torchaudio` | Neural nets, GPU/MPS backends |
| **SB3 core** | `gymnasium`, `numpy`, `cloudpickle` | RL env API, arrays, model serialization |
| **SB3 `[extra]`** | `tensorboard`, `pandas`, `matplotlib`, `tqdm`, `rich`, `psutil`, `opencv-python`, `pillow`, `pygame-ce`, `ale-py` | Logging, plots, progress bars, Atari (optional) |
| **Kaggriculture** | `kaggle-environments`, `kagglehub` | Simulation + dataset/kernel downloads |
| **Notebooks** | `jupyter`, `ipykernel` | Local kernel + Kaggle notebook parity |

### One-time setup

```bash
# Create env (miniforge)
conda create -n kagg python=3.12 -y
conda activate kagg

# uv drives all pip installs (pin env when another conda env is active)
conda install -c conda-forge uv -y
KAGG_PY="$(conda info --base)/envs/kagg/bin/python"
uv pip install --python "$KAGG_PY" torch torchvision torchaudio
uv pip install --python "$KAGG_PY" "stable-baselines3[extra]" kaggle-environments kagglehub jupyter ipykernel

# Register Jupyter kernel for notebooks in this repo
python -m ipykernel install --user --name kagg --display-name "Python (kagg)"
```

### Cursor / VS Code

This repo pins the interpreter in `.vscode/settings.json` to:

`/Users/sweeden/miniforge3/envs/kagg/bin/python`

Reload the window after first setup so notebooks and the integrated terminal pick up the `kagg` env automatically.

---

## ⛔ Do not add more files

`experiments/*` should be the only place submissions are generated and submitted from. There should always be one active experiment and that experiment should be documented below this line:

**Mappings**
1. ~/kagg = /kaggle/input
2. ~/kagg/working = /kaggle/working  & ~/kagg/experiments is player 1
3. ~/kagg/datasets = /kaggle/input/dataset
4. ~/kagg/opponents = adversarial opponents is player 2

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


---

## Evaluation

```bash
# One agent dir vs full ladder
python pair_match.py --agent-dir datasets/scottweeden/self-training-code

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
