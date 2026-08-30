#!/usr/bin/env python
"""Run notebook §1a→§1c→§2 pipeline to produce ladder_eval.json (E2E smoke)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Line-buffer stdout/stderr so ``tee`` and log files show progress immediately.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "datasets/scottweeden/self-training-code"
KAGGLE_INPUT = ROOT
KAGGLE_WORKING = ROOT / "working"
METADATA_DIR = KAGGLE_WORKING / "kaggle_episodes"
METADATA_PATH = METADATA_DIR / "metadata.json"
EXPERIMENT_DIR = KAGGLE_WORKING / "run"


def log(msg: str) -> None:
    print(msg, flush=True)


log(f"E2E smoke starting (cwd will be {KAGGLE_WORKING})")

os.environ.setdefault("KAGGLE_TRAINING_MODE", "dry_run")
os.environ["KAGGLE_FRESH_RUN"] = "1"  # no resume — clean self-play + ladder target

sys.path.insert(0, str(KAGGLE_WORKING))
sys.path.insert(0, str(CODE_SRC))
os.chdir(KAGGLE_WORKING)

from episode_catalog import (  # noqa: E402
    configure_local_datasets_root,
    ensure_kaggle_index_dataset,
    merge_episode_metadata,
    save_metadata,
)
from eval_policy import count_reference_opponent_files, resolve_opponents_dir  # noqa: E402
from kaggriculture_adapter import (  # noqa: E402
    CYCLES_PER_EPISODE,
    EPISODE_STEPS,
    TURNS_PER_CYCLE,
)
from kaggriculture_self_play_training import train_self_play  # noqa: E402
from notebook_paths import fresh_run_requested  # noqa: E402
from path_b_bootstrap import save_bootstrap_state  # noqa: E402

TRAINING_MODE = os.environ["KAGGLE_TRAINING_MODE"]

# Dry-run phase counts (each phase exercised more than once except the last,
# which is the agent/ladder output): 2, 3, 4, 6, 4, 3, 2, 1
_DRY_RUN_PHASES = {
    "bootstrap_days_per_run": 2,   # 1 — bootstrap days
    "bootstrap_passes": 3,         # 2 — corpus/pass loops
    "bc_epochs_per_pass": 4,       # 3 — BC epochs per pass/day
    "bc_steps_per_epoch": 6,       # 4 — BC optimizer steps
    "total_episodes": 4,           # 5 — self-play episodes
    "learning_start_episodes": 3,  # 6 — warmup before learning
    "n_eval_episodes": 2,          # 7 — win-rate eval vs baseline
    "ladder_eval_episodes": 1,     # 8 — agent/ladder output (once)
}

# Cap expert JSONs per day so smoke BC stays seconds-not-hours.
_SMOKE_TOP_PER_DAY = 3

_mode = {
    "bootstrap_mode": "daily_incremental",
    "bootstrap_episodes": None,
    "bootstrap_top_per_day": _SMOKE_TOP_PER_DAY,
    "bootstrap_transitions": None,
    "buffer_capacity": 50_000,
    "min_self_play_episodes": 0,
    # Kinematic season: one cycle is enough for smoke; full season is EPISODE_STEPS.
    "turns_per_cycle": TURNS_PER_CYCLE,
    "max_episode_steps": TURNS_PER_CYCLE,  # 72 = one 3×4×6 refresh cycle
    **_DRY_RUN_PHASES,
}

assert list(_DRY_RUN_PHASES.values()) == [2, 3, 4, 6, 4, 3, 2, 1], _DRY_RUN_PHASES
assert _mode["learning_start_episodes"] < _mode["total_episodes"]
assert TURNS_PER_CYCLE == 3 * 4 * 6
assert EPISODE_STEPS == TURNS_PER_CYCLE * CYCLES_PER_EPISODE

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

opp = resolve_opponents_dir(code_src=str(CODE_SRC))
if opp is None or not opp.is_dir():
    raise SystemExit("opponents/ not found")
log(f"Opponents: {opp} ({count_reference_opponent_files(opp)} agents)")

LOCAL_DATASETS = KAGGLE_INPUT / "datasets" / "kaggle"
configure_local_datasets_root(LOCAL_DATASETS)

log("Ensuring episodes index (local only; no CLI download in E2E smoke)...")
index_dir = LOCAL_DATASETS / "kaggriculture-episodes-index"
if not (index_dir / "manifest.csv").exists():
    ensure_kaggle_index_dataset(local_root=LOCAL_DATASETS)
else:
    log(f"  index already present: {index_dir}")


def _local_days_with_json(root: Path) -> list[str]:
    """Chronological dates that already have extracted episode JSONs locally."""
    days: list[str] = []
    for day_dir in sorted(root.glob("kaggriculture-episodes-????-??-??")):
        if day_dir.is_dir() and any(day_dir.glob("*.json")):
            days.append(day_dir.name.replace("kaggriculture-episodes-", "", 1))
    return days


_wanted_days = int(_mode["bootstrap_days_per_run"])
local_available = _local_days_with_json(LOCAL_DATASETS)
log(f"Local episode days with *.json: {local_available or '(none)'}")

# Smoke must not inherit "already bootstrapped" from the code-dataset watermark,
# otherwise phases 1–4 are no-ops (next calendar days lack local JSON).
if fresh_run_requested():
    save_bootstrap_state(
        EXPERIMENT_DIR,
        {"bootstrapped_dates": [], "runs": [], "total_transitions": 0},
    )
    log("KAGGLE_FRESH_RUN=1: reset experiment bootstrap_state for smoke re-bootstrap")

local_bootstrap_days = local_available[:_wanted_days]
if len(local_bootstrap_days) < _wanted_days:
    raise SystemExit(
        f"E2E smoke needs {_wanted_days} local bootstrap day(s) with *.json under "
        f"{LOCAL_DATASETS}; found {len(local_available)}: {local_available}"
    )

bootstrap_days = len(local_bootstrap_days)
log(f"Will bootstrap {bootstrap_days}/{_wanted_days} local day(s): {local_bootstrap_days}")

# Metadata window = only the smoke days so daily_incremental picks them first.
METADATA_DIR.mkdir(parents=True, exist_ok=True)
_metadata_start = min(local_bootstrap_days)
_metadata_end = max(local_bootstrap_days)
log(f"Merging metadata {_metadata_start} → {_metadata_end} (smoke local days only) ...")
metadata = merge_episode_metadata(
    data_dir=METADATA_DIR, start_date=_metadata_start, end_date=_metadata_end
)
save_metadata(metadata, METADATA_PATH)
log(f"Metadata saved: {METADATA_PATH} ({len(metadata.get('episodes', []))} episodes indexed)")

config = {
    "experiment_dir": str(EXPERIMENT_DIR),
    "code_src": str(CODE_SRC),
    "use_kaggle_env": True,
    "bootstrap_mode": _mode["bootstrap_mode"],
    "bootstrap_days_per_run": bootstrap_days,
    "bootstrap_episodes": _mode["bootstrap_episodes"],
    "bootstrap_top_per_day": _mode["bootstrap_top_per_day"],
    "bootstrap_passes": _mode["bootstrap_passes"],
    "bootstrap_transitions": _mode["bootstrap_transitions"],
    "buffer_capacity": _mode["buffer_capacity"],
    "bc_epochs": 0,
    "bc_epochs_per_pass": _mode["bc_epochs_per_pass"],
    "bc_steps_per_epoch": _mode["bc_steps_per_epoch"],
    "data_dir": str(METADATA_DIR),
    "metadata_path": str(METADATA_PATH),
    "download_bootstrap": False,
    "bc_batch_size": 64,
    "total_episodes": _mode["total_episodes"],
    "learning_start_episodes": _mode["learning_start_episodes"],
    "batch_size": 32,
    "checkpoint_interval": 10,
    "n_eval_episodes": _mode["n_eval_episodes"],
    "ladder_eval_episodes": _mode["ladder_eval_episodes"],
    "ladder_win_rate_target": 0.5,
    "min_self_play_episodes": 0,
    "opponents_dir": str(opp),
    "turns_per_cycle": _mode["turns_per_cycle"],
    "max_episode_steps": _mode["max_episode_steps"],
    "device_name": "auto",
    "seed": 42,
    "resume": None,
    "publish_code_dataset": False,
    "verbose": True,
}

log("=== E2E TRAINING_CONFIG ===")
log(json.dumps(config, indent=2))
log(
    "Dry-run phases (2,3,4,6,4,3,2,1): "
    + ", ".join(f"{k}={v}" for k, v in _DRY_RUN_PHASES.items())
)
log(f"Planned bootstrap day(s): {local_bootstrap_days}")
log(f"Smoke bootstrap_top_per_day={_SMOKE_TOP_PER_DAY}")

log("Starting train_self_play ...")
train_self_play(**config)

ladder_path = EXPERIMENT_DIR / "metrics" / "ladder_eval.json"
if not ladder_path.exists():
    raise SystemExit(f"FAIL: missing {ladder_path}")

report = json.loads(ladder_path.read_text())
n_opponents = len(report.get("results", {}))
log(f"\n=== ladder_eval.json OK ({n_opponents} opponents) ===")
for slug, row in report.get("results", {}).items():
    log(
        f"  {slug:16s} win={row.get('win_rate', 0):.0%} "
        f"p0={row.get('avg_p0_money', 0):.0f} p1={row.get('avg_p1_money', 0):.0f}"
    )
tier6 = report.get("results", {}).get("broker_bea", {})
if tier6.get("avg_p1_money", 0) == 0:
    log("WARN: broker_bea avg_p1_money is 0 — check step field")
log(f"beats_all_opponents: {report.get('beats_all_opponents')}")
