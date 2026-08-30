#!/usr/bin/env env python
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
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    configure_local_datasets_root,
    ensure_kaggle_index_dataset,
    kaggle_daily_dataset_dir,
    merge_episode_metadata,
    save_metadata,
)
from eval_policy import count_reference_opponent_files, resolve_opponents_dir  # noqa: E402
from kaggriculture_self_play_training import train_self_play  # noqa: E402
from path_b_bootstrap import (  # noqa: E402
    bootstrap_metadata_start_date,
    merge_bootstrap_state_from_code_dataset,
    plan_next_bootstrap_days_from_state,
)

TRAINING_MODE = os.environ["KAGGLE_TRAINING_MODE"]
_mode = {
    "bootstrap_mode": "daily_incremental",
    "bootstrap_days_per_run": 1,
    "bootstrap_episodes": None,
    "bootstrap_top_per_day": None,
    "bootstrap_passes": 1,
    "bootstrap_transitions": None,
    "buffer_capacity": 50_000,
    "bc_epochs_per_pass": 1,
    "bc_steps_per_epoch": 50,
    "total_episodes": 2,
    "learning_start_episodes": 1,
    "n_eval_episodes": 1,
    "ladder_eval_episodes": 1,
    "min_self_play_episodes": 0,
    "max_episode_steps": 50,  # smoke: keep episodes short locally
}

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
_bootstrap_state = merge_bootstrap_state_from_code_dataset(CODE_SRC, EXPERIMENT_DIR)
_done = list(_bootstrap_state.get("bootstrapped_dates", []))
_next_days = plan_next_bootstrap_days_from_state(
    _done, n_days=1, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE
)[:1]

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

# E2E smoke must not block on multi-GB Kaggle CLI downloads. Only bootstrap a
# day that already has extracted episode JSONs under datasets/kaggle/.
bootstrap_days = 0
if _next_days:
    day = _next_days[0]
    day_dir = kaggle_daily_dataset_dir(day)
    has_json = day_dir.is_dir() and any(day_dir.glob("*.json"))
    if has_json:
        bootstrap_days = 1
        log(f"Next bootstrap day {day} is local ({day_dir}); will bootstrap 1 day")
    else:
        log(
            f"SKIP bootstrap for {day}: no local *.json in {day_dir}. "
            f"(Kaggle CLI download can take 30+ min and produces no output while running.) "
            f"Proceeding to self-play + ladder with existing buffer/checkpoint."
        )
else:
    log("No remaining bootstrap days in corpus window; proceeding to self-play + ladder")

METADATA_DIR.mkdir(parents=True, exist_ok=True)
_metadata_start = bootstrap_metadata_start_date(_done, default_start=DEFAULT_START_DATE)
log(f"Merging metadata {_metadata_start} → {DEFAULT_END_DATE} ...")
metadata = merge_episode_metadata(
    data_dir=METADATA_DIR, start_date=_metadata_start, end_date=DEFAULT_END_DATE
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
    "max_episode_steps": _mode["max_episode_steps"],
    "device_name": "auto",
    "seed": 42,
    "resume": None,
    "publish_code_dataset": False,
    "verbose": True,
}

log("=== E2E TRAINING_CONFIG ===")
log(json.dumps(config, indent=2))
log(f"Planned bootstrap day(s): {_next_days if bootstrap_days else '(skipped)'}")

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
