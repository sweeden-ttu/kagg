#!/usr/bin/env python
"""Farm-upweighted BC → Finn money gate → low-ε self-play → competition ladder.

Goal: clear ladder (≥75% win rate vs every opponent including fallow_finn) at
competition settings (720 steps, turnsPerDay=24).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path(__file__).resolve().parents[1]
CODE_SRC = ROOT / "datasets/scottweeden/self-training-code"
EXP_ROOT = ROOT / (
    "experiments/Hinesight Experience Replay for Double Q-Learning DQN SB3 "
    "Kaggriculture reinforcement learning"
)
EXP = EXP_ROOT / "farm_bc_ladder"
OPPONENTS = ROOT / "opponents"
sys.path.insert(0, str(CODE_SRC))

from eval_policy import evaluate_win_rate, load_kaggle_agent_policy  # noqa: E402
from kaggriculture_self_play_training import train_self_play  # noqa: E402
from visualize import update_experiment_plots  # noqa: E402

FINN_MONEY_GATE = 3000.0
FINN_SMOKE_EPISODES = 4
SELF_PLAY_EPISODES = 36
LADDER_EPISODES = 10


def _log(msg: str) -> None:
    print(msg, flush=True)


def _finn_smoke(policy, *, n_episodes: int = FINN_SMOKE_EPISODES, seed: int = 42) -> dict:
    finn = load_kaggle_agent_policy(OPPONENTS / "fallow_finn.py")
    stats = evaluate_win_rate(
        policy,
        finn,
        n_episodes=n_episodes,
        max_steps=720,
        base_seed=seed,
        turns_per_day=24,
    )
    moneys = [e["p0_money"] for e in stats.get("episodes") or []]
    avg_money = sum(moneys) / len(moneys) if moneys else 0.0
    report = {
        "win_rate": stats["win_rate"],
        "avg_p0_money": avg_money,
        "avg_p1_money": (
            sum(e["p1_money"] for e in stats["episodes"]) / len(stats["episodes"])
            if stats.get("episodes")
            else 0.0
        ),
        "wins": stats["wins"],
        "losses": stats["losses"],
        "ties": stats["ties"],
        "n_episodes": stats["n_episodes"],
        "p0_moneys": moneys,
        "cleared_gate": avg_money > FINN_MONEY_GATE,
    }
    gate_path = EXP / "metrics" / "finn_smoke.json"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _policy_from_exported_agent():
    """Load the exported experiment Agent after train_self_play writes agent.py."""
    agent_path = EXP / "agent.py"
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing exported agent: {agent_path}")
    spec = importlib.util.spec_from_file_location("farm_bc_exported_agent", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "agent"):
        return module.agent
    inst = module.Agent()
    return inst.act


def main() -> int:
    EXP.mkdir(parents=True, exist_ok=True)
    _log(f"=== Farm-BC ladder run → {EXP} ===")
    _log(f"Python: {sys.executable}")

    # Phase 1: bootstrap + farm-upweighted BC only (no self-play wipe).
    train_self_play(
        experiment_dir=str(EXP),
        code_src=str(CODE_SRC),
        use_kaggle_env=True,
        seed=42,
        device_name="auto",
        total_episodes=0,
        learning_start_episodes=0,
        batch_size=32,
        checkpoint_interval=5,
        turns_per_cycle=24,
        max_episode_steps=720,
        n_eval_episodes=0,
        bootstrap_episodes=None,
        data_dir=str(ROOT / "working/kaggle_episodes"),
        metadata_path=str(ROOT / "working/kaggle_episodes/metadata.json"),
        download_bootstrap=False,
        bc_epochs=25,
        bc_batch_size=64,
        bc_steps_per_epoch=300,
        buffer_capacity=50_000,
        bootstrap_top_per_day=40,
        bootstrap_passes=1,
        bc_epochs_per_pass=2,
        verbose=True,
        bootstrap_mode="daily_incremental",
        bootstrap_days_per_run=1,
        publish_code_dataset=False,
        opponents_dir=str(OPPONENTS),
        ladder_eval_episodes=0,
        ladder_win_rate_target=0.75,
        resume=None,
    )

    policy = _policy_from_exported_agent()
    finn = _finn_smoke(policy, n_episodes=FINN_SMOKE_EPISODES, seed=42)
    _log(
        f"Finn smoke: avg_p0=${finn['avg_p0_money']:.1f} "
        f"win={finn['win_rate']:.0%} ties={finn['ties']} "
        f"gate={'PASS' if finn['cleared_gate'] else 'FAIL'} (need >{FINN_MONEY_GATE})"
    )

    if not finn["cleared_gate"]:
        _log("Finn money gate failed — skipping long self-play. Inspect finn_smoke.json.")
        summary = update_experiment_plots(EXP)
        _log(f"Plots: {summary.get('plots_dir')}")
        return 2

    # Phase 2: low-ε self-play then full competition ladder.
    _log(
        f"Finn gate cleared — self-play {SELF_PLAY_EPISODES}×720 then "
        f"ladder {LADDER_EPISODES} ep/opponent"
    )
    train_self_play(
        experiment_dir=str(EXP),
        code_src=str(CODE_SRC),
        use_kaggle_env=True,
        seed=42,
        device_name="auto",
        total_episodes=SELF_PLAY_EPISODES,
        learning_start_episodes=2,
        batch_size=32,
        checkpoint_interval=6,
        turns_per_cycle=24,
        max_episode_steps=720,
        n_eval_episodes=LADDER_EPISODES,
        bootstrap_episodes=None,
        data_dir=str(ROOT / "working/kaggle_episodes"),
        metadata_path=str(ROOT / "working/kaggle_episodes/metadata.json"),
        download_bootstrap=False,
        bc_epochs=0,  # keep BC weights from phase 1; resume buffer/model
        bc_batch_size=64,
        bc_steps_per_epoch=0,
        buffer_capacity=50_000,
        bootstrap_top_per_day=40,
        bootstrap_passes=1,
        bc_epochs_per_pass=0,
        verbose=True,
        bootstrap_mode="daily_incremental",
        bootstrap_days_per_run=1,
        publish_code_dataset=False,
        opponents_dir=str(OPPONENTS),
        ladder_eval_episodes=LADDER_EPISODES,
        ladder_win_rate_target=0.75,
        min_self_play_episodes=SELF_PLAY_EPISODES,
        resume=str(EXP),
    )

    summary = update_experiment_plots(EXP)
    ladder_path = EXP / "metrics" / "ladder_eval.json"
    if ladder_path.exists():
        ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
        _log(
            f"Ladder cleared={ladder.get('opponents_cleared')}/{ladder.get('n_opponents')} "
            f"beats_all={ladder.get('beats_all_opponents')}"
        )
        finn_row = (ladder.get("results") or {}).get("fallow_finn") or {}
        _log(
            f"vs fallow_finn: win={finn_row.get('win_rate')} "
            f"p0=${finn_row.get('avg_p0_money')} p1=${finn_row.get('avg_p1_money')}"
        )
    _log(f"Plots: {summary.get('plots_dir')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
