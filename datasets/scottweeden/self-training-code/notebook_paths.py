"""Shared path and environment helpers for the training notebook."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

CODE_DATASET_DIR_NAMES = (
    "kaggriculture-self-training-code",
    "self-training-code",
)

READ_ONLY_MODULES = frozenset(
    {
        "kaggriculture_adapter.py",
        "kaggriculture_path_b_rebuild.py",
        "notebook_paths.py",
    }
)

TRAINING_MODULES = (
    "kaggriculture_self_play_training.py",
    "train_orchestrator.py",
    "train_loop.py",
    "agent_coordinator.py",
    "agent_export.py",
    "checkpoints.py",
    "setup_experiment.py",
    "replay_buffer.py",
    "environment.py",
    "cli.py",
    "_resolve_code_src.py",
    "kaggriculture_dataset_publish.py",
    "episode_catalog.py",
    "path_b_bootstrap.py",
    "kaggriculture_adapter.py",
    "kaggriculture_path_b_rebuild.py",
    "kaggle_env_wrapper.py",
    "dataset_loader.py",
    "eval_policy.py",
    "training_metrics.py",
    "visualize.py",
    "notebook_paths.py",
)


def truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def fresh_run_requested() -> bool:
    """When set, skip resume from checkpoints / code-dataset artifacts."""
    return truthy_env("KAGGLE_FRESH_RUN")


def dry_run_resume_requested() -> bool:
    """Opt-in resume for dry_run (default: fresh self-play target each run)."""
    return truthy_env("KAGGLE_RESUME")


# ── README Mappings ────────────────────────────────────────────────────────
# 1. ~/kagg = /kaggle/input
# 2. ~/kagg/working = /kaggle/working & ~/kagg/experiments is player 1
# 3. ~/kagg/datasets = /kaggle/input/dataset (or /kaggle/input/datasets)
# 4. ~/kagg/opponents = adversarial opponents is player 2


def kaggle_input_root() -> Path:
    """Mapping 1: ~/kagg = /kaggle/input."""
    if Path("/kaggle/input").exists():
        return Path("/kaggle/input").resolve()
    return Path("~/kagg").expanduser().resolve()


def kaggle_working_root() -> Path:
    """Mapping 2: ~/kagg/working = /kaggle/working."""
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working").resolve()
    return Path("~/kagg/working").expanduser().resolve()


def kaggle_experiments_root() -> Path:
    """Mapping 2 (Player 1): ~/kagg/experiments (or /kaggle/working/run)."""
    if Path("/kaggle/working/run").exists():
        return Path("/kaggle/working/run").resolve()
    if Path("~/kagg/experiments").expanduser().exists():
        return Path("~/kagg/experiments").expanduser().resolve()
    if Path("/kaggle/working").exists():
        return Path("/kaggle/working/run").resolve()
    return Path("~/kagg/experiments").expanduser().resolve()


def kaggle_datasets_root() -> Path:
    """Mapping 3: ~/kagg/datasets = /kaggle/input/datasets or /kaggle/input/dataset."""
    for candidate in (
        Path("/kaggle/input/datasets"),
        Path("/kaggle/input/dataset"),
        Path("~/kagg/datasets").expanduser(),
    ):
        if candidate.exists():
            return candidate.resolve()
    return Path("~/kagg/datasets").expanduser().resolve()


def kaggle_opponents_root() -> Path:
    """Mapping 4 (Player 2): ~/kagg/opponents = /kaggle/input/opponents."""
    for candidate in (
        Path("/kaggle/input/opponents"),
        Path("/kaggle/input/kaggriculture-reference-agents"),
        Path("/kaggle/input/datasets/raykkretzschmar/kaggriculture-reference-agents"),
        Path("/kaggle/input/dataset/raykkretzschmar/kaggriculture-reference-agents"),
        Path("~/kagg/opponents").expanduser(),
        Path("~/kagg/datasets/reference").expanduser(),
    ):
        if candidate.exists() and any(candidate.glob("*.py")):
            return candidate.resolve()
    return Path("~/kagg/opponents").expanduser().resolve()


def code_src_candidates(kaggle_input: Optional[Path] = None) -> List[Path]:
    in_root = kaggle_input or kaggle_input_root()
    ds_root = kaggle_datasets_root()
    candidates = [
        ds_root / "scottweeden" / "self-training-code",
        ds_root / "scottweeden" / "kaggriculture-self-training-code",
        in_root / "datasets" / "scottweeden" / "self-training-code",
        in_root / "datasets" / "scottweeden" / "kaggriculture-self-training-code",
        in_root / "scottweeden" / "self-training-code",
        in_root / "scottweeden" / "kaggriculture-self-training-code",
        in_root / "self-training-code",
        Path("/kaggle/input/datasets/scottweeden/kaggriculture-self-training-code"),
        Path("/kaggle/input/dataset/scottweeden/kaggriculture-self-training-code"),
        Path("/kaggle/input/kaggriculture-self-training-code"),
        Path("~/kagg/datasets/scottweeden/self-training-code").expanduser(),
        Path("~/kagg/datasets/scottweeden/kaggriculture-self-training-code").expanduser(),
        Path(__file__).resolve().parent,
    ]
    seen: set[Path] = set()
    ordered: List[Path] = []
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def resolve_code_src(kaggle_input: Optional[Path] = None) -> Path:
    for candidate in code_src_candidates(kaggle_input):
        if (candidate / "episode_catalog.py").exists():
            return candidate
    tried = ", ".join(str(p) for p in code_src_candidates(kaggle_input))
    raise FileNotFoundError(
        "Missing code dataset. Tried: "
        f"{tried}. Attach scottweeden/kaggriculture-self-training-code."
    )


def deploy_training_code(
    code_src: Path,
    kaggle_working: Path,
) -> None:
    """Copy all training modules into ``/kaggle/working`` (or local mirror)."""
    kaggle_working.mkdir(parents=True, exist_ok=True)
    for name in TRAINING_MODULES:
        src = code_src / name
        if not src.exists():
            continue
        dst = kaggle_working / name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    rl_src = code_src / "kaggriculture_rl"
    rl_dst = kaggle_working / "kaggriculture_rl"
    if rl_src.exists() and rl_src.resolve() != rl_dst.resolve():
        shutil.copytree(rl_src, rl_dst, dirs_exist_ok=True)


def configure_notebook_sys_path(code_src: Path, kaggle_working: Path) -> None:
    os.chdir(kaggle_working)
    for path in (kaggle_working, code_src):
        path_str = str(path.resolve())
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)


def bust_stale_modules() -> None:
    for mod in list(sys.modules.keys()):
        if (
            mod.startswith("kaggriculture")
            or mod in (
                "episode_catalog",
                "path_b_bootstrap",
                "kaggle_env_wrapper",
                "dataset_loader",
                "eval_policy",
                "training_metrics",
                "visualize",
                "notebook_paths",
                "train_orchestrator",
                "train_loop",
                "agent_coordinator",
                "agent_export",
                "checkpoints",
                "setup_experiment",
                "replay_buffer",
                "environment",
                "cli",
                "_resolve_code_src",
            )
        ):
            sys.modules.pop(mod, None)


def ensure_sys_paths(*paths: Path) -> None:
    for path in paths:
        if path is None:
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def experiment_dir(kaggle_working: Optional[Path] = None) -> Path:
    return (kaggle_working or kaggle_working_root()) / "run"


def metadata_dir(kaggle_working: Optional[Path] = None) -> Path:
    return (kaggle_working or kaggle_working_root()) / "kaggle_episodes"
