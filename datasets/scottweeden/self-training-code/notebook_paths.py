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


def kaggle_input_root() -> Path:
    return (
        Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("~/kagg").expanduser()
    ).resolve()


def kaggle_working_root() -> Path:
    return (
        Path("/kaggle/working")
        if Path("/kaggle/working").exists()
        else Path("~/kagg/working").expanduser()
    ).resolve()


def code_src_candidates(kaggle_input: Optional[Path] = None) -> List[Path]:
    root = kaggle_input or kaggle_input_root()
    candidates = [
        root / "datasets" / "scottweeden" / "self-training-code",
        root / "self-training-code",
        Path("/kaggle/input/datasets/scottweeden/kaggriculture-self-training-code"),
        Path("/kaggle/input/kaggriculture-self-training-code"),
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
    """Copy writable training modules into ``/kaggle/working`` (or local mirror)."""
    for name in READ_ONLY_MODULES:
        if not (code_src / name).exists():
            raise FileNotFoundError(f"Missing read-only module in code dataset: {code_src / name}")

    kaggle_working.mkdir(parents=True, exist_ok=True)
    writable = [m for m in TRAINING_MODULES if m not in READ_ONLY_MODULES]
    for name in writable:
        src = code_src / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}")
        shutil.copy2(src, kaggle_working / name)
    shutil.copytree(code_src / "kaggriculture_rl", kaggle_working / "kaggriculture_rl", dirs_exist_ok=True)


def configure_notebook_sys_path(code_src: Path, kaggle_working: Path) -> None:
    os.chdir(kaggle_working)
    for path in (kaggle_working, code_src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def bust_stale_modules() -> None:
    for mod in (
        "episode_catalog",
        "kaggriculture_adapter",
        "path_b_bootstrap",
        "kaggriculture_dataset_publish",
        "kaggriculture_path_b_rebuild",
        "kaggle_env_wrapper",
        "dataset_loader",
        "eval_policy",
        "training_metrics",
        "visualize",
        "kaggriculture_self_play_training",
        "notebook_paths",
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
