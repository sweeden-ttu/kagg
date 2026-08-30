"""Publish training artifacts into scottweeden/kaggriculture-self-training-code via Kaggle CLI."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

CODE_DATASET_SLUG = "scottweeden/kaggriculture-self-training-code"
DEFAULT_PUBLISH_DIR = Path("/kaggle/working/kaggriculture-self-training-code-publish")
ARTIFACTS_SUBDIR = "training_artifacts"

# Relative paths under experiment_dir to copy into training_artifacts/
DEFAULT_ARTIFACT_REL_PATHS: Sequence[str] = (
    "models/model.pth",
    "checkpoints/training_state_latest.pt",
    "metrics/bootstrap_state.json",
    "metrics/bc_pretrain.json",
    "metrics/episode_metrics.json",
    "metrics/training_progress.json",
    "metrics/win_rate_eval.json",
    "metrics/ladder_eval.json",
    "config.json",
    "agent.py",
)

# Directories under experiment_dir copied recursively into training_artifacts/
DEFAULT_ARTIFACT_DIR_REL_PATHS: Sequence[str] = ("plots",)

DATASET_METADATA_TEMPLATE: Dict[str, Any] = {
    "title": "kaggriculture-self-training-code",
    "id": CODE_DATASET_SLUG,
    "licenses": [{"name": "CC0-1.0"}],
}


def _run_kaggle(cmd: List[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None, text=True)


def download_code_dataset(dest_dir: Path, *, dataset_slug: str = CODE_DATASET_SLUG) -> Path:
    """Download and unzip the code dataset into ``dest_dir`` using the Kaggle CLI."""
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_kaggle(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(dest_dir),
            "--unzip",
        ]
    )
    return dest_dir


def ensure_dataset_metadata(publish_dir: Path, *, dataset_slug: str = CODE_DATASET_SLUG) -> Path:
    """Write upload-format dataset-metadata.json if missing."""
    publish_dir = Path(publish_dir)
    meta_path = publish_dir / "dataset-metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("id") != dataset_slug:
            meta["id"] = dataset_slug
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
        return meta_path

    meta = dict(DATASET_METADATA_TEMPLATE)
    meta["id"] = dataset_slug
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


def copy_training_artifacts(
    experiment_dir: Path,
    publish_dir: Path,
    artifact_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_REL_PATHS,
    artifact_dir_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_DIR_REL_PATHS,
) -> List[str]:
    """Copy model/checkpoint files from ``experiment_dir`` into ``publish_dir/training_artifacts/``."""
    experiment_dir = Path(experiment_dir)
    publish_dir = Path(publish_dir)
    artifacts_root = publish_dir / ARTIFACTS_SUBDIR
    copied: List[str] = []

    for rel in artifact_rel_paths:
        src = experiment_dir / rel
        if not src.exists():
            logger.debug("Skip missing artifact: %s", src)
            continue
        dst = artifacts_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
        logger.info("Copied artifact → %s", dst)

    for rel_dir in artifact_dir_rel_paths:
        src_dir = experiment_dir / rel_dir
        if not src_dir.is_dir():
            logger.debug("Skip missing artifact dir: %s", src_dir)
            continue
        dst_dir = artifacts_root / rel_dir
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        n_files = sum(1 for p in dst_dir.rglob("*") if p.is_file())
        copied.append(f"{rel_dir}/ ({n_files} files)")
        logger.info("Copied artifact dir → %s (%d files)", dst_dir, n_files)

    if not copied:
        logger.warning("No training artifacts copied from %s", experiment_dir)
    return copied


def bootstrap_version_message(experiment_dir: Path) -> str:
    """Build a version message from bootstrap_state.json if present."""
    return training_version_message(experiment_dir)


def training_version_message(experiment_dir: Path) -> str:
    """Build a dataset version message from bootstrap, self-play, and eval metrics."""
    experiment_dir = Path(experiment_dir)
    parts: List[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    state_path = experiment_dir / "metrics" / "bootstrap_state.json"
    if state_path.exists():
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
        days = state.get("bootstrapped_dates", [])
        if days:
            parts.append(
                f"Bootstrap {len(days)} days through {days[-1]}"
            )

    config_path = experiment_dir / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        completed = config.get("last_completed_episode")
        target = config.get("total_episodes")
        if completed is not None:
            if target is not None:
                parts.append(f"self-play ep {completed}/{target}")
            else:
                parts.append(f"self-play ep {completed}")

    wr_path = experiment_dir / "metrics" / "win_rate_eval.json"
    if wr_path.exists():
        with open(wr_path, encoding="utf-8") as fh:
            wr = json.load(fh)
        win_rate = wr.get("win_rate")
        if win_rate is not None:
            parts.append(f"win vs random {win_rate:.0%}")

    plots_dir = experiment_dir / "plots"
    if plots_dir.is_dir() and any(plots_dir.glob("*.png")):
        parts.append("plots included")

    if not parts:
        return f"Training artifacts {stamp}"
    return f"{'; '.join(parts)}; {stamp}"


def publish_training_artifacts_to_code_dataset(
    experiment_dir: Path,
    publish_dir: Optional[Path] = None,
    *,
    dataset_slug: str = CODE_DATASET_SLUG,
    version_message: Optional[str] = None,
    artifact_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_REL_PATHS,
    artifact_dir_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_DIR_REL_PATHS,
    use_zip: bool = True,
) -> Dict[str, Any]:
    """Download code dataset, merge training artifacts, and push a new dataset version.

    Returns a summary dict with ``copied`` paths and CLI status.
    """
    experiment_dir = Path(experiment_dir)
    publish_dir = Path(publish_dir or DEFAULT_PUBLISH_DIR)

    download_code_dataset(publish_dir, dataset_slug=dataset_slug)
    copied = copy_training_artifacts(
        experiment_dir,
        publish_dir,
        artifact_rel_paths,
        artifact_dir_rel_paths,
    )
    ensure_dataset_metadata(publish_dir, dataset_slug=dataset_slug)

    message = version_message or training_version_message(experiment_dir)
    version_cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(publish_dir),
    ]
    if use_zip:
        version_cmd.extend(["-r", "zip"])
    version_cmd.extend(["-m", message])

    _run_kaggle(version_cmd, cwd=publish_dir)

    summary = {
        "dataset_slug": dataset_slug,
        "publish_dir": str(publish_dir),
        "copied_artifacts": copied,
        "version_message": message,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Published code dataset version: %s", summary)
    return summary


def restore_training_artifacts_from_code_dataset(
    code_src: Path,
    experiment_dir: Path,
    *,
    artifact_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_REL_PATHS,
    artifact_dir_rel_paths: Sequence[str] = DEFAULT_ARTIFACT_DIR_REL_PATHS,
) -> List[str]:
    """Copy ``training_artifacts/`` from mounted code dataset into ``experiment_dir``."""
    code_src = Path(code_src)
    artifacts_root = code_src / ARTIFACTS_SUBDIR
    if not artifacts_root.is_dir():
        return []

    experiment_dir = Path(experiment_dir)
    restored: List[str] = []

    try:
        from path_b_bootstrap import merge_bootstrap_state_from_code_dataset

        merged = merge_bootstrap_state_from_code_dataset(code_src, experiment_dir)
        if merged.get("bootstrapped_dates"):
            restored.append("metrics/bootstrap_state.json (merged)")
    except ImportError:
        logger.warning("path_b_bootstrap unavailable; bootstrap_state merge skipped")

    for rel in artifact_rel_paths:
        if rel == "metrics/bootstrap_state.json":
            continue
        src = artifacts_root / rel
        if not src.exists():
            continue
        dst = experiment_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored.append(rel)
        logger.info("Restored artifact from code dataset: %s", rel)

    for rel_dir in artifact_dir_rel_paths:
        src_dir = artifacts_root / rel_dir
        if not src_dir.is_dir():
            continue
        dst_dir = experiment_dir / rel_dir
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        restored.append(f"{rel_dir}/")
        logger.info("Restored artifact dir from code dataset: %s", rel_dir)

    return restored
