"""Build merged episode metadata and download expert replays for imitation training.

Workflow (mirrors Kaggle notebooks scottweeden/kaggriculture-*):
1. Read global ``manifest.csv`` (kaggriculture-episodes-index).
2. For each day in [start_date, end_date], fetch that day's ``manifest.csv``.
3. Merge into ``data/kaggle_episodes/metadata.json``.
4. Download top-scoring episode JSON files into ``data/kaggle_episodes/episodes/``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import date as date_cls, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "2026-07-30"
DEFAULT_END_DATE = (date_cls.today() - timedelta(days=1)).isoformat()

# Kaggle notebook mount layouts (both observed in the wild):
#   /kaggle/input/kaggriculture-episodes-2026-08-27/{id}.json   ← flat (current)
#   /kaggle/input/datasets/kaggle/kaggriculture-episodes-...    ← nested (legacy/docs)
# Local dev mirror (when /kaggle/input is absent):
#   ~/kagg/datasets/kaggle/kaggriculture-episodes-YYYY-MM-DD/{id}.json
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_DATASETS_ROOT = KAGGLE_INPUT_ROOT / "datasets/kaggle"
DAILY_DATASET_PREFIX = "kaggriculture-episodes-"
# Mounted daily episode JSONs: /kaggle/input/kaggriculture-episodes-YYYY-MM-DD/{id}.json
KAGGLE_EPISODES_MOUNT_TEMPLATE = "/kaggle/input/kaggriculture-episodes-{date}"

_LOCAL_DATASETS_ROOT: Optional[Path] = None


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.is_dir()


def configure_local_datasets_root(root: str | Path) -> Path:
    """Set offline episode dataset root (e.g. ~/kagg/datasets/kaggle)."""
    global _LOCAL_DATASETS_ROOT
    _LOCAL_DATASETS_ROOT = Path(root).expanduser().resolve()
    return _LOCAL_DATASETS_ROOT


def local_kaggle_datasets_root() -> Path:
    if _LOCAL_DATASETS_ROOT is not None:
        return _LOCAL_DATASETS_ROOT
    env_root = os.environ.get("KAGGLE_LOCAL_DATASETS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path("~/kagg/datasets/kaggle").expanduser().resolve()


def _daily_slug(date: str) -> str:
    return f"{DAILY_DATASET_PREFIX}{date}"


def _dates_between(start_date: str, end_date: str) -> List[str]:
    start = date_cls.fromisoformat(start_date)
    end = date_cls.fromisoformat(end_date)
    days: List[str] = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _daily_dataset_has_episodes(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.json"))


def _run_kaggle_dataset_download(dataset_slug: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        f"kaggle/{dataset_slug}",
        "-p", str(dest_dir),
        "--unzip", "-q",
    ]
    logger.info("Downloading kaggle/%s -> %s", dataset_slug, dest_dir)
    subprocess.run(cmd, check=True)


def ensure_daily_episode_dataset(
    date: str,
    local_root: Optional[str | Path] = None,
) -> Path:
    """Ensure a daily episode dataset exists (Kaggle mount or local CLI download)."""
    existing = kaggle_daily_dataset_dir(date)
    if _daily_dataset_has_episodes(existing):
        return existing

    root = Path(local_root).expanduser().resolve() if local_root else local_kaggle_datasets_root()
    dest = root / _daily_slug(date)
    if _daily_dataset_has_episodes(dest):
        return dest

    _run_kaggle_dataset_download(_daily_slug(date), dest)
    if not _daily_dataset_has_episodes(dest):
        raise FileNotFoundError(f"No episode JSONs found after download: {dest}")
    return dest


def ensure_episode_datasets_for_range(
    start_date: str,
    end_date: str,
    local_root: Optional[str | Path] = None,
) -> List[Path]:
    """Download any missing daily episode datasets in [start_date, end_date]."""
    if is_kaggle_runtime():
        return [kaggle_daily_dataset_dir(d) for d in _dates_between(start_date, end_date)]
    return [
        ensure_daily_episode_dataset(d, local_root=local_root)
        for d in _dates_between(start_date, end_date)
    ]


def ensure_kaggle_index_dataset(local_root: Optional[str | Path] = None) -> Path:
    """Ensure kaggriculture-episodes-index is available locally or on Kaggle input."""
    existing = kaggle_index_dataset_dir()
    if (existing / "manifest.csv").exists():
        return existing

    root = Path(local_root).expanduser().resolve() if local_root else local_kaggle_datasets_root()
    dest = root / "kaggriculture-episodes-index"
    if (dest / "manifest.csv").exists():
        return dest

    _run_kaggle_dataset_download("kaggriculture-episodes-index", dest)
    if not (dest / "manifest.csv").exists():
        raise FileNotFoundError(f"Missing index manifest after download: {dest / 'manifest.csv'}")
    return dest


def kaggle_daily_dataset_dir(date: str) -> Path:
    """Return daily dataset dir (Kaggle mount or local ~/kagg/datasets/kaggle mirror)."""
    slug = _daily_slug(date)
    if KAGGLE_INPUT_ROOT.is_dir():
        flat = KAGGLE_INPUT_ROOT / slug
        if flat.is_dir():
            return flat
        nested = KAGGLE_DATASETS_ROOT / slug
        if nested.is_dir():
            return nested
        return flat

    local = local_kaggle_datasets_root() / slug
    if local.is_dir():
        return local
    return local


def kaggle_index_dataset_dir() -> Path:
    slug = "kaggriculture-episodes-index"
    if KAGGLE_INPUT_ROOT.is_dir():
        for candidate in (
            KAGGLE_DATASETS_ROOT / slug,
            KAGGLE_INPUT_ROOT / slug,
        ):
            if candidate.is_dir():
                return candidate
        return KAGGLE_DATASETS_ROOT / slug

    local = local_kaggle_datasets_root() / slug
    if local.is_dir():
        return local
    return local


def kaggle_episode_path(date: str, episode_id: str) -> Path:
    """Absolute path to one episode JSON on a Kaggle notebook mount."""
    return kaggle_daily_dataset_dir(date) / f"{episode_id}.json"


def resolve_episode_json_path(
    date: str,
    episode_id: str,
    local_episodes_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Resolve episode JSON: Kaggle input first, then local ``episodes/`` copy."""
    kaggle_path = kaggle_episode_path(date, episode_id)
    if kaggle_path.exists():
        return kaggle_path
    if local_episodes_dir is not None:
        local_path = Path(local_episodes_dir) / f"{episode_id}.json"
        if local_path.exists() and local_path.stat().st_size > 0:
            return local_path
    return None


def kaggle_daily_manifest_path(date: str) -> Path:
    return kaggle_daily_dataset_dir(date) / "manifest.csv"


def kaggle_index_manifest_path() -> Path:
    return kaggle_index_dataset_dir() / "manifest.csv"


def _run_kaggle_download(dataset_slug: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", f"kaggle/{dataset_slug}",
        "-f", filename,
        "-p", str(dest_dir),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)
    out = dest_dir / filename
    if not out.exists():
        raise FileNotFoundError(f"Expected {out} after kaggle download")
    return out


def load_index_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def filter_days(
    index_rows: List[Dict[str, str]],
    start_date: str,
    end_date: str,
) -> List[Dict[str, str]]:
    return [
        row for row in index_rows
        if start_date <= row["date"] <= end_date
    ]


def fetch_daily_manifest(
    dataset_slug: str,
    cache_dir: Path,
    date: str,
    refresh: bool = False,
) -> List[Dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{date}.csv"
    if cached.exists() and not refresh:
        with open(cached, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    kaggle_manifest = kaggle_daily_manifest_path(date)
    if kaggle_manifest.exists() and not refresh:
        logger.info("Reading daily manifest from Kaggle input: %s", kaggle_manifest)
        rows = list(csv.DictReader(open(kaggle_manifest, encoding="utf-8")))
        with open(cached, "w", encoding="utf-8", newline="") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return rows

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _run_kaggle_download(dataset_slug, "manifest.csv", tmp_path)
        rows = list(csv.DictReader(open(tmp_path / "manifest.csv", encoding="utf-8")))
        with open(cached, "w", encoding="utf-8", newline="") as fh:
            if rows:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    return rows


def merge_episode_metadata(
    data_dir: str | Path = "./data/kaggle_episodes",
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    index_manifest: Optional[str | Path] = None,
    refresh_daily: bool = False,
) -> Dict[str, Any]:
    """Merge daily episode manifests into a single metadata.json structure."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(index_manifest) if index_manifest else data_path / "manifest.csv"
    kaggle_index_manifest = kaggle_index_manifest_path()
    if kaggle_index_manifest.exists():
        manifest_path = kaggle_index_manifest
        logger.info("Using Kaggle index manifest: %s", manifest_path)
    elif not manifest_path.exists():
        logger.info("Downloading global episodes index manifest...")
        _run_kaggle_download("kaggriculture-episodes-index", "manifest.csv", data_path)

    index_rows = load_index_manifest(manifest_path)
    day_rows = filter_days(index_rows, start_date, end_date)
    if not day_rows:
        raise ValueError(f"No datasets between {start_date} and {end_date}")

    cache_dir = data_path / "daily_manifests"
    merged_episodes: List[Dict[str, Any]] = []
    daily_stats: List[Dict[str, Any]] = []

    for day in day_rows:
        date = day["date"]
        slug = day["daily_dataset_slug"]
        logger.info("Fetching daily manifest for %s (%s)...", date, slug)
        episodes = fetch_daily_manifest(slug, cache_dir, date, refresh=refresh_daily)
        daily_dir = kaggle_daily_dataset_dir(date)
        for ep in episodes:
            ep_id = ep["episode_id"]
            merged_episodes.append(
                {
                    "episode_id": ep_id,
                    "date": date,
                    "dataset_slug": slug,
                    "kaggle_path": str(daily_dir / f"{ep_id}.json"),
                    "avg_score": float(ep.get("avg_score", 0)),
                    "min_score": float(ep.get("min_score", 0)),
                    "sum_score": float(ep.get("sum_score", 0)),
                    "agent_count": int(ep.get("agent_count", 2)),
                    "size_bytes": int(float(ep.get("size_bytes", 0))),
                    "create_time": ep.get("create_time", ""),
                }
            )
        daily_stats.append(
            {
                "date": date,
                "dataset_slug": slug,
                "episode_count": len(episodes),
                "top_avg_score": float(day.get("top_avg_score", 0)),
            }
        )

    merged_episodes.sort(key=lambda r: r["avg_score"], reverse=True)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "source": "kaggle/kaggriculture-episodes-index",
        "date_range": {"start": start_date, "end": end_date},
        "daily_datasets": daily_stats,
        "total_episodes_indexed": len(merged_episodes),
        "episodes": merged_episodes,
    }
    return metadata


def save_metadata(metadata: Dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Wrote merged metadata: %s (%d episodes)", out, len(metadata.get("episodes", [])))
    return out


def select_episodes_for_download(
    metadata: Dict[str, Any],
    top_per_day: Optional[int] = 20,
    max_episodes: Optional[int] = 500,
) -> List[Dict[str, Any]]:
    """Pick top-scoring episodes per day, capped globally.

    Pass ``top_per_day=None`` and/or ``max_episodes=None`` to include all indexed episodes.
    """
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for ep in metadata.get("episodes", []):
        by_date.setdefault(ep["date"], []).append(ep)

    selected: List[Dict[str, Any]] = []
    for date in sorted(by_date.keys()):
        day_eps = sorted(by_date[date], key=lambda r: r["avg_score"], reverse=True)
        if top_per_day is None:
            selected.extend(day_eps)
        else:
            selected.extend(day_eps[:top_per_day])

    selected.sort(key=lambda r: r["avg_score"], reverse=True)
    if max_episodes is None:
        return selected
    return selected[:max_episodes]


def resolve_episode_paths_from_metadata(
    metadata: Dict[str, Any],
    top_per_day: Optional[int] = 20,
    max_episodes: Optional[int] = 500,
    local_episodes_dir: Optional[str | Path] = None,
) -> List[Path]:
    """Return resolved JSON paths for top episodes (Kaggle input preferred)."""
    picks = select_episodes_for_download(metadata, top_per_day, max_episodes)
    resolved: List[Path] = []
    for ep in picks:
        path = resolve_episode_json_path(
            ep["date"],
            ep["episode_id"],
            local_episodes_dir=local_episodes_dir,
        )
        if path is not None:
            resolved.append(path)
    return resolved


def download_episodes_from_metadata(
    metadata: Dict[str, Any],
    episodes_dir: str | Path,
    top_per_day: int = 20,
    max_episodes: int = 500,
    skip_existing: bool = True,
    use_kaggle_input: Optional[bool] = None,
) -> List[Path]:
    """Resolve or download episode JSON files listed in merged metadata.

    When ``use_kaggle_input`` is true (default on Kaggle runtime), episode JSONs
    are read directly from mounted daily datasets, e.g.
    ``/kaggle/input/kaggriculture-episodes-2026-08-27/100453695.json``
    without copying to ``episodes_dir``.
    """
    if use_kaggle_input is None:
        use_kaggle_input = is_kaggle_runtime()

    episodes_path = Path(episodes_dir)
    picks = select_episodes_for_download(metadata, top_per_day, max_episodes)
    downloaded: List[Path] = []

    for idx, ep in enumerate(picks, 1):
        ep_id = ep["episode_id"]
        date = ep["date"]
        slug = ep["dataset_slug"]
        filename = f"{ep_id}.json"
        kaggle_src = kaggle_episode_path(date, ep_id)

        if use_kaggle_input and kaggle_src.exists():
            logger.info(
                "[%d/%d] using Kaggle input %s (avg_score=%.2f)",
                idx, len(picks), kaggle_src, ep["avg_score"],
            )
            downloaded.append(kaggle_src)
            continue

        dest = episodes_path / filename
        if skip_existing and dest.exists() and dest.stat().st_size > 0:
            logger.info("[%d/%d] skip existing %s", idx, len(picks), filename)
            downloaded.append(dest)
            continue

        if kaggle_src.exists():
            episodes_path.mkdir(parents=True, exist_ok=True)
            logger.info(
                "[%d/%d] copying %s from Kaggle input %s (avg_score=%.2f)",
                idx, len(picks), filename, kaggle_src.parent.name, ep["avg_score"],
            )
            shutil.copy2(kaggle_src, dest)
            downloaded.append(dest)
            continue

        episodes_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[%d/%d] downloading %s from %s (avg_score=%.2f)",
            idx, len(picks), filename, slug, ep["avg_score"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            _run_kaggle_download(slug, filename, Path(tmp))
            src = Path(tmp) / filename
            dest.write_bytes(src.read_bytes())
        downloaded.append(dest)

    logger.info("Resolved %d episode JSON paths", len(downloaded))
    return downloaded


def dates_in_metadata(metadata: Dict[str, Any]) -> List[str]:
    """Sorted list of calendar dates present in merged metadata."""
    return sorted({str(ep["date"]) for ep in metadata.get("episodes", [])})


def resolve_episode_paths_for_dates(
    metadata: Dict[str, Any],
    dates: List[str],
    local_episodes_dir: Optional[str | Path] = None,
) -> List[Path]:
    """Resolve **all** episode JSON paths for the given dates (no score cap)."""
    want = set(dates)
    resolved: List[Path] = []
    for ep in metadata.get("episodes", []):
        if str(ep["date"]) not in want:
            continue
        path = resolve_episode_json_path(
            ep["date"],
            ep["episode_id"],
            local_episodes_dir=local_episodes_dir,
        )
        if path is not None:
            resolved.append(path)
    resolved.sort(key=lambda p: (p.parent.name, p.stem))
    return resolved


def pick_next_bootstrap_days(
    metadata: Dict[str, Any],
    n_days: int = 3,
    exclude_dates: Optional[List[str]] = None,
) -> List[str]:
    """Pick the next ``n_days`` in chronological order not yet bootstrapped."""
    excluded = set(exclude_dates or [])
    available = [d for d in dates_in_metadata(metadata) if d not in excluded]
    picked = available[: max(0, n_days)]
    overlap = set(picked) & excluded
    if overlap:
        raise ValueError(f"pick_next_bootstrap_days would repeat bootstrapped days: {sorted(overlap)}")
    return picked


def pick_random_bootstrap_days(
    metadata: Dict[str, Any],
    n_days: int = 3,
    exclude_dates: Optional[List[str]] = None,
    seed: int = 42,
) -> List[str]:
    """Deprecated alias — use :func:`pick_next_bootstrap_days` (sequential days)."""
    return pick_next_bootstrap_days(metadata, n_days=n_days, exclude_dates=exclude_dates)


def build_and_download_catalog(
    data_dir: str = "./data/kaggle_episodes",
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    top_per_day: int = 20,
    max_episodes: int = 500,
    skip_existing: bool = True,
    refresh_daily: bool = False,
) -> Dict[str, Any]:
    """Merge metadata and download episodes; returns metadata dict."""
    data_path = Path(data_dir)
    metadata = merge_episode_metadata(
        data_dir=data_path,
        start_date=start_date,
        end_date=end_date,
        refresh_daily=refresh_daily,
    )
    save_metadata(metadata, data_path / "metadata.json")
    download_episodes_from_metadata(
        metadata,
        data_path / "episodes",
        top_per_day=top_per_day,
        max_episodes=max_episodes,
        skip_existing=skip_existing,
    )
    return metadata
