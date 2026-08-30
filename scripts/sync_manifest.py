#!/usr/bin/env python3
"""Scan, compare, and sync mirrored directory pairs (GitHub vs Kaggle/working).

Usage:
    python scripts/sync_manifest.py scan
    python scripts/sync_manifest.py report
    python scripts/sync_manifest.py sync --direction github-to-kaggle [--dry-run]
    python scripts/sync_manifest.py sync --direction kaggle-to-github [--dry-run] [--force github|kaggle]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("sync_pairs.json")
MANIFEST_PATH = ROOT / ".sync" / "manifest.json"

GITHUB_TO_KAGGLE = "github-to-kaggle"
KAGGLE_TO_GITHUB = "kaggle-to-github"


def _utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _match_any(rel_posix: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(rel_posix, pat) for pat in patterns)


def _should_include(rel_posix: str, include: List[str], exclude: List[str]) -> bool:
    if any(part == "__pycache__" for part in rel_posix.split("/")):
        return False
    if exclude and _match_any(rel_posix, exclude):
        return False
    if not include:
        return True
    return _match_any(rel_posix, include)


def _iter_files(base: Path, include: List[str], exclude: List[str]) -> Iterable[Path]:
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        if _should_include(rel, include, exclude):
            yield path


def _file_info(path: Optional[Path]) -> Tuple[Optional[str], Optional[str]]:
    if path is None or not path.is_file():
        return None, None
    st = path.stat()
    return _sha256(path), _utc_iso(st.st_mtime)


def _status(
    github: Optional[Path],
    kaggle: Optional[Path],
    github_hash: Optional[str],
    kaggle_hash: Optional[str],
) -> str:
    if github is None and kaggle is None:
        return "missing_both"
    if github is None:
        return "kaggle_only"
    if kaggle is None:
        return "github_only"
    if github_hash == kaggle_hash:
        return "equal"
    github_mtime = github.stat().st_mtime
    kaggle_mtime = kaggle.stat().st_mtime
    if github_mtime > kaggle_mtime:
        return "github_ahead"
    if kaggle_mtime > github_mtime:
        return "kaggle_ahead"
    return "conflict"


def scan_pair(pair: dict) -> List[dict]:
    name = pair["name"]
    github_base = ROOT / pair["github"]
    kaggle_base = ROOT / pair["kaggle"]
    include = list(pair.get("include") or ["**/*"])
    exclude = list(pair.get("exclude") or [])

    github_files: Dict[str, Path] = {}
    kaggle_files: Dict[str, Path] = {}

    for path in _iter_files(github_base, include, exclude):
        github_files[path.relative_to(github_base).as_posix()] = path
    for path in _iter_files(kaggle_base, include, exclude):
        kaggle_files[path.relative_to(kaggle_base).as_posix()] = path

    rows: List[dict] = []
    for rel in sorted(set(github_files) | set(kaggle_files)):
        gp = github_files.get(rel)
        kp = kaggle_files.get(rel)
        gh, gmt = _file_info(gp)
        kh, kmt = _file_info(kp)
        rows.append(
            {
                "path": rel,
                "pair": name,
                "github": str(pair["github"]),
                "kaggle": str(pair["kaggle"]),
                "github_sha256": gh,
                "kaggle_sha256": kh,
                "github_mtime": gmt,
                "kaggle_mtime": kmt,
                "status": _status(gp, kp, gh, kh),
            }
        )
    return rows


def cmd_scan(_: argparse.Namespace) -> None:
    config = _load_config()
    all_rows: List[dict] = []
    for pair in config["pairs"]:
        all_rows.extend(scan_pair(pair))

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _utc_iso(datetime.now(tz=timezone.utc).timestamp()),
        "root": str(ROOT),
        "entries": all_rows,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(all_rows)} entries → {MANIFEST_PATH}")


def cmd_report(_: argparse.Namespace) -> None:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Missing {MANIFEST_PATH}; run: python scripts/sync_manifest.py scan")

    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    by_pair: Dict[str, List[dict]] = {}
    for row in entries:
        by_pair.setdefault(row["pair"], []).append(row)

    print(f"Manifest: {MANIFEST_PATH} ({data.get('generated_at')})")
    for pair_name, rows in sorted(by_pair.items()):
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"\n[{pair_name}] {summary}")
        for row in rows:
            if row["status"] == "equal":
                continue
            print(
                f"  {row['status']:14s} {row['path']} "
                f"(GitHub={row['github_mtime'] or '-'} Kaggle={row['kaggle_mtime'] or '-'})"
            )


def _resolve_bases(pair_name: str) -> Tuple[dict, Path, Path]:
    config = _load_config()
    pair = next(p for p in config["pairs"] if p["name"] == pair_name)
    return pair, ROOT / pair["github"], ROOT / pair["kaggle"]


def cmd_sync(args: argparse.Namespace) -> None:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Missing {MANIFEST_PATH}; run scan first")

    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = data.get("entries") or []
    copied = 0
    skipped = 0

    for row in entries:
        status = row["status"]
        if status == "equal":
            continue

        _, github_base, kaggle_base = _resolve_bases(row["pair"])
        rel = row["path"]
        src: Optional[Path] = None
        dst: Optional[Path] = None

        if args.direction == GITHUB_TO_KAGGLE:
            if status in ("github_ahead", "github_only", "conflict"):
                if status == "conflict" and args.force != "github":
                    print(f"SKIP conflict {rel} (use --force github)")
                    skipped += 1
                    continue
                src, dst = github_base / rel, kaggle_base / rel
        else:
            if status in ("kaggle_ahead", "kaggle_only", "conflict"):
                if status == "conflict" and args.force != "kaggle":
                    print(f"SKIP conflict {rel} (use --force kaggle)")
                    skipped += 1
                    continue
                src, dst = kaggle_base / rel, github_base / rel

        if src is None or dst is None or not src.is_file():
            continue

        if args.dry_run:
            print(f"DRY-RUN copy {src} → {dst}")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"copied {src} → {dst}")
        copied += 1

    print(f"{'Would copy' if args.dry_run else 'Copied'} {copied} file(s); skipped {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scan", help="Rebuild .sync/manifest.json").set_defaults(func=cmd_scan)
    sub.add_parser("report", help="Print hash/mtime diff report").set_defaults(func=cmd_report)

    sync_p = sub.add_parser("sync", help="Copy newer/changed files for one direction")
    sync_p.add_argument(
        "--direction",
        required=True,
        choices=[GITHUB_TO_KAGGLE, KAGGLE_TO_GITHUB],
        help="github-to-kaggle copies GitHub canonical → Kaggle/working mirror",
    )
    sync_p.add_argument("--dry-run", action="store_true")
    sync_p.add_argument("--force", choices=["github", "kaggle"], default=None)
    sync_p.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
