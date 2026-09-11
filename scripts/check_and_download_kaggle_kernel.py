#!/usr/bin/env python3
"""
Automated Kaggle Download & Review Staging Job.
Safely stages downloaded artifacts in `incoming_review/` for inspection before
any baseline artifacts in `output/` are overwritten ("bounced").
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KERNEL_SLUG = "scottweeden/kaggriculture-self-training"
BASE_KERNEL_DIR = Path("/Users/sweeden/.local/share/kaggle/kernels/kaggriculture-self-training")
BASELINE_DIR = BASE_KERNEL_DIR / "output"
STAGING_DIR = BASE_KERNEL_DIR / "incoming_review"
ARTIFACT_REPORT_DIR = Path("/Users/sweeden/.gemini/antigravity-cli/brain/6bc95f94-233b-43b1-a212-5afa428b240a")

def get_status():
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    if res.returncode != 0:
        return "ERROR_QUERYING"
    return res.stdout.strip()

def file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    except Exception:
        return "error"

def download_and_stage():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading candidate artifacts into STAGING quarantine: {STAGING_DIR}...")
    res = subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(STAGING_DIR)], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)

    # Catalog candidate files
    candidate_files = {}
    for p in sorted(STAGING_DIR.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            rel = str(p.relative_to(STAGING_DIR))
            candidate_files[rel] = {
                "size_bytes": p.stat().st_size,
                "sha256_short": file_hash(p),
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            }

    # Catalog baseline files
    baseline_files = {}
    if BASELINE_DIR.exists():
        for p in sorted(BASELINE_DIR.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                rel = str(p.relative_to(BASELINE_DIR))
                baseline_files[rel] = {
                    "size_bytes": p.stat().st_size,
                    "sha256_short": file_hash(p),
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
                }

    # Compare differences
    added = [k for k in candidate_files if k not in baseline_files]
    removed = [k for k in baseline_files if k not in candidate_files]
    modified = [
        k for k in candidate_files
        if k in baseline_files and candidate_files[k]["sha256_short"] != baseline_files[k]["sha256_short"]
    ]
    unchanged = [
        k for k in candidate_files
        if k in baseline_files and candidate_files[k]["sha256_short"] == baseline_files[k]["sha256_short"]
    ]

    manifest = {
        "kernel_slug": KERNEL_SLUG,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "STAGED_FOR_REVIEW",
        "candidate_total_files": len(candidate_files),
        "baseline_total_files": len(baseline_files),
        "added": added,
        "modified": modified,
        "removed": removed,
        "unchanged_count": len(unchanged),
        "candidate_files": candidate_files
    }

    manifest_path = STAGING_DIR / "review_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate Review Gate Markdown Report
    report_lines = [
        "# Artifact Review Gate: Candidate vs Baseline",
        f"**Kernel:** `{KERNEL_SLUG}`  ",
        f"**Review Timestamp:** `{manifest['timestamp']}`  ",
        f"**Status:** `Awaiting Human Review Before Promotion`  ",
        "",
        "## Summary of Changes",
        f"- **Candidate Files (Staged):** {len(candidate_files)}",
        f"- **Baseline Files (Current):** {len(baseline_files)}",
        f"- **New Files Added:** {len(added)}",
        f"- **Modified Files:** {len(modified)}",
        f"- **Unchanged Files:** {len(unchanged)}",
        "",
        "### Modified Files (Would Bounce Existing Baseline)",
        "| File | Baseline SHA | Candidate SHA | Baseline Size | Candidate Size |",
        "| :--- | :---: | :---: | :---: | :---: |"
    ]

    for f in modified:
        b_info = baseline_files.get(f, {})
        c_info = candidate_files.get(f, {})
        report_lines.append(f"| `{f}` | `{b_info.get('sha256_short')}` | `{c_info.get('sha256_short')}` | {b_info.get('size_bytes')} B | {c_info.get('size_bytes')} B |")

    if not modified:
        report_lines.append("| *(None)* | - | - | - | - |")

    report_lines.extend([
        "",
        "### Added Files",
        "".join([f"- `{f}` ({candidate_files[f]['size_bytes']} bytes)\n" for f in added]) if added else "- *(None)*\n",
        "## Approval / Promotion Action",
        "To promote these staged artifacts and safely replace baseline, run:",
        "```bash",
        "/Users/sweeden/kagg/scripts/promote_artifacts.sh",
        "```"
    ])

    report_content = "\n".join(report_lines)
    with open(STAGING_DIR / "REVIEW_REPORT.md", "w") as f:
        f.write(report_content)

    if ARTIFACT_REPORT_DIR.exists():
        with open(ARTIFACT_REPORT_DIR / "artifact_review_gate.md", "w") as f:
            f.write(report_content)

    # Create promotion script
    promote_script = Path("/Users/sweeden/kagg/scripts/promote_artifacts.sh")
    with open(promote_script, "w") as f:
        f.write("""#!/bin/bash
set -euo pipefail
STAGING="/Users/sweeden/.local/share/kaggle/kernels/kaggriculture-self-training/incoming_review"
BASELINE="/Users/sweeden/.local/share/kaggle/kernels/kaggriculture-self-training/output"
BACKUP="/Users/sweeden/.local/share/kaggle/kernels/kaggriculture-self-training/backup_$(date +%Y%m%d_%H%M%S)"

if [ ! -d "$STAGING" ]; then
    echo "[-] Error: Staging directory does not exist: $STAGING"
    exit 1
fi

echo "[*] Backing up current baseline to $BACKUP..."
if [ -d "$BASELINE" ]; then
    cp -r "$BASELINE" "$BACKUP"
fi

echo "[*] Promoting staged artifacts to baseline..."
mkdir -p "$BASELINE"
rsync -av --exclude="review_manifest.json" --exclude="REVIEW_REPORT.md" "$STAGING/" "$BASELINE/"

echo "[+] Promotion complete! Baseline successfully updated."
""")
    promote_script.chmod(0o755)
    print(f"[+] Review gate initialized. Report: {STAGING_DIR / 'REVIEW_REPORT.md'}")
    return manifest

def main():
    status_str = get_status()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Kernel: {KERNEL_SLUG}")
    print(f"[{timestamp}] Status: {status_str}")

    if "COMPLETE" in status_str:
        print("[+] Kernel execution COMPLETE! Staging candidate outputs for review...")
        manifest = download_and_stage()
        print(f"[+] Download complete: {manifest['candidate_total_files']} candidate artifacts staged.")
        sys.exit(10)
    elif "ERROR" in status_str:
        print("[-] Kernel reported ERROR. Staging failure logs and partial artifacts for review...")
        manifest = download_and_stage()
        print(f"[-] Staged {manifest['candidate_total_files']} error trace artifacts for review.")
        sys.exit(20)
    elif "CANCELLED" in status_str:
        print("[!] Kernel was CANCELLED.")
        sys.exit(30)
    else:
        print(f"[*] Kernel is active ({status_str}). Safe polling continues.")
        sys.exit(0)

if __name__ == "__main__":
    main()
