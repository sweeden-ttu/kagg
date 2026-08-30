#!/usr/bin/env python3
"""Generate Kaggle packagemanager input_requirements.txt from repo imports.

Scans Python files under datasets/scottweeden/self-training-code (plus scripts/
and eval.py), maps third-party imports to PyPI names, fetches the latest
published version from PyPI JSON API, and writes one ``pip install pkg==ver``
line per package (no ``-U``, ``--upgrade``, or ``-r`` — required by Kaggle
packagemanager kernels).

Usage:
    python scripts/generate_pip_requirements.py
    python scripts/generate_pip_requirements.py --verify-only
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "datasets/scottweeden/self-training-code"
OUTPUT = ROOT / "kaggriculture-self-training" / "input_requirements.txt"
SHELL_OUTPUT = ROOT / "kaggriculture-self-training" / "pip_install.sh"

STDLIB = {
    "abc", "argparse", "ast", "base64", "collections", "copy", "csv", "datetime",
    "functools", "glob", "importlib", "json", "logging", "math", "os", "pathlib",
    "random", "re", "shutil", "subprocess", "sys", "tempfile", "typing", "urllib",
    "zlib", "__future__", "time", "tarfile", "zipfile", "enum", "itertools",
    "dataclasses", "textwrap", "warnings", "inspect", "io", "contextlib",
    "statistics", "heapq", "string", "struct", "threading", "traceback", "types",
    "uuid",
}

LOCAL_MODULES = {
    "kaggriculture_adapter", "kaggriculture_path_b_rebuild",
    "kaggriculture_self_play_training", "kaggriculture_dataset_publish",
    "kaggriculture_rl", "dataset_loader", "episode_catalog", "eval_policy",
    "kaggle_env_wrapper", "notebook_paths", "path_b_bootstrap", "visualize", "eval",
    "dqn",
}

IMPORT_TO_PYPI: dict[str, str] = {
    "numpy": "numpy",
    "torch": "torch",
    "kaggle_environments": "kaggle-environments",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "tensorboard": "tensorboard",
}

# Apple-only optional backend; not bundled for Kaggle GPU kernels.
KAGGLE_EXCLUDE = {"mlx"}


def scan_imports() -> dict[str, set[str]]:
    usage: dict[str, set[str]] = {}
    paths = sorted(LIB.rglob("*.py"))
    if (ROOT / "eval.py").exists():
        paths.append(ROOT / "eval.py")

    def add(mod: str, path: Path) -> None:
        top = mod.split(".")[0]
        if top in STDLIB or top in LOCAL_MODULES:
            return
        usage.setdefault(top, set()).add(str(path.relative_to(ROOT)))

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name, path)
            elif isinstance(node, ast.ImportFrom) and node.module:
                add(node.module, path)

    # torch.utils.tensorboard.SummaryWriter requires the tensorboard package.
    if any("dqn_sb3.py" in p for p in usage.get("torch", set())):
        usage.setdefault("tensorboard", set()).add(
            "datasets/scottweeden/self-training-code/kaggriculture_rl/dqn_sb3.py "
            "(torch.utils.tensorboard.SummaryWriter)"
        )

    return usage


def fetch_pypi_version(package: str) -> str:
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    return str(data["info"]["version"])


def build_requirements(usage: dict[str, set[str]]) -> list[tuple[str, str, list[str]]]:
    rows: list[tuple[str, str, list[str]]] = []
    unmapped: dict[str, list[str]] = {}

    for imp, files in sorted(usage.items()):
        if imp in KAGGLE_EXCLUDE:
            print(f"skip (not for Kaggle): {imp} -> {sorted(files)}", file=sys.stderr)
            continue
        pypi = IMPORT_TO_PYPI.get(imp)
        if not pypi:
            unmapped[imp] = sorted(files)
            continue
        version = fetch_pypi_version(pypi)
        rows.append((pypi, version, sorted(files)))

    if unmapped:
        msg = "Unmapped third-party imports (add to IMPORT_TO_PYPI or exclude):\n"
        for imp, files in unmapped.items():
            msg += f"  {imp}: {files}\n"
        raise SystemExit(msg)

    return rows


def write_outputs(rows: list[tuple[str, str, list[str]]]) -> None:
    lines = [f"pip install {pkg}=={ver}" for pkg, ver, _ in rows]
    OUTPUT.write_text("\n".join(lines) + "\n")

    shell = [
        "#!/usr/bin/env bash",
        "# Kaggle packagemanager-compatible installs (pip install only; no -U/--upgrade/-r).",
        "# Regenerate: python scripts/generate_pip_requirements.py",
        "set -euo pipefail",
        "",
    ]
    for pkg, ver, _ in rows:
        shell.append(f'pip install "{pkg}=={ver}"')
    SHELL_OUTPUT.write_text("\n".join(shell) + "\n")
    SHELL_OUTPUT.chmod(0o755)


def verify_lines(lines: list[str]) -> None:
    bad = [line for line in lines if re.search(r"(^|\s)(-U|--upgrade|-r|--requirement)\b", line)]
    if bad:
        raise SystemExit(f"Invalid packagemanager lines (upgrade/requirements not allowed):\n{bad}")
    for line in lines:
        if not line.startswith("pip install "):
            raise SystemExit(f"Each line must start with 'pip install ': {line!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate existing input_requirements.txt without rewriting.",
    )
    args = parser.parse_args()

    if args.verify_only:
        if not OUTPUT.is_file():
            raise SystemExit(f"Missing {OUTPUT}")
        lines = [ln.strip() for ln in OUTPUT.read_text().splitlines() if ln.strip()]
        verify_lines(lines)
        for line in lines:
            pkg = line.split()[2]
            fetch_pypi_version(pkg.split("==")[0])
            print(f"ok {line}")
        return

    usage = scan_imports()
    rows = build_requirements(usage)
    write_outputs(rows)

    lines = [f"pip install {pkg}=={ver}" for pkg, ver, _ in rows]
    verify_lines(lines)

    print(f"Wrote {OUTPUT} ({len(lines)} packages)")
    print(f"Wrote {SHELL_OUTPUT}")
    for pkg, ver, files in rows:
        print(f"\n{pkg}=={ver}  (PyPI verified)")
        for path in files:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
