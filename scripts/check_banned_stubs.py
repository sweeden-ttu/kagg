#!/usr/bin/env python
"""Warn on banned stub tokens unless the enclosing scope raises NotImplementedError.

Banned tokens (case-insensitive) are the usual unfinished-stand-in labels.
Scope: datasets/scottweeden/self-training-code/**/*.py and scripts/**/*.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = (
    ROOT / "datasets/scottweeden/self-training-code",
    ROOT / "scripts",
)
SELF = Path(__file__).resolve()

# Built from parts so this file does not self-match the ban list.
_BANNED_PARTS = ("dum" + "my", "mo" + "ck", "place" + "holder", "to" + "do")
BANNED_RE = re.compile(
    r"(?i)(?<![A-Za-z])(" + "|".join(re.escape(p) for p in _BANNED_PARTS) + r")(?![A-Za-z])"
)
NIE_RE = re.compile(r"\bNotImplementedError\b")


class Finding(NamedTuple):
    path: Path
    line: int
    col: int
    token: str
    text: str


def _iter_py_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if path.resolve() == SELF:
                continue
            yield path


def _line_starts(source: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line_col(starts: Sequence[int], offset: int) -> Tuple[int, int]:
    lo, hi = 0, len(starts) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if starts[mid] <= offset:
            lo = mid + 1
        else:
            hi = mid - 1
    line0 = hi
    return line0 + 1, offset - starts[line0] + 1


def _raising_nie_ranges(tree: ast.AST) -> List[Tuple[int, int]]:
    """Line ranges (1-based inclusive) of functions/classes that raise NotImplementedError."""
    ranges: List[Tuple[int, int]] = []

    class Visitor(ast.NodeVisitor):
        def _span(self, node: ast.AST) -> Optional[Tuple[int, int]]:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is None or end is None:
                return None
            return int(start), int(end)

        def _contains_nie(self, node: ast.AST) -> bool:
            for child in ast.walk(node):
                if isinstance(child, ast.Raise):
                    exc = child.exc
                    if exc is None:
                        continue
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                        if exc.func.id == "NotImplementedError":
                            return True
                    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                        return True
            return False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            span = self._span(node)
            if span and self._contains_nie(node):
                ranges.append(span)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            span = self._span(node)
            if span and self._contains_nie(node):
                ranges.append(span)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            span = self._span(node)
            if span and self._contains_nie(node):
                ranges.append(span)
            self.generic_visit(node)

    Visitor().visit(tree)
    return ranges


def _allowed(line_no: int, nie_ranges: Sequence[Tuple[int, int]], line_text: str) -> bool:
    if NIE_RE.search(line_text):
        return True
    for start, end in nie_ranges:
        if start <= line_no <= end:
            return True
    return False


def scan_file(path: Path) -> List[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=str(path))
        nie_ranges = _raising_nie_ranges(tree)
    except SyntaxError:
        nie_ranges = []

    findings: List[Finding] = []
    starts = _line_starts(source)
    for match in BANNED_RE.finditer(source):
        line_no, col = _offset_to_line_col(starts, match.start())
        line_text = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
        if _allowed(line_no, nie_ranges, line_text):
            continue
        findings.append(
            Finding(
                path=path,
                line=line_no,
                col=col,
                token=match.group(1),
                text=line_text.strip(),
            )
        )
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional files/dirs to scan (default: Path B lib + scripts)",
    )
    args = parser.parse_args(argv)

    if args.paths:
        files: List[Path] = []
        for p in args.paths:
            path = p if p.is_absolute() else ROOT / p
            if path.is_dir():
                files.extend(sorted(path.rglob("*.py")))
            elif path.suffix == ".py":
                files.append(path)
    else:
        files = list(_iter_py_files(SCAN_ROOTS))

    all_findings: List[Finding] = []
    for path in files:
        all_findings.extend(scan_file(path))

    if not all_findings:
        print("PASS: no banned stub tokens without NotImplementedError")
        return 0

    print(
        f"WARNING: {len(all_findings)} banned stub token(s) "
        f"without NotImplementedError"
    )
    for finding in all_findings:
        rel = finding.path.relative_to(ROOT) if finding.path.is_relative_to(ROOT) else finding.path
        print(
            f"  {rel}:{finding.line}:{finding.col}: "
            f"banned '{finding.token}' — replace with real code or raise NotImplementedError"
        )
        print(f"    {finding.text}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
