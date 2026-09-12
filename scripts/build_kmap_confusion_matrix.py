#!/usr/bin/env python3
"""Build 11×11 and 13×13 K-map decision confusion matrices over opponents.

For each farm tile, aggregate active minterms across the 9 commodities into an
11-dim opponent vector (expected vs packed mask). Labels:

  11×11: argmax opponent (skip tiles with empty vectors)
  13×13: same, plus ``none`` (empty) and ``tie`` (non-unique argmax)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
sys.path.insert(0, str(RVQ))

from proportional_kmap import (  # noqa: E402
    OPPONENT_LABELS,
    build_proportional_kmap_mask,
)

SHORT_OPP = [
    "Finn",
    "Walter",
    "Rosa",
    "Hana",
    "Mateo",
    "Rita",
    "Bea",
    "Silas",
    "Lena",
    "Cleo",
    "Self",
]
ASSERT_N = 11
assert len(OPPONENT_LABELS) == ASSERT_N == len(SHORT_OPP)

LABELS_11 = list(SHORT_OPP)
LABELS_13 = list(SHORT_OPP) + ["none", "tie"]

OUT_PNG = ROOT / "kmap_decision_confusion_matrix.png"
OUT_JSON = ROOT / "experiments" / "kmap_decision_confusion_matrix.json"


def tile_zone(ty: int, tx: int) -> str:
    is_center = (3 <= ty <= 6) and (3 <= tx <= 6)
    is_corner = (ty in (0, 9)) or (tx in (0, 9))
    if is_center:
        return "center"
    if is_corner:
        return "corner"
    return "expansion"


def expected_active(ty: int, tx: int, opp_idx: int, crop_idx: int) -> bool:
    zone = tile_zone(ty, tx)
    if zone == "center" and crop_idx in (3, 4) and opp_idx >= 4:
        return True
    if zone == "corner" and crop_idx in (0, 1) and opp_idx <= 3:
        return True
    if zone == "expansion" and crop_idx in (5, 6, 7) and opp_idx in (5, 6, 7, 8):
        return True
    return False


def classify_vector(vec: np.ndarray, *, allow_none_tie: bool) -> int:
    """Return class index into LABELS_11 or LABELS_13."""
    total = float(vec.sum())
    if total <= 0:
        if allow_none_tie:
            return 11  # none
        raise ValueError("empty vector")
    mx = float(vec.max())
    winners = np.flatnonzero(vec == mx)
    if len(winners) > 1:
        if allow_none_tie:
            return 12  # tie
        return int(winners[0])  # stable pick for 11×11
    return int(winners[0])


def build_opponent_cms(spatial_rows: int = 100, env_cols: int = 99):
    """Cell-level confusion: true=column opponent, pred=tile mask argmax opponent.

    Counts every cell that is expected-active or mask-active so the matrix
    shows how multi-opponent zone rules collapse onto a single predicted foe.
    """
    kmap = build_proportional_kmap_mask(spatial_rows=spatial_rows, env_cols=env_cols)
    mask = kmap.unpack(np.float32)

    cm11 = np.zeros((11, 11), dtype=np.int64)
    cm13 = np.zeros((13, 13), dtype=np.int64)
    n_cells_11 = 0
    n_cells_13 = 0

    for t_idx in range(spatial_rows):
        ty, tx = divmod(t_idx, 10)
        pred_vec = np.zeros(11, dtype=np.float64)
        for col in range(env_cols):
            opp_idx, _crop = divmod(col, 9)
            if mask[t_idx, col] >= 0.5:
                pred_vec[opp_idx] += 1.0

        pred11 = (
            classify_vector(pred_vec, allow_none_tie=False)
            if pred_vec.sum() > 0
            else None
        )
        pred13 = classify_vector(pred_vec, allow_none_tie=True)

        for col in range(env_cols):
            opp_idx, crop_idx = divmod(col, 9)
            exp = expected_active(ty, tx, opp_idx, crop_idx)
            act = bool(mask[t_idx, col] >= 0.5)
            if not (exp or act):
                continue

            # 13×13: always include; true opponent index, or none if inactive expected
            true13 = opp_idx if exp else 11  # unexpected mask-only → true=none
            cm13[true13, pred13] += 1
            n_cells_13 += 1

            # 11×11: only when expected-active and tile has a mask prediction
            if exp and pred11 is not None:
                cm11[opp_idx, pred11] += 1
                n_cells_11 += 1

    return {
        "cm11": cm11,
        "cm13": cm13,
        "n_tiles_11": n_cells_11,
        "n_tiles_13": n_cells_13,
        "minterms": int(kmap.minterms_count),
        "total_cells": int(spatial_rows * env_cols),
    }


def _annotate(ax, cm: np.ndarray) -> None:
    thresh = cm.max() * 0.55 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            if val == 0:
                continue
            color = "white" if val > thresh else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=7)


def plot_cms(cm11: np.ndarray, cm13: np.ndarray, save_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=140)

    ax = axes[0]
    im = ax.imshow(cm11, cmap="Blues", aspect="equal")
    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_xticklabels(LABELS_11, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(LABELS_11, fontsize=8)
    ax.set_xlabel("Predicted opponent (mask argmax)")
    ax.set_ylabel("True opponent (rule argmax)")
    ax.set_title(f"11×11 opponent confusion matrix\n(n={int(cm11.sum())} expected-active cells)")
    _annotate(ax, cm11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    im2 = ax2.imshow(cm13, cmap="Oranges", aspect="equal")
    ax2.set_xticks(range(13))
    ax2.set_yticks(range(13))
    ax2.set_xticklabels(LABELS_13, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(LABELS_13, fontsize=7)
    ax2.set_xlabel("Predicted class")
    ax2.set_ylabel("True class")
    diag = int(np.trace(cm13))
    acc = diag / max(1, int(cm13.sum()))
    ax2.set_title(
        f"13×13 opponent + none/tie confusion matrix\n"
        f"(n={int(cm13.sum())} active cells, accuracy={acc:.3f})"
    )
    _annotate(ax2, cm13)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _print_matrix(name: str, labels: list[str], cm: np.ndarray) -> None:
    print(f"\n{name}")
    width = max(6, max(len(x) for x in labels) + 1)
    header = f"{'':>{width}}" + "".join(f"{lab:>{width}}" for lab in labels)
    print(header)
    for i, lab in enumerate(labels):
        row = f"{lab:>{width}}" + "".join(f"{int(cm[i, j]):>{width}}" for j in range(len(labels)))
        print(row)
    total = int(cm.sum())
    acc = float(np.trace(cm) / total) if total else 0.0
    print(f"trace={int(np.trace(cm))}  total={total}  accuracy={acc:.4f}")


def main() -> int:
    result = build_opponent_cms()
    cm11 = result["cm11"]
    cm13 = result["cm13"]
    png = plot_cms(cm11, cm13, OUT_PNG)

    payload = {
        "labels_11": LABELS_11,
        "labels_13": LABELS_13,
        "opponent_labels_full": list(OPPONENT_LABELS),
        "confusion_11x11": cm11.tolist(),
        "confusion_13x13": cm13.tolist(),
        "n_tiles_11": result["n_tiles_11"],
        "n_tiles_13": result["n_tiles_13"],
        "accuracy_11": float(np.trace(cm11) / cm11.sum()) if cm11.sum() else 0.0,
        "accuracy_13": float(np.trace(cm13) / cm13.sum()) if cm13.sum() else 0.0,
        "minterms": result["minterms"],
        "total_cells": result["total_cells"],
        "png": str(png),
        "method": (
            "Per cell: true=opponent channel of the cell; "
            "pred=argmax opponent over that tile's mask activations "
            "(summed across 9 commodities). 13×13 adds none (empty mask) "
            "and tie (non-unique argmax)."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    _print_matrix("11×11 opponent confusion", LABELS_11, cm11)
    _print_matrix("13×13 opponent+none+tie confusion", LABELS_13, cm13)
    print(f"\nWrote {png}")
    print(f"Wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
