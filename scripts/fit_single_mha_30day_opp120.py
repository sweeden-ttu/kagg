#!/usr/bin/env python3
"""Fit single MHA with 30-day horizon + 120×120 opponent confusion matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from single_mha_30day_opp120 import fit_single_mha_30day_opp120


def main() -> int:
    result = fit_single_mha_30day_opp120(
        model_save_path=str(ROOT / "models" / "qkd_single_mha_30day_opp120.pkl"),
        feature_cache=str(ROOT / "experiments" / "qkd_120day_attention_features.npz"),
        confusion_png=str(ROOT / "opp120_confusion_heatmap.png"),
        max_samples=4000,
        verbose=True,
    )
    out = ROOT / "experiments" / "qkd_single_mha_30day_opp120_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    keys = (
        "final_model_size_mb",
        "horizon_days",
        "confusion_shape",
        "payload_kind",
        "status",
        "model_path",
        "confusion_png",
        "confusion_npz",
    )
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))
    print(f"Wrote {out}")
    return 0 if result.get("within_100mb") else 1


if __name__ == "__main__":
    raise SystemExit(main())
