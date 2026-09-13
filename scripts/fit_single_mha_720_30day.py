#!/usr/bin/env python3
"""Fit a single multi-headed attention matrix + 720×720 mask (30-day horizon).

Usage::

    /Users/sweeden/miniforge3/envs/kagg/bin/python scripts/fit_single_mha_720_30day.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from single_mha_720_30day import fit_single_mha_720_30day


def main() -> int:
    result = fit_single_mha_720_30day(
        model_save_path=str(ROOT / "models" / "qkd_single_mha_720_30day.pkl"),
        mask_path=str(ROOT / "artifacts" / "kmap_720x720" / "kmap_720x720_matrix.npz"),
        feature_cache=str(ROOT / "experiments" / "qkd_120day_attention_features.npz"),
        heatmap_path=str(ROOT / "single_mha_720_30day_heatmap.png"),
        max_samples=4000,
        verbose=True,
    )
    out_json = ROOT / "experiments" / "qkd_single_mha_720_30day_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    keys = (
        "final_model_size_mb",
        "within_100mb",
        "within_90mb_zip_budget",
        "status",
        "model_path",
        "heatmap_path",
        "horizon_days",
        "mask_shape",
        "payload_kind",
    )
    print(json.dumps({k: result.get(k) for k in keys}, indent=2))
    print(f"Wrote {out_json}")
    return 0 if result.get("within_100mb") else 1


if __name__ == "__main__":
    raise SystemExit(main())
