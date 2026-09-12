#!/usr/bin/env python3
"""Fit a multi-head attention matrix from the 120-day GBDT notebook pipeline and scale ≤100 MB.

Usage (conda env ``kagg``)::

    /Users/sweeden/miniforge3/envs/kagg/bin/python scripts/fit_multihead_attention_100mb.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
sys.path.insert(0, str(RVQ))
sys.path.insert(0, str(ROOT))

from fitted_multihead_attention import fit_multihead_attention_from_120day_pipeline


def main() -> int:
    result = fit_multihead_attention_from_120day_pipeline(
        model_save_path=str(ROOT / "models" / "qkd_multihead_attention_120day_100mb.pkl"),
        heatmap_path=str(ROOT / "multihead_attention_heatmap.png"),
        target_mb=99.0,
        episodes_per_opp=1,
        max_steps_per_match=720,
        reuse_gbdt_features=True,
        verbose=True,
    )
    out_json = ROOT / "experiments" / "qkd_multihead_attention_120day_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("final_model_size_mb", "within_100mb", "status", "model_path", "heatmap_path")}, indent=2))
    print(f"Wrote {out_json}")
    return 0 if result.get("within_100mb") else 1


if __name__ == "__main__":
    raise SystemExit(main())
