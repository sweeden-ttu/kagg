#!/usr/bin/env python3
"""Build and Train 720×720 Floating-Point Convergence K-Map from 10×10 trained questions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVQ = ROOT / "reasoning_vs_questioning"
if str(RVQ) not in sys.path:
    sys.path.insert(0, str(RVQ))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kmap_720x720 import build_and_train_720x720_kmap  # noqa: E402

DEFAULT_10X10_TRAINED = (
    Path("/Users/sweeden/.cursor/worktrees/kmap-approach1-4ff64ce5/kagg-7c2ea97d9d34/artifacts/kmap_10x10/kmap_10x10_trained.json")
    if Path("/Users/sweeden/.cursor/worktrees/kmap-approach1-4ff64ce5/kagg-7c2ea97d9d34/artifacts/kmap_10x10/kmap_10x10_trained.json").exists()
    else ROOT / "artifacts/kmap_10x10/kmap_10x10_trained.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and Train 720x720 Floating-Point Convergence K-Map")
    parser.add_argument(
        "--trained-10x10",
        default=str(DEFAULT_10X10_TRAINED),
        help="Path to kmap_10x10_trained.json",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "artifacts/kmap_720x720"),
        help="Directory to save 720x720 artifacts",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=25.0,
        help="Logistic polarization annealing temperature",
    )
    args = parser.parse_args()

    trained_path = Path(args.trained_10x10)
    if not trained_path.exists():
        print(f"Error: {trained_path} does not exist", file=sys.stderr)
        return 1

    print(f"Ingesting 10x10 trained question bank from: {trained_path}")
    results = build_and_train_720x720_kmap(
        trained_10x10_path=trained_path,
        out_dir=args.out_dir,
        temperature=args.temperature,
    )

    metrics = results["metrics"]
    print("\n=== 720×720 Floating-Point Convergence K-Map Successfully Trained ===")
    print(f"Shape: {metrics['shape'][0]} × {metrics['shape'][1]} ({metrics['total_cells']:,} cells)")
    print(f"Polarization Ratio: {metrics['polarization_ratio']*100:.2f}%")
    print(f"Shannon Entropy: {metrics['shannon_entropy_bits']:.4f} bits")
    print(f"Epistemic Variance: {metrics['epistemic_variance']:.6f}")
    print(f"Active Minterms: {metrics['active_minterms_count']:,} / {metrics['total_cells']:,}")
    print(f"Purity ({'{'}0, 1{'}'}): {metrics['purity_pct']:.1f}% (Zeros: {metrics['zeros_pct']}%, Ones: {metrics['ones_pct']}%)")
    print(f"\nSaved Artifacts:")
    print(f"  JSON Metadata: {results['json_path']}")
    print(f"  Matrix Array:  {results['matrix_path']}")
    print(f"  Log Heatmap:   {results['log_heatmap_path']}")
    print(f"  Linear Map:    {results['polarization_heatmap_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
