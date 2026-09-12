"""Execution runner for the 120-Day GBDT 100MB Scaling Self-Training Notebook."""

import io
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(".").resolve()
RVQ_DIR = ROOT_DIR / "reasoning_vs_questioning"
if str(RVQ_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_notebook(nb_path: str = "qkd_120day_gbdt_scaling_training.ipynb"):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Globals context for execution
    exec_globals = {
        "__name__": "__main__",
        "display": lambda x: print(x),
    }

    print(f"[*] Executing {nb_path} ({len(nb['cells'])} cells)...")
    t0_all = time.perf_counter()

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source_code = "".join(cell["source"])
        print(f"\\n{'='*70}")
        print(f"▶ Running Code Cell {idx + 1}...")
        print(f"{'='*70}")

        t0_cell = time.perf_counter()
        stdout_trap = io.StringIO()
        orig_stdout = sys.stdout
        sys.stdout = stdout_trap

        try:
            exec(source_code, exec_globals)
            out_str = stdout_trap.getvalue()
        except Exception as e:
            out_str = stdout_trap.getvalue()
            sys.stdout = orig_stdout
            print(out_str)
            print(f"❌ Error in Cell {idx + 1}: {e}")
            raise
        finally:
            sys.stdout = orig_stdout

        cell_time = time.perf_counter() - t0_cell
        print(out_str.strip())
        print(f"⏱️ Cell {idx + 1} completed in {cell_time:.2f}s")

        # Record output in notebook cell
        cell["execution_count"] = idx + 1
        cell["outputs"] = [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in out_str.split("\n")],
            }
        ]

    total_time = time.perf_counter() - t0_all
    print(f"\\n{'='*70}")
    print(f"🎉 All {len(nb['cells'])} cells in {nb_path} executed successfully in {total_time:.2f}s!")
    print(f"{'='*70}")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    # Also sync to self-training folder
    with open("kaggriculture-self-training/qkd_120day_gbdt_scaling_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)


if __name__ == "__main__":
    run_notebook()
