"""Execute notebook cells and populate cell outputs in place."""

import io
import json
import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

def execute_notebook(nb_path: str):
    path = Path(nb_path)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    print(f"🚀 Executing notebook {nb_path} ({len(nb['cells'])} cells)...")
    exec_globals = {
        "__name__": "__main__",
        "__file__": str(path.resolve()),
        "display": lambda x: print(x),
    }

    execution_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        src = "".join(cell["source"])
        # Clean magic commands like !pip install
        cleaned_lines = []
        for line in src.split("\n"):
            if line.strip().startswith("!") or line.strip().startswith("%"):
                cleaned_lines.append(f"# {line}")
            else:
                cleaned_lines.append(line)
        clean_code = "\n".join(cleaned_lines)

        print(f"\n--- Executing Code Cell {execution_count} ---")
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(clean_code, exec_globals)
            out_text = stdout_buf.getvalue()
            err_text = stderr_buf.getvalue()
            full_out = out_text + (f"\n[STDERR]\n{err_text}" if err_text else "")

            cell["execution_count"] = execution_count
            cell["outputs"] = [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [line + "\n" for line in full_out.strip().split("\n")] if full_out.strip() else [],
                }
            ]
            print(f"✅ Cell {execution_count} succeeded. Output preview: {full_out.strip()[:120]}...")
            execution_count += 1
        except Exception as e:
            print(f"❌ Error in cell {execution_count}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"\n🎉 Successfully executed and populated outputs in {nb_path}")

if __name__ == "__main__":
    execute_notebook("kaggriculture-self-training/qkd_90day_11opponents_self_training.ipynb")
    execute_notebook("qkd_90day_11opponents_self_training.ipynb")
