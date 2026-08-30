#!/usr/bin/env bash
# Kaggle packagemanager-compatible installs (pip install only; no -U/--upgrade/-r).
# Regenerate: python scripts/generate_pip_requirements.py
set -euo pipefail

pip install "kaggle-environments==1.32.7"
pip install "matplotlib==3.11.1"
pip install "numpy==2.5.2"
pip install "tensorboard==2.21.0"
pip install "torch==2.13.0"
pip install "tqdm==4.70.0"
