# Vendored pyahocorasick==2.3.1 binary wheels
#
# Linux C-extensions are architecture-specific: one wheel cannot load on both
# x86_64 and aarch64. This directory therefore contains BOTH:
#   - manylinux2014_x86_64 / manylinux_2_17_x86_64
#   - manylinux2014_aarch64 / manylinux_2_17_aarch64
# for CPython 3.10, 3.11, and 3.12.
#
# macOS ships a single universal2 wheel (x86_64 + arm64/aarch64) per CPython.
#
# Install (conda env kagg; never a separate pip venv):
#   KAGG_PY="$(conda info --base)/envs/kagg/bin/python"
#   uv pip install --find-links wheels/pyahocorasick --no-index pyahocorasick==2.3.1 --python "$KAGG_PY"
# Or with network fallback:
#   uv pip install --find-links wheels/pyahocorasick -r requirements.txt --python "$KAGG_PY"
#
# pip automatically selects the wheel matching the host arch + Python ABI.

pyahocorasick-2.3.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pyahocorasick-2.3.1-cp310-cp310-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
pyahocorasick-2.3.1-cp310-cp310-macosx_10_9_universal2.whl
pyahocorasick-2.3.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pyahocorasick-2.3.1-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
pyahocorasick-2.3.1-cp311-cp311-macosx_10_9_universal2.whl
pyahocorasick-2.3.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
pyahocorasick-2.3.1-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl
pyahocorasick-2.3.1-cp312-cp312-macosx_10_13_universal2.whl
