#!/bin/bash
# 生成 Python stub 文件

cd "$(dirname "$0")/.." || exit 1

PYO3_PYTHON="$(pwd)/.venv/bin/python" \
PYTHONHOME="$HOME/.local/share/uv/python/cpython-3.13.7-macos-aarch64-none" \
cargo run --bin stub_gen
