#!/bin/sh
set -eu

CONFIG="train/configs/mlx_lora.yaml"
PYTHON_BIN="${PYTHON:-python3}"

if "$PYTHON_BIN" -m mlx_lm.lora --help >/dev/null 2>&1; then
  "$PYTHON_BIN" -m mlx_lm.lora --config "$CONFIG"
  exit 0
fi

echo "mlx-lm is not installed. Install with:"
echo "  pip install mlx-lm"
echo "Then rerun: make train-mlx"
exit 1
