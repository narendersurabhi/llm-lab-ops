#!/bin/sh
set -eu

CONFIG="train/configs/cloud_lora.yaml"

if command -v llamafactory-cli >/dev/null 2>&1; then
  llamafactory-cli train "$CONFIG"
  exit 0
fi

echo "No trainer found. Install LlamaFactory (llamafactory-cli) or add your own cloud runner."
echo "Then rerun: make train-cloud"
exit 1
