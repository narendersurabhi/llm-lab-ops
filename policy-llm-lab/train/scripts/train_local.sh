#!/bin/sh
set -eu

CONFIG="train/configs/local_lora.yaml"

if command -v llamafactory-cli >/dev/null 2>&1; then
  llamafactory-cli train "$CONFIG"
  exit 0
fi

if command -v python >/dev/null 2>&1 && python -c "import unsloth" >/dev/null 2>&1; then
  echo "Unsloth detected. Provide a local training script if you prefer Unsloth."
  exit 0
fi

echo "No trainer found. Install one of:"
echo "  - LlamaFactory (llamafactory-cli)"
echo "  - Unsloth (python -m unsloth)"
echo "Then rerun: make train-local"
exit 1
