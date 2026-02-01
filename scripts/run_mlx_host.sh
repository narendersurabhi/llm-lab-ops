#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${ROOT_DIR}/.mlx_ops.pid"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/mlx_ops.log"
OPS_PORT="${OPS_PORT:-8002}"
RELEASE_PATH="${RELEASE_PATH:-${ROOT_DIR}/policy-llm-lab/dist/local-dev}"
ADAPTER_PATH="${MLX_ADAPTER_PATH:-${RELEASE_PATH}/model/adapter}"
MODEL_NAME="${MLX_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "MLX ops already running with PID $(cat "$PID_FILE")"
  exit 0
fi

mkdir -p "$LOG_DIR"

LLM_PROVIDER=mlx \
RELEASE_PATH="${RELEASE_PATH}" \
MLX_MODEL="${MODEL_NAME}" \
MLX_ADAPTER_PATH="${ADAPTER_PATH}" \
nohup "${ROOT_DIR}/.venv/bin/python" -m uvicorn llm_ops.gateway:app \
  --host 0.0.0.0 --port "$OPS_PORT" >"$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "MLX ops started (PID $(cat "$PID_FILE")), logs: $LOG_FILE"
