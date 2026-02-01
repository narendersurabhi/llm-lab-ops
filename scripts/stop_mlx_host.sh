#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="${ROOT_DIR}/.mlx_ops.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No MLX ops PID file found."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" >/dev/null 2>&1; then
  kill "$PID"
  echo "Stopped MLX ops (PID $PID)"
else
  echo "Process $PID not running."
fi

rm -f "$PID_FILE"
