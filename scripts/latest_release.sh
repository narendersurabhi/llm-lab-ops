#!/bin/sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/policy-llm-lab/dist"

if [ ! -d "$DIST_DIR" ]; then
  echo "dist directory not found: $DIST_DIR" >&2
  exit 1
fi

LATEST="$(ls -1dt "$DIST_DIR"/* 2>/dev/null | head -n 1 || true)"
if [ -z "$LATEST" ] || [ ! -d "$LATEST" ]; then
  echo "no release bundles found in $DIST_DIR" >&2
  exit 1
fi

echo "$LATEST"
