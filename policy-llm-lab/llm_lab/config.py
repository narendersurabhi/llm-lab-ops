from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample_docs"
ARTIFACTS_DIR = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"
MODEL_DIR = ARTIFACTS_DIR / "model"
CONTRACTS_DIR = ROOT / "contracts"
RELEASE_DIR = ROOT / "release"

INDEX_PATH = ARTIFACTS_DIR / "index.sqlite"
INDEX_META_PATH = ARTIFACTS_DIR / "index_meta.json"
