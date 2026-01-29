from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sample_docs"
ARTIFACTS_DIR = ROOT / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "index"
MODEL_DIR = ARTIFACTS_DIR / "model"
CONTRACTS_DIR = ROOT / "contracts"
RELEASE_DIR = ROOT / "release"

INDEX_PATH = INDEX_DIR / "bm25.pkl"
DOCS_PATH = INDEX_DIR / "docs.json"
