from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "policy-llm-lab"))


@pytest.fixture()
def sample_docs(tmp_path: Path) -> Path:
    data_dir = ROOT / "policy-llm-lab" / "data" / "sample_docs"
    dest = tmp_path / "sample_docs"
    shutil.copytree(data_dir, dest)
    return dest
