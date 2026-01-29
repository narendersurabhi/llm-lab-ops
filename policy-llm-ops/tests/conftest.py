from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "policy-llm-ops"))
sys.path.append(str(ROOT / "policy-llm-lab"))
