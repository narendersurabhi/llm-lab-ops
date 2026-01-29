from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from llm_lab.eval import write_eval_report
from llm_lab.config import CONTRACTS_DIR


def test_eval_report_schema_validation(tmp_path: Path) -> None:
    report_path = tmp_path / "eval_report.json"
    index_path = tmp_path / "index.sqlite"
    report = write_eval_report(report_path, index_path=index_path)
    schema = json.loads((CONTRACTS_DIR / "eval_report.schema.json").read_text())
    validate(instance=report, schema=schema)
