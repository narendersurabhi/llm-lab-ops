from __future__ import annotations

import shutil
from pathlib import Path

from llm_lab.config import DATA_DIR
from llm_lab.eval import EvalThresholds, run_eval


def test_citation_coverage_gate_fails_when_docs_removed(tmp_path: Path) -> None:
    data_dir = tmp_path / "sample_docs"
    data_dir.mkdir()
    for path in DATA_DIR.glob("*"):
        if path.stem == "rag":
            continue
        if path.is_file():
            shutil.copy(path, data_dir / path.name)

    thresholds = EvalThresholds(retrieval_recall_min=1.0, citation_coverage_min=1.0)
    report = run_eval(data_dir=data_dir, thresholds=thresholds)

    assert report["metrics"]["citation_coverage"] < thresholds.citation_coverage_min
    assert report["pass"] is False
