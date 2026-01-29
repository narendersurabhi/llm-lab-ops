from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_ops.canary import CanaryController
from llm_ops.config import settings


def _write_eval_report(path: Path, citation_threshold: float = 0.9) -> None:
    report = {
        "pass": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": "test",
        "metrics": {
            "retrieval_recall": 1.0,
            "retrieval_hit_rate": 1.0,
            "recall_at_5": 1.0,
            "citation_coverage": 1.0,
            "num_queries": 1,
        },
        "thresholds": {
            "retrieval_hit_rate_min": 0.5,
            "recall_at_5_min": 0.5,
            "citation_coverage_min": citation_threshold,
            "runtime_error_rate_max": 0.02,
            "runtime_p95_regression_max": 0.2,
            "runtime_tool_success_min": 0.95,
        },
        "baseline": {
            "p95_latency_ms": 100.0,
            "error_rate": 0.01,
            "tool_success_rate": 0.98,
        },
        "notes": "test",
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def test_canary_rolls_back_on_citation_drop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "canary_min_samples", 3)
    monkeypatch.setattr(settings, "canary_slo_window", 5)
    monkeypatch.setattr(settings, "canary_fraction", 0.05)

    eval_path = tmp_path / "eval_report.json"
    _write_eval_report(eval_path, citation_threshold=0.9)
    controller = CanaryController(tmp_path)
    assert controller.state.mode == "canary"

    for _ in range(3):
        controller.record(
            latency_ms=50.0,
            is_error=False,
            tool_success=True,
            citation_coverage=0.2,
        )

    assert controller.state.mode == "rolled_back"
    assert controller.state.fraction == 0.0
