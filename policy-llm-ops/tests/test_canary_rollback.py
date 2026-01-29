from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_ops.canary import CanaryController
from llm_ops.config import settings


def _write_eval_report(path: Path) -> None:
    report = {
        "release_id": "dev-abcdef1",
        "baseline_release_id": "none",
        "pass": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": "test",
        "metrics": {
            "latency_p50_ms": 30.0,
            "latency_p95_ms": 100.0,
            "ttft_ms": 25.0,
            "tokens_per_second": 20.0,
            "error_rate": 0.0,
            "tool_call_success_rate": 1.0,
            "retrieval_hit_rate": 1.0,
            "citation_coverage": 1.0,
            "retrieval_recall": 1.0,
            "recall_at_5": 1.0,
        },
        "thresholds": {
            "latency_p50_ms": {"max": 200.0},
            "latency_p95_ms": {"max": 150.0},
            "ttft_ms": {"max": 100.0},
            "tokens_per_second": {"min": 5.0},
            "error_rate": {"max": 0.02},
            "tool_call_success_rate": {"min": 0.95},
            "retrieval_hit_rate": {"min": 0.98},
            "citation_coverage": {"min": 0.75},
            "retrieval_recall": {"min": 0.85},
            "recall_at_5": {"min": 0.85},
        },
        "regressions": {
            "latency_p50_ms": None,
            "latency_p95_ms": None,
            "ttft_ms": None,
            "tokens_per_second": None,
            "error_rate": None,
            "tool_call_success_rate": None,
            "retrieval_hit_rate": None,
            "citation_coverage": None,
            "retrieval_recall": None,
            "recall_at_5": None,
        },
        "notes": "test",
    }
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.unit
def test_canary_rolls_back_on_latency_regression(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "canary_min_samples", 3)
    monkeypatch.setattr(settings, "canary_slo_window", 5)
    monkeypatch.setattr(settings, "canary_fraction", 0.05)
    monkeypatch.setattr(settings, "p95_regression_max", 0.2)

    eval_path = tmp_path / "eval_report.json"
    _write_eval_report(eval_path)

    controller = CanaryController(eval_path)
    assert controller.state.mode == "canary"

    for _ in range(3):
        controller.record(
            latency_ms=200.0,
            is_error=False,
            tool_success=True,
            citation_coverage=1.0,
        )

    assert controller.state.mode == "rolled_back"
    assert controller.state.fraction == 0.0
