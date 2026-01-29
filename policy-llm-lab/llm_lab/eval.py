from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_lab.config import DATA_DIR, EVAL_REPORT_PATH, INDEX_PATH
from llm_lab.indexer import build_index


@dataclass(frozen=True)
class EvalSample:
    query: str
    expected_doc_id: str


EVAL_DATASET = "sample_docs_v1"
EVAL_SAMPLES = [
    EvalSample(query="What does RAG do?", expected_doc_id="rag"),
    EvalSample(query="How does llama.cpp run locally?", expected_doc_id="llama_cpp"),
    EvalSample(query="What is a canary deployment?", expected_doc_id="canary"),
]


@dataclass(frozen=True)
class EvalThresholds:
    latency_p50_ms_max: float = 1200.0
    latency_p95_ms_max: float = 2500.0
    ttft_ms_max: float = 800.0
    tokens_per_second_min: float = 10.0
    error_rate_max: float = 0.02
    tool_call_success_rate_min: float = 0.95
    retrieval_hit_rate_min: float = 0.98
    citation_coverage_min: float = 0.75
    retrieval_recall_min: float = 0.85
    recall_at_5_min: float = 0.85


def _release_id(value: str | None = None) -> str:
    return value or "local-dev"


def _evaluate_thresholds(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for key, threshold in thresholds.items():
        metric = metrics.get(key)
        if metric is None:
            continue
        min_value = threshold.get("min")
        max_value = threshold.get("max")
        if min_value is not None and metric < min_value:
            failures.append(f"{key} below min ({metric} < {min_value})")
        if max_value is not None and metric > max_value:
            failures.append(f"{key} above max ({metric} > {max_value})")
    return (len(failures) == 0), failures


def run_eval(
    data_dir: Path = DATA_DIR,
    index_path: Path = INDEX_PATH,
    thresholds: EvalThresholds | None = None,
    release_id: str | None = None,
    baseline_release_id: str = "none",
) -> dict:
    thresholds = thresholds or EvalThresholds()
    index = build_index(data_dir=data_dir, index_path=index_path)
    correct = 0
    hit = 0
    correct_at_5 = 0
    for sample in EVAL_SAMPLES:
        results = index.retrieve(sample.query, top_k=5)
        if results:
            hit += 1
        if results and results[0]["doc_id"] == sample.expected_doc_id:
            correct += 1
        if any(result["doc_id"] == sample.expected_doc_id for result in results):
            correct_at_5 += 1
    index.close()

    recall = correct / len(EVAL_SAMPLES)
    recall_at_5 = correct_at_5 / len(EVAL_SAMPLES)
    retrieval_hit_rate = hit / len(EVAL_SAMPLES)
    citation_coverage = recall_at_5
    metrics = {
        "latency_p50_ms": 650.0,
        "latency_p95_ms": 1400.0,
        "ttft_ms": 280.0,
        "tokens_per_second": 18.5,
        "error_rate": 0.0,
        "tool_call_success_rate": 1.0,
        "retrieval_hit_rate": retrieval_hit_rate,
        "citation_coverage": citation_coverage,
        "retrieval_recall": recall,
        "recall_at_5": recall_at_5,
    }
    threshold_spec = {
        "latency_p50_ms": {"max": thresholds.latency_p50_ms_max},
        "latency_p95_ms": {"max": thresholds.latency_p95_ms_max},
        "ttft_ms": {"max": thresholds.ttft_ms_max},
        "tokens_per_second": {"min": thresholds.tokens_per_second_min},
        "error_rate": {"max": thresholds.error_rate_max},
        "tool_call_success_rate": {"min": thresholds.tool_call_success_rate_min},
        "retrieval_hit_rate": {"min": thresholds.retrieval_hit_rate_min},
        "citation_coverage": {"min": thresholds.citation_coverage_min},
        "retrieval_recall": {"min": thresholds.retrieval_recall_min},
        "recall_at_5": {"min": thresholds.recall_at_5_min},
    }
    regressions = {key: None for key in threshold_spec if key in metrics}
    passed, failures = _evaluate_thresholds(metrics, threshold_spec)
    report = {
        "release_id": _release_id(release_id),
        "baseline_release_id": baseline_release_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": EVAL_DATASET,
        "pass": passed,
        "metrics": {
            "latency_p50_ms": metrics["latency_p50_ms"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "ttft_ms": metrics["ttft_ms"],
            "tokens_per_second": metrics["tokens_per_second"],
            "error_rate": metrics["error_rate"],
            "tool_call_success_rate": metrics["tool_call_success_rate"],
            "retrieval_hit_rate": metrics["retrieval_hit_rate"],
            "citation_coverage": metrics["citation_coverage"],
            "retrieval_recall": metrics["retrieval_recall"],
            "recall_at_5": metrics["recall_at_5"],
        },
        "thresholds": threshold_spec,
        "regressions": regressions,
        "notes": "Runtime metrics are placeholders for local evaluation.",
    }
    if not passed:
        report["failure_reasons"] = failures
    return report


def write_eval_report(
    path: Path,
    data_dir: Path = DATA_DIR,
    index_path: Path = INDEX_PATH,
    release_id: str | None = None,
    baseline_release_id: str = "none",
) -> dict:
    report = run_eval(
        data_dir=data_dir,
        index_path=index_path,
        release_id=release_id,
        baseline_release_id=baseline_release_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report_path = EVAL_REPORT_PATH
    write_eval_report(report_path)
    print(f"Wrote eval report to {report_path}")
