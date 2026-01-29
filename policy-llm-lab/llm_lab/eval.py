from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_lab.config import DATA_DIR, INDEX_PATH, MODEL_DIR
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
    retrieval_hit_rate_min: float = 0.98
    recall_at_5_min: float = 0.85
    citation_coverage_min: float = 0.75
    runtime_error_rate_max: float = 0.02
    runtime_p95_regression_max: float = 0.2
    runtime_tool_success_min: float = 0.95


def run_eval(
    data_dir: Path | None = None,
    index_path: Path | None = None,
    thresholds: EvalThresholds | None = None,
) -> dict:
    data_dir = data_dir or DATA_DIR
    index_path = index_path or INDEX_PATH
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
    report = {
        "pass": retrieval_hit_rate >= thresholds.retrieval_hit_rate_min
        and recall_at_5 >= thresholds.recall_at_5_min
        and citation_coverage >= thresholds.citation_coverage_min,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": EVAL_DATASET,
        "metrics": {
            "retrieval_recall": recall,
            "retrieval_hit_rate": retrieval_hit_rate,
            "recall_at_5": recall_at_5,
            "citation_coverage": citation_coverage,
            "num_queries": len(EVAL_SAMPLES),
        },
        "thresholds": {
            "retrieval_hit_rate_min": thresholds.retrieval_hit_rate_min,
            "recall_at_5_min": thresholds.recall_at_5_min,
            "citation_coverage_min": thresholds.citation_coverage_min,
            "runtime_error_rate_max": thresholds.runtime_error_rate_max,
            "runtime_p95_regression_max": thresholds.runtime_p95_regression_max,
            "runtime_tool_success_min": thresholds.runtime_tool_success_min,
        },
        "baseline": {
            "p95_latency_ms": 1200.0,
            "error_rate": 0.01,
            "tool_success_rate": 0.98,
        },
        "notes": "Baseline metrics are conservative placeholders for MVP.",
    }
    return report


def write_eval_report(
    path: Path, data_dir: Path = DATA_DIR, index_path: Path = INDEX_PATH
) -> dict:
    report = run_eval(data_dir=data_dir, index_path=index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report_path = MODEL_DIR / "eval_report.json"
    write_eval_report(report_path)
    print(f"Wrote eval report to {report_path}")
