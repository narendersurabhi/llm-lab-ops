from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm_lab.config import MODEL_DIR
from llm_lab.indexer import build_index, load_documents


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


def run_eval() -> dict:
    docs = load_documents()
    index = build_index(docs)
    correct = 0
    for sample in EVAL_SAMPLES:
        results = index.retrieve(sample.query, top_k=1)
        if results and results[0]["doc_id"] == sample.expected_doc_id:
            correct += 1
    recall = correct / len(EVAL_SAMPLES)
    report = {
        "pass": recall >= 0.66,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": EVAL_DATASET,
        "metrics": {"retrieval_recall": recall, "num_queries": len(EVAL_SAMPLES)},
        "baseline": {
            "p95_latency_ms": 1200.0,
            "error_rate": 0.01,
            "tool_success_rate": 0.98,
        },
        "notes": "Baseline metrics are conservative placeholders for MVP.",
    }
    return report


def write_eval_report(path: Path) -> dict:
    report = run_eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report_path = MODEL_DIR / "eval_report.json"
    write_eval_report(report_path)
    print(f"Wrote eval report to {report_path}")
