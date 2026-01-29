from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_test_index(path: Path) -> None:
    import sqlite3

    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                chunk_id UNINDEXED,
                doc_id UNINDEXED
            )
            """
        )
        conn.execute(
            "INSERT INTO documents (doc_id, source, content) VALUES (?, ?, ?)",
            ("doc", "source.txt", "RAG combines retrieval and generation."),
        )
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, chunk_index, content) VALUES (?, ?, ?, ?)",
            ("doc-0000", "doc", 0, "RAG combines retrieval and generation."),
        )
        conn.execute(
            "INSERT INTO chunks_fts (content, chunk_id, doc_id) VALUES (?, ?, ?)",
            ("RAG combines retrieval and generation.", "doc-0000", "doc"),
        )
        conn.commit()


def build_minimal_release_bundle(release_dir: Path, contracts_dir: Path) -> dict:
    release_dir.mkdir(parents=True, exist_ok=True)
    model_dir = release_dir / "model"
    index_dir = release_dir / "index"
    eval_dir = release_dir / "eval"
    contracts_out = release_dir / "contracts"
    meta_dir = release_dir / "meta"
    model_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    contracts_out.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    build_test_index(index_dir / "index.sqlite")

    model_card = {
        "release_id": "dev-abcdef1",
        "bundle_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": "abcdef1234567890abcdef1234567890abcdef12",
        "model_name": "local-llama-gguf",
        "base_model": {"name": "llama", "version": "2-7b"},
        "tuning_method": "none",
        "training_config_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "dataset_fingerprint": {"algo": "sha256", "value": "a" * 64},
        "runtime_compatibility": ["llama.cpp"],
        "quantization": {"format": "Q4_K_M", "bits": 4, "group_size": 32},
        "parameters": 7000000000,
        "license": "unknown",
        "tags": ["test"],
        "intended_use": "Test",
        "limitations": "Test",
        "eval_report_path": "eval/eval_report.json",
    }

    eval_report = {
        "release_id": "dev-abcdef1",
        "baseline_release_id": "none",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "eval_dataset": "test",
        "pass": True,
        "metrics": {
            "latency_p50_ms": 30.0,
            "latency_p95_ms": 80.0,
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

    manifest = {
        "release_id": "dev-abcdef1",
        "bundle_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": "abcdef1234567890abcdef1234567890abcdef12",
        "artifacts": {
            "model": {
                "model_card": "model/model_card.json",
                "model_gguf": "model/model.gguf",
            },
            "index": {"index_sqlite": "index/index.sqlite"},
            "eval": {"eval_report": "eval/eval_report.json"},
            "contracts": {
                "manifest_schema": "contracts/manifest.schema.json",
                "model_card_schema": "contracts/model_card.schema.json",
                "eval_report_schema": "contracts/eval_report.schema.json",
            },
            "meta": {
                "sbom": {"sbom_json": "meta/sbom.json"},
                "checksums": {"checksums_json": "meta/checksums.json"},
                "attestation": {"attestation_json": "meta/attestation.json"},
                "changelog": {"changelog_md": "meta/CHANGELOG.md"},
            },
        },
        "fingerprints": {
            "dataset": {"algo": "sha256", "value": "a" * 64},
            "index": {"algo": "sha256", "value": "b" * 64},
            "model": {"algo": "sha256", "value": "c" * 64},
            "bundle": {"algo": "sha256", "value": "d" * 64},
            "eval": {"algo": "sha256", "value": "e" * 64},
        },
        "notes": "test",
    }

    (model_dir / "model_card.json").write_text(json.dumps(model_card), encoding="utf-8")
    (model_dir / "model.gguf").write_text("placeholder", encoding="utf-8")
    (eval_dir / "eval_report.json").write_text(json.dumps(eval_report), encoding="utf-8")
    (release_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    for schema_name in [
        "manifest.schema.json",
        "model_card.schema.json",
        "eval_report.schema.json",
    ]:
        (contracts_out / schema_name).write_text(
            (contracts_dir / schema_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    (meta_dir / "sbom.json").write_text("{}", encoding="utf-8")
    (meta_dir / "checksums.json").write_text("{}", encoding="utf-8")
    (meta_dir / "attestation.json").write_text("{}", encoding="utf-8")
    (meta_dir / "CHANGELOG.md").write_text("## 0.1.0\n", encoding="utf-8")

    return manifest
