from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient


def test_chat_completion_e2e(tmp_path, monkeypatch) -> None:
    release_dir = tmp_path / "release"
    _build_release_bundle(release_dir)

    monkeypatch.setenv("RELEASE_PATH", str(release_dir))
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    import llm_ops.config as config

    importlib.reload(config)
    import llm_ops.gateway as gateway

    importlib.reload(gateway)
    ops_app = gateway.app

    with TestClient(ops_app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-llama-gguf",
                "messages": [{"role": "user", "content": "What is RAG?"}],
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        assert content
        assert "Citations:" in content
        assert "doc-0000" in content


def _build_release_bundle(release_dir: Path) -> None:
    model_dir = release_dir / "model"
    index_dir = release_dir / "index"
    model_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    _build_test_index(index_dir / "index.sqlite")

    model_card = {
        "model_name": "local-llama-gguf",
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Test model card",
        "license": "unknown",
        "quantization": "Q4_K_M",
        "parameters": 7000000000,
    }
    eval_report = {
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
            "retrieval_hit_rate_min": 0.98,
            "recall_at_5_min": 0.85,
            "citation_coverage_min": 0.75,
            "runtime_error_rate_max": 0.02,
            "runtime_p95_regression_max": 0.2,
            "runtime_tool_success_min": 0.95,
        },
        "baseline": {
            "p95_latency_ms": 1200.0,
            "error_rate": 0.01,
            "tool_success_rate": 0.98,
        },
        "notes": "test",
    }
    manifest = {
        "bundle_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": "test",
        "artifacts": {
            "model": {
                "model_card": "model/model_card.json",
                "eval_report": "model/eval_report.json",
                "model_gguf": "model/model.gguf",
            },
            "index": {"index_sqlite": "index/index.sqlite"},
        },
        "fingerprints": {
            "model/model_card.json": "sha256:pending",
            "model/eval_report.json": "sha256:pending",
            "model/model.gguf": "sha256:pending",
            "index/index.sqlite": "sha256:pending",
        },
    }

    (model_dir / "model_card.json").write_text(json.dumps(model_card), encoding="utf-8")
    (model_dir / "eval_report.json").write_text(json.dumps(eval_report), encoding="utf-8")
    (model_dir / "model.gguf").write_text("placeholder", encoding="utf-8")
    (release_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _build_test_index(path: Path) -> None:
    import sqlite3

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
