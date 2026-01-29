from __future__ import annotations

from fastapi.testclient import TestClient

from llm_ops.agent import LangGraphAgent
from llm_ops.gateway import app as ops_app
from llm_ops.model import MockModelClient
from llm_ops.tools import RetrievalTool


def test_chat_completion_e2e(tmp_path) -> None:
    index_path = tmp_path / "index.sqlite"
    _build_test_index(index_path)
    retrieval = RetrievalTool(db_path=index_path)
    agent = LangGraphAgent(retrieval=retrieval, model=MockModelClient())

    # Inject test agent.
    ops_app.dependency_overrides = {}
    import llm_ops.gateway as gateway

    gateway.agent = agent
    gateway.canary = None

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


def _build_test_index(path) -> None:
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
