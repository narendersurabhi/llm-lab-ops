from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from llm_lab.indexer import IndexBuilder, RetrievalClient


def _build_index(data_dir: Path, tmp_path: Path) -> tuple[Path, Path]:
    index_path = tmp_path / "index.sqlite"
    meta_path = tmp_path / "index_meta.json"
    builder = IndexBuilder(
        data_dir=data_dir,
        index_path=index_path,
        meta_path=meta_path,
        chunk_size=80,
        chunk_overlap=20,
    )
    builder.build()
    return index_path, meta_path


@pytest.mark.component
def test_index_tables_and_counts(sample_docs: Path, tmp_path: Path) -> None:
    index_path, _meta_path = _build_index(sample_docs, tmp_path)
    with sqlite3.connect(index_path) as conn:
        table_names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        }
        assert {"documents", "chunks", "chunks_fts"} <= table_names

        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]

    assert doc_count > 0
    assert chunk_count > 0
    assert fts_count == chunk_count


@pytest.mark.component
def test_retrieval_returns_expected_chunk(sample_docs: Path, tmp_path: Path) -> None:
    index_path, _meta_path = _build_index(sample_docs, tmp_path)
    client = RetrievalClient(index_path)
    results = client.retrieve("What is RAG?", top_k=1)
    client.close()
    assert results
    assert results[0]["doc_id"] == "rag"
    assert results[0]["chunk_id"].startswith("rag-")


@pytest.mark.component
def test_bm25_ordering_stable(sample_docs: Path, tmp_path: Path) -> None:
    index_a, _meta_a = _build_index(sample_docs, tmp_path / "a")
    index_b, _meta_b = _build_index(sample_docs, tmp_path / "b")

    client_a = RetrievalClient(index_a)
    client_b = RetrievalClient(index_b)
    results_a = client_a.retrieve("llama cpp", top_k=3)
    results_b = client_b.retrieve("llama cpp", top_k=3)
    client_a.close()
    client_b.close()

    assert [row["chunk_id"] for row in results_a] == [row["chunk_id"] for row in results_b]
    scores = [row["score"] for row in results_a]
    assert scores == sorted(scores)
