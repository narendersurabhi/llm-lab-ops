from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from llm_lab.config import DATA_DIR
from llm_lab.indexer import IndexBuilder, RetrievalClient


def _build_index(tmp_path: Path) -> tuple[Path, Path]:
    index_path = tmp_path / "index.sqlite"
    meta_path = tmp_path / "index_meta.json"
    builder = IndexBuilder(
        data_dir=DATA_DIR,
        index_path=index_path,
        meta_path=meta_path,
        chunk_size=80,
        chunk_overlap=20,
    )
    builder.build()
    return index_path, meta_path


def _list_chunk_ids(index_path: Path) -> list[str]:
    with sqlite3.connect(index_path) as conn:
        rows = conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()
    return [row[0] for row in rows]


def test_deterministic_doc_and_chunk_ids(tmp_path: Path) -> None:
    builder = IndexBuilder(data_dir=DATA_DIR, index_path=tmp_path / "a.sqlite")
    docs = builder.ingest_documents()
    expected_doc_ids = sorted(path.stem for path in DATA_DIR.glob("*") if path.is_file())
    assert [doc.doc_id for doc in docs] == expected_doc_ids

    index_a, meta_a = _build_index(tmp_path / "run_a")
    index_b, _meta_b = _build_index(tmp_path / "run_b")

    chunk_ids_a = _list_chunk_ids(index_a)
    chunk_ids_b = _list_chunk_ids(index_b)

    assert chunk_ids_a == chunk_ids_b
    assert chunk_ids_a
    assert chunk_ids_a[0].endswith("-0000")
    assert len(chunk_ids_a) == len(set(chunk_ids_a))

    meta = json.loads(meta_a.read_text(encoding="utf-8"))
    assert meta["doc_count"] == len(expected_doc_ids)
    assert meta["chunk_count"] == len(chunk_ids_a)
    assert meta["avg_chunk_size"] > 0
    assert meta["build_time_ms"] >= 0


def test_retrieval_returns_expected_chunk(tmp_path: Path) -> None:
    index_path, _meta_path = _build_index(tmp_path)
    client = RetrievalClient(index_path)
    results = client.retrieve("What is RAG?", top_k=1)
    client.close()
    assert results
    assert results[0]["doc_id"] == "rag"
    assert results[0]["chunk_id"].startswith("rag-")


def test_bm25_ordering_stable(tmp_path: Path) -> None:
    index_a, _meta_a = _build_index(tmp_path / "a")
    index_b, _meta_b = _build_index(tmp_path / "b")

    client_a = RetrievalClient(index_a)
    client_b = RetrievalClient(index_b)
    results_a = client_a.retrieve("llama cpp", top_k=3)
    results_b = client_b.retrieve("llama cpp", top_k=3)
    client_a.close()
    client_b.close()

    assert [row["chunk_id"] for row in results_a] == [row["chunk_id"] for row in results_b]
    scores = [row["score"] for row in results_a]
    assert scores == sorted(scores)
