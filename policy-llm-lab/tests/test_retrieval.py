from __future__ import annotations

from pathlib import Path

from llm_lab.config import DATA_DIR
from llm_lab.indexer import IndexBuilder, load_index


def test_retrieval_top_doc_matches_expected(tmp_path: Path) -> None:
    index_path = tmp_path / "index.sqlite"
    builder = IndexBuilder(data_dir=DATA_DIR, index_path=index_path)
    builder.build()
    index = load_index(index_path)
    results = index.retrieve("What is RAG?", top_k=2)
    assert results
    assert results[0]["doc_id"] == "rag"
    if len(results) > 1:
        assert results[0]["score"] <= results[1]["score"]
    index.close()
