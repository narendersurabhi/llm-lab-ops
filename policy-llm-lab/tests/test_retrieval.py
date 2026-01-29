from __future__ import annotations

from llm_lab.indexer import build_index, load_documents


def test_retrieval_top_doc_matches_expected() -> None:
    docs = load_documents()
    index = build_index(docs)
    results = index.retrieve("What is RAG?", top_k=1)
    assert results
    assert results[0]["doc_id"] == "rag"
