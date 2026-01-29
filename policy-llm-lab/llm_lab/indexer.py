from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi

from llm_lab.config import DATA_DIR, DOCS_PATH, INDEX_PATH, INDEX_DIR

TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        docs.append(Document(doc_id=path.stem, source=str(path), text=text))
    return docs


@dataclass
class BM25Index:
    bm25: BM25Okapi
    docs: list[Document]
    tokens: list[list[str]]

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        if len(scores) == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[dict] = []
        for idx in top_indices:
            doc = self.docs[int(idx)]
            results.append(
                {
                    "doc_id": doc.doc_id,
                    "source": doc.source,
                    "text": doc.text,
                    "score": float(scores[int(idx)]),
                }
            )
        return results


def build_index(docs: Iterable[Document]) -> BM25Index:
    doc_list = list(docs)
    tokenized_corpus = [tokenize(doc.text) for doc in doc_list]
    bm25 = BM25Okapi(tokenized_corpus)
    return BM25Index(bm25=bm25, docs=doc_list, tokens=tokenized_corpus)


def persist_index(index: BM25Index) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("wb") as f:
        pickle.dump(index, f)
    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump([doc.__dict__ for doc in index.docs], f, indent=2)


def load_index() -> BM25Index:
    with INDEX_PATH.open("rb") as f:
        index: BM25Index = pickle.load(f)
    return index


def main() -> None:
    docs = load_documents()
    if not docs:
        raise SystemExit("No documents found in data/sample_docs")
    index = build_index(docs)
    persist_index(index)
    print(f"Indexed {len(docs)} documents")


if __name__ == "__main__":
    main()
