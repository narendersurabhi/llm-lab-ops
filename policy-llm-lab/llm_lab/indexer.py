from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from llm_lab.config import DATA_DIR, INDEX_PATH

WORD_PATTERN = re.compile(r"\S+")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")
HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
BLOCKQUOTE_RE = re.compile(r"^>\s+", re.MULTILINE)
BULLET_RE = re.compile(r"^[*-]\s+", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    chunk_index: int


class SQLiteIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        cursor = self.conn.execute(
            """
            SELECT chunk_id, doc_id, source, content as text, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (query, top_k),
        )
        rows = cursor.fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "source": row["source"],
                "text": row["text"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

    def close(self) -> None:
        self.conn.close()


class IndexBuilder:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        index_path: Path = INDEX_PATH,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.data_dir = data_dir
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_documents(self) -> list[Document]:
        docs: list[Document] = []
        for path in sorted(self.data_dir.glob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".markdown", ".pdf"}:
                continue
            text = self._read_file(path)
            if not text.strip():
                continue
            docs.append(Document(doc_id=path.stem, source=str(path), text=text))
        return docs

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() in {".txt"}:
            return path.read_text(encoding="utf-8").strip()
        if path.suffix.lower() in {".md", ".markdown"}:
            raw = path.read_text(encoding="utf-8")
            return self._strip_markdown(raw)
        if path.suffix.lower() == ".pdf":
            # PDF extraction can be added later.
            return ""
        return ""

    def _strip_markdown(self, text: str) -> str:
        text = CODE_FENCE_RE.sub("", text)
        text = IMAGE_RE.sub(r"\1", text)
        text = LINK_RE.sub(r"\1", text)
        text = INLINE_CODE_RE.sub(r"\1", text)
        text = HEADER_RE.sub("", text)
        text = BLOCKQUOTE_RE.sub("", text)
        text = BULLET_RE.sub("", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def chunk_document(self, doc: Document) -> list[Chunk]:
        words = WORD_PATTERN.findall(doc.text)
        if not words:
            return []
        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        index = 0
        for start in range(0, len(words), stride):
            chunk_words = words[start : start + self.chunk_size]
            if not chunk_words:
                continue
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{doc.doc_id}-{index:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source=doc.source,
                    text=chunk_text,
                    chunk_index=index,
                )
            )
            index += 1
        return chunks

    def build(self) -> Path:
        docs = self.ingest_documents()
        if not docs:
            raise RuntimeError("No documents found for indexing")
        if self.index_path.exists():
            self.index_path.unlink()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_path)
        conn.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                content,
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                source UNINDEXED
            )
            """
        )
        for doc in docs:
            for chunk in self.chunk_document(doc):
                conn.execute(
                    "INSERT INTO chunks_fts (content, chunk_id, doc_id, source) VALUES (?, ?, ?, ?)",
                    (chunk.text, chunk.chunk_id, chunk.doc_id, chunk.source),
                )
        conn.commit()
        conn.close()
        return self.index_path


def load_index(index_path: Path = INDEX_PATH) -> SQLiteIndex:
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    return SQLiteIndex(index_path)


def build_index(data_dir: Path = DATA_DIR, index_path: Path = INDEX_PATH) -> SQLiteIndex:
    builder = IndexBuilder(data_dir=data_dir, index_path=index_path)
    builder.build()
    return load_index(index_path)


def main() -> None:
    builder = IndexBuilder()
    path = builder.build()
    print(f"Indexed documents into {path}")


if __name__ == "__main__":
    main()
