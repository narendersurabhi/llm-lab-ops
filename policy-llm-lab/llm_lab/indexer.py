from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from llm_lab.config import DATA_DIR, INDEX_META_PATH, INDEX_PATH, INGEST_DIR

WORD_PATTERN = re.compile(r"\S+")
QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")
HEADER_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
BLOCKQUOTE_RE = re.compile(r"^>\s+", re.MULTILINE)
BULLET_RE = re.compile(r"^[*-]\s+", re.MULTILINE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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


@dataclass(frozen=True)
class IndexMeta:
    doc_count: int
    chunk_count: int
    avg_chunk_size: float
    build_time_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "avg_chunk_size": round(self.avg_chunk_size, 2),
            "build_time_ms": round(self.build_time_ms, 2),
        }


class SQLiteIndex:
    def __init__(self, path: Path, read_only: bool = False) -> None:
        self.path = path
        uri = f"file:{path}?mode=ro" if read_only else str(path)
        self.conn = sqlite3.connect(uri, uri=read_only)
        self.conn.row_factory = sqlite3.Row

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_text = _sanitize_query(query)
        if not query_text:
            return []
        cursor = self.conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, d.source, c.content as text, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (query_text, top_k),
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


class RetrievalClient:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_text = _sanitize_query(query)
        if not query_text:
            return []
        cursor = self._conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, d.source, c.content as text, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC, c.chunk_id ASC
            LIMIT ?
            """,
            (query_text, top_k),
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
        self._conn.close()


class IndexBuilder:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        extra_dirs: list[Path] | None = None,
        index_path: Path = INDEX_PATH,
        meta_path: Path = INDEX_META_PATH,
        chunk_size: int = 200,
        chunk_overlap: int = 40,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.data_dir = data_dir
        self.extra_dirs = extra_dirs or []
        self.index_path = index_path
        self.meta_path = meta_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_documents(self) -> list[Document]:
        docs: list[Document] = []
        for directory in [self.data_dir, *self.extra_dirs]:
            for path in sorted(directory.glob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in {".txt", ".md", ".markdown"}:
                    continue
                text = self._read_file(path)
                if not text.strip():
                    continue
                doc_id = self._doc_id_for(path, directory)
                docs.append(Document(doc_id=doc_id, source=str(path), text=text))
        return docs

    def _doc_id_for(self, path: Path, base_dir: Path) -> str:
        if base_dir == self.data_dir:
            return path.stem
        rel = path.relative_to(base_dir)
        safe = re.sub(r"[^A-Za-z0-9]+", "-", rel.stem).strip("-").lower()
        return f"{base_dir.name}-{safe}"

    def _read_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._normalize_text(path.read_text(encoding="utf-8"))
        if suffix in {".md", ".markdown"}:
            raw = path.read_text(encoding="utf-8")
            return self._normalize_text(self._strip_markdown(raw))
        return ""

    def _strip_markdown(self, text: str) -> str:
        text = CODE_FENCE_RE.sub("", text)
        text = IMAGE_RE.sub(r"\1", text)
        text = LINK_RE.sub(r"\1", text)
        text = INLINE_CODE_RE.sub(r"\1", text)
        text = HEADER_RE.sub("", text)
        text = BLOCKQUOTE_RE.sub("", text)
        text = BULLET_RE.sub("", text)
        return text

    def _normalize_text(self, text: str) -> str:
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
        start = time.perf_counter()
        docs = self.ingest_documents()
        if not docs:
            raise RuntimeError("No documents found for indexing")
        if self.index_path.exists():
            self.index_path.unlink()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if self.meta_path.exists():
            self.meta_path.unlink()

        conn = sqlite3.connect(self.index_path)
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
                content TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
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

        chunk_count = 0
        total_chunk_words = 0
        for doc in docs:
            conn.execute(
                "INSERT INTO documents (doc_id, source, content) VALUES (?, ?, ?)",
                (doc.doc_id, doc.source, doc.text),
            )
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                conn.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, chunk_index, content) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.doc_id, chunk.chunk_index, chunk.text),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (content, chunk_id, doc_id) VALUES (?, ?, ?)",
                    (chunk.text, chunk.chunk_id, chunk.doc_id),
                )
            chunk_count += len(chunks)
            total_chunk_words += sum(len(chunk.text.split()) for chunk in chunks)

        conn.commit()
        conn.close()

        avg_chunk_size = (total_chunk_words / chunk_count) if chunk_count else 0.0
        build_time_ms = (time.perf_counter() - start) * 1000
        meta = IndexMeta(
            doc_count=len(docs),
            chunk_count=chunk_count,
            avg_chunk_size=avg_chunk_size,
            build_time_ms=build_time_ms,
        )
        self.meta_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
        return self.index_path


def _sanitize_query(query: str) -> str:
    tokens = [
        token
        for token in QUERY_TOKEN_RE.findall(query.lower())
        if token not in STOPWORDS
    ]
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return " OR ".join(tokens)


def load_index(index_path: Path = INDEX_PATH) -> SQLiteIndex:
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found at {index_path}")
    return SQLiteIndex(index_path, read_only=True)


def build_index(data_dir: Path = DATA_DIR, index_path: Path = INDEX_PATH) -> SQLiteIndex:
    extra_dirs = []
    ingest_pdfs_dir = INGEST_DIR / "pdfs"
    if ingest_pdfs_dir.exists():
        extra_dirs.append(ingest_pdfs_dir)
    builder = IndexBuilder(data_dir=data_dir, extra_dirs=extra_dirs, index_path=index_path)
    builder.build()
    return load_index(index_path)


def main() -> None:
    extra_dirs = []
    ingest_pdfs_dir = INGEST_DIR / "pdfs"
    if ingest_pdfs_dir.exists():
        extra_dirs.append(ingest_pdfs_dir)
    builder = IndexBuilder(extra_dirs=extra_dirs)
    path = builder.build()
    print(f"Indexed documents into {path}")


if __name__ == "__main__":
    main()
