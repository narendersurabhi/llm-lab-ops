from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterable

from opentelemetry import trace

from llm_ops.config import settings
from llm_ops.metrics import RETRIEVAL_LATENCY_MS, TOOL_CALLS_SUCCESS, TOOL_CALLS_TOTAL, ROLLING


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    source: str
    text: str
    score: float
    chunk_id: str


@dataclass(frozen=True)
class Citation:
    doc_id: str
    source: str
    chunk_id: str
    snippet: str


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class RetrievalTool:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path or settings.retrieval_db_path)
        self._tracer = trace.get_tracer(__name__)

    async def run(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        TOOL_CALLS_TOTAL.labels(tool="retrieval").inc()
        start = time.perf_counter()
        with self._tracer.start_as_current_span("retrieval_tool"):
            try:
                with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        """
                        SELECT c.chunk_id, c.doc_id, d.source, c.content as text,
                               bm25(chunks_fts) as score
                        FROM chunks_fts
                        JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                        JOIN documents d ON d.doc_id = c.doc_id
                        WHERE chunks_fts MATCH ?
                        ORDER BY score ASC, c.chunk_id ASC
                        LIMIT ?
                        """,
                        (query, top_k),
                    )
                    rows = cursor.fetchall()
                results = [
                    RetrievalResult(
                        doc_id=row["doc_id"],
                        source=row["source"],
                        text=row["text"],
                        score=float(row["score"]),
                        chunk_id=row["chunk_id"],
                    )
                    for row in rows
                ]
                TOOL_CALLS_SUCCESS.labels(tool="retrieval").inc()
                ROLLING.record_tool_call(success=True)
                ROLLING.record_retrieval_hit(bool(results))
                return results
            except Exception:  # noqa: BLE001
                ROLLING.record_tool_call(success=False)
                ROLLING.record_retrieval_hit(False)
                raise
            finally:
                RETRIEVAL_LATENCY_MS.observe((time.perf_counter() - start) * 1000)

    async def close(self) -> None:
        return None


class QuoteTool:
    def __init__(self, max_quotes: int = 2, max_chars: int = 180) -> None:
        self.max_quotes = max_quotes
        self.max_chars = max_chars
        self._tracer = trace.get_tracer(__name__)

    def run(self, contexts: Iterable[RetrievalResult]) -> list[Citation]:
        TOOL_CALLS_TOTAL.labels(tool="quote").inc()
        with self._tracer.start_as_current_span("quote_tool"):
            results: list[Citation] = []
            for ctx in list(contexts)[: self.max_quotes]:
                snippet = ctx.text.strip().replace("\n", " ")
                if len(snippet) > self.max_chars:
                    snippet = snippet[: self.max_chars].rstrip() + "..."
                results.append(
                    Citation(
                        doc_id=ctx.doc_id,
                        source=ctx.source,
                        chunk_id=ctx.chunk_id,
                        snippet=snippet,
                    )
                )
            TOOL_CALLS_SUCCESS.labels(tool="quote").inc()
            ROLLING.record_tool_call(success=True)
            return results


class PolicyTool:
    _INJECTION_PATTERNS = [
        re.compile(r"ignore (all|previous) instructions", re.IGNORECASE),
        re.compile(r"system prompt", re.IGNORECASE),
        re.compile(r"developer message", re.IGNORECASE),
        re.compile(r"act as", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        self._tracer = trace.get_tracer(__name__)

    def check(self, query: str) -> PolicyDecision:
        TOOL_CALLS_TOTAL.labels(tool="policy").inc()
        with self._tracer.start_as_current_span("policy_tool"):
            for pattern in self._INJECTION_PATTERNS:
                if pattern.search(query):
                    ROLLING.record_tool_call(success=True)
                    TOOL_CALLS_SUCCESS.labels(tool="policy").inc()
                    return PolicyDecision(allowed=False, reason="prompt_injection_detected")
            ROLLING.record_tool_call(success=True)
            TOOL_CALLS_SUCCESS.labels(tool="policy").inc()
            return PolicyDecision(allowed=True)
