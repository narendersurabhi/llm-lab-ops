from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from opentelemetry import trace

from llm_ops.metrics import RETRIEVAL_LATENCY_MS, TOOL_CALLS_SUCCESS, TOOL_CALLS_TOTAL, ROLLING
from llm_ops.tools import RetrievalResult


class SQLiteFTS5RetrievalTool:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._tracer = trace.get_tracer(__name__)

    async def run(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        TOOL_CALLS_TOTAL.labels(tool="retrieval_sqlite").inc()
        start = time.perf_counter()
        with self._tracer.start_as_current_span("retrieval_sqlite"):
            try:
                with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
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
                results = [
                    RetrievalResult(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        source=row["source"],
                        text=row["text"],
                        score=float(row["score"]),
                    )
                    for row in rows
                ]
                TOOL_CALLS_SUCCESS.labels(tool="retrieval_sqlite").inc()
                ROLLING.record_tool_call(success=True)
                ROLLING.record_retrieval_hit(bool(results))
                return results
            except Exception:  # noqa: BLE001
                ROLLING.record_tool_call(success=False)
                ROLLING.record_retrieval_hit(False)
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                RETRIEVAL_LATENCY_MS.observe(elapsed_ms)

    async def close(self) -> None:
        return None
