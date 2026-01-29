from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Iterable

import httpx
from opentelemetry import trace

from llm_ops.config import settings
from llm_ops.metrics import RETRIEVAL_LATENCY_MS, TOOL_CALLS_SUCCESS, TOOL_CALLS_TOTAL, ROLLING


@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    source: str
    text: str
    score: float
    chunk_id: str | None = None


@dataclass(frozen=True)
class Citation:
    doc_id: str
    source: str
    snippet: str


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None


class RetrievalTool:
    def __init__(
        self, base_url: str | None = None, client: httpx.AsyncClient | None = None
    ) -> None:
        self.base_url = base_url or settings.retrieval_url
        self._client = client or httpx.AsyncClient(base_url=self.base_url, timeout=10.0)
        self._tracer = trace.get_tracer(__name__)

    async def run(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        TOOL_CALLS_TOTAL.labels(tool="retrieval").inc()
        start = time.perf_counter()
        with self._tracer.start_as_current_span("retrieval_tool"):
            try:
                resp = await self._client.post("/retrieve", json={"query": query, "top_k": top_k})
                resp.raise_for_status()
                data = resp.json()
                results = [
                    RetrievalResult(
                        doc_id=item.get("doc_id", ""),
                        source=item.get("source", ""),
                        text=item.get("text", ""),
                        score=float(item.get("score", 0.0)),
                        chunk_id=item.get("chunk_id"),
                    )
                    for item in data.get("results", [])
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
                elapsed_ms = (time.perf_counter() - start) * 1000
                RETRIEVAL_LATENCY_MS.observe(elapsed_ms)

    async def close(self) -> None:
        await self._client.aclose()


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
                results.append(Citation(doc_id=ctx.doc_id, source=ctx.source, snippet=snippet))
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
