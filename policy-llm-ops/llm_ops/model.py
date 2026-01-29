from __future__ import annotations

import time
from typing import Tuple

import httpx
from opentelemetry import trace

from llm_ops.config import settings


class ModelClient:
    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        raise NotImplementedError


class LlamaCppClient(ModelClient):
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or settings.llama_cpp_url
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self._tracer = trace.get_tracer(__name__)

    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.2}
        start = time.perf_counter()
        with self._tracer.start_as_current_span("model_inference"):
            resp = await self._client.post("/completion", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("content") or data.get("completion") or ""
        latency_ms = (time.perf_counter() - start) * 1000
        # For non-streaming llama.cpp requests, TTFT ~= total latency.
        return text.strip(), latency_ms

    async def close(self) -> None:
        await self._client.aclose()


class MockModelClient(ModelClient):
    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        start = time.perf_counter()
        text = "(mock) " + prompt.split("Question:")[-1].strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return text, latency_ms
