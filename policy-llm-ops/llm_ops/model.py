from __future__ import annotations

import asyncio
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
        self.timeout_s = max(1.0, settings.llama_timeout_s)
        self.max_retries = max(0, settings.llama_max_retries)
        self.retry_backoff_s = max(0.0, settings.llama_retry_backoff_ms) / 1000.0
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_s)
        self._tracer = trace.get_tracer(__name__)

    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.2}
        start = time.perf_counter()
        attempts = self.max_retries + 1
        with self._tracer.start_as_current_span("model_inference"):
            for attempt in range(attempts):
                try:
                    resp = await self._client.post("/completion", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    text = data.get("content") or data.get("completion") or ""
                    latency_ms = (time.perf_counter() - start) * 1000
                    # For non-streaming llama.cpp requests, TTFT ~= total latency.
                    return text.strip(), latency_ms
                except Exception as exc:  # noqa: BLE001
                    is_last = attempt >= (attempts - 1)
                    if is_last or not self._is_retriable(exc):
                        raise
                    if self.retry_backoff_s > 0:
                        await asyncio.sleep(self.retry_backoff_s * (2**attempt))
        raise RuntimeError("model_inference_unreachable")

    @staticmethod
    def _is_retriable(exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 429, 500, 502, 503, 504}
        return False

    async def close(self) -> None:
        await self._client.aclose()


class MockModelClient(ModelClient):
    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        start = time.perf_counter()
        text = "(mock) " + prompt.split("Question:")[-1].strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return text, latency_ms


class FakeModelClient(ModelClient):
    def __init__(
        self,
        mode: str = "normal",
        delay_ms: float | None = None,
        error_every: int | None = None,
    ) -> None:
        self.mode = mode
        if delay_ms is None:
            delay_ms = settings.fake_model_delay_ms
        if error_every is None:
            error_every = settings.fake_model_error_every
        if self.mode == "regression" and delay_ms == 0:
            delay_ms = 250.0
        self.delay_ms = delay_ms
        self.error_every = error_every or 0
        self._counter = 0

    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        start = time.perf_counter()
        self._counter += 1
        if self.error_every and (self._counter % self.error_every == 0):
            raise RuntimeError("Simulated model error")
        if self.delay_ms:
            await asyncio.sleep(self.delay_ms / 1000.0)
        text = "(fake) " + prompt.split("Question:")[-1].strip()
        latency_ms = (time.perf_counter() - start) * 1000
        return text, latency_ms


class MlxClient(ModelClient):
    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.mlx_model
        self.adapter_path = adapter_path or settings.mlx_adapter_path
        self._model = None
        self._tokenizer = None
        self._tracer = trace.get_tracer(__name__)

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from mlx_lm.utils import load as mlx_load
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "mlx-lm is not installed. Install with `pip install mlx-lm` "
                "and run on Apple Silicon."
            ) from exc
        tokenizer_config = {"trust_remote_code": settings.mlx_trust_remote_code}
        self._model, self._tokenizer = mlx_load(
            self.model_name, tokenizer_config=tokenizer_config, adapter_path=self.adapter_path
        )

    async def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[str, float]:
        start = time.perf_counter()
        with self._tracer.start_as_current_span("model_inference"):
            self._ensure_loaded()
            from mlx_lm.generate import generate as mlx_generate

            text = mlx_generate(
                self._model,  # type: ignore[arg-type]
                self._tokenizer,  # type: ignore[arg-type]
                prompt,
                max_tokens=max_tokens,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        return text.strip(), latency_ms
