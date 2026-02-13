from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from opentelemetry import trace

from llm_ops.agent import LangGraphAgent
from llm_ops.canary import CanaryController
from llm_ops.config import settings
from llm_ops.logging import request_id_var, setup_logging, log_event, should_sample_prompt
from llm_ops.metrics import (
    CITATION_COVERAGE,
    LATENCY_HISTOGRAM,
    REQUEST_COUNT,
    ROLLING,
    TTFT,
    TOKENS_IN,
    TOKENS_OUT,
)
from llm_ops.model import LlamaCppClient, MockModelClient, ModelClient
from llm_ops.observability import init_tracing, instrument_app
from llm_ops.release_manager import ReleaseManager, ReleaseBundle
from llm_ops.tools import RetrievalTool

logger = setup_logging()
init_tracing()

app = FastAPI(title="Policy LLM Ops Gateway", version="0.1.0")
instrument_app(app)

agent: LangGraphAgent | None = None
canary_agent: LangGraphAgent | None = None
canary: CanaryController | None = None
release_bundle: ReleaseBundle | None = None
release_manager = ReleaseManager()
tracer = trace.get_tracer(__name__)


class OverloadController:
    def __init__(self, max_inflight: int, max_queue: int, queue_timeout_ms: float) -> None:
        self.max_inflight = max(1, max_inflight)
        self.max_queue = max(self.max_inflight, max_queue)
        self.queue_timeout_s = max(0.001, queue_timeout_ms / 1000.0)
        self._semaphore = asyncio.Semaphore(self.max_inflight)
        self._pending = 0
        self._lock = asyncio.Lock()

    async def try_enter(self) -> str | None:
        async with self._lock:
            if self._pending >= self.max_queue:
                return "queue_full"
            self._pending += 1
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.queue_timeout_s)
            return None
        except TimeoutError:
            async with self._lock:
                self._pending = max(0, self._pending - 1)
            return "queue_timeout"

    async def exit(self) -> None:
        self._semaphore.release()
        async with self._lock:
            self._pending = max(0, self._pending - 1)


overload_controller = OverloadController(
    max_inflight=settings.gateway_max_inflight,
    max_queue=settings.gateway_max_queue,
    queue_timeout_ms=settings.gateway_queue_timeout_ms,
)


def _build_model(is_canary: bool) -> ModelClient:
    if settings.llm_provider == "mock":
        return MockModelClient()
    if is_canary:
        return MockModelClient()
    return LlamaCppClient()


def init_runtime() -> None:
    global agent, canary_agent, canary, release_bundle
    release_bundle = release_manager.load()
    if not release_bundle.allowed:
        raise RuntimeError("Release gate failed: eval_report.pass is false")
    retrieval = RetrievalTool(db_path=release_bundle.index_path)
    agent = LangGraphAgent(retrieval=retrieval, model=_build_model(False))
    if settings.canary_enabled:
        canary_agent = LangGraphAgent(
            retrieval=RetrievalTool(db_path=release_bundle.index_path),
            model=_build_model(True),
        )
        canary = CanaryController(release_bundle.model_dir)
    else:
        canary_agent = None
        canary = None


@app.on_event("startup")
async def startup() -> None:
    init_runtime()


@app.on_event("shutdown")
async def shutdown() -> None:
    for active_agent in [agent, canary_agent]:
        if active_agent is None:
            continue
        try:
            await active_agent.retrieval.close()
        except Exception:  # noqa: BLE001
            pass
        if hasattr(active_agent.model, "close"):
            try:
                await active_agent.model.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass


class ChatMessage(BaseModel):
    role: str
    content: str
    model_config = ConfigDict(extra="allow")


class ChatCompletionRequest(BaseModel):
    model: str = "local-llama-gguf"
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=256, ge=1, le=1024)
    temperature: float | None = Field(default=0.2, ge=0, le=2)
    stream: bool | None = False
    model_config = ConfigDict(extra="allow")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


@app.middleware("http")
async def add_request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
    req_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    token = request_id_var.set(req_id)
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        response.headers["x-request-id"] = req_id
        return response
    finally:
        request_id_var.reset(token)
        latency_ms = (time.perf_counter() - start) * 1000
        status = getattr(response, "status_code", 500)
        REQUEST_COUNT.labels(endpoint=request.url.path, status=str(status)).inc()
        LATENCY_HISTOGRAM.labels(endpoint=request.url.path).observe(latency_ms)
        ROLLING.record_request(is_error=status >= 400)
        log_event(
            logger,
            "request_complete",
            path=request.url.path,
            status=status,
            latency_ms=round(latency_ms, 2),
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest, response: Response) -> ChatCompletionResponse:
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent runtime not initialized")
    queue_status = await overload_controller.try_enter()
    if queue_status == "queue_full":
        raise HTTPException(status_code=503, detail="Gateway overloaded: request queue is full")
    if queue_status == "queue_timeout":
        raise HTTPException(status_code=503, detail="Gateway overloaded: request queue timeout")

    req_id = request_id_var.get() or str(uuid.uuid4())
    try:
        start = time.perf_counter()
        variant = canary.choose_variant() if canary else "stable"
        response.headers["x-llm-variant"] = variant
        active_agent = canary_agent if variant == "canary" and canary_agent else agent
        if should_sample_prompt():
            log_event(logger, "prompt_sample", messages=[m.model_dump() for m in req.messages])

        with tracer.start_as_current_span("langgraph_run"):
            try:
                response = await active_agent.run([m.model_dump() for m in req.messages])
            except Exception as exc:  # noqa: BLE001
                if canary and variant == "canary":
                    canary.record(
                        latency_ms=(time.perf_counter() - start) * 1000,
                        is_error=True,
                        tool_success=False,
                        citation_coverage=0.0,
                    )
                log_event(logger, "agent_error", error=str(exc))
                raise HTTPException(status_code=500, detail="Agent failure") from exc

        latency_ms = (time.perf_counter() - start) * 1000
        TTFT.observe(response.ttft_ms)
        TOKENS_IN.inc(response.tokens_in)
        TOKENS_OUT.inc(response.tokens_out)
        CITATION_COVERAGE.set(response.citation_coverage)

        if canary and variant == "canary":
            canary.record(
                latency_ms=latency_ms,
                is_error=False,
                tool_success=response.tool_success,
                citation_coverage=response.citation_coverage,
            )

        payload = ChatCompletionResponse(
            id=f"chatcmpl-{req_id}",
            object="chat.completion",
            created=int(datetime.now(timezone.utc).timestamp()),
            model=req.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": response.tokens_in,
                "completion_tokens": response.tokens_out,
                "total_tokens": response.tokens_in + response.tokens_out,
            },
        )
        log_event(
            logger,
            "chat_completion",
            variant=variant,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
            retrieval_hit=response.retrieval_hit,
            citation_coverage=round(response.citation_coverage, 3),
        )
        return payload
    finally:
        await overload_controller.exit()


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
