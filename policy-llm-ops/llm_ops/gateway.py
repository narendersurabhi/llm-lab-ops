from __future__ import annotations

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
from llm_ops.observability import init_tracing, instrument_app

logger = setup_logging()
init_tracing()

app = FastAPI(title="Policy LLM Ops Gateway", version="0.1.0")
instrument_app(app)

agent = LangGraphAgent()
canary_agent = LangGraphAgent() if settings.canary_enabled else None
canary = CanaryController(settings.model_dir) if settings.canary_enabled else None
tracer = trace.get_tracer(__name__)


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
    req_id = request_id_var.get() or str(uuid.uuid4())
    start = time.perf_counter()
    variant = canary.choose_variant() if canary else "stable"
    active_agent = canary_agent if variant == "canary" and canary_agent else agent
    response.headers["x-llm-variant"] = variant
    if should_sample_prompt():
        log_event(logger, "prompt_sample", messages=[m.model_dump() for m in req.messages])

    with tracer.start_as_current_span("langgraph_run"):
        try:
            response = await active_agent.run([m.model_dump() for m in req.messages])
        except Exception as exc:  # noqa: BLE001
            if canary:
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

    if canary:
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


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
