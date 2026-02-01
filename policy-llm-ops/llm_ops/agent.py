from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_ops.config import settings
from llm_ops.graph import build_graph
from llm_ops.model import FakeModelClient, LlamaCppClient, MockModelClient, ModelClient
from llm_ops.logging import setup_logging, log_event

logger = setup_logging()
from llm_ops.tools import PolicyTool, QuoteTool, RetrievalTool


@dataclass
class AgentResponse:
    content: str
    tokens_in: int
    tokens_out: int
    ttft_ms: float
    tool_success: bool
    retrieval_hit: bool
    citation_coverage: float


class LangGraphAgent:
    def __init__(
        self,
        retrieval: RetrievalTool | None = None,
        model: ModelClient | None = None,
        policy: PolicyTool | None = None,
        quote: QuoteTool | None = None,
    ) -> None:
        self.retrieval = retrieval or RetrievalTool()
        if model is not None:
            self.model = model
        elif settings.llm_provider == "mock":
            self.model = MockModelClient()
        elif settings.llm_provider == "fake":
            self.model = FakeModelClient(mode="normal")
        elif settings.llm_provider == "fake_regression":
            self.model = FakeModelClient(mode="regression")
        else:
            self.model = LlamaCppClient()
        self.policy = policy or PolicyTool()
        self.quote = quote or QuoteTool()
        self.graph = build_graph(self.retrieval, self.policy, self.quote, self.model)

    async def run(self, messages: list[dict[str, Any]]) -> AgentResponse:
        try:
            result = await self.graph.ainvoke({"messages": messages})
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "agent_exception", error=str(exc))
            fallback = "The system encountered an error while processing the request."
            return AgentResponse(
                content=fallback,
                tokens_in=0,
                tokens_out=len(fallback.split()),
                ttft_ms=0.0,
                tool_success=False,
                retrieval_hit=False,
                citation_coverage=0.0,
            )

        return AgentResponse(
            content=str(result.get("final_response", "")),
            tokens_in=int(result.get("tokens_in", 0)),
            tokens_out=int(result.get("tokens_out", 0)),
            ttft_ms=float(result.get("ttft_ms", 0.0)),
            tool_success=bool(result.get("tool_success", False)),
            retrieval_hit=bool(result.get("retrieval_hit", False)),
            citation_coverage=float(result.get("citation_coverage", 0.0)),
        )
