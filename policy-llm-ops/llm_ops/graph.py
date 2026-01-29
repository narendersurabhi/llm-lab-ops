from __future__ import annotations

from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, StateGraph
from opentelemetry import trace

from llm_ops.model import ModelClient
from llm_ops.retrieval_tool import RetrievalToolProtocol
from llm_ops.tools import Citation, PolicyDecision, PolicyTool, QuoteTool, RetrievalResult


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    query: str
    contexts: list[RetrievalResult]
    citations: list[Citation]
    response_text: str
    final_response: str
    tokens_in: int
    tokens_out: int
    ttft_ms: float
    tool_success: bool
    retrieval_hit: bool
    citation_coverage: float
    error: str
    fallback_reason: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def extract_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def build_prompt(query: str, contexts: list[RetrievalResult]) -> str:
    context_blocks = [f"[{ctx.doc_id}] {ctx.text}" for ctx in contexts]
    context_text = "\n".join(context_blocks)
    return (
        "You are a helpful assistant. Answer the question using the context below. "
        "Cite sources with [doc_id].\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )


def format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = ["\n\nCitations:"]
    for cite in citations:
        lines.append(f"[{cite.doc_id}] {cite.snippet} ({cite.source})")
    return "\n".join(lines)


def precheck_node(state: AgentState) -> dict[str, Any]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("node_precheck"):
        messages = state.get("messages", [])
        query = extract_user_query(messages)
        if not query.strip():
            return {"fallback_reason": "empty_query", "error": "Empty user query"}
        return {"query": query}


def route_precheck(state: AgentState) -> Literal["retrieve", "fallback"]:
    return "fallback" if state.get("fallback_reason") else "retrieve"


def make_retrieve_node(retrieval: RetrievalToolProtocol) -> Callable[[AgentState], Any]:
    tracer = trace.get_tracer(__name__)

    async def retrieve_node(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("node_retrieve"):
            query = state.get("query", "")
            contexts = await retrieval.run(query, top_k=3)
            return {"contexts": contexts, "retrieval_hit": bool(contexts)}

    return retrieve_node


def make_anti_injection_node(policy: PolicyTool) -> Callable[[AgentState], dict[str, Any]]:
    tracer = trace.get_tracer(__name__)

    def anti_injection_node(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("node_anti_injection"):
            query = state.get("query", "")
            decision: PolicyDecision = policy.check(query)
            if not decision.allowed:
                return {
                    "fallback_reason": decision.reason or "policy_block",
                    "error": "Prompt injection detected",
                }
            return {}

    return anti_injection_node


def route_guard(state: AgentState) -> Literal["generate", "fallback"]:
    return "fallback" if state.get("fallback_reason") else "generate"


def make_generate_node(model: ModelClient, quote: QuoteTool) -> Callable[[AgentState], Any]:
    tracer = trace.get_tracer(__name__)

    async def generate_node(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("node_generate"):
            query = state.get("query", "")
            contexts = state.get("contexts", [])
            prompt = build_prompt(query, contexts)
            response_text, ttft_ms = await model.generate(prompt)
            citations = quote.run(contexts)
            content_tokens = estimate_tokens(response_text)
            return {
                "response_text": response_text,
                "citations": citations,
                "tokens_in": estimate_tokens(prompt),
                "tokens_out": content_tokens,
                "ttft_ms": ttft_ms,
                "tool_success": True,
            }

    return generate_node


def cite_check_node(state: AgentState) -> dict[str, Any]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("node_cite_check"):
        contexts = state.get("contexts", [])
        citations = state.get("citations", [])
        coverage = (len(citations) / len(contexts)) if contexts else 0.0
        return {"citation_coverage": coverage}


def finalize_node(state: AgentState) -> dict[str, Any]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("node_finalize"):
        response_text = state.get("response_text", "")
        citations = state.get("citations", [])
        final_response = f"{response_text.strip()}{format_citations(citations)}"
        return {"final_response": final_response}


def fallback_node(state: AgentState) -> dict[str, Any]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("node_fallback"):
        reason = state.get("fallback_reason", "fallback")
        final_response = (
            "I was unable to answer that request safely. "
            f"Reason: {reason}. Please rephrase your question."
        )
        return {
            "final_response": final_response,
            "tokens_in": 0,
            "tokens_out": estimate_tokens(final_response),
            "ttft_ms": 0.0,
            "tool_success": False,
            "citation_coverage": 0.0,
        }


def build_graph(
    retrieval: RetrievalToolProtocol,
    policy: PolicyTool,
    quote: QuoteTool,
    model: ModelClient,
):
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("precheck", precheck_node)
    graph.add_node("retrieve", make_retrieve_node(retrieval))
    graph.add_node("anti_injection", make_anti_injection_node(policy))
    graph.add_node("generate", make_generate_node(model, quote))
    graph.add_node("cite_check", cite_check_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("fallback", fallback_node)

    graph.set_entry_point("precheck")
    graph.add_conditional_edges(
        "precheck",
        route_precheck,
        {"retrieve": "retrieve", "fallback": "fallback"},
    )
    graph.add_edge("retrieve", "anti_injection")
    graph.add_conditional_edges(
        "anti_injection",
        route_guard,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", "cite_check")
    graph.add_edge("cite_check", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)

    return graph.compile()
