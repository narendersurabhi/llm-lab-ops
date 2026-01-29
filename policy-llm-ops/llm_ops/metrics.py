from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests",
    ["endpoint", "status"],
)
ERROR_RATE = Gauge(
    "error_rate",
    "Rolling error rate (errors / requests)",
)
LATENCY_HISTOGRAM = Histogram(
    "latency_histogram",
    "Request latency in milliseconds",
    ["endpoint"],
    buckets=(25, 50, 100, 200, 400, 800, 1200, 2000, 5000),
)
TTFT = Histogram(
    "ttft_ms",
    "Time to first token in milliseconds",
    buckets=(25, 50, 100, 200, 400, 800, 1200, 2000, 5000),
)
TOKENS_IN = Counter(
    "tokens_in",
    "Approximate input tokens",
)
TOKENS_OUT = Counter(
    "tokens_out",
    "Approximate output tokens",
)
RETRIEVAL_LATENCY_MS = Histogram(
    "retrieval_latency_ms",
    "Retrieval tool latency in milliseconds",
    buckets=(5, 10, 25, 50, 100, 200, 400, 800),
)
TOOL_CALLS_TOTAL = Counter(
    "tool_calls_total",
    "Total tool calls",
    ["tool"],
)
TOOL_CALLS_SUCCESS = Counter(
    "tool_calls_success",
    "Successful tool calls",
    ["tool"],
)
TOOL_CALL_SUCCESS_RATE = Gauge(
    "tool_call_success_rate",
    "Rolling tool call success rate",
)
RETRIEVAL_HIT_RATE = Gauge(
    "retrieval_hit_rate",
    "Rolling retrieval hit rate",
)
CITATION_COVERAGE = Gauge(
    "citation_coverage",
    "Citation coverage ratio for retrieved contexts",
)


class RollingCounters:
    def __init__(self) -> None:
        self.requests = 0
        self.errors = 0
        self.tool_calls = 0
        self.tool_success = 0
        self.retrieval_requests = 0
        self.retrieval_hits = 0

    def record_request(self, is_error: bool) -> None:
        self.requests += 1
        if is_error:
            self.errors += 1
        if self.requests > 0:
            ERROR_RATE.set(self.errors / self.requests)

    def record_tool_call(self, success: bool) -> None:
        self.tool_calls += 1
        if success:
            self.tool_success += 1
        if self.tool_calls > 0:
            TOOL_CALL_SUCCESS_RATE.set(self.tool_success / self.tool_calls)

    def record_retrieval_hit(self, hit: bool) -> None:
        self.retrieval_requests += 1
        if hit:
            self.retrieval_hits += 1
        if self.retrieval_requests > 0:
            RETRIEVAL_HIT_RATE.set(self.retrieval_hits / self.retrieval_requests)


ROLLING = RollingCounters()
