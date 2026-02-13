# Failure Modes and Runtime Behavior

This document covers the gateway request path in `/v1/chat/completions`, model inference, and retrieval tooling.

## Timeout behavior

| Area | Trigger | Current behavior | Knobs |
|---|---|---|---|
| llama.cpp HTTP call | Upstream model call exceeds timeout | Request fails; gateway returns HTTP 500 from agent failure path | `LLAMA_TIMEOUT_S` |
| Gateway queue wait | Request cannot acquire inflight slot before queue timeout | Request is rejected with HTTP 503 (`Gateway overloaded: request queue timeout`) | `GATEWAY_QUEUE_TIMEOUT_MS`, `GATEWAY_MAX_INFLIGHT`, `GATEWAY_MAX_QUEUE` |

## Retry behavior

| Area | Trigger | Current behavior | Knobs |
|---|---|---|---|
| llama.cpp transient failures | Timeout/transport error or HTTP 408/429/5xx | Automatic retries with exponential backoff; final failure returns HTTP 500 | `LLAMA_MAX_RETRIES`, `LLAMA_RETRY_BACKOFF_MS` |
| Retrieval SQL query | SQLite read/query error | No retry; request fails and is counted as tool failure | none (intentional fail-fast) |

Notes:
- Retry latency is included in end-to-end latency and TTFT approximations.
- Retries are intentionally limited to avoid retry storms under sustained overload.

## Overload behavior

| Condition | Behavior | Why |
|---|---|---|
| Queue full (`pending >= GATEWAY_MAX_QUEUE`) | Immediate HTTP 503 (`queue is full`) | Protects process memory and avoids unbounded backlog |
| Queue wait timeout | HTTP 503 (`queue timeout`) | Caps tail latency and preserves capacity for new work |
| Inflight saturation | New requests queue until an inflight slot is available | Smooths short bursts without dropping all traffic |

## Other notable failures

| Failure mode | Behavior | Detection |
|---|---|---|
| Release gate fail (`eval_report.pass=false`) | Service startup fails; gateway does not serve traffic | Container restart loop + startup logs |
| Retrieval DB missing/corrupt | Request fails via tool exception path | 5xx errors + `tool_call_success_rate` drop |
| Canary regression | Canary fraction drops to `0.0` (`rolled_back`) | Runtime logs + canary status metrics/events |

## Operational checks

1. Confirm model endpoint health and timeout budget before raising retry count.
2. Watch `Gateway Error Rate (%)`, `Gateway Latency Quantiles (ms)`, and `TTFT p95 (ms)` in Grafana during incidents.
3. If 503 overload rises: first lower ingress RPS or increase replicas, then tune `GATEWAY_MAX_INFLIGHT` and queue settings conservatively.
4. Keep `LLAMA_MAX_RETRIES` low (typically `1-2`) to avoid amplifying load during outages.
