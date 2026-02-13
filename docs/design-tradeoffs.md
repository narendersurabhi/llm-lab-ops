# Design Tradeoffs: Retrieval Indexes, Caching, and Queue Bounds

This document captures current design choices and the tradeoffs behind them.

## 1) DB index strategy (SQLite + FTS5)

Current implementation:
- `chunks_fts` (FTS5) is the primary search index.
- `chunks` and `documents` are row tables joined after FTS match.
- Ranking uses `bm25(chunks_fts)` and returns top-K by score.

Tradeoffs:
- Pros: lightweight local deployment, predictable single-file ops, very fast text search for moderate corpus size.
- Cons: no sharding, write amplification during large reindex, limited concurrency compared with distributed vector/search engines.

Alternatives considered:
- Add secondary B-tree indexes for additional filters (for example `chunks(doc_id)`) when filter-heavy queries are introduced.
- Move to dedicated search infra (OpenSearch/pgvector) when corpus size or write concurrency outgrows SQLite limits.

Decision:
- Keep FTS5-first design for local-first simplicity.
- Revisit index strategy only when retrieval latency p95 or reindex windows exceed SLO.

## 2) Cache policy

Current implementation:
- No persistent response cache in gateway path.
- Retrieval queries execute directly against SQLite for each request.

Tradeoffs:
- Pros: no stale-answer risk, minimal invalidation complexity, deterministic behavior across deploys.
- Cons: repeated hot queries pay full retrieval/model cost each time.

Policy recommendation:
- Introduce optional in-memory cache only for retrieval results (not final model output).
- Suggested policy: LRU + TTL, keyed by normalized query and `top_k`.
- Safe starting point: TTL `30-120s`, cache size `1k-10k` entries.

Decision:
- Keep cache disabled by default.
- Enable only when load-test data shows repeated query patterns and CPU pressure on retrieval path.

## 3) Queue bounds and overload control

Current implementation:
- Gateway uses bounded pending queue and bounded inflight concurrency.
- Requests are rejected with HTTP 503 when queue is full or queue-wait timeout is hit.

Primary knobs:
- `GATEWAY_MAX_INFLIGHT`
- `GATEWAY_MAX_QUEUE`
- `GATEWAY_QUEUE_TIMEOUT_MS`

Tradeoffs:
- Higher inflight: improves throughput if CPU/headroom exists; risks latency collapse and context switching overhead.
- Higher queue bound: reduces immediate drops during bursts; risks long tail latency and memory growth.
- Longer queue timeout: fewer dropped requests; can increase p95/p99 significantly.

Decision:
- Prefer strict bounds to protect tail latency and service stability.
- Scale horizontally before increasing queue depth aggressively.

## Acceptance criteria for revisiting decisions

Revisit this design when any of the following persist for 3+ measurement windows:
1. `p95` or `p99` latency breaches SLO under expected RPS.
2. Overload 503s remain elevated after horizontal scaling.
3. Retrieval p95 dominates total latency and query repetition is high.
4. Reindex/build times impact release cadence.
