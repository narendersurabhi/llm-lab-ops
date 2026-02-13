from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--prompt", default="Explain RAG")
    parser.add_argument("--output", default="logs/loadtest_report.json")
    return parser.parse_args()


@dataclass(frozen=True)
class RequestResult:
    latency_ms: float
    status_code: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.status_code is not None and 200 <= self.status_code < 400


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def _split_url(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(f"Invalid URL: {url}")
    base_url = f"{parts.scheme}://{parts.netloc}"
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return base_url, path


async def _send_request(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> RequestResult:
    start = time.perf_counter()
    try:
        resp = await client.post(path, json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(latency_ms=latency_ms, status_code=resp.status_code, error=None)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(latency_ms=latency_ms, status_code=None, error=type(exc).__name__)


async def worker(
    client: httpx.AsyncClient,
    path: str,
    payload: dict[str, Any],
    timeout_s: float,
    queue: asyncio.Queue[int | None],
    results: list[RequestResult],
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        result = await _send_request(client=client, path=path, payload=payload, timeout_s=timeout_s)
        results.append(result)
        queue.task_done()


def _build_report(results: list[RequestResult], total: int, concurrency: int, duration_s: float) -> dict[str, Any]:
    latencies = [result.latency_ms for result in results]
    ok_count = sum(1 for result in results if result.ok)
    error_count = total - ok_count
    status_codes: dict[str, int] = {}
    exception_counts: dict[str, int] = {}
    for result in results:
        if result.status_code is not None:
            code = str(result.status_code)
            status_codes[code] = status_codes.get(code, 0) + 1
        elif result.error:
            exception_counts[result.error] = exception_counts.get(result.error, 0) + 1

    report = {
        "requests": {
            "total": total,
            "completed": len(results),
            "concurrency": concurrency,
            "ok": ok_count,
            "errors": error_count,
            "error_rate": (error_count / total) if total else 0.0,
            "status_codes": status_codes,
            "exceptions": exception_counts,
        },
        "throughput": {
            "duration_s": duration_s,
            "rps": (total / duration_s) if duration_s > 0 else 0.0,
        },
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
        },
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    req = report["requests"]
    throughput = report["throughput"]
    latency = report["latency_ms"]
    print("")
    print("Load Test Report")
    print("================")
    print(
        f"Requests: total={req['total']} completed={req['completed']} "
        f"ok={req['ok']} errors={req['errors']} error_rate={req['error_rate'] * 100:.2f}%"
    )
    print(
        f"Throughput: duration={throughput['duration_s']:.3f}s "
        f"rps={throughput['rps']:.2f}"
    )
    print(
        f"Latency (ms): p50={latency['p50']:.2f} "
        f"p95={latency['p95']:.2f} p99={latency['p99']:.2f}"
    )
    if req["status_codes"]:
        print(f"Status codes: {req['status_codes']}")
    if req["exceptions"]:
        print(f"Exceptions: {req['exceptions']}")


def _write_report(path: str, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {output_path}")


async def run_loadtest(
    url: str,
    total: int,
    concurrency: int,
    timeout_s: float,
    payload: dict[str, Any],
) -> dict[str, Any]:
    base_url, path = _split_url(url)
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    results: list[RequestResult] = []
    for i in range(total):
        queue.put_nowait(i)
    for _ in range(concurrency):
        queue.put_nowait(None)

    async with httpx.AsyncClient(base_url=base_url) as client:
        tasks = [
            asyncio.create_task(
                worker(
                    client=client,
                    path=path,
                    payload=payload,
                    timeout_s=timeout_s,
                    queue=queue,
                    results=results,
                )
            )
            for _ in range(concurrency)
        ]
        start = time.perf_counter()
        await queue.join()
        elapsed = time.perf_counter() - start
        await asyncio.gather(*tasks)
    return _build_report(results=results, total=total, concurrency=concurrency, duration_s=elapsed)


if __name__ == "__main__":
    args = parse_args()
    report = asyncio.run(
        run_loadtest(
            url=args.url,
            total=args.requests,
            concurrency=args.concurrency,
            timeout_s=args.timeout_s,
            payload={
                "model": "local-llama-gguf",
                "messages": [{"role": "user", "content": args.prompt}],
            },
        )
    )
    _print_report(report)
    if args.output:
        _write_report(args.output, report)
