from __future__ import annotations

import os
import time

import httpx
import pytest


OPS_BASE_URL = os.getenv("OPS_BASE_URL", "http://localhost:8000")
JAEGER_URL = os.getenv("JAEGER_URL", "http://localhost:16686")


def _wait_for_ready(url: str, timeout_s: int = 60) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise AssertionError(f"Service not ready: {url}")


@pytest.mark.integration
def test_gateway_health_and_metrics() -> None:
    _wait_for_ready(f"{OPS_BASE_URL}/health")
    health = httpx.get(f"{OPS_BASE_URL}/health", timeout=5.0)
    assert health.status_code == 200

    metrics = httpx.get(f"{OPS_BASE_URL}/metrics", timeout=5.0)
    assert metrics.status_code == 200
    body = metrics.text
    for key in [
        "request_count",
        "error_rate",
        "latency_histogram",
        "ttft_ms",
        "tokens_in",
        "tokens_out",
        "tool_call_success_rate",
        "retrieval_hit_rate",
        "citation_coverage",
    ]:
        assert key in body


@pytest.mark.integration
def test_chat_completions_returns_citations() -> None:
    _wait_for_ready(f"{OPS_BASE_URL}/health")
    resp = httpx.post(
        f"{OPS_BASE_URL}/v1/chat/completions",
        json={
            "model": "local-llama-gguf",
            "messages": [{"role": "user", "content": "What is RAG?"}],
        },
        timeout=10.0,
    )
    assert resp.status_code == 200
    payload = resp.json()
    content = payload["choices"][0]["message"]["content"]
    assert "Citations:" in content

    # Best-effort trace check. If Jaeger API is unavailable, skip.
    try:
        services = httpx.get(f"{JAEGER_URL}/api/services", timeout=5.0)
        if services.status_code == 200:
            data = services.json()
            service_names = data.get("data")
            if not service_names:
                pytest.skip("Jaeger returned no services yet")
            assert any("policy-llm-ops" in name for name in service_names)
        else:
            pytest.skip("Jaeger API not available")
    except httpx.HTTPError:
        pytest.skip("Jaeger API not reachable")
