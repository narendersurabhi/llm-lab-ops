from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from llm_lab.retrieval_api import app as retrieval_app
from llm_ops.agent import LangGraphAgent
from llm_ops.gateway import app as ops_app
from llm_ops.model import MockModelClient
from llm_ops.tools import RetrievalTool


def test_chat_completion_e2e() -> None:
    transport = ASGITransport(app=retrieval_app)
    async_client = AsyncClient(transport=transport, base_url="http://test")
    retrieval = RetrievalTool(client=async_client)
    agent = LangGraphAgent(retrieval=retrieval, model=MockModelClient())

    # Inject test agent.
    ops_app.dependency_overrides = {}
    import llm_ops.gateway as gateway

    gateway.agent = agent
    gateway.canary = None

    with TestClient(ops_app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "local-llama-gguf",
                "messages": [{"role": "user", "content": "What is RAG?"}],
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        assert content
        assert "Citations:" in content

    asyncio.run(async_client.aclose())
