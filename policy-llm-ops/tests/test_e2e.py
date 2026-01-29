from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from utils import build_minimal_release_bundle


@pytest.mark.e2e
def test_canary_rolls_back_with_regression(tmp_path: Path, monkeypatch) -> None:
    release_dir = tmp_path / "release"
    contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
    build_minimal_release_bundle(release_dir, contracts_dir)

    monkeypatch.setenv("RELEASE_PATH", str(release_dir))
    monkeypatch.setenv("CANARY_ENABLED", "true")
    monkeypatch.setenv("CANARY_FRACTION", "1.0")
    monkeypatch.setenv("CANARY_MIN_SAMPLES", "5")
    monkeypatch.setenv("CANARY_SLO_WINDOW", "10")
    monkeypatch.setenv("P95_REGRESSION_MAX", "0.2")
    monkeypatch.setenv("LLM_PROVIDER", "fake_regression")

    import llm_ops.config as config

    importlib.reload(config)
    import llm_ops.canary as canary
    import llm_ops.release_manager as release_manager

    importlib.reload(canary)
    importlib.reload(release_manager)
    import llm_ops.gateway as gateway

    importlib.reload(gateway)

    with TestClient(gateway.app) as client:
        for _ in range(5):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "local-llama-gguf",
                    "messages": [{"role": "user", "content": "What is RAG?"}],
                },
            )
            assert resp.status_code == 200

        assert gateway.canary is not None
        assert gateway.canary.state.mode == "rolled_back"
        assert gateway.canary.state.fraction == 0.0
