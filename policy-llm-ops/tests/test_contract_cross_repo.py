from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_lab.release.packager import build_release
from llm_ops.release_manager import ReleaseManager


@pytest.mark.contract
def test_ops_consumes_lab_release_bundle(sample_docs: Path, tmp_path: Path, monkeypatch) -> None:
    contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
    release_path_env = os.getenv("RELEASE_PATH")

    if release_path_env:
        release_dir = Path(release_path_env)
    else:
        release_id = "contract-bundle"
        release_dir = tmp_path / "dist" / release_id
        model_dir = tmp_path / "artifacts" / "model"
        eval_dir = tmp_path / "artifacts" / "eval"
        index_path = tmp_path / "artifacts" / "index.sqlite"

        monkeypatch.setenv("RELEASE_ID", release_id)
        build_release(
            output_dir=release_dir,
            data_dir=sample_docs,
            model_dir=model_dir,
            eval_dir=eval_dir,
            index_path=index_path,
            contracts_dir=contracts_dir,
        )

    manager = ReleaseManager(contracts_dir=contracts_dir, base_dir=release_dir.parent)
    bundle = manager.load(release_path=release_dir)
    assert bundle.allowed is True

    with sqlite3.connect(f"file:{bundle.index_path}?mode=ro", uri=True) as conn:
        conn.execute("SELECT COUNT(*) FROM chunks").fetchone()

    monkeypatch.setenv("RELEASE_PATH", str(release_dir))
    monkeypatch.setenv("CANARY_ENABLED", "false")
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    import llm_ops.config as config

    importlib.reload(config)
    import llm_ops.gateway as gateway

    importlib.reload(gateway)

    with TestClient(gateway.app) as client:
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
        assert "Citations:" in content
