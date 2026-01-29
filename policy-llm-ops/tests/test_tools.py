from __future__ import annotations

from pathlib import Path

import pytest

from llm_ops.release_manager import ReleaseManager
from llm_ops.tools import RetrievalTool
from utils import build_minimal_release_bundle, build_test_index


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retrieval_tool_reads_sqlite(tmp_path: Path) -> None:
    index_path = tmp_path / "index.sqlite"
    build_test_index(index_path)
    tool = RetrievalTool(db_path=index_path)
    results = await tool.run("RAG", top_k=1)
    assert results
    assert results[0].doc_id == "doc"
    await tool.close()


@pytest.mark.unit
def test_release_bundle_schema_validation(tmp_path: Path) -> None:
    contracts_dir = Path(__file__).resolve().parents[2] / "contracts"
    release_dir = tmp_path / "release"
    build_minimal_release_bundle(release_dir, contracts_dir)
    manager = ReleaseManager(contracts_dir=contracts_dir, base_dir=tmp_path)
    bundle = manager.load(release_path=release_dir)
    assert bundle.allowed is True
