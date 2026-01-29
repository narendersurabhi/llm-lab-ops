from __future__ import annotations

import sqlite3
from pathlib import Path

from llm_lab.config import DATA_DIR
from llm_lab.indexer import IndexBuilder


def _chunk_ids(index_path: Path) -> list[str]:
    conn = sqlite3.connect(index_path)
    rows = conn.execute("SELECT chunk_id FROM chunks_fts ORDER BY chunk_id").fetchall()
    conn.close()
    return [row[0] for row in rows]


def test_index_reproducible(tmp_path: Path) -> None:
    index_path_a = tmp_path / "index_a.sqlite"
    index_path_b = tmp_path / "index_b.sqlite"

    IndexBuilder(data_dir=DATA_DIR, index_path=index_path_a).build()
    IndexBuilder(data_dir=DATA_DIR, index_path=index_path_b).build()

    assert _chunk_ids(index_path_a) == _chunk_ids(index_path_b)
