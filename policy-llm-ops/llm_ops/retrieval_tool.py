from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm_ops.config import settings
from llm_ops.retrieval_sqlite import SQLiteFTS5RetrievalTool
from llm_ops.tools import RetrievalResult, RetrievalTool


class RetrievalToolProtocol(Protocol):
    async def run(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        ...

    async def close(self) -> None:
        ...


@dataclass
class RetrievalToolFactory:
    def create(self) -> RetrievalToolProtocol:
        if settings.retrieval_mode == "sqlite":
            db_path = settings.retrieval_db_path
            if db_path is None:
                raise ValueError("RETRIEVAL_DB_PATH must be set for sqlite retrieval")
            return SQLiteFTS5RetrievalTool(Path(db_path))
        return RetrievalTool()
