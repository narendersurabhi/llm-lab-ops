from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llm_lab.indexer import IndexBuilder, SQLiteIndex, load_index

app = FastAPI(title="LLM Lab Retrieval API")


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class RetrievalResponse(BaseModel):
    query: str
    results: list[dict]


INDEX: SQLiteIndex | None = None


def ensure_index() -> SQLiteIndex:
    global INDEX
    if INDEX is not None:
        return INDEX
    try:
        INDEX = load_index()
        return INDEX
    except FileNotFoundError:
        builder = IndexBuilder()
        builder.build()
        INDEX = load_index()
        return INDEX


@app.on_event("startup")
async def startup() -> None:
    ensure_index()


@app.on_event("shutdown")
async def shutdown() -> None:
    if INDEX is not None:
        INDEX.close()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(req: RetrievalRequest) -> RetrievalResponse:
    try:
        index = ensure_index()
        results = index.retrieve(req.query, req.top_k)
        return RetrievalResponse(query=req.query, results=results)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
