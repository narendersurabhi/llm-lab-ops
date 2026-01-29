from __future__ import annotations

import argparse
import asyncio
import time

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    return parser.parse_args()


async def worker(client: httpx.AsyncClient, queue: asyncio.Queue[int]) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "local-llama-gguf",
                "messages": [{"role": "user", "content": "Explain RAG"}],
            },
            timeout=10.0,
        )
        queue.task_done()


async def run_loadtest(url: str, total: int, concurrency: int) -> None:
    queue: asyncio.Queue[int | None] = asyncio.Queue()
    for i in range(total):
        queue.put_nowait(i)
    for _ in range(concurrency):
        queue.put_nowait(None)

    async with httpx.AsyncClient(base_url=url.replace("/v1/chat/completions", "")) as client:
        tasks = [asyncio.create_task(worker(client, queue)) for _ in range(concurrency)]
        start = time.perf_counter()
        await queue.join()
        for task in tasks:
            task.cancel()
        elapsed = time.perf_counter() - start
        print(f"Completed {total} requests in {elapsed:.2f}s")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_loadtest(args.url, args.requests, args.concurrency))
