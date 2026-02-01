from __future__ import annotations

import json
import os
from typing import Any

import asyncio
import time
import httpx
import streamlit as st

DEFAULT_BASE_URL = os.getenv("OPS_BASE_URL", "http://localhost:8002")
DEFAULT_RELEASES_DIR = os.getenv("RELEASES_DIR", "")


async def _call_api_async(
    base_url: str, messages: list[dict[str, Any]], model: str, stream: bool
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": messages, "stream": stream},
        )
        resp.raise_for_status()
        return resp.json()


def _stream_api(
    base_url: str, messages: list[dict[str, Any]], model: str
):
    async def _runner():
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={"model": model, "messages": messages, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.replace("data: ", "").strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except Exception:  # noqa: BLE001
                        continue
                    delta = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta

    # Consume async generator into a blocking generator for Streamlit.
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        agen = _runner()
        while True:
            try:
                chunk = loop.run_until_complete(agen.__anext__())
                yield chunk
            except StopAsyncIteration:
                break
    finally:
        loop.close()


def _parse_citations(content: str) -> list[str]:
    if "Citations:" not in content:
        return []
    _, citations = content.split("Citations:", 1)
    return [line.strip("- ").strip() for line in citations.strip().splitlines() if line]


def _stream_text(text: str, chunk_size: int = 32, delay_s: float = 0.01):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        time.sleep(delay_s)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _discover_releases(releases_dir: str) -> list[dict[str, Any]]:
    if not releases_dir:
        return []
    if not os.path.isdir(releases_dir):
        return []
    rows: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(releases_dir)):
        path = os.path.join(releases_dir, entry)
        if not os.path.isdir(path):
            continue
        manifest_path = os.path.join(path, "manifest.json")
        model_card_path = os.path.join(path, "model", "model_card.json")
        eval_path = os.path.join(path, "eval", "eval_report.json")
        if not os.path.exists(manifest_path):
            continue
        row: dict[str, Any] = {"release_dir": path, "release_id": entry}
        try:
            manifest = _load_json(manifest_path)
            row["release_id"] = manifest.get("release_id", entry)
            row["bundle_version"] = manifest.get("bundle_version", "")
        except Exception:  # noqa: BLE001
            pass
        try:
            model_card = _load_json(model_card_path)
            row["model_name"] = model_card.get("model_name")
            row["tuning_method"] = model_card.get("tuning_method")
            row["runtime"] = ", ".join(model_card.get("runtime_compatibility", []))
            quant = model_card.get("quantization", {})
            if quant:
                row["quantization"] = f"{quant.get('format')} ({quant.get('bits')})"
            row["parameters"] = model_card.get("parameters")
        except Exception:  # noqa: BLE001
            pass
        try:
            eval_report = _load_json(eval_path)
            row["eval_pass"] = eval_report.get("pass")
            metrics = eval_report.get("metrics", {})
            row["p95_ms"] = metrics.get("latency_p95_ms")
            row["ttft_ms"] = metrics.get("ttft_ms")
            row["error_rate"] = metrics.get("error_rate")
            row["retrieval_hit_rate"] = metrics.get("retrieval_hit_rate")
            row["citation_coverage"] = metrics.get("citation_coverage")
        except Exception:  # noqa: BLE001
            pass
        rows.append(row)
    return rows


st.set_page_config(page_title="Policy LLM Ops", layout="wide")
st.title("Policy LLM Ops — RAG Demo")

tab_chat, tab_models = st.tabs(["Chat", "Models"])

with st.sidebar:
    base_url = st.text_input("OPS_BASE_URL", value=DEFAULT_BASE_URL)
    model_name = st.text_input("Model", value="local-llama-gguf")
    show_raw = st.checkbox("Show raw response", value=False)
    stream_output = st.checkbox("Stream output", value=True)

with tab_models:
    st.subheader("Available Releases")
    releases_dir = st.text_input("Releases directory", value=DEFAULT_RELEASES_DIR)
    rows = _discover_releases(releases_dir)
    if not rows:
        st.info("No releases found. Build one with `make pipeline-mlx` or `make serve-latest-mlx`.")
    else:
        st.dataframe(rows, use_container_width=True)

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your docs…")
    if prompt:
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                payload = None
                if stream_output:
                    content = st.write_stream(
                        _stream_api(base_url, st.session_state.messages, model_name)
                    )
                else:
                    payload = asyncio.run(
                        _call_api_async(
                            base_url, st.session_state.messages, model_name, False
                        )
                    )
                    content = payload["choices"][0]["message"]["content"]
                    st.markdown(content)

                citations = _parse_citations(content)
                if citations:
                    st.caption("Citations")
                    for item in citations:
                        st.write(f"- {item}")

                if payload is not None:
                    usage = payload.get("usage", {})
                    if usage:
                        st.caption(
                            f"tokens_in={usage.get('prompt_tokens', 0)} "
                            f"tokens_out={usage.get('completion_tokens', 0)}"
                        )

                if show_raw and not stream_output:
                    st.code(json.dumps(payload, indent=2), language="json")

                st.session_state.messages.append({"role": "assistant", "content": content})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
