from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import validate

from llm_lab.config import MODEL_DIR, CONTRACTS_DIR
from llm_lab.eval import write_eval_report

MODEL_CARD_SCHEMA = CONTRACTS_DIR / "model_card.schema.json"
EVAL_REPORT_SCHEMA = CONTRACTS_DIR / "eval_report.schema.json"


def build_model_card() -> dict:
    return {
        "model_name": "local-llama-gguf",
        "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Local llama.cpp GGUF model for LLMOps MVP.",
        "license": "unknown",
        "quantization": "Q4_K_M",
        "parameters": 7000000000,
        "tags": ["local", "gguf", "llama.cpp"],
    }


def ensure_placeholder_model(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "PLACEHOLDER: add a GGUF model file named model.gguf to this directory.",
        encoding="utf-8",
    )


def validate_schema(data: dict, schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(instance=data, schema=schema)


def release() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_card = build_model_card()
    eval_report = write_eval_report(MODEL_DIR / "eval_report.json")

    validate_schema(model_card, MODEL_CARD_SCHEMA)
    validate_schema(eval_report, EVAL_REPORT_SCHEMA)

    (MODEL_DIR / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )
    ensure_placeholder_model(MODEL_DIR / "model.gguf")


if __name__ == "__main__":
    release()
    print(f"Release artifacts written to {MODEL_DIR}")
