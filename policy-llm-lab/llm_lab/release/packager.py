from __future__ import annotations

import json
import shutil
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import validate

from llm_lab.config import CONTRACTS_DIR, DATA_DIR, INDEX_PATH, MODEL_DIR, RELEASE_DIR
from llm_lab.eval import write_eval_report

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_CARD_SCHEMA = CONTRACTS_DIR / "model_card.schema.json"
EVAL_REPORT_SCHEMA = CONTRACTS_DIR / "eval_report.schema.json"
MANIFEST_SCHEMA = CONTRACTS_DIR / "manifest.schema.json"

RELEASE_LAYOUT = {
    "model": ["model_card.json", "eval_report.json", "model.gguf"],
    "index": ["index.sqlite"],
    "manifest": "manifest.json",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha
    if not (REPO_ROOT / ".git").exists():
        return "unknown"
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_model_card() -> dict[str, Any]:
    return {
        "model_name": "local-llama-gguf",
        "version": "0.1.0",
        "created_at": _now_iso(),
        "description": "Local llama.cpp GGUF model for LLMOps release bundle.",
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


def ensure_placeholder_index(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("PLACEHOLDER: index.sqlite not built yet.", encoding="utf-8")


def validate_schema(data: dict[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validate(instance=data, schema=schema)


def build_manifest(release_dir: Path) -> dict[str, Any]:
    model_dir = release_dir / "model"
    index_dir = release_dir / "index"
    manifest_paths = [
        model_dir / "model_card.json",
        model_dir / "eval_report.json",
        model_dir / "model.gguf",
        index_dir / "index.sqlite",
    ]
    fingerprints = {
        str(path.relative_to(release_dir)): "sha256:pending" for path in manifest_paths
    }
    return {
        "bundle_version": "0.1.0",
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "artifacts": {
            "model": {
                "model_card": "model/model_card.json",
                "eval_report": "model/eval_report.json",
                "model_gguf": "model/model.gguf",
            },
            "index": {"index_sqlite": "index/index.sqlite"},
        },
        "fingerprints": fingerprints,
        "notes": "Fingerprints are placeholders until signing is implemented.",
    }


def build_release(
    output_dir: Path = RELEASE_DIR,
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    index_path: Path = INDEX_PATH,
    contracts_dir: Path = CONTRACTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_card = build_model_card()
    model_card_path = model_dir / "model_card.json"
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    eval_report_path = model_dir / "eval_report.json"
    eval_report = write_eval_report(eval_report_path, data_dir=data_dir, index_path=index_path)

    ensure_placeholder_model(model_dir / "model.gguf")
    ensure_placeholder_index(index_path)

    validate_schema(model_card, contracts_dir / "model_card.schema.json")
    validate_schema(eval_report, contracts_dir / "eval_report.schema.json")

    release_model_dir = output_dir / "model"
    release_index_dir = output_dir / "index"
    release_model_dir.mkdir(parents=True, exist_ok=True)
    release_index_dir.mkdir(parents=True, exist_ok=True)

    for artifact in [model_card_path, eval_report_path, model_dir / "model.gguf"]:
        shutil.copy2(artifact, release_model_dir / artifact.name)

    shutil.copy2(index_path, release_index_dir / "index.sqlite")

    manifest = build_manifest(output_dir)
    validate_schema(manifest, contracts_dir / "manifest.schema.json")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return output_dir


def main() -> None:
    release_dir = build_release()
    print(f"Release bundle written to {release_dir}")


if __name__ == "__main__":
    main()
