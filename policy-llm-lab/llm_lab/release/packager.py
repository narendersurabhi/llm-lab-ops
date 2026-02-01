from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib
from jsonschema import validate

from llm_lab.config import (
    CONTRACTS_DIR,
    DATA_DIR,
    EVAL_DIR,
    INDEX_PATH,
    MODEL_DIR,
    RELEASE_DIR,
)
from llm_lab.eval import write_eval_report

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_PATH = REPO_ROOT / "policy-llm-lab" / "pyproject.toml"

MODEL_CARD_SCHEMA = CONTRACTS_DIR / "model_card.schema.json"
EVAL_REPORT_SCHEMA = CONTRACTS_DIR / "eval_report.schema.json"
MANIFEST_SCHEMA = CONTRACTS_DIR / "manifest.schema.json"

RELEASE_LAYOUT = {
    "manifest": "manifest.json",
    "model": ["model_card.json", "model.gguf"],
    "index": ["index.sqlite"],
    "eval": ["eval_report.json"],
    "contracts": [
        "manifest.schema.json",
        "model_card.schema.json",
        "eval_report.schema.json",
    ],
    "meta": ["sbom.json", "checksums.json", "attestation.json", "CHANGELOG.md"],
}

RELEASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
SEMVER_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_sha() -> str:
    env_sha = os.getenv("GIT_SHA")
    if env_sha:
        return env_sha
    if not (REPO_ROOT / ".git").exists():
        return "0000000"
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return "0000000"


def _release_id() -> str:
    env_value = os.getenv("RELEASE_ID", "").strip().lower()
    if env_value and RELEASE_ID_PATTERN.fullmatch(env_value):
        return env_value
    sha = _git_sha()
    candidate = f"dev-{sha[:7]}"
    if RELEASE_ID_PATTERN.fullmatch(candidate):
        return candidate
    return "local-dev"


def _bundle_version() -> str:
    env_value = os.getenv("BUNDLE_VERSION", "").strip()
    if env_value and SEMVER_PATTERN.fullmatch(env_value):
        return env_value
    return "0.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_value(value: str) -> dict[str, str]:
    return {"algo": "sha256", "value": value}


def _fingerprint_file(path: Path) -> dict[str, str]:
    return _fingerprint_value(_sha256(path))


def _fingerprint_directory(path: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    if path.exists():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        for file_path in files:
            rel = file_path.relative_to(path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return _fingerprint_value(digest.hexdigest())


def _fingerprint_bundle(fingerprints: dict[str, dict[str, str]]) -> dict[str, str]:
    digest = hashlib.sha256()
    for key in sorted(fingerprints):
        if key == "bundle":
            continue
        value = fingerprints[key]["value"]
        digest.update(f"{key}:{value}\n".encode("utf-8"))
    return _fingerprint_value(digest.hexdigest())


def _load_dependencies() -> list[dict[str, Any]]:
    if not PYPROJECT_PATH.exists():
        return []
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    return [{"name": str(dep)} for dep in deps]


def build_model_card(
    release_id: str,
    bundle_version: str,
    dataset_fingerprint: dict[str, str],
    eval_report_path: str,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    training_hash = hashlib.sha256(b"local-dev-training").hexdigest()
    tuning_method = "sft_lora" if adapter_path else "none"
    runtime_compatibility = ["mlx"] if adapter_path else ["llama.cpp"]
    model_name = "local-mlx-lora" if adapter_path else "local-llama-gguf"
    base_model = (
        {"name": "Qwen/Qwen2.5-3B-Instruct"}
        if adapter_path
        else {"name": "llama", "version": "2-7b"}
    )
    quantization = (
        {"format": "fp16", "bits": 16} if adapter_path else {"format": "Q4_K_M", "bits": 4, "group_size": 32}
    )
    parameters = 3085000000 if adapter_path else 7000000000

    model_card = {
        "release_id": release_id,
        "bundle_version": bundle_version,
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "model_name": model_name,
        "base_model": base_model,
        "tuning_method": tuning_method,
        "training_config_hash": training_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "runtime_compatibility": runtime_compatibility,
        "quantization": quantization,
        "parameters": parameters,
        "license": "unknown",
        "tags": ["local", "gguf", "llama.cpp"],
        "intended_use": "Local agentic RAG evaluation and release bundle demo.",
        "limitations": "Demo-only model card; not tuned for production use.",
        "eval_report_path": eval_report_path,
    }
    if adapter_path:
        model_card["adapter_path"] = adapter_path
        model_card["tags"] = ["local", "mlx", "lora"]
    return model_card


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


def build_manifest(
    release_id: str,
    bundle_version: str,
    fingerprints: dict[str, dict[str, str]],
    include_meta: bool = True,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {
        "model": {
            "model_card": "model/model_card.json",
            "model_gguf": "model/model.gguf",
        },
        "index": {"index_sqlite": "index/index.sqlite"},
        "eval": {"eval_report": "eval/eval_report.json"},
        "contracts": {
            "manifest_schema": "contracts/manifest.schema.json",
            "model_card_schema": "contracts/model_card.schema.json",
            "eval_report_schema": "contracts/eval_report.schema.json",
        },
    }
    if adapter_path:
        artifacts["model"]["adapter"] = adapter_path
    if include_meta:
        artifacts["meta"] = {
            "sbom": {"sbom_json": "meta/sbom.json"},
            "checksums": {"checksums_json": "meta/checksums.json"},
            "attestation": {"attestation_json": "meta/attestation.json"},
            "changelog": {"changelog_md": "meta/CHANGELOG.md"},
        }
    return {
        "release_id": release_id,
        "bundle_version": bundle_version,
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "artifacts": artifacts,
        "fingerprints": fingerprints,
        "notes": "Fingerprints cover dataset, model, index, eval, and bundle.",
    }


def build_sbom(release_dir: Path, files: list[Path]) -> dict[str, Any]:
    file_entries = []
    for path in files:
        rel = path.relative_to(release_dir)
        file_entries.append(
            {
                "path": str(rel),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "generator": "llm_lab.release.packager",
        "python": sys.version.split()[0],
        "packages": _load_dependencies(),
        "files": file_entries,
    }


def build_checksums(paths: list[Path], base_dir: Path) -> dict[str, str]:
    return {str(path.relative_to(base_dir)): _sha256(path) for path in paths}


def build_attestation(manifest_path: Path, checksums_path: Path) -> dict[str, Any]:
    return {
        "created_at": _now_iso(),
        "git_sha": _git_sha(),
        "algorithm": "sha256",
        "manifest_sha256": _sha256(manifest_path),
        "checksums_sha256": _sha256(checksums_path),
        "signer": "local-dev",
        "notes": "Checksum-based attestation for local demo purposes.",
    }


def build_changelog(
    changelog_path: Path,
    previous_manifest: dict[str, Any] | None,
    current_manifest: dict[str, Any],
) -> None:
    created_at = current_manifest.get("created_at", _now_iso())
    bundle_version = current_manifest.get("bundle_version", "unknown")
    current_fp = current_manifest.get("fingerprints", {})
    lines = [f"## {bundle_version} - {created_at}"]

    if not previous_manifest:
        lines.append("- Initial release")
    else:
        prev_fp = previous_manifest.get("fingerprints", {})
        added = sorted(set(current_fp) - set(prev_fp))
        removed = sorted(set(prev_fp) - set(current_fp))
        changed = sorted(
            key for key in current_fp if key in prev_fp and current_fp[key] != prev_fp[key]
        )
        if added:
            lines.append(f"- Added: {', '.join(added)}")
        if removed:
            lines.append(f"- Removed: {', '.join(removed)}")
        if changed:
            lines.append(f"- Changed: {', '.join(changed)}")
        if not added and not removed and not changed:
            lines.append("- No artifact fingerprint changes detected")

    entry = "\n".join(lines) + "\n"
    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
        changelog_path.write_text(entry + "\n" + existing, encoding="utf-8")
    else:
        changelog_path.write_text(entry, encoding="utf-8")


def build_release(
    output_dir: Path = RELEASE_DIR,
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
    eval_dir: Path = EVAL_DIR,
    index_path: Path = INDEX_PATH,
    contracts_dir: Path = CONTRACTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest = None
    previous_manifest_path = output_dir / RELEASE_LAYOUT["manifest"]
    if previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))

    release_id = _release_id()
    bundle_version = _bundle_version()
    dataset_fingerprint = _fingerprint_directory(data_dir)

    adapter_candidates = [
        REPO_ROOT / "policy-llm-lab" / "adapters",
        REPO_ROOT / "policy-llm-lab" / "train" / "outputs" / "adapters" / "mlx",
    ]
    adapter_dir = next((p for p in adapter_candidates if p.exists()), None)
    adapter_source = None
    adapter_config = None
    if adapter_dir:
        candidate_weights = adapter_dir / "adapters.safetensors"
        candidate_config = adapter_dir / "adapter_config.json"
        if candidate_weights.exists() and candidate_config.exists():
            adapter_source = candidate_weights
            adapter_config = candidate_config
    adapter_rel_path = "model/adapter" if adapter_source else None

    model_card = build_model_card(
        release_id=release_id,
        bundle_version=bundle_version,
        dataset_fingerprint=dataset_fingerprint,
        eval_report_path="eval/eval_report.json",
        adapter_path=adapter_rel_path,
    )
    model_card_path = model_dir / "model_card.json"
    model_card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    baseline_release_id = (
        previous_manifest.get("release_id", "none") if previous_manifest else "none"
    )
    eval_report_path = eval_dir / "eval_report.json"
    eval_report = write_eval_report(
        eval_report_path,
        data_dir=data_dir,
        index_path=index_path,
        release_id=release_id,
        baseline_release_id=baseline_release_id,
    )

    ensure_placeholder_model(model_dir / "model.gguf")
    ensure_placeholder_index(index_path)

    validate_schema(model_card, contracts_dir / "model_card.schema.json")
    validate_schema(eval_report, contracts_dir / "eval_report.schema.json")

    release_model_dir = output_dir / "model"
    release_index_dir = output_dir / "index"
    release_eval_dir = output_dir / "eval"
    release_contracts_dir = output_dir / "contracts"
    release_meta_dir = output_dir / "meta"
    release_model_dir.mkdir(parents=True, exist_ok=True)
    release_index_dir.mkdir(parents=True, exist_ok=True)
    release_eval_dir.mkdir(parents=True, exist_ok=True)
    release_contracts_dir.mkdir(parents=True, exist_ok=True)
    release_meta_dir.mkdir(parents=True, exist_ok=True)

    for artifact in [model_card_path, model_dir / "model.gguf"]:
        shutil.copy2(artifact, release_model_dir / artifact.name)
    if adapter_source and adapter_config:
        adapter_release_dir = release_model_dir / "adapter"
        adapter_release_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(adapter_source, adapter_release_dir / "adapters.safetensors")
        shutil.copy2(adapter_config, adapter_release_dir / "adapter_config.json")

    shutil.copy2(index_path, release_index_dir / "index.sqlite")
    shutil.copy2(eval_report_path, release_eval_dir / "eval_report.json")
    for schema_name in RELEASE_LAYOUT["contracts"]:
        shutil.copy2(contracts_dir / schema_name, release_contracts_dir / schema_name)

    sbom_path = release_meta_dir / "sbom.json"
    sbom_files = [
        release_model_dir / "model_card.json",
        release_model_dir / "model.gguf",
        release_eval_dir / "eval_report.json",
        release_index_dir / "index.sqlite",
        release_contracts_dir / "manifest.schema.json",
        release_contracts_dir / "model_card.schema.json",
        release_contracts_dir / "eval_report.schema.json",
    ]
    if adapter_source:
        sbom_files.append(release_model_dir / "adapter" / "adapters.safetensors")
        sbom_files.append(release_model_dir / "adapter" / "adapter_config.json")
    sbom = build_sbom(output_dir, sbom_files)
    sbom_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")

    fingerprints = {
        "dataset": dataset_fingerprint,
        "index": _fingerprint_file(release_index_dir / "index.sqlite"),
        "model": _fingerprint_file(release_model_dir / "model.gguf"),
        "eval": _fingerprint_file(release_eval_dir / "eval_report.json"),
    }
    if adapter_source:
        fingerprints["adapter"] = _fingerprint_directory(release_model_dir / "adapter")
    fingerprints["bundle"] = _fingerprint_bundle(fingerprints)

    manifest = build_manifest(
        release_id=release_id,
        bundle_version=bundle_version,
        fingerprints=fingerprints,
        include_meta=True,
        adapter_path=adapter_rel_path,
    )
    validate_schema(manifest, contracts_dir / "manifest.schema.json")
    manifest_path = output_dir / RELEASE_LAYOUT["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    changelog_path = release_meta_dir / "CHANGELOG.md"
    build_changelog(changelog_path, previous_manifest, manifest)

    checksums_path = release_meta_dir / "checksums.json"
    checksum_targets = [
        manifest_path,
        sbom_path,
        changelog_path,
        release_model_dir / "model_card.json",
        release_model_dir / "model.gguf",
        release_eval_dir / "eval_report.json",
        release_index_dir / "index.sqlite",
        release_contracts_dir / "manifest.schema.json",
        release_contracts_dir / "model_card.schema.json",
        release_contracts_dir / "eval_report.schema.json",
    ]
    if adapter_source:
        checksum_targets.append(release_model_dir / "adapter" / "adapters.safetensors")
        checksum_targets.append(release_model_dir / "adapter" / "adapter_config.json")
    checksums = build_checksums(checksum_targets, output_dir)
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    attestation_path = release_meta_dir / "attestation.json"
    attestation = build_attestation(manifest_path, checksums_path)
    attestation_path.write_text(json.dumps(attestation, indent=2), encoding="utf-8")

    return output_dir


def main() -> None:
    release_dir_env = os.getenv("RELEASE_DIR")
    output_dir = Path(release_dir_env) if release_dir_env else RELEASE_DIR
    release_dir = build_release(output_dir=output_dir)
    print(f"Release bundle written to {release_dir}")


if __name__ == "__main__":
    main()
