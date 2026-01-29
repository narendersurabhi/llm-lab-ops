from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib
from jsonschema import validate

from llm_lab.config import CONTRACTS_DIR, DATA_DIR, INDEX_PATH, MODEL_DIR, RELEASE_DIR
from llm_lab.eval import write_eval_report

REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT_PATH = REPO_ROOT / "policy-llm-lab" / "pyproject.toml"

MODEL_CARD_SCHEMA = CONTRACTS_DIR / "model_card.schema.json"
EVAL_REPORT_SCHEMA = CONTRACTS_DIR / "eval_report.schema.json"
MANIFEST_SCHEMA = CONTRACTS_DIR / "manifest.schema.json"

RELEASE_LAYOUT = {
    "model": ["model_card.json", "eval_report.json", "model.gguf"],
    "index": ["index.sqlite"],
    "sbom": "sbom.json",
    "checksums": "checksums.json",
    "attestation": "attestation.json",
    "changelog": "CHANGELOG.md",
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_dependencies() -> list[dict[str, Any]]:
    if not PYPROJECT_PATH.exists():
        return []
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    return [{"name": str(dep)} for dep in deps]


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


def build_manifest(release_dir: Path, fingerprints: dict[str, str]) -> dict[str, Any]:
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
            "sbom": {"sbom_json": "sbom.json"},
            "checksums": {"checksums_json": "checksums.json"},
            "attestation": {"attestation_json": "attestation.json"},
            "changelog": {"changelog_md": "CHANGELOG.md"},
        },
        "fingerprints": fingerprints,
        "notes": "Fingerprints cover model/index/sbom/changelog artifacts.",
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


def build_checksums(paths: list[Path]) -> dict[str, str]:
    return {str(path.name): _sha256(path) for path in paths}


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
    index_path: Path = INDEX_PATH,
    contracts_dir: Path = CONTRACTS_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    previous_manifest = None
    previous_manifest_path = output_dir / RELEASE_LAYOUT["manifest"]
    if previous_manifest_path.exists():
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))

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

    sbom_path = output_dir / RELEASE_LAYOUT["sbom"]
    sbom_files = [
        release_model_dir / "model_card.json",
        release_model_dir / "eval_report.json",
        release_model_dir / "model.gguf",
        release_index_dir / "index.sqlite",
    ]
    sbom = build_sbom(output_dir, sbom_files)
    sbom_path.write_text(json.dumps(sbom, indent=2), encoding="utf-8")

    fingerprints = {
        "model/model_card.json": _sha256(release_model_dir / "model_card.json"),
        "model/eval_report.json": _sha256(release_model_dir / "eval_report.json"),
        "model/model.gguf": _sha256(release_model_dir / "model.gguf"),
        "index/index.sqlite": _sha256(release_index_dir / "index.sqlite"),
        "sbom.json": _sha256(sbom_path),
    }

    manifest = build_manifest(output_dir, fingerprints)

    changelog_path = output_dir / RELEASE_LAYOUT["changelog"]
    build_changelog(changelog_path, previous_manifest, manifest)
    fingerprints["CHANGELOG.md"] = _sha256(changelog_path)

    manifest = build_manifest(output_dir, fingerprints)
    validate_schema(manifest, contracts_dir / "manifest.schema.json")
    manifest_path = output_dir / RELEASE_LAYOUT["manifest"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    checksums_path = output_dir / RELEASE_LAYOUT["checksums"]
    checksum_targets = [
        manifest_path,
        sbom_path,
        changelog_path,
        release_model_dir / "model_card.json",
        release_model_dir / "eval_report.json",
        release_model_dir / "model.gguf",
        release_index_dir / "index.sqlite",
    ]
    checksums = build_checksums(checksum_targets)
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")

    attestation_path = output_dir / RELEASE_LAYOUT["attestation"]
    attestation = build_attestation(manifest_path, checksums_path)
    attestation_path.write_text(json.dumps(attestation, indent=2), encoding="utf-8")

    return output_dir


def main() -> None:
    release_dir = build_release()
    print(f"Release bundle written to {release_dir}")


if __name__ == "__main__":
    main()
