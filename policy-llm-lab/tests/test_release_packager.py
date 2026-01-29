from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from llm_lab.config import CONTRACTS_DIR, DATA_DIR
from llm_lab.release.packager import RELEASE_LAYOUT, build_release


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contract_examples_validate() -> None:
    examples_dir = CONTRACTS_DIR / "examples"
    manifest = _load_json(examples_dir / "manifest.json")
    model_card = _load_json(examples_dir / "model_card.json")
    eval_report = _load_json(examples_dir / "eval_report.json")

    validate(instance=manifest, schema=_load_json(CONTRACTS_DIR / "manifest.schema.json"))
    validate(instance=model_card, schema=_load_json(CONTRACTS_DIR / "model_card.schema.json"))
    validate(instance=eval_report, schema=_load_json(CONTRACTS_DIR / "eval_report.schema.json"))


def test_release_bundle_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "release"
    model_dir = tmp_path / "artifacts" / "model"
    eval_dir = tmp_path / "artifacts" / "eval"
    index_path = tmp_path / "artifacts" / "index.sqlite"

    build_release(
        output_dir=output_dir,
        data_dir=DATA_DIR,
        model_dir=model_dir,
        eval_dir=eval_dir,
        index_path=index_path,
        contracts_dir=CONTRACTS_DIR,
    )

    manifest_path = output_dir / RELEASE_LAYOUT["manifest"]
    assert manifest_path.exists()

    for name in RELEASE_LAYOUT["model"]:
        assert (output_dir / "model" / name).exists()

    for name in RELEASE_LAYOUT["index"]:
        assert (output_dir / "index" / name).exists()

    for name in RELEASE_LAYOUT["eval"]:
        assert (output_dir / "eval" / name).exists()

    for name in RELEASE_LAYOUT["contracts"]:
        assert (output_dir / "contracts" / name).exists()

    for name in RELEASE_LAYOUT["meta"]:
        assert (output_dir / "meta" / name).exists()

    sbom_path = output_dir / "meta" / "sbom.json"
    checksums_path = output_dir / "meta" / "checksums.json"
    attestation_path = output_dir / "meta" / "attestation.json"
    changelog_path = output_dir / "meta" / "CHANGELOG.md"

    assert sbom_path.exists()
    assert checksums_path.exists()
    assert attestation_path.exists()
    assert changelog_path.exists()

    checksums = _load_json(checksums_path)
    assert RELEASE_LAYOUT["manifest"] in checksums
    assert "meta/sbom.json" in checksums

    attestation = _load_json(attestation_path)
    assert attestation.get("manifest_sha256")
    assert attestation.get("checksums_sha256")
