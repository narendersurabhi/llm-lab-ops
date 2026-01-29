from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import validate

from llm_ops.config import settings

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"


@dataclass(frozen=True)
class ReleaseBundle:
    path: Path
    manifest: dict[str, Any]
    model_card: dict[str, Any]
    eval_report: dict[str, Any]
    model_dir: Path
    index_path: Path
    allowed: bool


class ReleaseManager:
    def __init__(self, contracts_dir: Path | None = None, base_dir: Path | None = None) -> None:
        self.contracts_dir = contracts_dir or CONTRACTS_DIR
        self.base_dir = base_dir or Path(settings.release_base_dir)

    def resolve_path(self, release_path: str | Path | None, release_id: str | None) -> Path:
        if release_path is not None:
            return Path(release_path)
        if release_id is not None:
            return self.base_dir / release_id
        if settings.release_path:
            return Path(settings.release_path)
        if settings.release_id:
            return self.base_dir / settings.release_id
        return self.base_dir

    def load(self, release_path: str | Path | None = None, release_id: str | None = None) -> ReleaseBundle:
        path = self.resolve_path(release_path, release_id)
        manifest_path = path / "manifest.json"
        model_card_path = path / "model" / "model_card.json"
        eval_report_path = path / "model" / "eval_report.json"
        index_path = path / "index" / "index.sqlite"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Release manifest not found at {manifest_path}")
        if not model_card_path.exists():
            raise FileNotFoundError(f"Model card not found at {model_card_path}")
        if not eval_report_path.exists():
            raise FileNotFoundError(f"Eval report not found at {eval_report_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")

        manifest = self._load_json(manifest_path)
        model_card = self._load_json(model_card_path)
        eval_report = self._load_json(eval_report_path)

        self._validate(manifest, self.contracts_dir / "manifest.schema.json")
        self._validate(model_card, self.contracts_dir / "model_card.schema.json")
        self._validate(eval_report, self.contracts_dir / "eval_report.schema.json")

        allowed = bool(eval_report.get("pass"))
        return ReleaseBundle(
            path=path,
            manifest=manifest,
            model_card=model_card,
            eval_report=eval_report,
            model_dir=path / "model",
            index_path=index_path,
            allowed=allowed,
        )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _validate(data: dict[str, Any], schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validate(instance=data, schema=schema)
