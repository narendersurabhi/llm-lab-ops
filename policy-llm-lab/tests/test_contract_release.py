from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.contract
def test_make_release_bundle_contract(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lab_root = repo_root / "policy-llm-lab"
    release_id = "contract-test"
    output_dir = tmp_path / "dist" / release_id

    env = os.environ.copy()
    env["RELEASE_ID"] = release_id
    env["RELEASE_DIR"] = str(output_dir)
    env["PYTHON"] = sys.executable

    subprocess.run(["make", "release"], cwd=lab_root, env=env, check=True)

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()

    manifest = _load_json(manifest_path)
    _assert_relative_paths(manifest)

    for section, value in manifest["artifacts"].items():
        if section == "meta":
            for entry in value.values():
                for rel_path in entry.values():
                    assert (output_dir / rel_path).exists()
            continue
        for rel_path in value.values():
            assert (output_dir / rel_path).exists()

    index_path = output_dir / "index" / "index.sqlite"
    assert index_path.exists()
    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as conn:
        conn.execute("SELECT COUNT(*) FROM chunks").fetchone()


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _assert_relative_paths(manifest: dict) -> None:
    from pathlib import Path as FsPath

    for section in ("model", "index", "eval", "contracts"):
        for value in manifest["artifacts"][section].values():
            assert not FsPath(value).is_absolute()
    meta = manifest["artifacts"].get("meta", {})
    for entry in meta.values():
        for value in entry.values():
            assert not FsPath(value).is_absolute()
