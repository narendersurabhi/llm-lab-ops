from __future__ import annotations

from llm_lab.config import DATA_DIR


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit("Sample docs directory not found.")
    print(f"Ingest stub: found {len(list(DATA_DIR.glob('*')))} files in {DATA_DIR}.")


if __name__ == "__main__":
    main()
