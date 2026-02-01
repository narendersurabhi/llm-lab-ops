from __future__ import annotations

from llm_lab.config import DATA_DIR, PDF_DIR, INGEST_DIR
from llm_lab.ingest_pdfs import ingest_pdfs


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit("Sample docs directory not found.")
    print(f"Ingest stub: found {len(list(DATA_DIR.glob('*')))} files in {DATA_DIR}.")
    if PDF_DIR.exists():
        count = ingest_pdfs(PDF_DIR, INGEST_DIR / "pdfs")
        print(f"Ingested {count} PDFs into {INGEST_DIR / 'pdfs'}.")


if __name__ == "__main__":
    main()
