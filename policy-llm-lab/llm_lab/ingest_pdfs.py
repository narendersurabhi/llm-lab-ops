from __future__ import annotations

import re
from pathlib import Path
import json

from pypdf import PdfReader

from llm_lab.config import INGEST_DIR, PDF_DIR

WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = _clean_text(page_text)
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def ingest_pdfs(pdf_dir: Path = PDF_DIR, output_dir: Path = INGEST_DIR / "pdfs") -> int:
    if not pdf_dir.exists():
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = INGEST_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "pdf_ingest_report.json"
    report: dict[str, list[str]] = {"ingested": [], "skipped_empty": [], "errors": []}
    count = 0
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            text = _extract_pdf_text(pdf_path)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{pdf_path.name}: {exc}")
            continue
        if not text:
            report["skipped_empty"].append(pdf_path.name)
            continue
        out_path = output_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        report["ingested"].append(pdf_path.name)
        count += 1
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return count


def main() -> None:
    count = ingest_pdfs()
    if count == 0:
        print(f"No PDFs found in {PDF_DIR}.")
    else:
        print(f"Ingested {count} PDFs into {INGEST_DIR / 'pdfs'}.")
        print(f"Report: {INGEST_DIR / 'pdf_ingest_report.json'}")


if __name__ == "__main__":
    main()
