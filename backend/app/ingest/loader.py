"""Estrae testo per-pagina da un PDF (pymupdf)."""
import fitz


class EmptyPdfError(Exception):
    """Nessun testo estraibile (PDF vuoto o scannerizzato senza OCR)."""


def load_pdf(path: str) -> list[tuple[int, str]]:
    doc = fitz.open(path)
    try:
        pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    finally:
        doc.close()
    if not any(text for _, text in pages):
        raise EmptyPdfError(f"Nessun testo estraibile da {path} (PDF scannerizzato?)")
    return pages
