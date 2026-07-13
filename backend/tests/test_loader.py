import fitz  # pymupdf
import pytest
from app.ingest.loader import load_pdf, EmptyPdfError


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_pdf_returns_pages_with_text(tmp_path):
    p = tmp_path / "a.pdf"
    _make_pdf(p, ["Pagina uno testo", "Pagina due testo"])
    pages = load_pdf(str(p))
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "uno" in pages[0][1]


def test_load_empty_pdf_raises(tmp_path):
    p = tmp_path / "empty.pdf"
    _make_pdf(p, ["", ""])
    with pytest.raises(EmptyPdfError):
        load_pdf(str(p))
