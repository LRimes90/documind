import pytest

pytestmark = pytest.mark.slow

import fitz
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_indexes_chunks(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p, ["Il gatto dorme sul divano.", "Il cane corre nel parco."])
    emb = Embedder()
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=emb.dense_dim)
    info = ingest_pdf(str(p), "doc.pdf", emb, store)
    assert info.pages == 2
    assert info.n_chunks >= 2
    assert store.count() == info.n_chunks
