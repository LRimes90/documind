import pytest

pytestmark = pytest.mark.slow

import fitz
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.pipeline import retrieve


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for t in pages_text:
        doc.new_page().insert_text((72, 72), t)
    doc.save(str(path))
    doc.close()


def test_retrieve_returns_relevant_chunk(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p, [
        "La capitale della Francia è Parigi.",
        "Le api producono il miele nelle arnie.",
    ])
    emb = Embedder()
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=emb.dense_dim)
    ingest_pdf(str(p), "doc.pdf", emb, store)
    reranker = Reranker()
    chunks = retrieve("Qual è la capitale della Francia?", emb, store, reranker)
    assert len(chunks) >= 1
    assert "Parigi" in chunks[0].text
