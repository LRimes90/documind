"""Orchestrazione ingest: PDF → chunk → embeddings → store."""
import uuid
from app.config import settings
from app.models import DocumentInfo
from app.ingest.loader import load_pdf
from app.ingest.chunker import chunk_pages
from app.ingest.embeddings import Embedder
from app.store import VectorStore


def ingest_pdf(
    path: str, doc_name: str, embedder: Embedder, store: VectorStore
) -> DocumentInfo:
    pages = load_pdf(path)
    doc_id = uuid.uuid4().hex[:8]
    chunks = chunk_pages(
        pages,
        doc_id=doc_id,
        doc_name=doc_name,
        max_words=settings.chunk_words,
        overlap=settings.chunk_overlap,
    )
    texts = [c.text for c in chunks]
    dense = embedder.embed_dense(texts)
    sparse = embedder.embed_sparse(texts)
    store.upsert(chunks, dense, sparse)
    return DocumentInfo(
        doc_id=doc_id, doc_name=doc_name, n_chunks=len(chunks), pages=len(pages)
    )
