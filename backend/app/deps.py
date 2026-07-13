"""Stato applicativo lazy: modelli e store caricati una sola volta."""
from functools import lru_cache
from app.config import settings
from app.ingest.embeddings import Embedder
from app.store import VectorStore
from app.retrieval.rerank import Reranker


class AppState:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore(
            path=settings.qdrant_path,
            collection=settings.collection,
            dense_dim=self.embedder.dense_dim,
        )
        self.reranker = Reranker()


@lru_cache(maxsize=1)
def get_state() -> AppState:
    return AppState()
