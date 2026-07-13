"""Wrapper fastembed: dense + sparse, con lazy loading dei modelli.

Il modello dense (paraphrase-multilingual-MiniLM) NON usa prefissi query/passage
(a differenza dei modelli e5), quindi gli input vengono passati così come sono.
"""
from fastembed import TextEmbedding, SparseTextEmbedding
from app.config import settings


class Embedder:
    def __init__(self) -> None:
        self._dense = TextEmbedding(model_name=settings.dense_model)
        self._sparse = SparseTextEmbedding(model_name=settings.sparse_model)
        # dimensione ricavata dal primo embedding
        probe = next(iter(self._dense.embed(["x"])))
        self.dense_dim = len(probe)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._dense.embed(texts)]

    def embed_sparse(self, texts: list[str]):
        return list(self._sparse.embed(texts))

    def embed_query_dense(self, text: str) -> list[float]:
        return next(iter(self._dense.embed([text]))).tolist()

    def embed_query_sparse(self, text: str):
        return next(iter(self._sparse.embed([text])))
