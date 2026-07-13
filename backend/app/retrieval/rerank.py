"""Reranking con cross-encoder ONNX (fastembed)."""
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.config import settings


class Reranker:
    def __init__(self) -> None:
        self._model = TextCrossEncoder(model_name=settings.reranker_model)

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_n: int
    ) -> list[str]:
        if not candidates:
            return []
        docs = [text for _, text in candidates]
        scores = list(self._model.rerank(query, docs))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [cid for (cid, _text), _score in ranked[:top_n]]
