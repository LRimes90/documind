"""Pipeline: embed query → hybrid (RRF) → rerank → chunk completi."""
from app.config import settings
from app.models import Chunk
from app.ingest.embeddings import Embedder
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.hybrid import hybrid_candidates


def retrieve(
    question: str, embedder: Embedder, store: VectorStore, reranker: Reranker
) -> list[Chunk]:
    dense_vec = embedder.embed_query_dense(question)
    sparse_vec = embedder.embed_query_sparse(question)
    candidate_ids = hybrid_candidates(
        store, dense_vec, sparse_vec, top_k=settings.top_k_dense
    )
    if not candidate_ids:
        return []
    candidates = [(cid, store.get_chunk(cid)) for cid in candidate_ids]
    pairs = [(cid, chunk.text) for cid, chunk in candidates]
    top_ids = reranker.rerank(question, pairs, top_n=settings.top_n_rerank)
    by_id = {cid: chunk for cid, chunk in candidates}
    return [by_id[cid] for cid in top_ids]
