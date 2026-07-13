"""Hybrid retrieval: Reciprocal Rank Fusion su ricerche dense + sparse."""
from app.store import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fonde più classifiche (liste di chunk_id ordinate) in una sola lista ordinata.

    Formula RRF: per ogni documento d,
        score(d) = Σ_i  1 / (k + rank_i(d))
    dove rank_i(d) è la posizione 0-based di d nella i-esima classifica.
    Un documento assente da una classifica non riceve contributo per quella.
    Al termine, ordinare i documenti per score decrescente e ritornare la
    lista dei loro chunk_id.

    Perché per rango e non per punteggio grezzo: i punteggi dense (cosine) e
    sparse (BM25) vivono su scale incompatibili; fondere per posizione li rende
    confrontabili.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def hybrid_candidates(
    store: VectorStore, dense_vec, sparse_vec, top_k: int
) -> list[str]:
    dense_hits = store.search_dense(dense_vec, top_k=top_k)
    sparse_hits = store.search_sparse(sparse_vec, top_k=top_k)
    return reciprocal_rank_fusion([dense_hits, sparse_hits])
