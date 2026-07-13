"""Eval retrieval: Hit@k, MRR, Recall@k su naive / hybrid / hybrid+rerank."""
import tempfile
from pathlib import Path
from app.config import settings
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.hybrid import hybrid_candidates
from eval.dataset import GOLD

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"
K = 5


def hit_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> int:
    return int(any(p in gold_pages for p in retrieved_pages[:k]))


def mrr(retrieved_pages: list[int], gold_pages: list[int]) -> float:
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_pages:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    found = {p for p in retrieved_pages[:k] if p in gold_pages}
    return len(found) / len(gold_pages) if gold_pages else 0.0


def _keys_for(ids: list[str], store: VectorStore) -> list[str]:
    # chiave doc_name:page — evita collisioni tra pagine omonime di doc diversi
    return [f"{store.get_chunk(cid).doc_name}:{store.get_chunk(cid).page}" for cid in ids]


def run() -> None:
    emb = Embedder()
    reranker = Reranker()
    tmp = tempfile.mkdtemp()
    store = VectorStore(path=tmp, collection="eval", dense_dim=emb.dense_dim)
    for pdf in SAMPLE_DIR.glob("*.pdf"):
        ingest_pdf(str(pdf), pdf.name, emb, store)

    configs = ["naive", "hybrid", "hybrid+rerank"]
    _provider = None
    try:
        from app.generation.llm import get_provider
        from app.retrieval.query_rewrite import expand_query

        _provider = get_provider()
        configs = configs + ["hybrid+rerank+hyde"]
    except ValueError:
        print("[eval] Provider LLM non disponibile: 4a config (HyDE) saltata.")
    agg = {c: {"hit1": 0.0, "hit": 0.0, "mrr": 0.0, "recall": 0.0} for c in configs}

    for item in GOLD:
        dvec = emb.embed_query_dense(item.question)
        svec = emb.embed_query_sparse(item.question)
        naive_ids = store.search_dense(dvec, top_k=settings.top_k_dense)
        hybrid_ids = hybrid_candidates(store, dvec, svec, top_k=settings.top_k_dense)
        pairs = [(cid, store.get_chunk(cid).text) for cid in hybrid_ids]
        rerank_ids = reranker.rerank(item.question, pairs, top_n=settings.top_n_rerank)

        gold = [f"{item.doc_name}:{p}" for p in item.pages]
        base = zip(["naive", "hybrid", "hybrid+rerank"], [naive_ids, hybrid_ids, rerank_ids])
        for cfg, ids in base:
            keys = _keys_for(ids, store)
            agg[cfg]["hit1"] += hit_at_k(keys, gold, 1)
            agg[cfg]["hit"] += hit_at_k(keys, gold, K)
            agg[cfg]["mrr"] += mrr(keys, gold)
            agg[cfg]["recall"] += recall_at_k(keys, gold, K)

        if _provider is not None:
            exp = expand_query(item.question, _provider)
            hyde_ids = hybrid_candidates(
                store, emb.embed_query_dense(exp), svec, top_k=settings.top_k_dense
            )
            pairs2 = [(cid, store.get_chunk(cid).text) for cid in hyde_ids]
            hyde_rr = reranker.rerank(item.question, pairs2, top_n=settings.top_n_rerank)
            keys = _keys_for(hyde_rr, store)
            agg["hybrid+rerank+hyde"]["hit1"] += hit_at_k(keys, gold, 1)
            agg["hybrid+rerank+hyde"]["hit"] += hit_at_k(keys, gold, K)
            agg["hybrid+rerank+hyde"]["mrr"] += mrr(keys, gold)
            agg["hybrid+rerank+hyde"]["recall"] += recall_at_k(keys, gold, K)

    n = len(GOLD) or 1
    n_pdf = len(list(SAMPLE_DIR.glob("*.pdf")))
    lines = [
        "# Eval results — DocuMind retrieval\n",
        f"Dataset: {len(GOLD)} domande · sample_docs: {n_pdf} PDF\n",
        "| Config | Hit@1 | Hit@5 | MRR | Recall@5 |",
        "|--------|-------|-------|-----|----------|",
    ]
    for cfg in configs:
        m = agg[cfg]
        lines.append(
            f"| {cfg} | {m['hit1']/n:.2%} | {m['hit']/n:.2%} | {m['mrr']/n:.3f} | {m['recall']/n:.2%} |"
        )
    (Path(__file__).parent / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
