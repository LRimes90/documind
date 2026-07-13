import pytest

pytestmark = pytest.mark.slow

from app.retrieval.rerank import Reranker


def test_rerank_puts_relevant_first():
    rr = Reranker()
    query = "Chi ha dipinto la Gioconda?"
    candidates = [
        ("c0", "La ricetta della pasta al pomodoro richiede basilico."),
        ("c1", "La Gioconda fu dipinta da Leonardo da Vinci."),
    ]
    ranked = rr.rerank(query, candidates, top_n=2)
    assert ranked[0] == "c1"
