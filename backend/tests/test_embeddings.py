import pytest

pytestmark = pytest.mark.slow

from app.ingest.embeddings import Embedder


def test_dense_dim_and_shape():
    emb = Embedder()
    vecs = emb.embed_dense(["ciao mondo", "hello world"])
    assert emb.dense_dim > 0
    assert len(vecs) == 2
    assert len(vecs[0]) == emb.dense_dim


def test_sparse_has_indices_and_values():
    emb = Embedder()
    sv = emb.embed_query_sparse("hello world")
    assert len(sv.indices) == len(sv.values)
    assert len(sv.indices) > 0
