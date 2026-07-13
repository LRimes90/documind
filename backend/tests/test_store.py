from app.models import Chunk
from app.store import VectorStore


def _chunk(cid, text):
    return Chunk(doc_id="d1", doc_name="x.pdf", page=1, text=text, chunk_id=cid)


class _FakeSparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values


def test_upsert_and_dense_search(tmp_path):
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=3)
    chunks = [_chunk("c0", "gatto"), _chunk("c1", "cane")]
    dense = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    sparse = [_FakeSparse([0], [1.0]), _FakeSparse([1], [1.0])]
    store.upsert(chunks, dense, sparse)
    assert store.count() == 2
    hits = store.search_dense([0.9, 0.1, 0.0], top_k=2)
    assert hits[0] == "c0"
    assert store.get_chunk("c1").text == "cane"
