from app.models import Chunk, Citation, QueryResponse


def test_chunk_roundtrip():
    c = Chunk(doc_id="d1", doc_name="a.pdf", page=3, text="ciao", chunk_id="d1:3:0")
    assert c.page == 3
    assert c.model_dump()["chunk_id"] == "d1:3:0"


def test_query_response_nested_citations():
    resp = QueryResponse(
        answer="risposta [1]",
        citations=[Citation(n=1, doc_name="a.pdf", page=3, snippet="ciao")],
    )
    assert resp.citations[0].page == 3
