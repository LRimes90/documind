from app.ingest.chunker import chunk_pages


def test_chunk_preserves_page_number():
    pages = [(1, "a " * 10), (2, "b " * 10)]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=100, overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_id == "d1" for c in chunks)


def test_chunk_windows_with_overlap():
    pages = [(1, " ".join(str(i) for i in range(10)))]  # 10 parole
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=4, overlap=1)
    # window di 4, passo 3 → start 0,3,6; il chunk finale [6..9] copre già la fine
    # (un chunk a start=9 sarebbe ridondante) → 3 chunk
    assert len(chunks) == 3
    assert chunks[0].text.split() == ["0", "1", "2", "3"]
    assert chunks[1].text.split()[0] == "3"  # overlap di 1
    assert chunks[2].text.split() == ["6", "7", "8", "9"]


def test_chunk_skips_empty_pages():
    pages = [(1, "   "), (2, "parola")]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=50, overlap=0)
    assert len(chunks) == 1 and chunks[0].page == 2


def test_chunk_ids_unique():
    pages = [(1, " ".join(str(i) for i in range(20)))]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=5, overlap=0)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
