"""Chunking page-aware: finestra scorrevole di parole, per-pagina."""
from app.models import Chunk


def chunk_pages(
    pages: list[tuple[int, str]],
    doc_id: str,
    doc_name: str,
    max_words: int,
    overlap: int,
) -> list[Chunk]:
    step = max(1, max_words - overlap)
    chunks: list[Chunk] = []
    for page_no, text in pages:
        words = text.split()
        if not words:
            continue
        idx = 0
        start = 0
        while start < len(words):
            window = words[start : start + max_words]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page_no,
                    text=" ".join(window),
                    chunk_id=f"{doc_id}:{page_no}:{idx}",
                )
            )
            idx += 1
            if start + max_words >= len(words):
                break
            start += step
    return chunks
