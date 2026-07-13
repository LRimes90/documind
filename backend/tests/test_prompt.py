from app.models import Chunk
from app.generation.prompt import build_context, build_prompt


def _c(cid, page, text):
    return Chunk(doc_id="d1", doc_name="a.pdf", page=page, text=text, chunk_id=cid)


def test_build_context_numbers_sources():
    chunks = [_c("c0", 1, "alpha"), _c("c1", 5, "beta")]
    context, citations = build_context(chunks)
    assert "[1]" in context and "[2]" in context
    assert citations[0].n == 1 and citations[0].page == 1
    assert citations[1].n == 2 and citations[1].doc_name == "a.pdf"


def test_build_prompt_contains_grounding_rule():
    prompt = build_prompt("Domanda?", "[1] (a.pdf p.1): testo")
    assert "Domanda?" in prompt
    assert "non" in prompt.lower()  # regola anti-allucinazione presente
    assert "[n]" in prompt or "[1]" in prompt
