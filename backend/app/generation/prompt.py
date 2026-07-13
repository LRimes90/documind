"""Costruzione contesto numerato e prompt con grounding."""
from app.models import Chunk, Citation

_SNIPPET_LEN = 200


def build_context(chunks: list[Chunk]) -> tuple[str, list[Citation]]:
    lines: list[str] = []
    citations: list[Citation] = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[{i}] ({c.doc_name} p.{c.page}): {c.text}")
        citations.append(
            Citation(n=i, doc_name=c.doc_name, page=c.page, snippet=c.text[:_SNIPPET_LEN])
        )
    return "\n\n".join(lines), citations


def build_prompt(question: str, context: str) -> str:
    return (
        "Sei un assistente che risponde SOLO usando il CONTESTO fornito.\n"
        "Regole:\n"
        "1. Cita le fonti pertinenti nel formato [n] subito dopo l'affermazione.\n"
        "2. Se il contesto non contiene la risposta, dichiara che l'informazione "
        "non è presente nei documenti. NON inventare.\n\n"
        f"CONTESTO:\n{context}\n\n"
        f"DOMANDA: {question}\n\nRISPOSTA:"
    )
