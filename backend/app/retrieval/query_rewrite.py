"""HyDE / query expansion: genera un passaggio ipotetico per migliorare il recall dense.

Idea (Hypothetical Document Embeddings): invece di embeddare solo la domanda,
si chiede all'LLM di scrivere un breve passaggio che *potrebbe* contenere la
risposta, e si concatena alla domanda. L'embedding risultante è più vicino ai
chunk realmente rilevanti. Attivo solo con settings.use_hyde=True.
"""
from app.generation.llm import LLMProvider

_HYDE_PROMPT = (
    "Scrivi un breve paragrafo (2-3 frasi) che potrebbe contenere la risposta "
    "alla seguente domanda, come se fosse un estratto di documento. "
    "Non dichiarare che è ipotetico.\n\nDomanda: {q}\n\nParagrafo:"
)


def expand_query(question: str, provider: LLMProvider) -> str:
    hypothetical = "".join(provider.generate(_HYDE_PROMPT.format(q=question)))
    return f"{question}\n{hypothetical}".strip()
