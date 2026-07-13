"""Gold set etichettato a mano sui sample_docs.

Ogni GoldItem mappa una domanda alla/e pagina/e (del doc indicato) che
contiene la risposta. Popolare con ≥15 item allineati ai PDF in sample_docs/.
"""
from dataclasses import dataclass, field


@dataclass
class GoldItem:
    question: str
    doc_name: str
    pages: list[int] = field(default_factory=list)


# Popolato allo step di eval reale (Task 14), coerente coi sample_docs generati.
GOLD: list[GoldItem] = []
