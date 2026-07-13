"""Gold set etichettato a mano sui sample_docs (corpus sintetico multilingue).

Ogni GoldItem mappa una domanda alla/e pagina/e (del doc indicato) che contiene
la risposta. Include 2 domande cross-lingua (19-20) per esercitare il modello
multilingue. Rigenerare i PDF con: uv run python sample_docs/generate.py
"""
from dataclasses import dataclass, field


@dataclass
class GoldItem:
    question: str
    doc_name: str
    pages: list[int] = field(default_factory=list)


GOLD: list[GoldItem] = [
    GoldItem("Qual è la capitale della Francia?", "geografia.pdf", [1]),
    GoldItem("Con quali nazioni confina la Francia?", "geografia.pdf", [1]),
    GoldItem("Qual è il fiume più lungo d'Italia?", "geografia.pdf", [2]),
    GoldItem("Qual è la capitale d'Italia?", "geografia.pdf", [2]),
    GoldItem("Quanto è alto il Monte Bianco?", "geografia.pdf", [3]),
    GoldItem("Qual è la capitale della Spagna?", "geografia.pdf", [4]),
    GoldItem("Qual è la capitale della Germania?", "geografia.pdf", [5]),
    GoldItem("Who created Python?", "technology.pdf", [1]),
    GoldItem("In which year was Python first released?", "technology.pdf", [1]),
    GoldItem("What does HTTP status code 404 mean?", "technology.pdf", [2]),
    GoldItem("What is a vector database used for?", "technology.pdf", [3]),
    GoldItem("Who created Git?", "technology.pdf", [4]),
    GoldItem("What is JSON?", "technology.pdf", [5]),
    GoldItem("Quali sono gli ingredienti della carbonara?", "cucina.pdf", [1]),
    GoldItem("Cosa contiene il tiramisù?", "cucina.pdf", [2]),
    GoldItem("Come si prepara il pesto genovese?", "cucina.pdf", [3]),
    GoldItem("Quali ingredienti servono per il risotto alla milanese?", "cucina.pdf", [4]),
    GoldItem("Dove è nata la pizza margherita?", "cucina.pdf", [5]),
    # cross-lingua
    GoldItem("What is the capital of France?", "geografia.pdf", [1]),
    GoldItem("Chi ha creato il linguaggio di programmazione Python?", "technology.pdf", [1]),
]
