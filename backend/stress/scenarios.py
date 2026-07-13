"""Generatore di ≥100 scenari di stress su 5 categorie (vedi design §12)."""
from dataclasses import dataclass, field
from eval.dataset import GOLD


@dataclass
class Scenario:
    category: str  # retrieval | grounding | ingest | concurrency | api_edge
    kind: str
    payload: dict = field(default_factory=dict)
    expect: str = ""


# domande palesemente fuori dal corpus (geografia/technology/cucina)
_OUT_OF_CORPUS = [
    "Chi ha vinto il campionato di Formula 1 nel 2010?",
    "Qual è la distanza tra la Terra e la Luna?",
    "Come si cura il raffreddore?",
    "Qual è il PIL del Giappone?",
    "Chi ha scritto la Divina Commedia?",
    "What is the boiling point of mercury?",
    "How do vaccines work?",
    "Qual è la capitale dell'Australia?",
    "What is the speed of light?",
    "Chi ha dipinto la Cappella Sistina?",
    "Come funziona un motore a scoppio?",
    "In what year did World War II end?",
    "Qual è la formula chimica dell'acqua?",
    "Who wrote Romeo and Juliet?",
    "Come si addestra un cane?",
    "What is the largest planet in the solar system?",
    "Quanto costa un iPhone?",
    "Chi era Giulio Cesare?",
    "How tall is Mount Everest?",
    "Qual è il numero atomico dell'oro?",
]


def build_scenarios() -> list[Scenario]:
    scen: list[Scenario] = []

    # retrieval (~40): ogni gold in forma originale + una variante riformulata
    for item in GOLD:
        gold = [f"{item.doc_name}:{p}" for p in item.pages]
        scen.append(
            Scenario("retrieval", "gold_in_citations",
                     {"question": item.question, "gold": gold},
                     expect="gold_page_cited")
        )
        scen.append(
            Scenario("retrieval", "gold_in_citations_variant",
                     {"question": f"Rispondi alla domanda: {item.question}", "gold": gold},
                     expect="gold_page_cited")
        )

    # grounding (~20): fuori-corpus → in live deve dire "non nei documenti"
    for q in _OUT_OF_CORPUS:
        scen.append(
            Scenario("grounding", "out_of_corpus", {"question": q},
                     expect="not_found_or_structural")
        )

    # ingest (~12)
    ingest_kinds = [
        ("empty_pdf", "http_422"),
        ("non_pdf", "http_422"),
        ("normal_pdf", "http_200"),
        ("unicode_pdf", "http_200"),
        ("large_pdf", "http_200"),
        ("duplicate_pdf", "http_200"),
    ]
    for kind, expect in ingest_kinds * 2:  # ×2 → 12
        scen.append(Scenario("ingest", kind, {}, expect=expect))

    # concurrency (~20): gold query ripetute in parallelo
    for i in range(20):
        item = GOLD[i % len(GOLD)]
        scen.append(
            Scenario("concurrency", "parallel_query",
                     {"question": item.question}, expect="http_200")
        )

    # api_edge (~8)
    scen.append(Scenario("api_edge", "query_before_ingest", {}, expect="http_400"))
    scen.append(Scenario("api_edge", "empty_question", {"question": ""}, expect="ok_any"))
    scen.append(Scenario("api_edge", "long_question", {"question": "perché? " * 500}, expect="http_200"))
    scen.append(Scenario("api_edge", "malformed_payload", {}, expect="http_422"))
    scen.append(Scenario("api_edge", "sse_completes", {"question": "Capitale d'Italia?"}, expect="has_done_event"))
    scen.append(Scenario("api_edge", "missing_field", {}, expect="http_422"))
    scen.append(Scenario("api_edge", "whitespace_question", {"question": "   "}, expect="ok_any"))
    scen.append(Scenario("api_edge", "numeric_question", {"question": "404"}, expect="http_200"))

    return scen
