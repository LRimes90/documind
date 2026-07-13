"""Genera il corpus sintetico multilingue (IT+EN) usato da demo, eval e stress.

Contenuto neutro e riproducibile: nessun dato sensibile, adatto a un repo pubblico.
Rilancia con:  uv run python sample_docs/generate.py
"""
import fitz
from pathlib import Path

DOCS: dict[str, list[str]] = {
    "geografia.pdf": [
        "La Francia è uno Stato dell'Europa occidentale. La sua capitale è Parigi, "
        "attraversata dal fiume Senna. La Francia confina con Spagna, Italia, "
        "Svizzera, Germania, Belgio e Lussemburgo. La lingua ufficiale è il francese "
        "e la moneta è l'euro.",
        "L'Italia è una repubblica dell'Europa meridionale. La sua capitale è Roma. "
        "Il fiume più lungo d'Italia è il Po, che scorre per circa 652 chilometri. "
        "La catena montuosa degli Appennini percorre la penisola da nord a sud.",
        "Le Alpi sono la più importante catena montuosa d'Europa. La vetta più alta "
        "delle Alpi è il Monte Bianco, che raggiunge 4808 metri di altitudine, "
        "al confine tra Italia e Francia.",
        "La Spagna occupa gran parte della penisola iberica. La sua capitale è Madrid, "
        "situata nel centro del paese. Il fiume Tago attraversa la Spagna e il "
        "Portogallo prima di sfociare nell'oceano Atlantico presso Lisbona.",
        "La Germania è uno Stato dell'Europa centrale. La sua capitale è Berlino. "
        "Il fiume Reno è una delle principali vie d'acqua tedesche. La Foresta Nera "
        "è una celebre regione montuosa e boschiva nel sud-ovest del paese.",
    ],
    "technology.pdf": [
        "Python is a high-level programming language created by Guido van Rossum and "
        "first released in 1991. It is dynamically typed and known for its readable "
        "syntax. Python is widely used in data science, web development and automation.",
        "HTTP status codes indicate the result of a request. Code 200 means OK, the "
        "request succeeded. Code 404 means Not Found, the resource does not exist. "
        "Code 500 means Internal Server Error, a generic server failure.",
        "A vector database stores high-dimensional embeddings and enables similarity "
        "search. Qdrant is an open-source vector database. Similarity is often measured "
        "with cosine distance between vectors. Such databases power retrieval-augmented "
        "generation.",
        "Git is a distributed version control system created by Linus Torvalds in 2005. "
        "It tracks changes in source code and supports branching and merging. GitHub is "
        "a popular platform for hosting Git repositories.",
        "JSON, JavaScript Object Notation, is a lightweight data-interchange format. "
        "It represents data as key-value pairs and ordered lists. JSON is "
        "language-independent and widely used in web APIs.",
    ],
    "cucina.pdf": [
        "La pasta alla carbonara è un piatto romano. Gli ingredienti tradizionali sono "
        "guanciale, tuorli d'uovo, pecorino romano e pepe nero. La ricetta autentica "
        "non prevede l'uso della panna.",
        "Il tiramisù è un dolce al cucchiaio molto diffuso in Italia. Si prepara con "
        "mascarpone, savoiardi inzuppati nel caffè, uova, zucchero e cacao amaro "
        "in polvere.",
        "Il pesto genovese è una salsa a base di basilico. Gli ingredienti sono "
        "basilico, pinoli, aglio, parmigiano, pecorino e olio extravergine di oliva, "
        "tradizionalmente pestati nel mortaio.",
        "Il risotto alla milanese è un primo piatto lombardo. Si prepara con riso, "
        "zafferano che gli conferisce il colore giallo, brodo, burro e parmigiano. "
        "È spesso servito con l'ossobuco.",
        "La pizza margherita è nata a Napoli. È condita con pomodoro, mozzarella e "
        "basilico fresco, che ricordano i colori della bandiera italiana. Fu dedicata "
        "alla regina Margherita di Savoia.",
    ],
}


def make(path: Path, pages: list[str]) -> None:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
        page.insert_textbox(rect, text, fontsize=13, fontname="helv")
    doc.save(str(path))
    doc.close()


def main() -> None:
    here = Path(__file__).parent
    for name, pages in DOCS.items():
        make(here / name, pages)
    print("Generati:", ", ".join(DOCS))


if __name__ == "__main__":
    main()
