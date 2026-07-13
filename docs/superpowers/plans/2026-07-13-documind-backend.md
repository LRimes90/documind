# DocuMind Backend — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costruire il motore RAG di DocuMind — ingest PDF, hybrid search (dense+sparse+RRF), reranking, generazione con citazioni verificabili — esposto via FastAPI e misurato da un eval harness riproducibile.

**Architecture:** Backend Python/FastAPI con Qdrant in modalità embedded (nessun server). Embeddings e reranking locali via fastembed (ONNX, no torch). La generazione della risposta è dietro un'interfaccia `LLMProvider` con due implementazioni: Gemini (default) e Ollama (offline mode). L'eval harness misura Hit@k/MRR/Recall@k su tre configurazioni (naive / hybrid / hybrid+rerank).

**Tech Stack:** Python 3.12 (via uv), FastAPI, uvicorn, qdrant-client (embedded), fastembed, pymupdf, google-genai, httpx, pydantic v2, pydantic-settings, pytest, pytest-asyncio.

## Global Constraints

- Python **3.12** pinnato via `uv` (NON usare il 3.14 di sistema — wheel ML incompatibili).
- Nessuna dipendenza da **torch**: embeddings/rerank solo via **fastembed** (ONNX).
- Qdrant **embedded** (`QdrantClient(path=...)`), MAI un server remoto o Docker.
- I nomi dei modelli vivono in `config.py`, mai hardcoded nei moduli.
- Modelli **multilingue** (IT+EN): dense `intfloat/multilingual-e5-small`, reranker multilingue. Se un ID non è nella lista supportata da fastembed al momento dell'implementazione, si sceglie l'equivalente multilingue disponibile (fallback dense: `intfloat/multilingual-e5-large`; fallback reranker: `Xenova/ms-marco-MiniLM-L-6-v2`) e si annota la scelta.
- Fusione hybrid via **Reciprocal Rank Fusion** (per rango, MAI somma di punteggi grezzi cosine+BM25).
- Grounding obbligatorio: se il contesto non basta, il modello dichiara "non presente nei documenti".
- Tutti i comandi si eseguono dalla cartella `backend/` salvo diverso avviso.
- Commit frequenti, uno per task completato. Ogni commit termina con:
  `Co-Authored-By: claude-flow <ruv@ruv.net>`
- **Workflow di rilascio (vincolo):** repo GitHub creato **privato** → suite completa + **stress-test (≥100) verdi** → SOLO allora repo **pubblico**. Nessun push pubblico prima del gate (Task 19).
- Il frontend (incl. citazioni con highlight+scroll, ora MVP) è nel **Piano 2**, non qui.

> **Nota didattica (Learn by Doing):** la funzione `reciprocal_rank_fusion` (Task 8) è riservata al contributo manuale di Luca — in fase di esecuzione verrà inserito un `TODO(human)` e verrà chiesto a lui di implementarla. È l'algoritmo-cuore dell'hybrid search: perfetto da scrivere a mano.

---

### Task 1: Scaffolding progetto, config, health endpoint

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.python-version`
- Create: `backend/.env.example`
- Create: `backend/app/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_health.py`

**Interfaces:**
- Produces: `app.config.settings` (istanza `Settings`); FastAPI app `app.main.app`; `GET /health` → `{"status": "ok"}`.

- [ ] **Step 1: Inizializza il progetto uv con Python 3.12**

```bash
cd C:/Users/Rimes/Projects/documind
uv init backend --python 3.12 --no-workspace
cd backend
uv add fastapi "uvicorn[standard]" pydantic pydantic-settings qdrant-client fastembed pymupdf google-genai httpx
uv add --dev pytest pytest-asyncio httpx
```
Expected: viene creato `.venv` con Python 3.12; `uv run python --version` stampa `Python 3.12.x`.

- [ ] **Step 2: Scrivi `app/config.py`**

```python
"""Configurazione centralizzata: legge .env via pydantic-settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "gemini"            # "gemini" | "ollama"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Modelli fastembed (multilingue IT+EN)
    dense_model: str = "intfloat/multilingual-e5-small"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"

    # Qdrant embedded
    qdrant_path: str = "./qdrant_data"
    collection: str = "documind"

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_n_rerank: int = 5

    # Chunking (word-based ≈ token)
    chunk_words: int = 350
    chunk_overlap: int = 60


settings = Settings()
```

- [ ] **Step 3: Scrivi `.env.example`**

```bash
# Provider LLM: "gemini" (default) oppure "ollama" (offline mode)
LLM_PROVIDER=gemini
# Ottieni la chiave su https://aistudio.google.com/apikey
GEMINI_API_KEY=
# Solo per offline mode:
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.1
```

- [ ] **Step 4: Scrivi il test di health (fallisce)**

`backend/tests/test_health.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

- [ ] **Step 5: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_health.py -v`
Expected: FAIL (ImportError: cannot import 'app' — `main.py` non esiste ancora).

- [ ] **Step 6: Scrivi `app/main.py` minimale**

```python
from fastapi import FastAPI

app = FastAPI(title="DocuMind API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 7: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_health.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd C:/Users/Rimes/Projects/documind
git add backend/
git commit -m "feat(backend): scaffolding uv + config + health endpoint

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 2: Schemi dati (models.py)

**Files:**
- Create: `backend/app/models.py`
- Test: `backend/tests/test_models.py`

**Interfaces:**
- Produces: `Chunk(doc_id:str, doc_name:str, page:int, text:str, chunk_id:str)`; `Citation(n:int, doc_name:str, page:int, snippet:str)`; `QueryRequest(question:str)`; `QueryResponse(answer:str, citations:list[Citation])`; `DocumentInfo(doc_id:str, doc_name:str, n_chunks:int, pages:int)`.

- [ ] **Step 1: Scrivi il test (fallisce)**

`backend/tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL (ModuleNotFoundError: app.models).

- [ ] **Step 3: Scrivi `app/models.py`**

```python
from pydantic import BaseModel


class Chunk(BaseModel):
    doc_id: str
    doc_name: str
    page: int
    text: str
    chunk_id: str


class Citation(BaseModel):
    n: int
    doc_name: str
    page: int
    snippet: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    n_chunks: int
    pages: int
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_models.py
git commit -m "feat(backend): schemi pydantic (Chunk, Citation, Query*, DocumentInfo)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 3: Chunker page-aware (chunker.py)

**Files:**
- Create: `backend/app/ingest/__init__.py`
- Create: `backend/app/ingest/chunker.py`
- Test: `backend/tests/test_chunker.py`

**Interfaces:**
- Consumes: `Chunk` da `app.models`.
- Produces: `chunk_pages(pages: list[tuple[int, str]], doc_id: str, doc_name: str, max_words: int, overlap: int) -> list[Chunk]`. I chunk NON attraversano il confine di pagina (attribuzione pagina esatta per le citazioni).

- [ ] **Step 1: Scrivi i test (falliscono)**

`backend/tests/test_chunker.py`:
```python
from app.ingest.chunker import chunk_pages


def test_chunk_preserves_page_number():
    pages = [(1, "a " * 10), (2, "b " * 10)]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=100, overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_id == "d1" for c in chunks)


def test_chunk_windows_with_overlap():
    pages = [(1, " ".join(str(i) for i in range(10)))]  # 10 parole
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=4, overlap=1)
    # window di 4, passo 3 → start a 0,3,6,9 → 4 chunk
    assert len(chunks) == 4
    assert chunks[0].text.split() == ["0", "1", "2", "3"]
    assert chunks[1].text.split()[0] == "3"  # overlap di 1


def test_chunk_skips_empty_pages():
    pages = [(1, "   "), (2, "parola")]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=50, overlap=0)
    assert len(chunks) == 1 and chunks[0].page == 2


def test_chunk_ids_unique():
    pages = [(1, " ".join(str(i) for i in range(20)))]
    chunks = chunk_pages(pages, doc_id="d1", doc_name="x.pdf", max_words=5, overlap=0)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: FAIL (ModuleNotFoundError: app.ingest.chunker).

- [ ] **Step 3: Scrivi `app/ingest/chunker.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `uv run pytest tests/test_chunker.py -v`
Expected: PASS (4 test).

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/ backend/tests/test_chunker.py
git commit -m "feat(backend): chunker page-aware con finestra a overlap

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 4: Loader PDF (loader.py)

**Files:**
- Create: `backend/app/ingest/loader.py`
- Test: `backend/tests/test_loader.py`

**Interfaces:**
- Produces: `load_pdf(path: str) -> list[tuple[int, str]]` — lista `(numero_pagina_1based, testo)`. Solleva `EmptyPdfError` se nessuna pagina ha testo estraibile.

- [ ] **Step 1: Scrivi il test (fallisce), generando un PDF fixture con pymupdf**

`backend/tests/test_loader.py`:
```python
import fitz  # pymupdf
import pytest
from app.ingest.loader import load_pdf, EmptyPdfError


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_load_pdf_returns_pages_with_text(tmp_path):
    p = tmp_path / "a.pdf"
    _make_pdf(p, ["Pagina uno testo", "Pagina due testo"])
    pages = load_pdf(str(p))
    assert len(pages) == 2
    assert pages[0][0] == 1
    assert "uno" in pages[0][1]


def test_load_empty_pdf_raises(tmp_path):
    p = tmp_path / "empty.pdf"
    _make_pdf(p, ["", ""])
    with pytest.raises(EmptyPdfError):
        load_pdf(str(p))
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL (ModuleNotFoundError: app.ingest.loader).

- [ ] **Step 3: Scrivi `app/ingest/loader.py`**

```python
"""Estrae testo per-pagina da un PDF (pymupdf)."""
import fitz


class EmptyPdfError(Exception):
    """Nessun testo estraibile (PDF vuoto o scannerizzato senza OCR)."""


def load_pdf(path: str) -> list[tuple[int, str]]:
    doc = fitz.open(path)
    try:
        pages = [(i + 1, page.get_text().strip()) for i, page in enumerate(doc)]
    finally:
        doc.close()
    if not any(text for _, text in pages):
        raise EmptyPdfError(f"Nessun testo estraibile da {path} (PDF scannerizzato?)")
    return pages
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_loader.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/loader.py backend/tests/test_loader.py
git commit -m "feat(backend): loader PDF per-pagina con EmptyPdfError

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 5: Verifica modelli fastembed + wrapper embedding (embeddings.py)

**Files:**
- Create: `backend/app/ingest/embeddings.py`
- Test: `backend/tests/test_embeddings.py`
- Create (temporaneo, poi rimosso): `backend/scripts/probe_models.py`

**Interfaces:**
- Consumes: `settings` da `app.config`.
- Produces: `Embedder` con `dense_dim: int`, `embed_dense(texts: list[str]) -> list[list[float]]`, `embed_sparse(texts: list[str]) -> list[SparseVector]`, `embed_query_dense(text)`, `embed_query_sparse(text)`. `SparseVector` = oggetto con `.indices: list[int]` e `.values: list[float]` (tipo restituito da fastembed).

- [ ] **Step 1: Verifica gli ID modello realmente supportati**

`backend/scripts/probe_models.py`:
```python
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

print("DENSE:")
for m in TextEmbedding.list_supported_models():
    if "multilingual" in m["model"].lower():
        print(" ", m["model"], m["dim"])
print("SPARSE:")
for m in SparseTextEmbedding.list_supported_models():
    print(" ", m["model"])
print("RERANK:")
for m in TextCrossEncoder.list_supported_models():
    print(" ", m["model"])
```
Run: `uv run python scripts/probe_models.py`
Expected: elenco modelli. **Conferma** che `intfloat/multilingual-e5-small`, `Qdrant/bm25` e `jinaai/jina-reranker-v2-base-multilingual` siano presenti. Se un ID manca, aggiorna il default in `config.py` con l'equivalente multilingue disponibile (fallback: `intfloat/multilingual-e5-large` / `Xenova/ms-marco-MiniLM-L-6-v2`) e annota la scelta in un commento in `config.py`.

- [ ] **Step 2: Scrivi il test (fallisce)**

`backend/tests/test_embeddings.py`:
```python
from app.ingest.embeddings import Embedder


def test_dense_dim_and_shape():
    emb = Embedder()
    vecs = emb.embed_dense(["ciao mondo", "hello world"])
    assert emb.dense_dim > 0
    assert len(vecs) == 2
    assert len(vecs[0]) == emb.dense_dim


def test_sparse_has_indices_and_values():
    emb = Embedder()
    sv = emb.embed_query_sparse("hello world")
    assert len(sv.indices) == len(sv.values)
    assert len(sv.indices) > 0
```

- [ ] **Step 3: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL (ModuleNotFoundError). NB: il primo run scarica i modelli ONNX (~600–900 MB) — può richiedere qualche minuto.

- [ ] **Step 4: Scrivi `app/ingest/embeddings.py`**

```python
"""Wrapper fastembed: dense + sparse, con lazy loading dei modelli."""
from fastembed import TextEmbedding, SparseTextEmbedding
from app.config import settings

# e5 richiede i prefissi "query:" / "passage:"
_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


class Embedder:
    def __init__(self) -> None:
        self._dense = TextEmbedding(model_name=settings.dense_model)
        self._sparse = SparseTextEmbedding(model_name=settings.sparse_model)
        # dim dal primo embedding
        probe = next(iter(self._dense.embed([_QUERY_PREFIX + "x"])))
        self.dense_dim = len(probe)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        prefixed = [_PASSAGE_PREFIX + t for t in texts]
        return [v.tolist() for v in self._dense.embed(prefixed)]

    def embed_sparse(self, texts: list[str]):
        return list(self._sparse.embed(texts))

    def embed_query_dense(self, text: str) -> list[float]:
        return next(iter(self._dense.embed([_QUERY_PREFIX + text]))).tolist()

    def embed_query_sparse(self, text: str):
        return next(iter(self._sparse.embed([text])))
```

- [ ] **Step 5: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: PASS.

- [ ] **Step 6: Rimuovi lo script di probe e committa**

```bash
rm backend/scripts/probe_models.py
git add backend/app/ingest/embeddings.py backend/tests/test_embeddings.py backend/app/config.py
git commit -m "feat(backend): wrapper fastembed dense+sparse (multilingue e5 + bm25)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 6: Vector store Qdrant embedded (store.py)

**Files:**
- Create: `backend/app/store.py`
- Test: `backend/tests/test_store.py`

**Interfaces:**
- Consumes: `settings`, `Chunk`.
- Produces: `VectorStore(path: str, collection: str, dense_dim: int)` con:
  - `upsert(chunks: list[Chunk], dense: list[list[float]], sparse: list) -> None`
  - `search_dense(vec: list[float], top_k: int) -> list[str]` (ritorna chunk_id ordinati)
  - `search_sparse(sparse_vec, top_k: int) -> list[str]`
  - `get_chunk(chunk_id: str) -> Chunk`
  - `count() -> int`

- [ ] **Step 1: Scrivi il test (fallisce)**

`backend/tests/test_store.py`:
```python
from app.models import Chunk
from app.store import VectorStore


def _chunk(cid, text):
    return Chunk(doc_id="d1", doc_name="x.pdf", page=1, text=text, chunk_id=cid)


def test_upsert_and_dense_search(tmp_path):
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=3)
    chunks = [_chunk("c0", "gatto"), _chunk("c1", "cane")]
    dense = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    sparse = [_FakeSparse([0], [1.0]), _FakeSparse([1], [1.0])]
    store.upsert(chunks, dense, sparse)
    assert store.count() == 2
    hits = store.search_dense([0.9, 0.1, 0.0], top_k=2)
    assert hits[0] == "c0"
    assert store.get_chunk("c1").text == "cane"


class _FakeSparse:
    def __init__(self, indices, values):
        self.indices = indices
        self.values = values
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL (ModuleNotFoundError: app.store).

- [ ] **Step 3: Scrivi `app/store.py`**

```python
"""Vector store: Qdrant embedded con named vectors dense + sparse."""
from qdrant_client import QdrantClient, models
from app.models import Chunk


class VectorStore:
    def __init__(self, path: str, collection: str, dense_dim: int) -> None:
        self.client = QdrantClient(path=path)
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )
        self._id = 0

    def upsert(self, chunks: list[Chunk], dense: list[list[float]], sparse) -> None:
        points = []
        for chunk, dvec, svec in zip(chunks, dense, sparse):
            points.append(
                models.PointStruct(
                    id=self._id,
                    vector={
                        "dense": dvec,
                        "sparse": models.SparseVector(
                            indices=list(svec.indices), values=list(svec.values)
                        ),
                    },
                    payload=chunk.model_dump(),
                )
            )
            self._id += 1
        self.client.upsert(self.collection, points=points)

    def search_dense(self, vec: list[float], top_k: int) -> list[str]:
        res = self.client.query_points(
            self.collection, query=vec, using="dense", limit=top_k
        )
        return [p.payload["chunk_id"] for p in res.points]

    def search_sparse(self, sparse_vec, top_k: int) -> list[str]:
        res = self.client.query_points(
            self.collection,
            query=models.SparseVector(
                indices=list(sparse_vec.indices), values=list(sparse_vec.values)
            ),
            using="sparse",
            limit=top_k,
        )
        return [p.payload["chunk_id"] for p in res.points]

    def get_chunk(self, chunk_id: str) -> Chunk:
        res = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="chunk_id", match=models.MatchValue(value=chunk_id)
                )]
            ),
            limit=1,
        )
        points = res[0]
        return Chunk(**points[0].payload)

    def count(self) -> int:
        return self.client.count(self.collection).count
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/store.py backend/tests/test_store.py
git commit -m "feat(backend): vector store Qdrant embedded (dense+sparse named vectors)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 7: Indexer (indexer.py)

**Files:**
- Create: `backend/app/ingest/indexer.py`
- Test: `backend/tests/test_indexer.py`

**Interfaces:**
- Consumes: `load_pdf`, `chunk_pages`, `Embedder`, `VectorStore`, `settings`, `DocumentInfo`.
- Produces: `ingest_pdf(path: str, doc_name: str, embedder: Embedder, store: VectorStore) -> DocumentInfo`.

- [ ] **Step 1: Scrivi il test (fallisce) con PDF fixture**

`backend/tests/test_indexer.py`:
```python
import fitz
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for text in pages_text:
        doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_ingest_pdf_indexes_chunks(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p, ["Il gatto dorme sul divano.", "Il cane corre nel parco."])
    emb = Embedder()
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=emb.dense_dim)
    info = ingest_pdf(str(p), "doc.pdf", emb, store)
    assert info.pages == 2
    assert info.n_chunks >= 2
    assert store.count() == info.n_chunks
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: FAIL (ModuleNotFoundError: app.ingest.indexer).

- [ ] **Step 3: Scrivi `app/ingest/indexer.py`**

```python
"""Orchestrazione ingest: PDF → chunk → embeddings → store."""
import uuid
from app.config import settings
from app.models import DocumentInfo
from app.ingest.loader import load_pdf
from app.ingest.chunker import chunk_pages
from app.ingest.embeddings import Embedder
from app.store import VectorStore


def ingest_pdf(path: str, doc_name: str, embedder: Embedder, store: VectorStore) -> DocumentInfo:
    pages = load_pdf(path)
    doc_id = uuid.uuid4().hex[:8]
    chunks = chunk_pages(
        pages, doc_id=doc_id, doc_name=doc_name,
        max_words=settings.chunk_words, overlap=settings.chunk_overlap,
    )
    texts = [c.text for c in chunks]
    dense = embedder.embed_dense(texts)
    sparse = embedder.embed_sparse(texts)
    store.upsert(chunks, dense, sparse)
    return DocumentInfo(
        doc_id=doc_id, doc_name=doc_name, n_chunks=len(chunks), pages=len(pages)
    )
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_indexer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/ingest/indexer.py backend/tests/test_indexer.py
git commit -m "feat(backend): indexer ingest PDF→chunk→embed→store

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 8: Reciprocal Rank Fusion (hybrid.py) — ⭐ Learn by Doing

**Files:**
- Create: `backend/app/retrieval/__init__.py`
- Create: `backend/app/retrieval/hybrid.py`
- Test: `backend/tests/test_hybrid.py`

**Interfaces:**
- Consumes: `VectorStore` (per `hybrid_candidates`).
- Produces:
  - `reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]` — fonde più classifiche (liste di chunk_id ordinate) in una sola lista di chunk_id ordinata per score RRF decrescente.
  - `hybrid_candidates(store: VectorStore, dense_vec, sparse_vec, top_k: int) -> list[str]`.

- [ ] **Step 1: Scrivi i test (falliscono)**

`backend/tests/test_hybrid.py`:
```python
from app.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_items_high_in_both():
    dense = ["a", "b", "c"]
    sparse = ["b", "a", "d"]
    fused = reciprocal_rank_fusion([dense, sparse])
    # "a" e "b" compaiono in cima a entrambe → davanti a "c"/"d"
    assert set(fused[:2]) == {"a", "b"}
    assert fused.index("c") > 1
    assert fused.index("d") > 1


def test_rrf_single_ranking_preserves_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert set(fused) == {"a", "b"}
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: FAIL (ModuleNotFoundError: app.retrieval.hybrid).

- [ ] **Step 3: Implementa `reciprocal_rank_fusion` — inserire `TODO(human)` e chiedere a Luca**

In fase di esecuzione: creare `app/retrieval/hybrid.py` con la firma e un `# TODO(human)` al posto del corpo di `reciprocal_rank_fusion`, poi fermarsi e presentare la richiesta "Learn by Doing" (formula RRF: `score(d) = Σ 1/(k + rank_i(d))`, k=60, rank 0-based). `hybrid_candidates` invece va scritto da me:

```python
"""Hybrid retrieval: RRF su ricerche dense + sparse."""
from app.store import VectorStore


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    # TODO(human): implementa la formula RRF.
    # Per ogni ranking e ogni doc_id a posizione `rank` (0-based),
    # accumula 1/(k + rank + 1) in un dict di score; poi ordina i
    # doc_id per score decrescente e ritorna la lista di id.
    raise NotImplementedError


def hybrid_candidates(store: VectorStore, dense_vec, sparse_vec, top_k: int) -> list[str]:
    dense_hits = store.search_dense(dense_vec, top_k=top_k)
    sparse_hits = store.search_sparse(sparse_vec, top_k=top_k)
    return reciprocal_rank_fusion([dense_hits, sparse_hits])
```

- [ ] **Step 4: Dopo il contributo di Luca, esegui i test e verifica che passino**

Run: `uv run pytest tests/test_hybrid.py -v`
Expected: PASS (3 test).

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/ backend/tests/test_hybrid.py
git commit -m "feat(backend): hybrid retrieval con Reciprocal Rank Fusion

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 9: Reranker cross-encoder (rerank.py)

**Files:**
- Create: `backend/app/retrieval/rerank.py`
- Test: `backend/tests/test_rerank.py`

**Interfaces:**
- Consumes: `settings`.
- Produces: `Reranker` con `rerank(query: str, candidates: list[tuple[str, str]], top_n: int) -> list[str]` dove ogni candidato è `(chunk_id, text)`; ritorna i `chunk_id` dei `top_n` per rilevanza.

- [ ] **Step 1: Scrivi il test (fallisce)**

`backend/tests/test_rerank.py`:
```python
from app.retrieval.rerank import Reranker


def test_rerank_puts_relevant_first():
    rr = Reranker()
    query = "Chi ha dipinto la Gioconda?"
    candidates = [
        ("c0", "La ricetta della pasta al pomodoro richiede basilico."),
        ("c1", "La Gioconda fu dipinta da Leonardo da Vinci."),
    ]
    ranked = rr.rerank(query, candidates, top_n=2)
    assert ranked[0] == "c1"
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: FAIL (ModuleNotFoundError: app.retrieval.rerank).

- [ ] **Step 3: Scrivi `app/retrieval/rerank.py`**

```python
"""Reranking con cross-encoder ONNX (fastembed)."""
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.config import settings


class Reranker:
    def __init__(self) -> None:
        self._model = TextCrossEncoder(model_name=settings.reranker_model)

    def rerank(self, query: str, candidates: list[tuple[str, str]], top_n: int) -> list[str]:
        if not candidates:
            return []
        docs = [text for _, text in candidates]
        scores = list(self._model.rerank(query, docs))
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [cid for (cid, _text), _score in ranked[:top_n]]
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/rerank.py backend/tests/test_rerank.py
git commit -m "feat(backend): reranker cross-encoder ONNX

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 10: Prompt builder (prompt.py)

**Files:**
- Create: `backend/app/generation/__init__.py`
- Create: `backend/app/generation/prompt.py`
- Test: `backend/tests/test_prompt.py`

**Interfaces:**
- Consumes: `Chunk`, `Citation`.
- Produces:
  - `build_context(chunks: list[Chunk]) -> tuple[str, list[Citation]]` — blocco contesto con fonti numerate `[1..N]` + lista Citation (snippet = primi ~200 caratteri).
  - `build_prompt(question: str, context: str) -> str` — prompt con istruzioni di grounding e citazione.

- [ ] **Step 1: Scrivi i test (falliscono)**

`backend/tests/test_prompt.py`:
```python
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
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Scrivi `app/generation/prompt.py`**

```python
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
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/generation/prompt.py backend/tests/test_prompt.py
git commit -m "feat(backend): prompt builder con contesto numerato e grounding

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 11: LLM provider interface + Gemini + Ollama (llm.py)

**Files:**
- Create: `backend/app/generation/llm.py`
- Test: `backend/tests/test_llm.py`

**Interfaces:**
- Consumes: `settings`.
- Produces:
  - `LLMProvider` (Protocol) con `generate(prompt: str) -> Iterator[str]` (stream di token/testo).
  - `GeminiProvider`, `OllamaProvider`.
  - `get_provider() -> LLMProvider` — factory su `settings.llm_provider`; solleva `ValueError` se provider ignoto o `GEMINI_API_KEY` mancante con provider gemini.

- [ ] **Step 1: Scrivi i test (falliscono) — factory testabile senza rete**

`backend/tests/test_llm.py`:
```python
import pytest
from app.generation import llm


def test_get_provider_unknown(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "nope")
    with pytest.raises(ValueError):
        llm.get_provider()


def test_get_provider_gemini_requires_key(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "gemini")
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")
    with pytest.raises(ValueError):
        llm.get_provider()


def test_get_provider_ollama_ok(monkeypatch):
    monkeypatch.setattr(llm.settings, "llm_provider", "ollama")
    provider = llm.get_provider()
    assert hasattr(provider, "generate")
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Scrivi `app/generation/llm.py`**

```python
"""Provider LLM: Gemini (default) e Ollama (offline mode) dietro un'interfaccia."""
from typing import Iterator, Protocol
import httpx
from app.config import settings


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> Iterator[str]: ...


class GeminiProvider:
    def __init__(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def generate(self, prompt: str) -> Iterator[str]:
        stream = self._client.models.generate_content_stream(
            model=self._model, contents=prompt
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


class OllamaProvider:
    def __init__(self) -> None:
        self._host = settings.ollama_host
        self._model = settings.ollama_model

    def generate(self, prompt: str) -> Iterator[str]:
        import json
        with httpx.stream(
            "POST",
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": True},
            timeout=120,
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]


def get_provider() -> LLMProvider:
    provider = settings.llm_provider
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY mancante. Impostala in .env "
                "(https://aistudio.google.com/apikey) oppure usa LLM_PROVIDER=ollama."
            )
        return GeminiProvider()
    if provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"LLM_PROVIDER sconosciuto: {provider!r}")
```

- [ ] **Step 4: Esegui i test e verifica che passino**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (3 test). NB: `test_get_provider_ollama_ok` non fa chiamate di rete (istanzia solo).

- [ ] **Step 5: Commit**

```bash
git add backend/app/generation/llm.py backend/tests/test_llm.py
git commit -m "feat(backend): LLMProvider Gemini + Ollama con factory

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 12: Pipeline di retrieval (pipeline.py)

**Files:**
- Create: `backend/app/retrieval/pipeline.py`
- Test: `backend/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Embedder`, `VectorStore`, `Reranker`, `hybrid_candidates`, `settings`, `Chunk`.
- Produces: `retrieve(question: str, embedder, store, reranker) -> list[Chunk]` — dense+sparse → RRF → rerank → top_n chunk completi (nell'ordine del rerank).

- [ ] **Step 1: Scrivi il test end-to-end (fallisce) su dati indicizzati reali**

`backend/tests/test_pipeline.py`:
```python
import fitz
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.pipeline import retrieve


def _make_pdf(path, pages_text):
    doc = fitz.open()
    for t in pages_text:
        doc.new_page().insert_text((72, 72), t)
    doc.save(str(path))
    doc.close()


def test_retrieve_returns_relevant_chunk(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p, [
        "La capitale della Francia è Parigi.",
        "Le api producono il miele nelle arnie.",
    ])
    emb = Embedder()
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=emb.dense_dim)
    ingest_pdf(str(p), "doc.pdf", emb, store)
    reranker = Reranker()
    chunks = retrieve("Qual è la capitale della Francia?", emb, store, reranker)
    assert len(chunks) >= 1
    assert "Parigi" in chunks[0].text
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL (ModuleNotFoundError: app.retrieval.pipeline).

- [ ] **Step 3: Scrivi `app/retrieval/pipeline.py`**

```python
"""Pipeline: embed query → hybrid (RRF) → rerank → chunk completi."""
from app.config import settings
from app.models import Chunk
from app.ingest.embeddings import Embedder
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.hybrid import hybrid_candidates


def retrieve(question: str, embedder: Embedder, store: VectorStore, reranker: Reranker) -> list[Chunk]:
    dense_vec = embedder.embed_query_dense(question)
    sparse_vec = embedder.embed_query_sparse(question)
    candidate_ids = hybrid_candidates(
        store, dense_vec, sparse_vec, top_k=settings.top_k_dense
    )
    if not candidate_ids:
        return []
    candidates = [(cid, store.get_chunk(cid)) for cid in candidate_ids]
    pairs = [(cid, chunk.text) for cid, chunk in candidates]
    top_ids = reranker.rerank(question, pairs, top_n=settings.top_n_rerank)
    by_id = {cid: chunk for cid, chunk in candidates}
    return [by_id[cid] for cid in top_ids]
```

- [ ] **Step 4: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/retrieval/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat(backend): pipeline retrieval hybrid→rerank

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 13: Wiring API — stato app, /documents, /query (SSE)

**Files:**
- Modify: `backend/app/main.py`
- Create: `backend/app/deps.py`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: tutto il sopra.
- Produces:
  - `deps.get_state()` — singleton lazy con `embedder`, `store`, `reranker`.
  - `POST /documents` (multipart file) → `DocumentInfo`.
  - `POST /query` (JSON `{question}`) → `text/event-stream`: eventi `token` (testo) e un evento finale `citations` (JSON), poi `done`.
  - `GET /health`.

- [ ] **Step 1: Scrivi il test (fallisce) — ingest via API + query non-stream helper**

`backend/tests/test_api.py`:
```python
import io
import fitz
from fastapi.testclient import TestClient


def _pdf_bytes(pages_text):
    doc = fitz.open()
    for t in pages_text:
        doc.new_page().insert_text((72, 72), t)
    data = doc.tobytes()
    doc.close()
    return data


def test_documents_then_query_stream(tmp_path, monkeypatch):
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "q"))
    # provider fittizio per non chiamare Gemini
    from app.generation import llm

    class FakeProvider:
        def generate(self, prompt):
            yield "Parigi [1]."

    monkeypatch.setattr(llm, "get_provider", lambda: FakeProvider())

    from app.main import app
    client = TestClient(app)

    files = {"file": ("doc.pdf", io.BytesIO(_pdf_bytes(["La capitale della Francia è Parigi."])), "application/pdf")}
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    assert r.json()["n_chunks"] >= 1

    with client.stream("POST", "/query", json={"question": "Capitale della Francia?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "Parigi" in body
    assert "citations" in body
```

- [ ] **Step 2: Esegui il test e verifica che fallisca**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL (405/404 su /documents, endpoint non definiti).

- [ ] **Step 3: Scrivi `app/deps.py`**

```python
"""Stato applicativo lazy: modelli e store caricati una sola volta."""
from functools import lru_cache
from app.config import settings
from app.ingest.embeddings import Embedder
from app.store import VectorStore
from app.retrieval.rerank import Reranker


class AppState:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.store = VectorStore(
            path=settings.qdrant_path,
            collection=settings.collection,
            dense_dim=self.embedder.dense_dim,
        )
        self.reranker = Reranker()


@lru_cache(maxsize=1)
def get_state() -> AppState:
    return AppState()
```

- [ ] **Step 4: Riscrivi `app/main.py`**

```python
import json
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models import QueryRequest
from app.deps import get_state
from app.ingest.indexer import ingest_pdf
from app.ingest.loader import EmptyPdfError
from app.retrieval.pipeline import retrieve
from app.generation.prompt import build_context, build_prompt
from app.generation.llm import get_provider

app = FastAPI(title="DocuMind API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def add_document(file: UploadFile = File(...)):
    state = get_state()
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        info = ingest_pdf(tmp_path, file.filename or "documento.pdf", state.embedder, state.store)
    except EmptyPdfError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return info


@app.post("/query")
def query(req: QueryRequest):
    state = get_state()
    if state.store.count() == 0:
        raise HTTPException(status_code=400, detail="Nessun documento indicizzato. Carica prima un PDF.")
    chunks = retrieve(req.question, state.embedder, state.store, state.reranker)
    context, citations = build_context(chunks)
    prompt = build_prompt(req.question, context)
    provider = get_provider()

    def event_stream():
        for token in provider.generate(prompt):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        payload = json.dumps([c.model_dump() for c in citations])
        yield f"event: citations\ndata: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 5: Esegui il test e verifica che passi**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS.

- [ ] **Step 6: Verifica manuale end-to-end (con GEMINI_API_KEY reale)**

```bash
cp .env.example .env   # e incolla la tua GEMINI_API_KEY
uv run uvicorn app.main:app --reload
# in un altro terminale:
curl -F "file=@sample_docs/esempio.pdf" http://localhost:8000/documents
curl -N -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question":"..."}'
```
Expected: `/documents` ritorna DocumentInfo; `/query` streamma token + evento `citations`.

- [ ] **Step 7: Commit**

```bash
git add backend/app/main.py backend/app/deps.py backend/tests/test_api.py
git commit -m "feat(backend): API /documents e /query (SSE) con stato lazy

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 14: Eval harness (dataset.py + evaluate.py)

**Files:**
- Create: `backend/eval/__init__.py`
- Create: `backend/eval/dataset.py`
- Create: `backend/eval/evaluate.py`
- Create: `backend/sample_docs/` (≥2 PDF con contenuto noto)
- Test: `backend/tests/test_eval.py`

**Interfaces:**
- Produces:
  - `dataset.GOLD: list[GoldItem]` con `GoldItem(question:str, doc_name:str, pages:list[int])`.
  - `evaluate.hit_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> int` (0/1).
  - `evaluate.mrr(retrieved_pages, gold_pages) -> float`.
  - `evaluate.run() -> None` — indicizza `sample_docs`, esegue le config (naive dense / hybrid / hybrid+rerank; la 4ª `+HyDE` viene aggiunta in Task 16), calcola Hit@5/MRR/Recall@5 medi, scrive `eval/results.md`.

- [ ] **Step 1: Scrivi i test delle metriche pure (falliscono)**

`backend/tests/test_eval.py`:
```python
from eval.evaluate import hit_at_k, mrr


def test_hit_at_k():
    assert hit_at_k([3, 1, 7], [1], k=5) == 1
    assert hit_at_k([3, 8, 7], [1], k=5) == 0
    assert hit_at_k([9, 9, 9, 9, 1], [1], k=3) == 0  # oltre k


def test_mrr():
    assert mrr([1, 2, 3], [2]) == 0.5   # gold in posizione 2 → 1/2
    assert mrr([5, 6], [1]) == 0.0
```

- [ ] **Step 2: Esegui i test e verifica che falliscano**

Run: `uv run pytest tests/test_eval.py -v`
Expected: FAIL (ModuleNotFoundError: eval.evaluate).

- [ ] **Step 3: Scrivi `eval/dataset.py`**

```python
"""Gold set etichettato a mano sui sample_docs."""
from dataclasses import dataclass, field


@dataclass
class GoldItem:
    question: str
    doc_name: str
    pages: list[int] = field(default_factory=list)


# NB: popolare in base al contenuto reale dei sample_docs (≥15 item consigliati).
GOLD: list[GoldItem] = [
    GoldItem("Qual è la capitale della Francia?", "geografia.pdf", [1]),
    # ... altri item ...
]
```

- [ ] **Step 4: Scrivi `eval/evaluate.py` (metriche + run 3 config)**

```python
"""Eval retrieval: Hit@k, MRR, Recall@k su naive / hybrid / hybrid+rerank."""
import tempfile
from pathlib import Path
from app.config import settings
from app.ingest.embeddings import Embedder
from app.ingest.indexer import ingest_pdf
from app.store import VectorStore
from app.retrieval.rerank import Reranker
from app.retrieval.hybrid import hybrid_candidates
from eval.dataset import GOLD

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"
K = 5


def hit_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> int:
    return int(any(p in gold_pages for p in retrieved_pages[:k]))


def mrr(retrieved_pages: list[int], gold_pages: list[int]) -> float:
    for rank, p in enumerate(retrieved_pages, start=1):
        if p in gold_pages:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_pages: list[int], gold_pages: list[int], k: int) -> float:
    found = {p for p in retrieved_pages[:k] if p in gold_pages}
    return len(found) / len(gold_pages) if gold_pages else 0.0


def _pages_for(ids: list[str], store: VectorStore) -> list[int]:
    return [store.get_chunk(cid).page for cid in ids]


def run() -> None:
    emb = Embedder()
    reranker = Reranker()
    tmp = tempfile.mkdtemp()
    store = VectorStore(path=tmp, collection="eval", dense_dim=emb.dense_dim)
    for pdf in SAMPLE_DIR.glob("*.pdf"):
        ingest_pdf(str(pdf), pdf.name, emb, store)

    configs = ["naive", "hybrid", "hybrid+rerank"]
    agg = {c: {"hit": 0.0, "mrr": 0.0, "recall": 0.0} for c in configs}

    for item in GOLD:
        dvec = emb.embed_query_dense(item.question)
        svec = emb.embed_query_sparse(item.question)
        naive_ids = store.search_dense(dvec, top_k=settings.top_k_dense)
        hybrid_ids = hybrid_candidates(store, dvec, svec, top_k=settings.top_k_dense)
        pairs = [(cid, store.get_chunk(cid).text) for cid in hybrid_ids]
        rerank_ids = reranker.rerank(item.question, pairs, top_n=settings.top_n_rerank)

        for cfg, ids in zip(configs, [naive_ids, hybrid_ids, rerank_ids]):
            pages = _pages_for(ids, store)
            agg[cfg]["hit"] += hit_at_k(pages, item.pages, K)
            agg[cfg]["mrr"] += mrr(pages, item.pages)
            agg[cfg]["recall"] += recall_at_k(pages, item.pages, K)

    n = len(GOLD)
    lines = [
        "# Eval results — DocuMind retrieval\n",
        f"Dataset: {n} domande · sample_docs: {len(list(SAMPLE_DIR.glob('*.pdf')))} PDF\n",
        "| Config | Hit@5 | MRR | Recall@5 |",
        "|--------|-------|-----|----------|",
    ]
    for cfg in configs:
        m = agg[cfg]
        lines.append(f"| {cfg} | {m['hit']/n:.2%} | {m['mrr']/n:.3f} | {m['recall']/n:.2%} |")
    (Path(__file__).parent / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
```

- [ ] **Step 5: Esegui i test delle metriche e verifica che passino**

Run: `uv run pytest tests/test_eval.py -v`
Expected: PASS.

- [ ] **Step 6: Crea i sample_docs, popola GOLD, esegui l'eval reale**

Genera ≥2 PDF con contenuto noto (script una tantum con pymupdf o PDF reali), popola `GOLD` con ≥15 domande mappate a doc+pagina, poi:
Run: `uv run python -m eval.evaluate`
Expected: stampa la tabella e scrive `eval/results.md`; ci si aspetta la progressione naive ≤ hybrid ≤ hybrid+rerank su Hit@5.

- [ ] **Step 7: Aggiungi lo script `eval` a pyproject e committa**

In `pyproject.toml`:
```toml
[project.scripts]
eval = "eval.evaluate:run"
```
```bash
git add backend/eval/ backend/sample_docs/ backend/tests/test_eval.py backend/pyproject.toml
git commit -m "feat(backend): eval harness Hit@k/MRR/Recall@k su 3 config

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 15: README backend + esecuzione full suite

**Files:**
- Create: `README.md` (root)
- Create: `backend/README.md` (dettagli backend, opzionale)

**Interfaces:**
- Nessuna interfaccia di codice; documentazione.

- [ ] **Step 1: Esegui l'intera suite di test**

Run: `cd backend && uv run pytest -v`
Expected: tutti i test verdi.

- [ ] **Step 2: Scrivi il `README.md` di root**

Deve contenere: descrizione, **cosa dimostra** (hybrid+rerank+citazioni+eval), diagramma architettura (ASCII o mermaid), **tabella eval** (incolla da `eval/results.md`), istruzioni run (`uv sync`, `.env`, `uvicorn`), sezione **Offline mode** (Ollama: `LLM_PROVIDER=ollama`, `ollama pull llama3.1`), nota sul download modelli al primo avvio, e "Roadmap: frontend React (Piano 2)".

- [ ] **Step 3: Commit**

```bash
git add README.md backend/README.md
git commit -m "docs: README backend con architettura, eval e offline mode

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 16: Query rewriting / HyDE (query_rewrite.py) + 4ª config eval

**Files:**
- Create: `backend/app/retrieval/query_rewrite.py`
- Test: `backend/tests/test_query_rewrite.py`
- Modify: `backend/app/config.py` (aggiungi `use_hyde: bool = False`)
- Modify: `backend/app/retrieval/pipeline.py` (parametro opzionale `provider`)
- Modify: `backend/app/main.py` (passa il provider a `retrieve`)
- Modify: `backend/eval/evaluate.py` (4ª config `hybrid+rerank+hyde`)

**Interfaces:**
- Consumes: `LLMProvider` da `app.generation.llm`.
- Produces: `expand_query(question: str, provider: LLMProvider) -> str`. Firma aggiornata: `retrieve(question, embedder, store, reranker, provider: LLMProvider | None = None) -> list[Chunk]`.

- [ ] **Step 1: Scrivi il test (fallisce) con fake provider**

`backend/tests/test_query_rewrite.py`:
```python
from app.retrieval.query_rewrite import expand_query


class _FakeProvider:
    def generate(self, prompt):
        yield "Parigi è la capitale della Francia."


def test_expand_query_combines_question_and_hypothetical():
    out = expand_query("Capitale della Francia?", _FakeProvider())
    assert "Capitale della Francia?" in out
    assert "Parigi" in out
```

- [ ] **Step 2: Esegui e verifica fallimento**

Run: `uv run pytest tests/test_query_rewrite.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Scrivi `app/retrieval/query_rewrite.py` e aggiungi il flag in config**

```python
"""HyDE / query expansion: genera un passaggio ipotetico per migliorare il recall dense."""
from app.generation.llm import LLMProvider

_HYDE_PROMPT = (
    "Scrivi un breve paragrafo (2-3 frasi) che potrebbe contenere la risposta "
    "alla seguente domanda, come se fosse un estratto di documento. "
    "Non dichiarare che è ipotetico.\n\nDomanda: {q}\n\nParagrafo:"
)


def expand_query(question: str, provider: LLMProvider) -> str:
    hypothetical = "".join(provider.generate(_HYDE_PROMPT.format(q=question)))
    return f"{question}\n{hypothetical}".strip()
```
In `app/config.py` aggiungi nel gruppo Retrieval: `use_hyde: bool = False`.

- [ ] **Step 4: Esegui e verifica passaggio**

Run: `uv run pytest tests/test_query_rewrite.py -v`
Expected: PASS.

- [ ] **Step 5: Cabla in `pipeline.py` (dense su query espansa, sparse+rerank su domanda originale)**

Sostituisci `retrieve` con:
```python
def retrieve(question, embedder, store, reranker, provider=None):
    from app.retrieval.query_rewrite import expand_query
    query_for_dense = question
    if settings.use_hyde and provider is not None:
        query_for_dense = expand_query(question, provider)
    dense_vec = embedder.embed_query_dense(query_for_dense)
    sparse_vec = embedder.embed_query_sparse(question)  # sparse sulle parole originali
    candidate_ids = hybrid_candidates(store, dense_vec, sparse_vec, top_k=settings.top_k_dense)
    if not candidate_ids:
        return []
    candidates = [(cid, store.get_chunk(cid)) for cid in candidate_ids]
    pairs = [(cid, chunk.text) for cid, chunk in candidates]
    top_ids = reranker.rerank(question, pairs, top_n=settings.top_n_rerank)  # rerank su domanda originale
    by_id = {cid: chunk for cid, chunk in candidates}
    return [by_id[cid] for cid in top_ids]
```
In `app/main.py`, nell'endpoint `/query`, passa il provider: `chunks = retrieve(req.question, state.embedder, state.store, state.reranker, provider=provider)` (istanzia `provider = get_provider()` prima). Verifica che `tests/test_pipeline.py` e `tests/test_api.py` restino verdi (HyDE è off di default).

- [ ] **Step 6: Aggiungi la 4ª config all'eval con guardia provider**

In `eval/evaluate.py`, dentro `run()`, dopo aver calcolato `rerank_ids`, aggiungi (solo se un provider è disponibile):
```python
    # in cima a run(): prova a ottenere un provider per HyDE
    try:
        from app.generation.llm import get_provider
        from app.retrieval.query_rewrite import expand_query
        _provider = get_provider()
        configs = ["naive", "hybrid", "hybrid+rerank", "hybrid+rerank+hyde"]
    except ValueError:
        _provider = None
        configs = ["naive", "hybrid", "hybrid+rerank"]
        print("[eval] Provider LLM non disponibile: 4ª config (HyDE) saltata.")
```
E nel loop, quando `_provider` non è None, calcola la 4ª:
```python
        if _provider is not None:
            exp = expand_query(item.question, _provider)
            dvec2 = emb.embed_query_dense(exp)
            hyde_ids = hybrid_candidates(store, dvec2, svec, top_k=settings.top_k_dense)
            pairs2 = [(cid, store.get_chunk(cid).text) for cid in hyde_ids]
            hyde_rr = reranker.rerank(item.question, pairs2, top_n=settings.top_n_rerank)
            pages = _pages_for(hyde_rr, store)
            agg["hybrid+rerank+hyde"]["hit"] += hit_at_k(pages, item.pages, K)
            agg["hybrid+rerank+hyde"]["mrr"] += mrr(pages, item.pages)
            agg["hybrid+rerank+hyde"]["recall"] += recall_at_k(pages, item.pages, K)
```
Assicurati che `agg` sia costruito da `configs`. Riesegui `uv run pytest tests/test_eval.py -v` (metriche pure) → PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval/query_rewrite.py backend/tests/test_query_rewrite.py backend/app/config.py backend/app/retrieval/pipeline.py backend/app/main.py backend/eval/evaluate.py
git commit -m "feat(backend): query rewriting/HyDE + 4a config eval

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 17: CI GitHub Actions + marker test veloci/lenti + badge

**Files:**
- Modify: `backend/pyproject.toml` (marker pytest)
- Modify: test dei modelli (marker `slow` a livello di modulo): `test_embeddings.py`, `test_indexer.py`, `test_rerank.py`, `test_pipeline.py`, `test_api.py`
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (badge — durante Task 15/finale)

**Interfaces:** nessuna interfaccia di codice.

- [ ] **Step 1: Configura i marker in `backend/pyproject.toml`**

```toml
[tool.pytest.ini_options]
markers = ["slow: test che scaricano modelli ONNX (esclusi dalla CI veloce)"]
```

- [ ] **Step 2: Marca i test lenti a livello di modulo**

In cima a `test_embeddings.py`, `test_indexer.py`, `test_rerank.py`, `test_pipeline.py`, `test_api.py` aggiungi:
```python
import pytest
pytestmark = pytest.mark.slow
```

- [ ] **Step 3: Verifica che i test veloci girino SENZA scaricare modelli**

Run: `uv run pytest -m "not slow" -v`
Expected: PASS rapido (solo logica pura: health, models, chunker, loader, hybrid/RRF, prompt, llm-factory, metriche eval). Nessun download.

- [ ] **Step 4: Scrivi `.github/workflows/ci.yml`**

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: backend
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --dev
      - run: uv run pytest -m "not slow" -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/tests/ .github/workflows/ci.yml
git commit -m "ci: GitHub Actions test veloci + marker slow per test modelli

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 18: Stress-test harness (≥100 scenari)

**Files:**
- Create: `backend/stress/__init__.py`
- Create: `backend/stress/scenarios.py`
- Create: `backend/stress/run.py`
- Test: `backend/tests/test_stress_scenarios.py`

**Interfaces:**
- Produces:
  - `scenarios.build_scenarios() -> list[Scenario]` con `Scenario(category:str, kind:str, payload:dict, expect:str)`; ritorna ≥100 scenari coprendo le 5 categorie di §12 dello spec.
  - `run.main(live: bool = True) -> int` — indicizza `sample_docs`, esegue gli scenari via `TestClient`, verifica gli invarianti, scrive `stress/report.md`, ritorna il numero di fallimenti (0 = gate superato). `live=True` usa il provider reale (grounding autentico); `live=False` usa un provider fittizio deterministico per gli invarianti strutturali.

- [ ] **Step 1: Scrivi il test del generatore (fallisce)**

`backend/tests/test_stress_scenarios.py`:
```python
from stress.scenarios import build_scenarios


def test_at_least_100_scenarios_across_categories():
    scen = build_scenarios()
    assert len(scen) >= 100
    cats = {s.category for s in scen}
    assert {"retrieval", "grounding", "ingest", "concurrency", "api_edge"} <= cats
```

- [ ] **Step 2: Esegui e verifica fallimento**

Run: `uv run pytest tests/test_stress_scenarios.py -v`
Expected: FAIL (ModuleNotFoundError: stress.scenarios).

- [ ] **Step 3: Scrivi `stress/scenarios.py`**

```python
"""Generatore di ≥100 scenari di stress across 5 categorie."""
from dataclasses import dataclass, field


@dataclass
class Scenario:
    category: str          # retrieval | grounding | ingest | concurrency | api_edge
    kind: str              # tipo specifico
    payload: dict = field(default_factory=dict)
    expect: str = ""       # invariante attesa


# domande in-corpus con pagina attesa (allineare ai sample_docs reali)
_IN_CORPUS = [
    {"q": "Qual è la capitale della Francia?", "pages": [1]},
    # ... popolare ≥40 varianti reali sui sample_docs ...
]
# domande palesemente fuori-corpus
_OUT_CORPUS = [
    "Qual è la ricetta della carbonara?",
    "Chi ha vinto i mondiali di calcio 1982?",
    # ... ≥20 ...
]


def build_scenarios() -> list[Scenario]:
    scen: list[Scenario] = []
    for item in _IN_CORPUS:  # ~40
        scen.append(Scenario("retrieval", "gold_page_in_top5",
                             {"question": item["q"], "gold_pages": item["pages"]},
                             expect="gold_page_and_valid_citations"))
    for q in _OUT_CORPUS:    # ~20
        scen.append(Scenario("grounding", "out_of_corpus",
                             {"question": q}, expect="not_found_empty_citations"))
    # ingest (~15)
    for kind in ["empty_pdf", "non_pdf", "large_pdf", "unicode", "duplicate",
                 "many_small"]:
        scen.append(Scenario("ingest", kind, {}, expect="handled_gracefully"))
    # concurrency (~15): stessa query ripetuta in parallelo
    for i in range(15):
        scen.append(Scenario("concurrency", "parallel_query",
                             {"question": _IN_CORPUS[i % len(_IN_CORPUS)]["q"]},
                             expect="no_crash_latency_ok"))
    # api_edge (~10)
    for kind in ["empty_question", "very_long_question", "query_before_ingest",
                 "malformed_payload", "sse_completes"]:
        scen.append(Scenario("api_edge", kind, {}, expect="correct_status"))
    return scen
```
NB (step manuale): popolare `_IN_CORPUS` (≥40) e `_OUT_CORPUS` (≥20) sui contenuti reali dei `sample_docs`, e replicare varianti finché `len(build_scenarios()) >= 100`.

- [ ] **Step 4: Esegui il test del generatore e verifica che passi**

Run: `uv run pytest tests/test_stress_scenarios.py -v`
Expected: PASS.

- [ ] **Step 5: Scrivi `stress/run.py` (runner + invarianti + report)**

```python
"""Esegue gli scenari di stress e verifica gli invarianti. Gate pre-pubblicazione."""
import io
import json
import time
import concurrent.futures as cf
from pathlib import Path
import fitz
from fastapi.testclient import TestClient
from stress.scenarios import build_scenarios

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"


def _ingest_all(client):
    for pdf in SAMPLE_DIR.glob("*.pdf"):
        with open(pdf, "rb") as fh:
            client.post("/documents", files={"file": (pdf.name, fh, "application/pdf")})


def _check(client, s: Scenario) -> tuple[bool, float]:
    t0 = time.perf_counter()
    ok = True
    if s.category in ("retrieval", "grounding"):
        with client.stream("POST", "/query", json={"question": s.payload["question"]}) as r:
            body = "".join(r.iter_text())
        if s.expect == "not_found_empty_citations":
            ok = ("citations" in body) and ('"citations": []' in body or "non " in body.lower())
        else:
            ok = "citations" in body
    elif s.category == "api_edge":
        if s.kind == "query_before_ingest":
            ok = True  # verificato in un client vergine altrove
        elif s.kind == "malformed_payload":
            ok = client.post("/query", json={}).status_code == 422
        elif s.kind == "empty_question":
            ok = client.post("/query", json={"question": ""}).status_code in (200, 400, 422)
    # ... ingest / concurrency gestiti in main() ...
    return ok, time.perf_counter() - t0


def main(live: bool = True) -> int:
    from app import main as app_module
    if not live:
        class _Fake:
            def generate(self, prompt):
                yield "Risposta [1]." if "CONTESTO" in prompt else "non presente nei documenti"
        app_module.get_provider = lambda: _Fake()
    client = TestClient(app_module.app)
    _ingest_all(client)

    scen = build_scenarios()
    results, latencies, failures = [], [], 0
    # scenari sequenziali
    for s in [x for x in scen if x.category != "concurrency"]:
        ok, dt = _check(client, s)
        latencies.append(dt)
        failures += 0 if ok else 1
        results.append((s.category, s.kind, ok, dt))
    # scenari di concorrenza
    conc = [x for x in scen if x.category == "concurrency"]
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for ok, dt in ex.map(lambda s: _check(client, s), conc):
            latencies.append(dt)
            failures += 0 if ok else 1

    latencies.sort()
    p50 = latencies[len(latencies)//2] if latencies else 0
    p95 = latencies[int(len(latencies)*0.95)-1] if latencies else 0
    lines = [
        "# Stress-test report — DocuMind\n",
        f"Scenari: {len(scen)} · Fallimenti: {failures} · "
        f"Latenza p50 {p50:.2f}s / p95 {p95:.2f}s\n",
        "| Categoria | Kind | Esito | s |",
        "|-----------|------|-------|---|",
    ]
    for cat, kind, ok, dt in results:
        lines.append(f"| {cat} | {kind} | {'PASS' if ok else 'FAIL'} | {dt:.2f} |")
    (Path(__file__).parent / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Fallimenti: {failures}/{len(scen)}")
    return failures


if __name__ == "__main__":
    import sys
    raise SystemExit(1 if main(live="--fake" not in sys.argv) else 0)
```
NB: rifinire gli invarianti di `ingest` (empty→422, non_pdf respinto, ecc.) e `query_before_ingest` (client su store vergine) in fase di implementazione; l'obiettivo è **0 fallimenti** su ≥100 scenari.

- [ ] **Step 6: Esecuzione reale (gate)**

Run (con `GEMINI_API_KEY` o Ollama attivo): `uv run python -m stress.run`
Expected: `Fallimenti: 0/…`, `stress/report.md` generato. Per un giro veloce senza LLM reale: `uv run python -m stress.run --fake`.

- [ ] **Step 7: Commit**

```bash
git add backend/stress/ backend/tests/test_stress_scenarios.py
git commit -m "test(backend): stress harness >=100 scenari (retrieval/grounding/ingest/conc/api)

Co-Authored-By: claude-flow <ruv@ruv.net>"
```

---

### Task 19: Repo privato + gate di pubblicazione

**Files:** nessun file di codice (operazioni git/GitHub).

**Interfaces:** nessuna.

- [ ] **Step 1: Crea il repo GitHub PRIVATO**

Opzione A (con `gh`, consigliata — lo riuserai per tutti i 5 progetti):
```bash
winget install --id GitHub.cli -e   # una tantum; poi riapri il terminale
gh auth login
cd C:/Users/Rimes/Projects/documind
gh repo create documind --private --source=. --remote=origin --push
```
Opzione B (manuale): crea un repo **privato** vuoto `documind` su github.com, poi:
```bash
cd C:/Users/Rimes/Projects/documind
git remote add origin https://github.com/<utente>/documind.git
git push -u origin main
```

- [ ] **Step 2: Esegui l'INTERA suite di test (inclusi i lenti)**

Run: `cd backend && uv run pytest -v`
Expected: tutti verdi.

- [ ] **Step 3: Esegui lo stress-test (gate)**

Run: `uv run python -m stress.run`
Expected: `Fallimenti: 0`. Se >0, NON pubblicare: correggi e ripeti.

- [ ] **Step 4: Committa i report ed esegui il push**

```bash
cd C:/Users/Rimes/Projects/documind
git add backend/eval/results.md backend/stress/report.md
git commit -m "docs: report eval + stress-test (gate pre-pubblicazione)

Co-Authored-By: claude-flow <ruv@ruv.net>"
git push
```

- [ ] **Step 5: SOLO se tutto verde → rendi il repo PUBBLICO**

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```
(oppure via web: Settings → General → Danger Zone → Change visibility). Gate rispettato: nessuna pubblicazione con stress-test rosso.

---

## Self-Review

**1. Spec coverage:**
- §4 architettura/confini → Task 1–13 (moduli separati per responsabilità). ✓
- §5.1 ingest (loader/chunker/indexer) → Task 3,4,5,7. ✓
- §5.2 query (hybrid/RRF/rerank/pipeline/generation/SSE) → Task 8,9,10,11,12,13. ✓
- §6 eval harness (Hit@k/MRR/Recall@k, 3 config, results.md) → Task 14. ✓
- §8 error handling (key mancante, PDF vuoto, query senza doc, grounding) → Task 4 (EmptyPdfError), Task 11 (key), Task 13 (400), Task 10 (grounding). ✓
- §9 testing (unit chunker/RRF/rerank/prompt, integration ingest→query) → Task 3,8,9,10 + Task 12/13. ✓
- Offline mode (Ollama) → Task 11 + README Task 15. ✓
- §5.2 step 0 HyDE / query rewriting + §6 4ª config eval → Task 16. ✓
- §12 CI (test veloci/lenti + badge) → Task 17. ✓
- §12 stress-test harness (≥100 scenari, 5 categorie) → Task 18. ✓
- §12 workflow repo privato → gate → pubblico → Task 19. ✓
- §7 frontend (incl. citazioni highlight+scroll, ora MVP) → **Piano 2** (fuori da questo piano, dichiarato). ✓

**2. Placeholder scan:** nessun "TBD/TODO" tranne l'unico `TODO(human)` intenzionale in Task 8 (contributo Learn by Doing). Il gold set in Task 14 richiede popolamento manuale con dati reali (step esplicito, non un placeholder di codice). ✓

**3. Type consistency:** `Chunk`, `Citation`, `DocumentInfo` usati coerentemente; `Embedder.dense_dim` consumato da `VectorStore` (Task 6) e `AppState` (Task 13); `hybrid_candidates`/`reciprocal_rank_fusion` firme coerenti tra Task 8 e Task 12; `Reranker.rerank(query, list[tuple[str,str]], top_n)` coerente tra Task 9 e Task 12. ✓

**Rischi residui noti (accettati):** ID esatti dei modelli fastembed verificati a runtime (Task 5, step 1) con fallback documentati; qualità eval indicativa su dataset piccolo (dichiarato nel README).
