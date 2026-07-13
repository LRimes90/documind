# DocuMind — Design Document

**Data:** 2026-07-13
**Autore:** Luca Rimediotti (con Claude Code)
**Stato:** Approvato — pronto per il piano di implementazione
**Obiettivo:** Progetto #1 pubblico su GitHub. Deve dimostrare competenza RAG di livello senior (hybrid search + reranking + citazioni verificabili + eval riproducibile), non essere l'ennesimo wrapper LLM da tutorial.

---

## 1. Sintesi

DocuMind è un'applicazione di **RAG multi-documento**: l'utente carica dei PDF e pone domande in linguaggio naturale; il sistema risponde citando **documento e pagina reali**. La qualità del retrieval è ottenuta con **hybrid search** (dense + sparse fusi via Reciprocal Rank Fusion) seguito da **reranking** con cross-encoder, e viene **misurata** da un eval harness riproducibile. I modelli dense e reranker sono **multilingue (IT+EN)**: si possono porre domande in inglese su documenti italiani e viceversa (cross-lingual retrieval).

**Non-goal (esplicitamente fuori scope MVP):** autenticazione/multi-utente, OCR di PDF scannerizzati, persistenza della chat history su DB, deploy in cloud.

---

## 2. Vincoli d'ambiente (rilevati)

| Vincolo | Decisione di design |
|--------|---------------------|
| Docker assente | Qdrant in **modalità embedded** (`QdrantClient(path=...)`), nessun server da avviare |
| Python 3.14 (troppo recente per alcune wheel ML) | `uv` pinna **Python 3.12** solo per questo progetto |
| Nessuna dipendenza da torch desiderata | Embeddings e reranking via **fastembed** (ONNX runtime) |
| `gh` CLI assente | Repo creato a mano su GitHub, push via `git remote` (step separato, post-MVP) |
| Nessuna API key in env | `GEMINI_API_KEY` richiesta a runtime; documentata in `.env.example` |

**Obiettivo di portabilità:** `git clone` → `uv sync` → `uv run` funziona a freddo, con la sola aggiunta di una `GEMINI_API_KEY` (o, in offline mode, un Ollama locale).

---

## 3. Approccio scelto

**Approccio A — Full-local**: embedded Qdrant + fastembed per dense/sparse/rerank, Gemini solo per la sintesi finale della risposta.

**Offline mode (Approccio C) — opzionale, documentato nel README**: la generazione della risposta passa da Gemini a un modello locale via **Ollama**, dietro la stessa interfaccia provider. Nessun'altra parte del sistema cambia.

Approccio B (Qdrant Cloud + embeddings via API) scartato: moltiplica le API key, non ha sparse/hybrid nativo e tradisce l'obiettivo "clone-and-run".

---

## 4. Architettura

Monorepo unico (backend Python + frontend React):

```
documind/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI: POST /documents, POST /query (SSE), GET /health
│   │   ├── config.py          # pydantic-settings; legge .env
│   │   ├── store.py           # Qdrant embedded; collection con 2 named vectors (dense, sparse)
│   │   ├── ingest/
│   │   │   ├── loader.py      # PDF → testo per-pagina (pymupdf); la pagina è metadato
│   │   │   ├── chunker.py     # chunk ~500 token + overlap; ogni chunk porta {doc, page, text}
│   │   │   └── indexer.py     # fastembed dense+sparse → upsert in Qdrant
│   │   ├── retrieval/
│   │   │   ├── hybrid.py      # search dense + sparse → fusione RRF
│   │   │   ├── rerank.py      # cross-encoder ONNX riordina i top-N
│   │   │   └── pipeline.py    # orchestrazione: retrieve → rerank → costruzione contesto
│   │   ├── generation/
│   │   │   ├── prompt.py      # template con istruzioni di citazione + grounding
│   │   │   └── llm.py         # interfaccia provider: GeminiProvider (default) | OllamaProvider
│   │   └── models.py          # schemi pydantic: Chunk, Citation, QueryRequest, QueryResponse
│   ├── eval/
│   │   ├── dataset.py         # gold set etichettato: domanda → (doc, pagina/e)
│   │   └── evaluate.py        # Hit@k, MRR, Recall@k su 3 config; scrive eval/results.md
│   ├── tests/
│   ├── sample_docs/           # PDF demo (usati da README ed eval)
│   ├── .env.example
│   └── pyproject.toml         # uv; Python 3.12
├── frontend/                  # Vite + React + TS, tema dark
│   └── src/
│       ├── components/        # Uploader, ChatPanel, MessageBubble, CitationChip, SourceViewer
│       ├── hooks/useChat.ts   # streaming SSE
│       └── api.ts             # client tipizzato
├── docs/superpowers/specs/
├── .gitignore
└── README.md
```

### 4.1 Confini dei componenti

Ogni unità ha uno scopo unico, interfaccia definita e testabilità isolata:

- **`ingest/*`** — "PDF in, chunk con metadato pagina out". Non conosce Qdrant.
- **`store.py`** — unico proprietario della collection Qdrant (2 named vectors: `dense`, `sparse`). Espone `upsert(chunks)` e `search(dense_vec, sparse_vec, top_k)`.
- **`retrieval/*`** — "query in, chunk ordinati out". `hybrid.py` e `rerank.py` sono funzioni pure (input → output deterministico dato lo stato dello store); `pipeline.py` le compone.
- **`generation/llm.py`** — interfaccia astratta `LLMProvider.generate(prompt) -> stream[str]`. Punto d'innesto dell'offline mode.
- **`main.py`** — strato HTTP sottile: valida input, chiama i moduli, gestisce lo streaming SSE.
- **`frontend`** — comunica col backend solo via REST + SSE; nessuna logica di retrieval lato client.

---

## 5. Data flow

### 5.1 Ingest — `POST /documents`
1. Upload di uno o più PDF.
2. `loader` estrae il testo **per pagina** (numero di pagina conservato).
3. `chunker` spezza in chunk di ~500 token con overlap; ogni chunk = `{doc_id, doc_name, page, text}`.
4. `indexer` calcola vettore **dense** (`intfloat/multilingual-e5-small`, IT+EN) + **sparse** (BM25) via fastembed e fa upsert in Qdrant con entrambi i vettori nominati + payload dei metadati.
5. Risposta: `{doc_id, doc_name, n_chunks, pages}`.

### 5.2 Query — `POST /query` (risposta in streaming SSE)
1. Embedding della domanda (dense + sparse).
2. **Hybrid search**: due ricerche Qdrant (es. top-20 dense, top-20 sparse) → **Reciprocal Rank Fusion** fonde le due classifiche per *rango* (non per punteggio grezzo, perché cosine e BM25 hanno scale incompatibili) in un'unica lista di candidati.
3. **Rerank**: cross-encoder ONNX multilingue (`jina-reranker-v2-base-multilingual`) assegna un punteggio (domanda, chunk) ai candidati → tiene i **top-5**.
4. `pipeline` costruisce il blocco di contesto con fonti numerate `[1]…[5]` (con doc + pagina).
5. `generation`: prompt a Gemini — *"rispondi SOLO dal contesto fornito; cita le fonti come [n]; se il contesto non basta, dichiara che l'informazione non è nei documenti"* → token in streaming.
6. Output finale: `answer` + `citations: [{n, doc_name, page, snippet}]`. Il frontend rende i chip `[n]` cliccabili che aprono il PDF alla pagina citata.

---

## 6. Eval harness (differenziatore chiave)

- **Gold dataset** (`eval/dataset.py`): ~20 domande etichettate a mano, ognuna mappata alla/e `(doc, pagina)` che contiene la risposta, costruito sui `sample_docs`.
- **`eval/evaluate.py`**: calcola **Hit@k, MRR, Recall@k** su **3 configurazioni**:
  1. dense-only *naive* (baseline)
  2. hybrid + RRF
  3. hybrid + RRF + rerank
- Stampa una tabella comparativa e la salva in `eval/results.md`.
- README: *"esegui `uv run eval` per riprodurre questi numeri"*.

**Onestà dei numeri:** si riportano i valori **realmente misurati sui nostri dati**, non il "96%→100%" della card originale (marketing). Il valore è la **riproducibilità** e la progressione visibile naive → hybrid → rerank.

**Distinzione eval vs test:** l'eval misura la *qualità* del retrieval (metriche, non pass/fail) e non deve rompere la CI se peggiora di poco; i test proteggono la *correttezza* del codice. Un guard opzionale può verificare `Hit@5(hybrid+rerank) ≥ Hit@5(baseline)`.

---

## 7. Frontend (React curato, tema dark)

- **Stack:** Vite + React + TypeScript.
- **Componenti:**
  - `Uploader` — drag-drop PDF, mostra progresso ingest.
  - `ChatPanel` — lista messaggi + input.
  - `MessageBubble` — rende la risposta con chip citazione `[1][2]` inline.
  - `CitationChip` — click → apre `SourceViewer`.
  - `SourceViewer` — PDF.js che renderizza la pagina citata.
  - `useChat` — hook per lo streaming SSE.
  - `api.ts` — client HTTP tipizzato.
- **Stretch (non blocca l'MVP):** highlight esatto dello snippet dentro la pagina PDF. Per l'MVP è sufficiente aprire la pagina corretta.

---

## 8. Error handling & configurazione

- Config via `.env` + `pydantic-settings`; `.env.example` committato.
- `GEMINI_API_KEY` mancante → errore esplicito all'avvio con istruzioni.
- PDF senza testo estraibile (scannerizzato) → messaggio chiaro (OCR fuori scope).
- Query senza documenti indicizzati → HTTP 400 con messaggio parlante.
- **Grounding:** se il contesto non contiene la risposta, il modello dichiara "non presente nei documenti" e le citazioni sono vuote → nessuna allucinazione.
- Offline mode: Ollama non raggiungibile → messaggio chiaro con istruzioni.
- Rate-limit Gemini free tier → retry con backoff esponenziale.

---

## 9. Testing

- **Backend (`pytest`):**
  - Unit: `chunker` (metadato pagina preservato), `hybrid`/RRF (fusione corretta su classifiche note), `rerank` (riordino corretto), `prompt` (fonti numerate coerenti).
  - Integration: ciclo ingest→query su un PDF piccolo con Qdrant embedded temporaneo.
- **Frontend (Vitest):** un paio di test sul rendering delle citazioni (leggero, è MVP).
- **TDD:** applicato alle unità a logica pura (RRF, chunker) — test prima, implementazione dopo.

---

## 10. Criteri di successo (Definition of Done MVP)

1. `git clone` → `uv sync` → avvio backend e frontend funziona con la sola `GEMINI_API_KEY`.
2. Upload di ≥2 PDF, domanda, risposta in streaming con citazioni cliccabili che aprono la pagina corretta.
3. `uv run eval` produce `eval/results.md` con la tabella naive/hybrid/hybrid+rerank.
4. Suite `pytest` verde; test frontend verdi.
5. README con: cosa fa, GIF/demo, diagramma architettura, tabella eval, istruzioni run, sezione "offline mode".
6. Offline mode (Ollama) documentata e funzionante come alternativa a Gemini.

---

## 11. Rischi noti

- **Download modelli ONNX al primo avvio** (dense multilingue ~470 MB + reranker → indicativamente ~600–900 MB totali): documentato nel README; mitigato dal fatto che fastembed cache-a i modelli dopo il primo download.
- **Compatibilità wheel su Python 3.12**: mitigato dal pinning via `uv`.
- **Qualità eval su dataset piccolo**: ~20 domande è indicativo, non statisticamente robusto; dichiarato apertamente nel README.
