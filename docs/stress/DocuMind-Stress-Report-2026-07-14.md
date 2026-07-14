# DocuMind — Report campagna stress multi-modello offline

**Data:** 14 luglio 2026 · **Macchina:** Mac ufficio (Apple Silicon, Ollama 0.19) · **Repo:** `LRimes90/documind` (privato), commit `6e7c295`

## Esito in una riga

**10.400 esecuzioni di scenario valide, 10.391 passate (99,91%)** su 5 modelli locali + soak strutturale. Le 9 mancanze sono tutte lapse di grounding dei modelli 3–4B; l'infrastruttura (retrieval, ingest, concurrency, API) non ha mai fallito.

## Verifica offline preliminare (gate handoff)

| Verifica | Risultato |
|---|---|
| pytest | 28/28 (identico a Windows) |
| Smoke provider Ollama | OK |
| e2e offline | risposta grounded + citazione `geografia.pdf` p.1 |
| Eval retrieval | Hit@1 90→95→100%, MRR .938→.967→1.000 (identico a Windows, HyDE neutro) |
| Grounding | 20/20 (llama3.1 e qwen2.5:7b-instruct) |
| Stress `--fake` / live | 100/100 / 100/100 |

## Campagna estesa — struttura

- **Per modello:** 5 run grounding (20 domande fuori-corpus) + 2 stress live completi (100 scenari: 40 retrieval, 20 grounding, 12 ingest, 20 concurrency, 8 api_edge)
- **Soak strutturale:** 89 run `--fake` da 100 scenari (LLM-independent; caccia a flakiness, race condition, leak)
- Driver resiliente con resume (`campaign.jsonl` come journal append-only)

## Risultati per modello

| Modello | Grounding | Live | Totale | t medio grounding | t medio live |
|---|---|---|---|---|---|
| llama3.1 (8B) | 100/100 | 200/200 | **300/300** | 77s | 242s |
| qwen2.5:7b-instruct | 100/100 | 200/200 | **300/300** | 54s | 191s |
| gemma3:4b | 100/100 | 200/200 | **300/300** | 54s | 221s |
| llama3.2:3b | 98/100 | 199/200 | 297/300 | 63s | 364s |
| phi4-mini (3.8B) | 94/100 | 200/200 | 294/300 | 71s | 210s |
| **Soak strutturale** | — | — | **8.900/8.900** | — | ~67s/run |

**Lettura:** da 7B in su il grounding è perfetto; a 3–4B compare qualche risposta a domande fuori-corpus (94–98%). gemma3:4b è l'eccezione: 300/300, miglior rapporto qualità/dimensione. Nessun fallimento mai in retrieval/ingest/concurrency/api_edge, su nessun modello, in nessuna ripetizione.

## qwen3.5-9b (GGUF LM Studio): non testabile via Ollama

Import GGUF riuscito (con `SYSTEM /no_think` + stop token nel Modelfile: gli smoke test brevi rispondevano), ma su prompt RAG reali il modello **carica in GPU e non genera mai token** → `httpx.ReadTimeout`. Causa: architettura `qwen35` più recente del runner llama.cpp incluso in Ollama 0.19; LM Studio funziona perché aggiorna il runtime più spesso. 300 esecuzioni marcate invalide e compensate con run soak extra.

## Bug trovati e corretti durante il lavoro

1. **Check grounding su body SSE grezzo** (`stress/run.py`, fix in `cfab569`): il verdetto cercava `"non "` nel transport SSE, ma Ollama streamma token che spezzano le parole tra eventi (`data: "non"` + `data: " è"`) → falsi negativi (9/20 apparenti vs 20/20 reali). Con Gemini era invisibile (chunk grandi). Fix: la risposta viene ricomposta dai token prima del check.

## Finding aperti (backlog suggerito)

1. **SSE senza `event: error`:** se il provider LLM si blocca, lo stream muore con eccezione grezza lato server. Un evento di errore pulito renderebbe il client (ora che c'è il frontend React) più robusto.
2. **Provider OpenAI-compatible:** aprirebbe LM Studio (`localhost:1234`), vLLM, LocalAI — e permetterebbe di testare qwen3.5-9b davvero.

## Riproducibilità

```bash
cd backend
uv run python -m stress.campaign     # rilancia l'intera campagna (resume automatico)
```

Il driver è in `backend/stress/campaign.py` (committato); journal e aggregato vengono scritti in `backend/stress/`. Dati grezzi di questa campagna: `DocuMind-campaign-2026-07-14.jsonl` accanto a questo report.
