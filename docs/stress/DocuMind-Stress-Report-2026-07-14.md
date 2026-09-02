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

## qwen3.5-9b (GGUF LM Studio): non testabile via Ollama — ⚠️ diagnosi ritrattata

Import GGUF riuscito (con `SYSTEM /no_think` + stop token nel Modelfile: gli smoke test brevi rispondevano), ma su prompt RAG reali il modello **carica in GPU e non genera mai token** → `httpx.ReadTimeout`. Causa: architettura `qwen35` più recente del runner llama.cpp incluso in Ollama 0.19; LM Studio funziona perché aggiorna il runtime più spesso. 300 esecuzioni marcate invalide e compensate con run soak extra.

> **Rettifica del 21 ago 2026 — la causa qui sopra è sbagliata.** Rimisurato su Ollama
> 0.32.5 via `/v1/chat/completions`: qwen3.5 emette **194 chunk `delta.reasoning`** e poi
> 2 chunk `delta.content`, chiude con `finish_reason: "stop"` e `[DONE]` regolare. I token
> c'erano dall'inizio: il chain-of-thought viaggia in `delta.reasoning`, campo fuori dallo
> standard OpenAI, mentre `delta.content` resta `""` finché il modello non ha finito di
> pensare. Un client conforme che legge solo `content` conta zero e conclude che il modello
> sia rotto — è quello che è successo a luglio. Non era il runner llama.cpp, e il timeout
> era il nostro client in attesa di un campo che non sarebbe mai arrivato. Gestito in
> `eff0f50`; le 300 esecuzioni restano marcate invalide perché la misura di luglio non è
> recuperabile, non perché il modello non funzioni.

## Bug trovati e corretti durante il lavoro

1. **Check grounding su body SSE grezzo** (`stress/run.py`, fix in `cfab569`): il verdetto cercava `"non "` nel transport SSE, ma Ollama streamma token che spezzano le parole tra eventi (`data: "non"` + `data: " è"`) → falsi negativi (9/20 apparenti vs 20/20 reali). Con Gemini era invisibile (chunk grandi). Fix: la risposta viene ricomposta dai token prima del check.

## Finding aperti (backlog suggerito) — ✅ entrambi chiusi il 21 ago 2026

1. ✅ **SSE senza `event: error`** — chiuso in `8f3412d`. Lo status HTTP è impegnato al primo
   byte, quindi un `raise` dentro il generatore già avviato produce un 200 troncato, non un
   500: l'unico canale d'errore residuo è un evento dentro il protocollo. Il payload
   distingue solo transitorio/permanente, per non far uscire host, porte e frammenti di
   chiave API in `str(exc)`. Lato client, uno stream che chiude senza `done` né `error` non
   è più «fine normale».
2. ✅ **Provider OpenAI-compatible** — chiuso in `eff0f50`. Validato contro un server reale
   (Ollama 0.32.5, che espone anch'esso `/v1/chat/completions`): le fixture dei test sono
   righe SSE registrate con `curl`, non inventate. Il protocollo vero differiva dal doppio
   su due punti — `role` e `content` arrivano nello **stesso** chunk, e i reasoning model
   usano `delta.reasoning`. Terzo bug emerso qui e non previsto in questo report: uno stream
   che chiude senza testo arrivava alla UI come un successo.

Aperto, ma è un limite dell'hardware e non del progetto: la suite frontend va lanciata con
`--maxWorkers=1` (o `--pool=threads`) su macchine dove N worker fork che caricano jsdom
insieme sforano il timeout di handshake del pool.

## Riproducibilità

```bash
cd backend
uv run python -m stress.campaign     # rilancia l'intera campagna (resume automatico)
```

Il driver è in `backend/stress/campaign.py` (committato); journal e aggregato vengono scritti in `backend/stress/`. Dati grezzi di questa campagna: `DocuMind-campaign-2026-07-14.jsonl` accanto a questo report.
