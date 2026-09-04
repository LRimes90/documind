import json
import logging
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.models import QueryRequest
from app.deps import get_state
from app.ingest.indexer import ingest_pdf
from app.ingest.loader import EmptyPdfError
from app.retrieval.pipeline import retrieve
from app.generation.prompt import build_context, build_prompt
from app.generation.llm import get_provider, is_transient

app = FastAPI(title="DocuMind API")

logger = logging.getLogger(__name__)


def _error_payload(exc: Exception) -> dict[str, str]:
    """Costruisce il corpo dell'evento `error` inviato al client durante lo streaming.

    L'eccezione completa e' sempre gia' finita nel log del server (con stack trace):
    questa funzione decide soltanto **quanto** di quel guasto il browser deve vedere.
    """
    # `str(exc)` non esce mai dal server: puo' contenere host e porta del provider,
    # percorsi del filesystem, frammenti di chiave API. Al client va solo la
    # distinzione che gli serve per decidere cosa fare — riprovare o smettere.
    if is_transient(exc):
        return {"detail": "Il modello non ha risposto. Riprova tra qualche istante."}
    return {
        "detail": "Errore interno durante la generazione della risposta. "
        "I dettagli sono nel log del server."
    }


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
        info = ingest_pdf(
            tmp_path, file.filename or "documento.pdf", state.embedder, state.store
        )
    except EmptyPdfError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:  # file corrotto / non-PDF / illeggibile
        raise HTTPException(status_code=422, detail=f"File PDF non valido o illeggibile: {e}")
    finally:
        # su Windows un fitz.open fallito può lasciare il file bloccato:
        # la pulizia non deve mai mascherare la risposta HTTP
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return info


@app.post("/query")
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Domanda vuota.")
    state = get_state()
    if state.store.count() == 0:
        raise HTTPException(
            status_code=400, detail="Nessun documento indicizzato. Carica prima un PDF."
        )
    provider = get_provider()
    chunks = retrieve(
        req.question, state.embedder, state.store, state.reranker, provider=provider
    )
    context, citations = build_context(chunks)
    prompt = build_prompt(req.question, context)

    def event_stream():
        try:
            for token in provider.generate(prompt):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
            payload = json.dumps([c.model_dump() for c in citations])
            yield f"event: citations\ndata: {payload}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            # Lo status HTTP e' stato impegnato col primo byte: un HTTPException qui
            # produrrebbe un 200 troncato, non un 500. L'unico canale di errore
            # ancora aperto e' un evento dentro lo stream stesso.
            logger.exception("Streaming della risposta interrotto")
            yield f"event: error\ndata: {json.dumps(_error_payload(exc))}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
