import json
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
        info = ingest_pdf(
            tmp_path, file.filename or "documento.pdf", state.embedder, state.store
        )
    except EmptyPdfError as e:
        raise HTTPException(status_code=422, detail=str(e))
    finally:
        os.unlink(tmp_path)
    return info


@app.post("/query")
def query(req: QueryRequest):
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
        for token in provider.generate(prompt):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"
        payload = json.dumps([c.model_dump() for c in citations])
        yield f"event: citations\ndata: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
