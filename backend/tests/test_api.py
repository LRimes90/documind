import pytest

pytestmark = pytest.mark.slow

import io
import json
import fitz
import httpx
from fastapi.testclient import TestClient


def _pdf_bytes(pages_text):
    doc = fitz.open()
    for t in pages_text:
        doc.new_page().insert_text((72, 72), t)
    data = doc.tobytes()
    doc.close()
    return data


def test_documents_then_query_stream(tmp_path, monkeypatch):
    from app.config import settings as cfg
    import app.deps as deps

    monkeypatch.setattr(cfg, "qdrant_path", str(tmp_path / "q"))
    monkeypatch.setattr(cfg, "collection", "test_api")
    deps.get_state.cache_clear()

    class FakeProvider:
        def generate(self, prompt):
            yield "Parigi [1]."

    # patch al punto d'uso (main importa get_provider nel proprio namespace)
    monkeypatch.setattr("app.main.get_provider", lambda: FakeProvider())

    from app.main import app

    client = TestClient(app)
    files = {
        "file": (
            "doc.pdf",
            io.BytesIO(_pdf_bytes(["La capitale della Francia è Parigi."])),
            "application/pdf",
        )
    }
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    assert r.json()["n_chunks"] >= 1

    with client.stream("POST", "/query", json={"question": "Capitale della Francia?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "Parigi" in body
    assert "citations" in body

    deps.get_state.cache_clear()


def test_query_stream_emits_error_event_on_provider_failure(tmp_path, monkeypatch):
    """Provider che muore DOPO il primo token.

    Lo status HTTP e' gia' stato impegnato a 200 col primo byte: il guasto deve
    raggiungere il client come `event: error` dentro lo stream, e lo stream non
    deve chiudersi con un `done` (che il frontend leggerebbe come successo).
    """
    from app.config import settings as cfg
    import app.deps as deps

    monkeypatch.setattr(cfg, "qdrant_path", str(tmp_path / "q"))
    monkeypatch.setattr(cfg, "collection", "test_api_err")
    deps.get_state.cache_clear()

    class DyingProvider:
        def generate(self, prompt):
            yield "La capitale "
            raise httpx.ReadTimeout("il provider ha smesso di generare token")

    monkeypatch.setattr("app.main.get_provider", lambda: DyingProvider())

    from app.main import app

    client = TestClient(app)
    files = {
        "file": (
            "doc.pdf",
            io.BytesIO(_pdf_bytes(["La capitale della Francia è Parigi."])),
            "application/pdf",
        )
    }
    assert client.post("/documents", files=files).status_code == 200

    with client.stream("POST", "/query", json={"question": "Capitale della Francia?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    assert "event: error" in body
    assert "event: done" not in body
    marker = "event: error" + chr(10) + "data: "
    payload = json.loads(body.split(marker)[1].split(chr(10) * 2)[0])
    assert payload.get("detail")  # qualunque policy scelta, un detail non vuoto

    deps.get_state.cache_clear()
