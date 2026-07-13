import pytest

pytestmark = pytest.mark.slow

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
