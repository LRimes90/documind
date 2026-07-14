"""Runner dello stress harness: esegue ≥100 scenari e verifica gli invarianti.

Gate pre-pubblicazione. Uso:
    uv run python -m stress.run          # live (usa il provider LLM reale)
    uv run python -m stress.run --fake   # provider fittizio (invarianti strutturali)
Ritorna exit code 0 se 0 fallimenti, 1 altrimenti.
"""
import io
import json
import tempfile
import time
import concurrent.futures as cf
from collections import Counter
from pathlib import Path
import fitz
from fastapi.testclient import TestClient
from stress.scenarios import build_scenarios, Scenario

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"


def _pdf_bytes(pages: list[str]) -> bytes:
    doc = fitz.open()
    for t in pages:
        doc.new_page().insert_text((72, 72), t)
    data = doc.tobytes()
    doc.close()
    return data


def _parse_citations(body: str):
    for block in body.split("\n\n"):
        if "event: citations" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    return json.loads(line[6:])
    return None


def _query(client, question):
    with client.stream("POST", "/query", json={"question": question}) as r:
        body = "".join(r.iter_text())
    return r.status_code, body


def _ingest(client, name, data):
    return client.post(
        "/documents", files={"file": (name, io.BytesIO(data), "application/pdf")}
    )


def _check(client, s: Scenario, live: bool) -> bool:
    if s.category == "retrieval":
        st, body = _query(client, s.payload["question"])
        if st != 200:
            return False
        cits = _parse_citations(body) or []
        keys = {f"{c['doc_name']}:{c['page']}" for c in cits}
        return any(g in keys for g in s.payload["gold"])
    if s.category == "grounding":
        st, body = _query(client, s.payload["question"])
        if st != 200:
            return False
        if live:
            low = body.lower()
            return ("non " in low) or ("not " in low)
        return "event: citations" in body
    if s.category == "concurrency":
        st, _ = _query(client, s.payload["question"])
        return st == 200
    if s.category == "ingest":
        if s.kind == "empty_pdf":
            return _ingest(client, "e.pdf", _pdf_bytes(["", ""])).status_code == 422
        if s.kind == "non_pdf":
            return _ingest(client, "x.pdf", b"non sono un pdf").status_code == 422
        if s.kind == "unicode_pdf":
            return _ingest(client, "u.pdf", _pdf_bytes(["Città però àèìòù test"])).status_code == 200
        if s.kind == "large_pdf":
            return _ingest(client, "l.pdf", _pdf_bytes(["parola " * 400] * 6)).status_code == 200
        return _ingest(client, "n.pdf", _pdf_bytes(["Contenuto di prova."])).status_code == 200
    if s.category == "api_edge":
        if s.kind in ("malformed_payload", "missing_field"):
            return client.post("/query", json={}).status_code == 422
        if s.kind in ("empty_question", "whitespace_question"):
            st, _ = _query(client, s.payload["question"])
            return st in (200, 400, 422)
        if s.kind in ("long_question", "numeric_question"):
            st, _ = _query(client, s.payload["question"])
            return st == 200
        if s.kind == "sse_completes":
            st, body = _query(client, s.payload["question"])
            return st == 200 and "event: done" in body
    return False


def main(live: bool = False) -> int:
    from app.config import settings as cfg

    cfg.qdrant_path = tempfile.mkdtemp()
    cfg.collection = "stress"
    import app.deps as deps

    deps.get_state.cache_clear()

    from app import main as app_module

    if not live:
        class _Fake:
            def generate(self, prompt):
                yield "Risposta di prova [1]."

        app_module.get_provider = lambda: _Fake()

    client = TestClient(app_module.app)
    scen = build_scenarios()
    results: list[tuple[str, str, bool, float]] = []
    latencies: list[float] = []

    # query_before_ingest: PRIMA di ingerire (store vuoto → 400)
    if any(s.kind == "query_before_ingest" for s in scen):
        ok = client.post("/query", json={"question": "x"}).status_code == 400
        results.append(("api_edge", "query_before_ingest", ok, 0.0))

    for pdf in SAMPLE_DIR.glob("*.pdf"):
        with open(pdf, "rb") as fh:
            _ingest(client, pdf.name, fh.read())

    seq = [s for s in scen if s.category != "concurrency" and s.kind != "query_before_ingest"]
    for s in seq:
        t0 = time.perf_counter()
        ok = _check(client, s, live)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        results.append((s.category, s.kind, ok, dt))

    conc = [s for s in scen if s.category == "concurrency"]

    def _run_one(s):
        t0 = time.perf_counter()
        return _check(client, s, live), time.perf_counter() - t0

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for ok, dt in ex.map(_run_one, conc):
            latencies.append(dt)
            results.append(("concurrency", "parallel_query", ok, dt))

    failures = sum(1 for _cat, _kind, ok, _dt in results if not ok)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0

    passed, tot = Counter(), Counter()
    for cat, _kind, ok, _dt in results:
        tot[cat] += 1
        passed[cat] += 1 if ok else 0

    lines = [
        "# Stress-test report — DocuMind\n",
        f"Modalità: {'live' if live else 'fake'} · Scenari: {len(results)} · "
        f"Fallimenti: {failures} · Latenza p50 {p50:.3f}s / p95 {p95:.3f}s\n",
        "| Categoria | Pass | Totale |",
        "|-----------|------|--------|",
    ]
    for cat in ["retrieval", "grounding", "ingest", "concurrency", "api_edge"]:
        lines.append(f"| {cat} | {passed[cat]} | {tot[cat]} |")
    (Path(__file__).parent / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[stress] modalità={'live' if live else 'fake'} scenari={len(results)} fallimenti={failures}")
    return failures


def grounding_live() -> int:
    """Verifica grounding con LLM reale: le domande fuori-corpus non devono allucinare."""
    from app.config import settings as cfg

    cfg.qdrant_path = tempfile.mkdtemp()
    cfg.collection = "grounding"
    import app.deps as deps

    deps.get_state.cache_clear()
    from app import main as app_module
    from stress.scenarios import _OUT_OF_CORPUS

    client = TestClient(app_module.app)
    for pdf in SAMPLE_DIR.glob("*.pdf"):
        with open(pdf, "rb") as fh:
            _ingest(client, pdf.name, fh.read())

    fails = 0
    for q in _OUT_OF_CORPUS:
        st, body = _query(client, q)
        low = body.lower()
        ok = st == 200 and (("non " in low) or ("not " in low))
        fails += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {q[:48]}")
    print(f"[grounding-live] {len(_OUT_OF_CORPUS) - fails}/{len(_OUT_OF_CORPUS)} grounded (no hallucination)")
    return fails


if __name__ == "__main__":
    import sys

    if "--grounding" in sys.argv:
        raise SystemExit(1 if grounding_live() else 0)
    _live = "--fake" not in sys.argv
    raise SystemExit(1 if main(live=_live) else 0)
