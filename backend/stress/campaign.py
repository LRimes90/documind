"""Campagna stress multi-modello: N modelli Ollama x (grounding + live) + soak strutturale.

Uso:
    uv run python -m stress.campaign

Requisiti: Ollama in ascolto con i MODELS scaricati (`ollama pull <nome>`).
Journal append-only in stress/campaign.jsonl: rilanciare riprende da dove si era
(i run gia' presenti nel journal vengono saltati). Per una campagna nuova,
cancellare il journal. Aggregato finale stampato a fine corsa.

Campagna di riferimento (2026-07-14, Mac ufficio): 10.400 esecuzioni valide,
99,91% pass — vedi README "Offline mode".
"""
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

BACKEND = Path(__file__).parent.parent
LOG = Path(__file__).parent / "campaign.jsonl"

MODELS = ["llama3.1", "qwen2.5:7b-instruct", "llama3.2:3b", "gemma3:4b", "phi4-mini"]
LIVE_PER_MODEL = 2
GROUNDING_PER_MODEL = 5
FAKE_RUNS = 89
RUN_TIMEOUT = 1800  # ponytail: 30 min per run; il piu' lento osservato e' 430s


def _run(args, model=None):
    env = dict(os.environ)
    if model:
        env["OLLAMA_MODEL"] = model
    t0 = time.time()
    p = subprocess.run(
        ["uv", "run", "python", "-m", "stress.run", *args],
        cwd=BACKEND, env=env, capture_output=True, text=True, timeout=RUN_TIMEOUT,
    )
    return p.stdout + p.stderr, round(time.time() - t0, 1)


def _log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)


def _attempt(done_set, kind, args, model, i, n_scen):
    """Esegue un run; un errore/timeout viene loggato e la campagna prosegue."""
    if (kind, model, i) in done_set:
        return
    try:
        out, dt = _run(args, model)
    except subprocess.TimeoutExpired:
        _log({"kind": "error", "run": kind, "model": model, "i": i, "err": f"timeout {RUN_TIMEOUT}s"})
        return
    except Exception as e:  # noqa: BLE001
        _log({"kind": "error", "run": kind, "model": model, "i": i, "err": str(e)[:200]})
        return
    if kind == "grounding":
        m = re.search(r"\[grounding-live\] (\d+)/(\d+)", out)
        ok, tot = (int(m[1]), int(m[2])) if m else (-1, n_scen)  # -1 = output non parsabile
        _log({"kind": kind, "model": model, "i": i, "ok": ok, "tot": tot, "sec": dt})
    else:
        m = re.search(r"scenari=(\d+) fallimenti=(\d+)", out)
        scen, fail = (int(m[1]), int(m[2])) if m else (n_scen, -1)
        _log({"kind": kind, "model": model, "i": i, "scenari": scen, "fail": fail, "sec": dt})


def _aggregate():
    rows = [json.loads(l) for l in open(LOG)]
    g = defaultdict(lambda: [0, 0])
    lv = defaultdict(lambda: [0, 0])
    fake_pass = fake_tot = invalid = 0
    for r in rows:
        if r["kind"] == "grounding":
            if r["ok"] < 0:
                invalid += r["tot"]
            else:
                g[r["model"]][0] += r["ok"]
                g[r["model"]][1] += r["tot"]
        elif r["kind"] == "live":
            if r["fail"] < 0:
                invalid += r["scenari"]
            else:
                lv[r["model"]][0] += r["scenari"] - r["fail"]
                lv[r["model"]][1] += r["scenari"]
        elif r["kind"] == "fake":
            fake_pass += r["scenari"] - max(r["fail"], 0)
            fake_tot += r["scenari"]
    tot = sum(v[1] for v in g.values()) + sum(v[1] for v in lv.values()) + fake_tot
    ok = sum(v[0] for v in g.values()) + sum(v[0] for v in lv.values()) + fake_pass
    print(f"\n{'Modello':<22}{'Grounding':<14}{'Live'}")
    for m in g:
        print(f"{m:<22}{g[m][0]}/{g[m][1]:<11}{lv[m][0]}/{lv[m][1]}")
    print(f"Soak strutturale: {fake_pass}/{fake_tot}")
    print(f"TOTALE VALIDO: {ok}/{tot} pass ({ok / tot:.2%})" if tot else "nessun dato")
    if invalid:
        print(f"Esecuzioni invalide (modello non testabile): {invalid}")
    return tot - ok


def main() -> int:
    done_set = set()
    if LOG.exists():
        for line in open(LOG):
            r = json.loads(line)
            if r["kind"] in ("grounding", "live", "fake"):
                done_set.add((r["kind"], r.get("model"), r["i"]))
    for model in MODELS:
        for i in range(GROUNDING_PER_MODEL):
            _attempt(done_set, "grounding", ["--grounding"], model, i, 20)
        for i in range(LIVE_PER_MODEL):
            _attempt(done_set, "live", [], model, i, 100)
    for i in range(FAKE_RUNS):
        _attempt(done_set, "fake", ["--fake"], None, i, 100)
    return _aggregate()


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
