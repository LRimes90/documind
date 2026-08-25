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


class _FakeStream:
    """Rimpiazza httpx.stream: espone le righe SSE come le vedrebbe il client."""

    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter(self._lines)


def _openai_provider(monkeypatch, lines, api_key=""):
    monkeypatch.setattr(llm.settings, "llm_provider", "openai")
    monkeypatch.setattr(llm.settings, "openai_base_url", "http://localhost:1234/v1/")
    monkeypatch.setattr(llm.settings, "openai_model", "qwen3.5-9b")
    monkeypatch.setattr(llm.settings, "openai_api_key", api_key)
    sent = {}

    def fake_stream(method, url, **kwargs):
        sent["method"] = method
        sent["url"] = url
        sent.update(kwargs)
        return _FakeStream(lines)

    monkeypatch.setattr(llm.httpx, "stream", fake_stream)
    return llm.get_provider(), sent


def test_openai_provider_parses_sse_deltas(monkeypatch):
    lines = [
        'data: {"choices":[{"delta":{"content":"La capitale "}}]}',
        'data: {"choices":[{"delta":{"content":"e Parigi."}}]}',
        "data: [DONE]",
    ]
    provider, sent = _openai_provider(monkeypatch, lines)
    assert list(provider.generate("domanda")) == ["La capitale ", "e Parigi."]
    assert sent["url"] == "http://localhost:1234/v1/chat/completions"
    assert sent["json"]["model"] == "qwen3.5-9b"
    assert sent["json"]["stream"] is True
    assert sent["json"]["messages"] == [{"role": "user", "content": "domanda"}]


def test_openai_provider_ignores_noise(monkeypatch):
    """Keep-alive, righe vuote, delta di sole statistiche e ruolo senza contenuto."""
    lines = [
        ": ping",
        "",
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"Parigi"}}]}',
        'data: {"choices":[],"usage":{"total_tokens":42}}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"mai letto"}}]}',
    ]
    provider, _ = _openai_provider(monkeypatch, lines)
    assert list(provider.generate("q")) == ["Parigi"]


def test_openai_provider_auth_header_only_when_key_set(monkeypatch):
    provider, _ = _openai_provider(monkeypatch, ["data: [DONE]"])
    assert "Authorization" not in provider._headers
    provider, _ = _openai_provider(monkeypatch, ["data: [DONE]"], api_key="sk-abc")
    assert provider._headers["Authorization"] == "Bearer sk-abc"


# --- Righe catturate da Ollama 0.32.5 su /v1/chat/completions (21 ago 2026, Mac mini).
# Non sono inventate: il doppio riproduceva un protocollo plausibile, questo e' quello
# che il server manda davvero. Due differenze che il doppio non aveva: `role` e `content`
# arrivano nello STESSO chunk, e i reasoning model usano `delta.reasoning`.
_OLLAMA_CHUNK = (
    'data: {{"id":"chatcmpl-659","object":"chat.completion.chunk","model":"llama3.2:3b",'
    '"system_fingerprint":"fp_ollama","choices":[{{"index":0,"delta":{delta},'
    '"finish_reason":{finish}}}]}}'
)


def _ollama_lines(deltas, finish="null"):
    """Righe SSE nel formato reale: chunk, riga vuota, ..., `data: [DONE]`."""
    out = []
    for d in deltas:
        out += [_OLLAMA_CHUNK.format(delta=d, finish="null"), ""]
    out += [
        _OLLAMA_CHUNK.format(delta='{"role":"assistant","content":""}', finish=f'"{finish}"'),
        "",
        "data: [DONE]",
    ]
    return out


def test_openai_provider_on_real_ollama_stream(monkeypatch):
    """Il flusso reale di llama3.2 via Ollama: role e content nello stesso chunk."""
    lines = _ollama_lines(
        [
            '{"role":"assistant","content":"Par"}',
            '{"role":"assistant","content":"igi"}',
            '{"role":"assistant","content":"."}',
        ],
        finish="stop",
    )
    provider, _ = _openai_provider(monkeypatch, lines)
    assert "".join(provider.generate("capitale della Francia?")) == "Parigi."


def test_reasoning_is_not_emitted_but_content_is(monkeypatch):
    """qwen3.5 manda il chain-of-thought in `delta.reasoning`: fuori dalla risposta."""
    lines = _ollama_lines(
        [
            '{"role":"assistant","content":"","reasoning":"Thinking Process:"}',
            '{"role":"assistant","content":"","reasoning":" la capitale e Parigi"}',
            '{"role":"assistant","content":"Parigi"}',
        ],
        finish="stop",
    )
    provider, _ = _openai_provider(monkeypatch, lines)
    assert list(provider.generate("q")) == ["Parigi"]


def test_reasoning_without_content_is_an_error(monkeypatch):
    """Budget di token esaurito ragionando: `[DONE]` regolare, zero testo.

    Senza questa eccezione l'utente riceve `event: done` e una risposta vuota,
    cioe' un fallimento che la UI presenta come successo.
    """
    lines = _ollama_lines(
        ['{"role":"assistant","content":"","reasoning":"Thinking Process:"}'],
        finish="length",
    )
    provider, _ = _openai_provider(monkeypatch, lines)
    with pytest.raises(RuntimeError, match="max_tokens"):
        list(provider.generate("q"))


def test_empty_stream_without_reasoning_is_an_error(monkeypatch):
    """Anche senza reasoning, uno stream senza testo non e' un successo."""
    provider, _ = _openai_provider(monkeypatch, ["data: [DONE]"])
    with pytest.raises(RuntimeError, match="senza produrre testo"):
        list(provider.generate("q"))
