"""Provider LLM dietro un'interfaccia: Gemini (default), Ollama e qualunque
server OpenAI-compatible (LM Studio, vLLM, LocalAI) per il modo offline."""
import time
from typing import Iterator, Protocol
import httpx
from app.config import settings

_TRANSIENT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")


def is_transient(exc: Exception) -> bool:
    return any(m in str(exc) for m in _TRANSIENT_MARKERS)


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> Iterator[str]: ...


class GeminiProvider:
    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def generate(self, prompt: str) -> Iterator[str]:
        delay = 2.0
        for attempt in range(5):
            started = False
            try:
                stream = self._client.models.generate_content_stream(
                    model=self._model, contents=prompt
                )
                for chunk in stream:
                    started = True
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:  # retry solo su errori transitori e prima del 1° token
                if started or attempt == 4 or not is_transient(e):
                    raise
                time.sleep(delay)
                delay *= 2


class OllamaProvider:
    def __init__(self) -> None:
        self._host = settings.ollama_host
        self._model = settings.ollama_model

    def generate(self, prompt: str) -> Iterator[str]:
        import json

        with httpx.stream(
            "POST",
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": True},
            timeout=120,
        ) as r:
            r.raise_for_status()  # senza questo un modello inesistente da' 0 token, non un errore
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]


class OpenAICompatProvider:
    """Qualunque server che parla l'API `/v1/chat/completions` di OpenAI.

    Copre LM Studio, vLLM, LocalAI e llama.cpp server. Serve per i modelli la cui
    architettura e' piu' recente del runner llama.cpp incluso in Ollama (es.
    qwen3.5: carica in GPU e non genera mai token via Ollama, funziona in LM Studio).
    """

    def __init__(self) -> None:
        self._url = settings.openai_base_url.rstrip("/") + "/chat/completions"
        self._model = settings.openai_model
        self._headers = {"Content-Type": "application/json"}
        if settings.openai_api_key:
            self._headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    def generate(self, prompt: str) -> Iterator[str]:
        import json

        emitted = 0
        reasoned = False
        with httpx.stream(
            "POST",
            self._url,
            headers=self._headers,
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                # Il protocollo e' SSE: righe `data: {...}`, chiuse da `data: [DONE]`.
                # Tutto il resto (commenti keep-alive, righe vuote) va ignorato.
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                # Alcuni server chiudono con un chunk di sole statistiche: choices vuoto.
                choices = json.loads(payload).get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                # I reasoning model (qwen3.5 via Ollama) mandano il chain-of-thought
                # in `delta.reasoning` — campo fuori dallo standard OpenAI — e tengono
                # `content` a "" finche' non hanno finito di pensare. Il ragionamento
                # non deve entrare nella risposta citata: lo registriamo e lo scartiamo.
                if delta.get("reasoning"):
                    reasoned = True
                content = delta.get("content")
                if content:
                    emitted += 1
                    yield content
        if emitted == 0:
            # Lo stream si e' chiuso "bene" (finish_reason + [DONE]) ma senza testo:
            # senza questa eccezione l'utente riceverebbe `done` e una risposta vuota,
            # cioe' un fallimento indistinguibile da un successo.
            raise RuntimeError(
                "Il modello ha chiuso lo stream senza produrre testo: ha esaurito il "
                "budget di token ragionando (max_tokens troppo basso per un reasoning "
                "model)." if reasoned else
                "Il modello ha chiuso lo stream senza produrre testo."
            )

def get_provider() -> LLMProvider:
    provider = settings.llm_provider
    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY mancante. Impostala in .env "
                "(https://aistudio.google.com/apikey) oppure usa LLM_PROVIDER=ollama."
            )
        return GeminiProvider()
    if provider == "ollama":
        return OllamaProvider()
    if provider == "openai":
        return OpenAICompatProvider()
    raise ValueError(f"LLM_PROVIDER sconosciuto: {provider!r}")
