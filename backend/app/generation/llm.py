"""Provider LLM: Gemini (default) e Ollama (offline mode) dietro un'interfaccia."""
import time
from typing import Iterator, Protocol
import httpx
from app.config import settings

_TRANSIENT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")


def _is_transient(exc: Exception) -> bool:
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
                if started or attempt == 4 or not _is_transient(e):
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
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("response"):
                    yield data["response"]


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
    raise ValueError(f"LLM_PROVIDER sconosciuto: {provider!r}")
