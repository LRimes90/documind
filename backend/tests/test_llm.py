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
