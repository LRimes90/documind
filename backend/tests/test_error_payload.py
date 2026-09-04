import httpx
from app.main import _error_payload

SECRET = "http://192.168.1.50:11434 key=AIzaSyTOPSECRET /home/luca/documind"


def test_transient_suggests_retry():
    payload = _error_payload(httpx.HTTPStatusError("503 UNAVAILABLE", request=None, response=None))
    assert "iprova" in payload["detail"]


def test_definitive_does_not_suggest_retry():
    payload = _error_payload(ValueError("modello non configurato"))
    assert "iprova" not in payload["detail"]


def test_payload_never_leaks_exception_text():
    """Il contenuto dell'eccezione non deve raggiungere il browser."""
    for exc in (RuntimeError(SECRET), httpx.ReadTimeout("503 " + SECRET)):
        blob = " ".join(_error_payload(exc).values())
        assert "11434" not in blob
        assert "AIzaSy" not in blob
        assert "/home/luca" not in blob
