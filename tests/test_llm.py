"""llm.py with the Gemini client mocked: no network, no API key."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google import genai
from google.genai import errors, types

import llm


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def fake_gemini(monkeypatch):
    """Replace genai.Client; returns a function that installs canned outcomes."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("PAPERPILOT_MODEL", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    state = {}

    def install(*outcomes):
        models = FakeModels(outcomes)

        class FakeClient:
            def __init__(self, **kwargs):
                state["client_kwargs"] = kwargs
                self.models = models

        monkeypatch.setattr(genai, "Client", FakeClient)
        state["models"] = models
        return state

    return install


def _resp(text, finish="STOP"):
    return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)])


def _api_error(code):
    return errors.ClientError(code, {"error": {"code": code, "message": "x", "status": "S"}}) if code < 500 \
        else errors.ServerError(code, {"error": {"code": code, "message": "x", "status": "S"}})


def test_json_call_uses_default_model_json_mime_and_timeout(fake_gemini):
    state = fake_gemini(_resp('```json\n{"ok": true}\n```'))
    assert llm.generate_json("prompt") == '{"ok": true}'
    call = state["models"].calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["contents"] == "prompt"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].max_output_tokens == llm.MAX_OUTPUT_TOKENS
    assert state["client_kwargs"]["api_key"] == "test-key"
    assert state["client_kwargs"]["http_options"].timeout == 120_000


def test_model_comes_from_env(fake_gemini, monkeypatch):
    monkeypatch.setenv("PAPERPILOT_MODEL", "gemini-3.6-flash")
    state = fake_gemini(_resp("hello"))
    assert llm.generate_text("p") == "hello"
    assert state["models"].calls[0]["model"] == "gemini-3.6-flash"
    assert state["models"].calls[0]["config"].response_mime_type is None


def test_retries_once_on_rate_limit(fake_gemini):
    state = fake_gemini(_api_error(429), _resp("{}"))
    assert llm.generate_json("p") == "{}"
    assert len(state["models"].calls) == 2


def test_retries_once_on_server_error_then_gives_up(fake_gemini):
    state = fake_gemini(_api_error(503), _api_error(503))
    with pytest.raises(errors.ServerError):
        llm.generate_json("p")
    assert len(state["models"].calls) == 2


def test_does_not_retry_a_bad_request(fake_gemini):
    state = fake_gemini(_api_error(400), _resp("{}"))
    with pytest.raises(errors.ClientError):
        llm.generate_json("p")
    assert len(state["models"].calls) == 1


def test_retries_on_timeout(fake_gemini):
    import httpx

    state = fake_gemini(httpx.ReadTimeout("read timed out"), _resp("ok"))
    assert llm.generate_text("p") == "ok"
    assert len(state["models"].calls) == 2


def test_output_limit_is_reported_not_parsed(fake_gemini):
    fake_gemini(_resp('{"truncated": ', finish=types.FinishReason.MAX_TOKENS))
    with pytest.raises(RuntimeError, match="cut off") as err:
        llm.generate_json("p")
    # the app shows a "rate limit or timeout, retry" hint for these words; retrying won't help here
    assert not any(w in str(err.value).lower() for w in ("rate", "limit", "timeout"))


def test_empty_response_raises(fake_gemini):
    fake_gemini(_resp(None))
    with pytest.raises(RuntimeError, match="Empty response"):
        llm.generate_text("p")


def test_missing_sdk_says_how_to_install(monkeypatch):
    import sys

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "google.genai", None)
    with pytest.raises(ImportError, match="pip install google-genai"):
        llm.generate_json("p")


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        llm.generate_json("p")
