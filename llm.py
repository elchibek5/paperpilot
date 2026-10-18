"""
Gemini LLM wrapper (google-genai SDK): JSON/freeform generation, one retry on
rate limit / server error / timeout.
"""

from __future__ import annotations

import os
import time
from typing import Any

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Current stable Flash model (https://ai.google.dev/gemini-api/docs/models).
# Override with PAPERPILOT_MODEL.
DEFAULT_MODEL = "gemini-3.8-flash"

REQUEST_TIMEOUT_MS = 120_000
# Includes thinking tokens on current models, so it must leave room for the JSON.
MAX_OUTPUT_TOKENS = 32_768
RETRY_DELAY_S = 2.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
        )
    return key


def default_model() -> str:
    return (os.getenv("PAPERPILOT_MODEL") or "").strip() or DEFAULT_MODEL


def _client():
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "The Gemini SDK is not installed. Run `pip install google-genai` "
            "(if you installed with pipx: `pipx inject paperpilot google-genai`)."
        ) from e
    return genai.Client(
        api_key=_get_api_key(),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


# -----------------------------------------------------------------------------
# Fence stripping (same as paperpilot)
# -----------------------------------------------------------------------------

def _strip_markdown_fence(s: str) -> str:
    s = s.strip()
    stripped_opening = False
    for fmt in ("```json", "```"):
        if s.lower().startswith(fmt):
            s = s[len(fmt) :].strip()
            stripped_opening = True
            break
    if stripped_opening and s.endswith("```"):
        s = s[:-3].strip()
    return s


# -----------------------------------------------------------------------------
# Retry + generate
# -----------------------------------------------------------------------------

def _is_retryable(e: Exception) -> bool:
    code = getattr(e, "code", None)  # google.genai.errors.APIError carries the HTTP status
    if isinstance(code, int):
        return code in RETRYABLE_STATUS
    if isinstance(e, TimeoutError):
        return True
    try:
        import httpx
        if isinstance(e, httpx.TimeoutException):
            return True
    except ImportError:  # pragma: no cover
        pass
    return "timed out" in str(e).lower()


def _finish_reason(resp: Any) -> str:
    try:
        return str(resp.candidates[0].finish_reason or "")
    except (AttributeError, IndexError, TypeError):
        return ""


def _generate(prompt: str, model_name: str | None, json_mode: bool) -> str:
    client = _client()  # first: it explains how to install the SDK if it is missing
    from google.genai import types

    config = types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json" if json_mode else None,
    )
    model = model_name or default_model()
    for attempt in range(2):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            if _finish_reason(resp).endswith("MAX_TOKENS"):
                raise RuntimeError(
                    f"Gemini's response was cut off at the maximum output length ({MAX_OUTPUT_TOKENS} tokens)."
                )
            text = (resp.text or "").strip() if resp is not None else ""
            if not text:
                raise RuntimeError("Empty response from Gemini")
            return text
        except Exception as e:
            if attempt == 0 and _is_retryable(e):
                time.sleep(RETRY_DELAY_S)
                continue
            raise
    raise RuntimeError("Empty response from Gemini")  # pragma: no cover


def generate_json(prompt: str, model_name: str | None = None) -> str:
    """Call Gemini in JSON mode; retry once on rate limit/timeout. Returns raw JSON string."""
    return _strip_markdown_fence(_generate(prompt, model_name, json_mode=True))


def generate_text(prompt: str, model_name: str | None = None) -> str:
    """Freeform text generation; retry once on rate limit/timeout."""
    return _generate(prompt, model_name, json_mode=False)
