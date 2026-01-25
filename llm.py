"""
Gemini LLM wrapper: JSON/freeform generation, retry on rate limit/timeout, replication plan.
"""

from __future__ import annotations

import os
from typing import Any

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment or .env file."
        )
    return key


def _init_genai():
    import google.generativeai as genai
    genai.configure(api_key=_get_api_key())
    return genai


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
    msg = str(e).lower()
    return "rate" in msg or "limit" in msg or "timeout" in msg or "429" in msg or "503" in msg


def generate_json(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """Call Gemini with JSON prompt; retry once on rate limit/timeout. Returns raw JSON string."""
    genai = _init_genai()
    model = genai.GenerativeModel(model_name)
    cfg_kw: dict[str, Any] = {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_output_tokens": 8192,
    }
    try:
        cfg = genai.types.GenerationConfig(**{**cfg_kw, "response_mime_type": "application/json"})
    except (TypeError, AttributeError):
        cfg = genai.types.GenerationConfig(**cfg_kw)

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = model.generate_content(prompt, generation_config=cfg)
            if not resp or not resp.text:
                raise RuntimeError("Empty response from Gemini")
            raw = resp.text.strip()
            return _strip_markdown_fence(raw)
        except Exception as e:
            last_err = e
            if attempt == 0 and _is_retryable(e):
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Empty response from Gemini")


def generate_text(
    prompt: str,
    model_name: str = "gemini-1.5-flash",
) -> str:
    """Freeform text generation; retry once on rate limit/timeout."""
    genai = _init_genai()
    model = genai.GenerativeModel(model_name)
    cfg = genai.types.GenerationConfig(
        temperature=0.3,
        top_p=0.95,
        max_output_tokens=8192,
    )
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            resp = model.generate_content(prompt, generation_config=cfg)
            if not resp or not resp.text:
                raise RuntimeError("Empty response from Gemini")
            return resp.text.strip()
        except Exception as e:
            last_err = e
            if attempt == 0 and _is_retryable(e):
                continue
            raise
    if last_err:
        raise last_err
    raise RuntimeError("Empty response from Gemini")
