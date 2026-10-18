"""
PDF text extraction (pdfplumber → pypdf fallback) and chunking (~800–1200 tokens, overlap).
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any, BinaryIO, Union

# Below this many extracted characters the PDF is treated as scanned / image-only.
MIN_PDF_CHARS = 1500

PdfSource = Union[str, "os.PathLike[str]", BinaryIO]


# -----------------------------------------------------------------------------
# PDF extraction (pdfplumber → pypdf fallback)
# -----------------------------------------------------------------------------

def _read_pdf_bytes(file_or_path: PdfSource) -> bytes:
    """Read the whole PDF from a filesystem path or a binary file object."""
    if isinstance(file_or_path, (str, os.PathLike)):
        with open(file_or_path, "rb") as f:
            return f.read()
    raw = file_or_path.read()
    file_or_path.seek(0)
    return raw


def extract_text_from_pdf(file_or_path: PdfSource) -> str:
    raw = _read_pdf_bytes(file_or_path)

    try:
        import pdfplumber
        with pdfplumber.open(BytesIO(raw)) as pdf:
            parts = []
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            if not parts:
                raise ValueError("pdfplumber extracted no text")
            return "\n\n".join(parts)
    except Exception:
        pass

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(raw))
    parts = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            parts.append(t)
    if not parts:
        raise ValueError("Could not extract any text from PDF (pdfplumber and pypdf)")
    return "\n\n".join(parts)


# -----------------------------------------------------------------------------
# Chunking (~800–1200 tokens, overlap)
# -----------------------------------------------------------------------------

CHARS_PER_TOKEN = 4
TARGET_TOKENS = 1000
MIN_TOKENS = 800
MAX_TOKENS = 1200
OVERLAP_TOKENS = 200


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // CHARS_PER_TOKEN)


def chunk_text(text: str) -> list[dict[str, Any]]:
    """Split text into overlapping chunks: [{chunk_id (str), text, approx_tokens}]."""
    target_chars = TARGET_TOKENS * CHARS_PER_TOKEN
    overlap_chars = OVERLAP_TOKENS * CHARS_PER_TOKEN

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + target_chars, len(text))
        if end < len(text):
            last_space = text.rfind(" ", start, end + 1)
            if last_space > start:
                end = last_space + 1
        chunk_text_slice = text[start:end].strip()
        if chunk_text_slice:
            chunks.append({
                "chunk_id": str(chunk_id),
                "text": chunk_text_slice,
                "approx_tokens": _approx_tokens(chunk_text_slice),
            })
            chunk_id += 1
        if end >= len(text):
            break
        # Step back by the overlap, but always move forward: a space early in the
        # window followed by a long unbroken run would otherwise loop forever.
        next_start = max(0, end - overlap_chars)
        start = next_start if next_start > start else end

    return chunks
