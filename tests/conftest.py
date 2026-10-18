"""Shared test helpers. No test may need network access or an API key."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: os.PathLike | str, lines: list[str]) -> Path:
    """Write a minimal one-page PDF whose text stream draws `lines` in Helvetica."""
    ops = ["BT", "/F1 8 Tf", "10 TL", "36 770 Td"]
    for i, line in enumerate(lines):
        ops.append(("" if i == 0 else "T* ") + f"({_pdf_escape(line)}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    p = Path(path)
    p.write_bytes(bytes(out))
    return p


# ~70 lines of plain prose: well over the 1500-character scanned-PDF threshold.
LONG_LINES = [
    f"Line {i}: we train the model on the benchmark split and report accuracy." for i in range(70)
]


@pytest.fixture
def make_pdf(tmp_path):
    def _make(lines: list[str], name: str = "paper.pdf") -> Path:
        return write_pdf(tmp_path / name, lines)
    return _make
