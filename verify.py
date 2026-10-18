"""
Mechanical citation check: does every quoted citation appear in the chunk it cites?

A quote is compared with the chunk text after both go through normalize() (case,
curly quotes, dashes, ligatures, zero-width characters, line-break hyphenation,
whitespace). If that fails, a looser comparison that also ignores spaces and word
hyphens is tried, so PDF extraction quirks ("state-of-the-\\nart", run-together
words) don't count against an honest quote. Nothing is paraphrase-matched: the
characters must be there. Numbers are never loosened: signs, ranges, exponents,
decimals and digit groups must match, and a quote may not start or stop inside a
number ("batch size 6" is not "batch size 64", "0.42" is not "-0.42"). A quote may
not start or stop inside a word either ("stable" is not "unstable"), except where the
PDF text itself has lost its spaces.

Statuses:
  verified     quote found in the cited chunk
  wrong_chunk  quote found, but in a different chunk (see found_in)
  not_found    quote not found in any chunk (or too short to check, see below)
  empty        blank quote or "Not found in provided text" (allowed; counted as
               declared_missing)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Iterator

from pydantic import BaseModel

NOT_FOUND_TEXT = "Not found in provided text"

# Quotes shorter than this (after normalize), or with fewer letters and digits than
# MIN_QUOTE_ALNUM, only count when they equal the whole chunk text; otherwise "the",
# "0.1" or "Adam - - - -" would "verify" against almost anything.
MIN_QUOTE_CHARS = 12
MIN_QUOTE_ALNUM = 8

_CHAR_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'", "`": "'",
    "\u02bc": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
    "\u2212": "-", "\ufe58": "-", "\ufe63": "-", "\uff0d": "-",
})
_SOFT_HYPHEN = re.compile(r"\u00ad(?:[ \t]*\r?\n\s*)?")
# "reproduc-\nibility" -> "reproducibility"; before a digit the hyphen is a minus
# or a range ("10-\n12", "GPT-\n4"), so only the line break goes.
_EOL_HYPHEN_WORD = re.compile(r"([^\W\d_])-[ \t]*\r?\n\s*([^\W\d_])")
_EOL_HYPHEN_DIGIT = re.compile(r"-[ \t]*\r?\n\s*(?=\d)")
_WS = re.compile(r"\s+")
_SEP_RUN = re.compile(r"[ -]+")
_CHUNK_PREFIX = re.compile(r"^\[?\s*chunk[\s_:#-]*([^\]\s]+)\s*\]?$", re.IGNORECASE)


def normalize(s: str) -> str:
    """Casefold, unify quotes/dashes, drop soft hyphens and zero-width characters,
    join line-break hyphenation, collapse whitespace, strip."""
    if not s:
        return ""
    s = str(s).translate(_CHAR_MAP)  # before NFKC, which would split "\u00b4" and "\u2033"
    s = unicodedata.normalize("NFKC", s).translate(_CHAR_MAP)  # ligatures: "\ufb01" -> "fi"
    s = _SOFT_HYPHEN.sub("", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = _EOL_HYPHEN_WORD.sub(r"\1\2", s)
    s = _EOL_HYPHEN_DIGIT.sub("-", s)
    s = s.casefold()
    return _WS.sub(" ", s).strip()


def _loose_map(t: str) -> tuple[str, list[int]]:
    """Drop spaces and word hyphens from normalized text, keeping a "-" that precedes a
    digit (sign, range, exponent) and a space between two digits. Returns the loose
    string and, for each of its characters, the index in t it came from."""
    chars: list[str] = []
    pos: list[int] = []
    last = 0
    for m in _SEP_RUN.finditer(t):
        chars.extend(t[last:m.start()])
        pos.extend(range(last, m.start()))
        left = t[m.start() - 1] if m.start() > 0 else ""
        right = t[m.end()] if m.end() < len(t) else ""
        if right.isdigit() and "-" in m.group(0):
            chars.append("-")
            pos.append(m.start() + m.group(0).index("-"))
        elif left.isdigit() and right.isdigit():
            chars.append(" ")
            pos.append(m.start())
        last = m.end()
    chars.extend(t[last:])
    pos.extend(range(last, len(t)))
    return "".join(chars), pos


def _edges_ok(t: str, i: int, j: int, relaxed: bool) -> bool:
    """Does t[i:j] start and end on a boundary? Numbers must never be cut (no digit,
    sign, or ".,"-plus-digit next to an edge digit); words must not be cut unless
    `relaxed` (the matched PDF text has no spaces, so word boundaries are unknown);
    a letter-hyphen-letter compound ("non-reproducible") is never cut."""
    first, last = t[i], t[j - 1]
    prev = t[i - 1] if i > 0 else ""
    prev2 = t[i - 2] if i > 1 else ""
    nxt = t[j] if j < len(t) else ""
    nxt2 = t[j + 1] if j + 1 < len(t) else ""
    if first.isdigit():
        if prev.isdigit() or prev in ("-", "+") or (prev in (".", ",") and prev2.isdigit()):
            return False
        if prev.isalpha() and not relaxed:
            return False
    elif first.isalpha():
        if (prev.isalnum() and not relaxed) or (prev == "-" and prev2.isalpha()):
            return False
    if last.isdigit():
        if nxt.isdigit() or (nxt in (".", ",", "-") and nxt2.isdigit()):
            return False
        if nxt.isalpha() and not relaxed:
            return False
    elif last.isalpha():
        if (nxt.isalnum() and not relaxed) or (nxt == "-" and nxt2.isalpha()):
            return False
    return True


def _find_all(text: str, needle: str) -> Iterator[int]:
    i = text.find(needle)
    while i >= 0:
        yield i
        i = text.find(needle, i + 1)


def _is_short(nq: str) -> bool:
    return len(nq) < MIN_QUOTE_CHARS or sum(ch.isalnum() for ch in nq) < MIN_QUOTE_ALNUM


def normalize_chunk_id(chunk_id: Any) -> str:
    """'3', 3, 'CHUNK 3' and '[chunk 3]' all mean chunk '3'."""
    s = str(chunk_id).strip() if chunk_id is not None else ""
    m = _CHUNK_PREFIX.match(s)
    return m.group(1) if m else s


def is_declared_missing(quote: str | None) -> bool:
    q = normalize(quote or "").rstrip(".").strip()
    return not q or q == normalize(NOT_FOUND_TEXT)


class ChunkIndex:
    """Normalized chunk texts, built once and reused for every citation."""

    def __init__(self, chunks: Iterable[dict]):
        self._norm: dict[str, str] = {}
        self._loose: dict[str, tuple[str, list[int]]] = {}
        for c in chunks:
            cid = normalize_chunk_id(c.get("chunk_id"))
            n = normalize(c.get("text") or "")
            self._norm[cid] = n
            self._loose[cid] = _loose_map(n)

    def __contains__(self, chunk_id: Any) -> bool:
        return normalize_chunk_id(chunk_id) in self._norm

    def _hit(self, cid: str, nq: str, lq: str) -> bool:
        text = self._norm[cid]
        loose, pos = self._loose[cid]
        if _is_short(nq):
            return nq == text or (bool(lq) and lq == loose)
        if any(_edges_ok(text, i, i + len(nq), relaxed=False) for i in _find_all(text, nq)):
            return True
        if not lq:
            return False
        for li in _find_all(loose, lq):
            i, j = pos[li], pos[li + len(lq) - 1] + 1
            if _edges_ok(text, i, j, relaxed=" " not in text[i:j]):
                return True
        return False

    def classify(self, chunk_id: Any, quote: str | None) -> tuple[str, list[str]]:
        """Return (status, found_in) for one citation."""
        if is_declared_missing(quote):
            return "empty", []
        nq = normalize(quote or "")
        lq = _loose_map(nq)[0]
        cid = normalize_chunk_id(chunk_id)
        if cid in self._norm and self._hit(cid, nq, lq):
            return "verified", []
        others = [k for k in self._norm if k != cid and self._hit(k, nq, lq)]
        if others:
            return "wrong_chunk", others
        return "not_found", []


def _iter_citations(node: Any, path: str = "") -> Iterator[tuple[str, dict]]:
    """Yield (location, citation_dict) for every {chunk_id, quote} object in node."""
    if isinstance(node, dict):
        if "quote" in node and "chunk_id" in node:
            yield path or "$", node
            return
        for k, v in node.items():
            yield from _iter_citations(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_citations(v, f"{path}[{i}]")


def verify_citations(analysis_or_dict: Any, chunks: Iterable[dict] | ChunkIndex) -> dict:
    """
    Check every citation in a PaperAnalysis / ReplicationPlan (model or plain dict)
    against the chunks. Returns counts plus an `issues` list with one entry per
    wrong_chunk / not_found citation (declared-missing citations are counted, not listed).
    """
    data = analysis_or_dict.model_dump() if isinstance(analysis_or_dict, BaseModel) else analysis_or_dict
    index = chunks if isinstance(chunks, ChunkIndex) else ChunkIndex(chunks)
    out: dict[str, Any] = {
        "total": 0, "verified": 0, "wrong_chunk": 0, "not_found": 0, "declared_missing": 0, "issues": [],
    }
    for location, cit in _iter_citations(data):
        quote = cit.get("quote")
        quote = "" if quote is None else str(quote)
        status, found_in = index.classify(cit.get("chunk_id"), quote)
        out["total"] += 1
        if status == "empty":
            out["declared_missing"] += 1
            continue
        out[status] += 1
        if status != "verified":
            out["issues"].append({
                "location": location,
                "chunk_id": str(cit.get("chunk_id")),
                "quote": quote,
                "status": status,
                "found_in": found_in,
            })
    return out


def summary_line(result: dict) -> str:
    """One line for reports, e.g. '41/43 quotes verified against the paper text (2 not found)'."""
    checked = result["total"] - result["declared_missing"]
    line = f"{result['verified']}/{checked} quotes verified against the paper text"
    problems = []
    if result["not_found"]:
        problems.append(f"{result['not_found']} not found")
    if result["wrong_chunk"]:
        problems.append(f"{result['wrong_chunk']} in a different chunk")
    if problems:
        line += " (" + ", ".join(problems) + ")"
    if result["declared_missing"]:
        line += f"; {result['declared_missing']} marked not found in the paper"
    return line
