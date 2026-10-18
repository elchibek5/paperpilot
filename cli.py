"""
paperpilot command line.

No API key needed: extract, search, schema, check, export. A coding agent (or a
person) writes review.json; `check` validates it, recomputes the risk score and
verifies every quote against the paper text.

Gemini key needed: review, plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from export import combined_md
from pdftext import MIN_PDF_CHARS, NoTextError, chunk_text, extract_text_from_pdf
from retrieval import rank_chunks_tfidf
from schemas import PaperAnalysis, ReplicationPlan
from scoring import RUBRIC, RUBRIC_FACTORS, compute_risk_score_from_breakdown
from verify import NOT_FOUND_TEXT, ChunkIndex, summary_line, verify_citations

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NO_TEXT = 2


class CliError(Exception):
    def __init__(self, message: str, code: int = EXIT_FAIL):
        super().__init__(message)
        self.code = code


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _write(text: str, out_path: str | None, ascii_text=None) -> None:
    """Write UTF-8; if that cannot be encoded (a legacy Windows console, a lone
    surrogate), fall back to `ascii_text()` or backslash escapes."""
    def fallback() -> str:
        return ascii_text() if ascii_text else text.encode("ascii", "backslashreplace").decode("ascii")

    if out_path:
        try:
            Path(out_path).write_text(text, encoding="utf-8")
        except UnicodeEncodeError:
            Path(out_path).write_text(fallback(), encoding="utf-8")
        print(f"wrote {out_path}", file=sys.stderr)
        return
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(fallback())


def _emit_json(obj: Any, out_path: str | None) -> None:
    _write(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        out_path,
        ascii_text=lambda: json.dumps(obj, indent=2, ensure_ascii=True) + "\n",
    )


def _load_json(path: str, what: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise CliError(f"{what} not found: {path}")
    except UnicodeDecodeError:
        raise CliError(f"{what} is not UTF-8 text: {path}")
    except json.JSONDecodeError as e:
        raise CliError(f"{what} is not valid JSON ({path}): {e}")


def _load_chunks(path: str) -> list[dict]:
    data = _load_json(path, "chunks file")
    chunks = data.get("chunks") if isinstance(data, dict) else data
    if not isinstance(chunks, list) or not all(
        isinstance(c, dict) and "chunk_id" in c and isinstance(c.get("text"), str) for c in chunks
    ):
        raise CliError(f"{path} is not `paperpilot extract` output (expected chunks with chunk_id and text)")
    ids = [str(c["chunk_id"]) for c in chunks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise CliError(f"{path} has duplicate chunk ids: {', '.join(dupes)}")
    return [{**c, "chunk_id": cid} for c, cid in zip(chunks, ids)]


def _loc(loc: tuple) -> str:
    """('main_claims', 2, 'citations', 0) -> 'main_claims[2].citations[0]'"""
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "$"


def _schema_errors(e: ValidationError) -> list[dict]:
    return [{"loc": _loc(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in e.errors()]


def _validate(model, data: Any, what: str):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        lines = "\n".join(f"  {x['loc']}: {x['msg']}" for x in _schema_errors(e))
        raise CliError(f"{what} does not match the schema (run `paperpilot check` for details):\n{lines}")


def _load_env() -> None:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))


# -----------------------------------------------------------------------------
# Pure functions behind the commands
# -----------------------------------------------------------------------------

def schema_for(plan: bool = False) -> dict:
    """JSON Schema of PaperAnalysis (or ReplicationPlan) plus the rubric an agent must use."""
    model = ReplicationPlan if plan else PaperAnalysis
    schema = model.model_json_schema()
    schema["x-declared-missing"] = NOT_FOUND_TEXT
    if not plan:
        schema["description"] = (
            "PaperPilot review. Every citation is {chunk_id, quote} with the quote copied verbatim from that "
            f"chunk, or quote \"{NOT_FOUND_TEXT}\". risk_breakdown has one entry per x-rubric factor, "
            "named exactly; risk_score is recomputed from it by `paperpilot check`."
        )
        schema["x-rubric"] = [{"factor": f, "points": p} for f, p in RUBRIC.items()]
    return schema


def rubric_report(breakdown: list[dict]) -> dict:
    names = [str(f.get("factor", "")).strip() for f in breakdown]
    mismatch = []
    for f, name in zip(breakdown, names):
        pts = f.get("points")
        if name in RUBRIC and pts != RUBRIC[name]:
            mismatch.append({"factor": name, "points": pts, "rubric_points": RUBRIC[name]})
    return {
        "missing_factors": [k for k in RUBRIC_FACTORS if k not in names],
        "unknown_factors": [n for n in names if n not in RUBRIC],
        "duplicate_factors": sorted({n for n in names if n in RUBRIC and names.count(n) > 1}),
        "points_mismatch": mismatch,
    }


# Claims, results and the method overview state what the paper says, so each needs a
# verified quote. Checklist items and risk factors need one when marked "present".
# Every cited item needs at least one citation, even if only the declared-missing one.
_CITED = ("main_claims", "results", "reproducibility_checklist", "missing_details", "risk_breakdown")
_MUST_QUOTE = {"main_claims", "results", "method_overview"}
_QUOTE_IF_PRESENT = {"reproducibility_checklist", "risk_breakdown"}


def evidence_report(data: Any, index: ChunkIndex) -> list[dict]:
    """Items with no citations at all, or with no verified quote where one is required."""
    if not isinstance(data, dict):
        return []
    items: list[tuple[str, str, dict]] = []
    for key in _CITED:
        value = data.get(key)
        for i, item in enumerate(value if isinstance(value, list) else []):
            if isinstance(item, dict):
                items.append((key, f"{key}[{i}]", item))
    if isinstance(data.get("method_overview"), dict):
        items.append(("method_overview", "method_overview", data["method_overview"]))
    problems = []
    for kind, loc, item in items:
        cites = item.get("citations")
        cites = [c for c in cites if isinstance(c, dict)] if isinstance(cites, list) else []
        if not cites:
            problems.append({"location": loc, "problem": f'no citations (cite the paper, or "{NOT_FOUND_TEXT}")'})
            continue
        if any(index.classify(c.get("chunk_id"), c.get("quote"))[0] == "verified" for c in cites):
            continue
        if kind in _MUST_QUOTE:
            problems.append({"location": loc, "problem": "no verified quote"})
        elif kind in _QUOTE_IF_PRESENT and item.get("status") == "present":
            problems.append({"location": loc, "problem": "marked present without a verified quote"})
    return problems


def check_review(data: Any, chunks: list[dict], strict: bool = False) -> dict:
    """Validate review JSON, recompute the risk score, check the rubric, every citation,
    and that each item is backed by a quote."""
    schema_errors: list[dict] = []
    try:
        PaperAnalysis.model_validate(data)
    except ValidationError as e:
        schema_errors = _schema_errors(e)
    breakdown = data.get("risk_breakdown") if isinstance(data, dict) else None
    breakdown = [f for f in breakdown if isinstance(f, dict)] if isinstance(breakdown, list) else []
    rubric = rubric_report(breakdown)
    index = ChunkIndex(chunks)
    cites = verify_citations(data, index)
    evidence = evidence_report(data, index)
    citation_problems = cites["wrong_chunk"] + cites["not_found"]
    rubric_problems = any(rubric.values())
    ok = not schema_errors and not (strict and (citation_problems or rubric_problems or evidence))
    return {
        "ok": ok,
        "strict": strict,
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "model_risk_score": data.get("risk_score") if isinstance(data, dict) else None,
        "computed_risk_score": compute_risk_score_from_breakdown(breakdown),
        "rubric": rubric,
        "citations": cites,
        "evidence": evidence,
        "summary": summary_line(cites),
    }


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_extract(args) -> int:
    path = Path(args.pdf)
    if not path.is_file():
        raise CliError(f"file not found: {path}")
    try:
        text = extract_text_from_pdf(path)
    except NoTextError:
        text = ""
    except Exception as e:
        raise CliError(f"could not read {path} as a PDF: {e}")
    if len(text) < MIN_PDF_CHARS:
        raise CliError(
            f"extracted only {len(text):,} characters from {path.name} (need {MIN_PDF_CHARS:,}); "
            "it is probably scanned or image-only. Run OCR first (for example `ocrmypdf in.pdf out.pdf`).",
            EXIT_NO_TEXT,
        )
    chunks = chunk_text(text)
    _emit_json({"source": path.name, "n_chars": len(text), "n_chunks": len(chunks), "chunks": chunks}, args.output)
    return EXIT_OK


def cmd_search(args) -> int:
    if args.k < 1:
        raise CliError("-k must be at least 1")
    chunks = _load_chunks(args.chunks)
    hits = rank_chunks_tfidf(chunks, args.query, top_k=args.k)
    _emit_json(
        [{"chunk_id": c["chunk_id"], "score": round(s, 4), "text": c["text"]} for c, s in hits if s > 0],
        args.output,
    )
    return EXIT_OK


def cmd_schema(args) -> int:
    _emit_json(schema_for(plan=args.plan), args.output)
    return EXIT_OK


def cmd_check(args) -> int:
    data = _load_json(args.review, "review")
    chunks = _load_chunks(args.chunks)
    report = check_review(data, chunks, strict=args.strict)
    _emit_json(report, args.output)
    schema = "schema ok" if report["schema_valid"] else f"schema INVALID ({len(report['schema_errors'])} errors)"
    rubric_issues = sum(len(v) for v in report["rubric"].values())
    rubric = "rubric ok" if not rubric_issues else f"{rubric_issues} rubric issue(s)"
    evidence = "evidence ok" if not report["evidence"] else f"{len(report['evidence'])} item(s) without evidence"
    print(
        f"check: {schema}; {rubric}; {evidence}; {report['summary']}; risk score {report['computed_risk_score']}"
        f" computed (review says {report['model_risk_score']})",
        file=sys.stderr,
    )
    return EXIT_OK if report["ok"] else EXIT_FAIL


def cmd_export(args) -> int:
    analysis = _validate(PaperAnalysis, _load_json(args.review, "review"), "review")
    plan = _validate(ReplicationPlan, _load_json(args.plan, "plan"), "plan") if args.plan else None
    if args.chunks:
        target = {"review": analysis.model_dump(), "plan": plan.model_dump()} if plan else analysis
        line = summary_line(verify_citations(target, _load_chunks(args.chunks)))
    else:
        line = "not run (pass --chunks chunks.json to check quotes against the paper text)"
    computed = compute_risk_score_from_breakdown(analysis.risk_breakdown)
    _write(combined_md(analysis, computed, plan, line) + "\n", args.output)
    return EXIT_OK


def _gemini(fn, *a):
    """Run a Gemini-backed call and turn its failures into one clear CLI error."""
    _load_env()
    try:
        return fn(*a)
    except ImportError as e:
        raise CliError(str(e))
    except ValueError as e:
        if isinstance(e, json.JSONDecodeError):
            raise CliError(f"Gemini returned invalid JSON: {e}")
        if isinstance(e, ValidationError):
            raise CliError(f"Gemini's JSON does not match the schema: {e}")
        raise CliError(str(e))
    except Exception as e:
        raise CliError(f"Gemini call failed: {e}")


def cmd_review(args) -> int:
    from llm import run_analysis

    chunks = _load_chunks(args.chunks)
    analysis = _gemini(run_analysis, chunks)
    data = analysis.model_dump()
    data["risk_score"] = compute_risk_score_from_breakdown(analysis.risk_breakdown)
    _emit_json(data, args.output)
    print(f"review: {summary_line(verify_citations(analysis, chunks))}", file=sys.stderr)
    return EXIT_OK


def cmd_plan(args) -> int:
    from llm import run_replication_plan

    chunks = _load_chunks(args.chunks)
    analysis = _validate(PaperAnalysis, _load_json(args.review, "review"), "review")
    plan = _gemini(run_replication_plan, chunks, analysis)
    _emit_json(plan.model_dump(), args.output)
    return EXIT_OK


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def _version() -> str:
    try:
        from importlib.metadata import version

        return version("paperpilot")
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paperpilot",
        description="Reproducibility review for ML papers, with every quote checked against the paper text.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name, func, help_):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(func=func)
        return sp

    sp = add("extract", cmd_extract, "Extract and chunk a PDF (exit 2 if it has no usable text).")
    sp.add_argument("pdf", metavar="PAPER.pdf")
    sp.add_argument("-o", "--output", metavar="chunks.json")

    sp = add("search", cmd_search, "TF-IDF search over the chunks.")
    sp.add_argument("chunks", metavar="chunks.json")
    sp.add_argument("query")
    sp.add_argument("-k", type=int, default=5, help="number of chunks to return (default 5)")
    sp.add_argument("-o", "--output")

    sp = add("schema", cmd_schema, "Print the review JSON Schema and the rubric (x-rubric).")
    sp.add_argument("--plan", action="store_true", help="print the replication plan schema instead")
    sp.add_argument("-o", "--output")

    sp = add("check", cmd_check, "Validate review.json, recompute the risk score, verify every quote.")
    sp.add_argument("review", metavar="review.json")
    sp.add_argument("--chunks", required=True, metavar="chunks.json")
    sp.add_argument("--strict", action="store_true",
                    help="also fail on unverified quotes, rubric problems, and items without evidence")
    sp.add_argument("-o", "--output")

    sp = add("export", cmd_export, "Render review.json (and optionally plan.json) as Markdown.")
    sp.add_argument("review", metavar="review.json")
    sp.add_argument("--plan", metavar="plan.json")
    sp.add_argument("--chunks", metavar="chunks.json", help="adds the citation check line")
    sp.add_argument("-o", "--output", metavar="review.md")

    sp = add("review", cmd_review, "Write review.json with Gemini (needs GEMINI_API_KEY).")
    sp.add_argument("chunks", metavar="chunks.json")
    sp.add_argument("-o", "--output", metavar="review.json")

    sp = add("plan", cmd_plan, "Write a replication plan with Gemini (needs GEMINI_API_KEY).")
    sp.add_argument("chunks", metavar="chunks.json")
    sp.add_argument("review", metavar="review.json")
    sp.add_argument("-o", "--output", metavar="plan.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
