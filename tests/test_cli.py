from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import LONG_LINES

from paperpilot import cli
from paperpilot.scoring import RUBRIC

FIX = Path(__file__).parent / "fixtures"
CHUNKS = str(FIX / "chunks.json")


def run(capsys, *argv):
    code = cli.main([str(a) for a in argv])
    out, err = capsys.readouterr()
    return code, out, err


def test_extract_writes_chunks(capsys, make_pdf, tmp_path):
    pdf = make_pdf(LONG_LINES)
    out_file = tmp_path / "chunks.json"
    code, out, err = run(capsys, "extract", pdf, "-o", out_file)
    assert code == 0 and out == ""
    data = json.loads(out_file.read_text())
    assert data["source"] == "paper.pdf"
    assert data["n_chars"] >= 1500 and data["n_chunks"] == len(data["chunks"]) >= 1
    assert set(data["chunks"][0]) == {"chunk_id", "text", "approx_tokens"}
    assert data["chunks"][0]["chunk_id"] == "0"


def test_extract_scanned_pdf_exits_2(capsys, make_pdf):
    code, out, err = run(capsys, "extract", make_pdf(["Too short to be a paper."]))
    assert code == 2 and out == ""
    assert "scanned" in err


def test_extract_tells_no_text_from_unreadable(capsys, make_pdf, tmp_path):
    code, _, err = run(capsys, "extract", make_pdf([]))  # valid PDF, no text at all
    assert code == 2 and "scanned" in err
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"this is not a pdf")
    code, _, err = run(capsys, "extract", junk)
    assert code == 1 and "could not read" in err


def test_search_returns_scored_chunks(capsys):
    code, out, _ = run(capsys, "search", CHUNKS, "NVIDIA GPU batch size", "-k", "2")
    hits = json.loads(out)
    assert code == 0
    assert hits[0]["chunk_id"] == "1" and hits[0]["score"] > 0
    assert len(hits) <= 2 and set(hits[0]) == {"chunk_id", "score", "text"}


def test_schema_includes_the_rubric(capsys):
    code, out, _ = run(capsys, "schema")
    schema = json.loads(out)
    assert code == 0
    assert {r["factor"]: r["points"] for r in schema["x-rubric"]} == RUBRIC
    assert "risk_breakdown" in schema["properties"]
    plan = json.loads(run(capsys, "schema", "--plan")[1])
    assert plan["title"] == "ReplicationPlan" and "x-rubric" not in plan


def test_check_valid_review(capsys):
    code, out, err = run(capsys, "check", FIX / "review_valid.json", "--chunks", CHUNKS, "--strict")
    report = json.loads(out)
    assert code == 0 and report["ok"] and report["schema_valid"]
    assert report["computed_risk_score"] == 10 and report["model_risk_score"] == 10
    assert report["citations"]["verified"] == 14 and report["citations"]["declared_missing"] == 2
    assert not any(report["rubric"].values()) and report["evidence"] == []
    assert "14/14 quotes verified" in err


def test_check_invalid_review_lists_schema_errors(capsys):
    code, out, _ = run(capsys, "check", FIX / "review_invalid.json", "--chunks", CHUNKS)
    report = json.loads(out)
    assert code == 1 and not report["schema_valid"]
    assert {e["loc"] for e in report["schema_errors"]} == {"paper_summary", "confidence", "risk_breakdown[0].status"}


def test_fake_quote_fails_only_under_strict(capsys):
    review = FIX / "review_fake_quote.json"
    code, out, _ = run(capsys, "check", review, "--chunks", CHUNKS)
    assert code == 0
    code, out, _ = run(capsys, "check", review, "--chunks", CHUNKS, "--strict")
    report = json.loads(out)
    assert code == 1 and not report["ok"]
    assert [(i["location"], i["status"], i["found_in"]) for i in report["citations"]["issues"]] == [
        ("main_claims[1].citations[0]", "not_found", []),
        ("results[0].citations[0]", "wrong_chunk", ["2"]),
    ]
    assert report["evidence"] == [
        {"location": "main_claims[1]", "problem": "no verified quote"},
        {"location": "results[0]", "problem": "no verified quote"},
    ]


def _review_with(tmp_path, change):
    review = json.loads((FIX / "review_valid.json").read_text())
    change(review)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    return path


def test_strict_rejects_present_factors_without_a_quote(capsys, tmp_path):
    def all_present_no_evidence(r):
        for f in r["risk_breakdown"]:
            f["status"] = "present"
            f["citations"] = [{"chunk_id": "", "quote": "Not found in provided text"}]
    path = _review_with(tmp_path, all_present_no_evidence)
    code, out, _ = run(capsys, "check", path, "--chunks", CHUNKS)
    assert code == 0  # reported, but only --strict fails on it
    code, out, err = run(capsys, "check", path, "--chunks", CHUNKS, "--strict")
    report = json.loads(out)
    assert code == 1 and report["computed_risk_score"] == 0
    assert len(report["evidence"]) == 9
    assert report["evidence"][0] == {"location": "risk_breakdown[0]", "problem": "marked present without a verified quote"}
    assert "9 item(s) without evidence" in err


def test_strict_rejects_items_with_no_citations(capsys, tmp_path):
    def strip(r):
        r["main_claims"][0]["citations"] = []
        r["missing_details"][0]["citations"] = []
    code, out, _ = run(capsys, "check", _review_with(tmp_path, strip), "--chunks", CHUNKS, "--strict")
    assert code == 1
    assert [e["location"] for e in json.loads(out)["evidence"]] == ["main_claims[0]", "missing_details[0]"]


def test_a_duplicated_factor_counts_once(capsys, tmp_path):
    path = _review_with(tmp_path, lambda r: r["risk_breakdown"].append(dict(r["risk_breakdown"][0])))
    code, out, _ = run(capsys, "check", path, "--chunks", CHUNKS, "--strict")
    report = json.loads(out)
    assert report["computed_risk_score"] == 10
    assert code == 1 and report["rubric"]["duplicate_factors"] == ["dataset defined & access described"]


def test_check_flags_rubric_problems(capsys, tmp_path):
    review = json.loads((FIX / "review_valid.json").read_text())
    review["risk_breakdown"][0]["factor"] = "dataset described"
    review["risk_breakdown"][1]["points"] = 3
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    code, out, _ = run(capsys, "check", path, "--chunks", CHUNKS, "--strict")
    report = json.loads(out)
    assert code == 1 and report["schema_valid"]
    assert report["rubric"]["missing_factors"] == ["dataset defined & access described"]
    assert report["rubric"]["unknown_factors"] == ["dataset described"]
    assert report["rubric"]["points_mismatch"] == [
        {"factor": "train/val/test split described", "points": 3, "rubric_points": 15}]
    assert report["computed_risk_score"] == 25  # the misspelled factor no longer counts


def test_export_has_the_citation_check_line(capsys, tmp_path):
    out_file = tmp_path / "review.md"
    code, _, _ = run(capsys, "export", FIX / "review_fake_quote.json", "--chunks", CHUNKS, "-o", out_file)
    md = out_file.read_text()
    assert code == 0
    assert "**Risk score (computed):** 10/100" in md
    assert "**Citation check:** 13/15 quotes verified against the paper text (1 not found, 1 in a different chunk)" in md
    code, out, _ = run(capsys, "export", FIX / "review_valid.json")
    assert "**Citation check:** not run" in out


def test_export_rejects_an_invalid_review(capsys):
    code, out, err = run(capsys, "export", FIX / "review_invalid.json")
    assert code == 1 and out == "" and "paper_summary" in err


def test_review_uses_gemini_and_writes_the_computed_score(capsys, monkeypatch):
    from paperpilot import llm

    canned = json.loads((FIX / "review_valid.json").read_text())
    canned["risk_score"] = 0
    prompts = []
    monkeypatch.setattr(llm, "generate_json", lambda prompt, model_name=None: prompts.append(prompt) or json.dumps(canned))
    code, out, err = run(capsys, "review", CHUNKS)
    assert code == 0
    assert "[CHUNK 1]\nInputs are standardized" in prompts[0]
    assert json.loads(out)["risk_score"] == 10
    assert "14/14 quotes verified" in err


def test_review_without_a_key_is_a_clear_error(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    code, out, err = run(capsys, "review", CHUNKS)
    assert code == 1 and out == ""
    assert "GEMINI_API_KEY" in err


def test_python_dash_m_runs_the_cli():
    import subprocess
    import sys

    from paperpilot import __version__

    out = subprocess.run([sys.executable, "-m", "paperpilot", "--version"], cwd=FIX.parent.parent,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == f"paperpilot {__version__}"


@pytest.mark.parametrize("argv", [["search", "missing.json", "q"], ["check", "missing.json", "--chunks", CHUNKS]])
def test_missing_files_are_reported(capsys, argv):
    code, _, err = run(capsys, *argv)
    assert code == 1 and "not found" in err


def test_bom_and_bad_chunk_files(capsys, tmp_path):
    review = tmp_path / "review.json"
    review.write_bytes(b"\xef\xbb\xbf" + (FIX / "review_valid.json").read_bytes())
    assert run(capsys, "check", review, "--chunks", CHUNKS)[0] == 0
    utf16 = tmp_path / "utf16.json"
    utf16.write_text("{}", encoding="utf-16")
    code, _, err = run(capsys, "check", utf16, "--chunks", CHUNKS)
    assert code == 1 and "not UTF-8" in err
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps([{"chunk_id": "0", "text": "a"}, {"chunk_id": 0, "text": "b"}]))
    code, _, err = run(capsys, "search", dup, "q")
    assert code == 1 and "duplicate chunk ids: 0" in err
    null = tmp_path / "null.json"
    null.write_text(json.dumps([{"chunk_id": "0", "text": None}]))
    assert run(capsys, "search", null, "q")[0] == 1


def test_json_stays_valid_when_stdout_cannot_encode(monkeypatch, capsys):
    import io

    raw = io.BytesIO()
    monkeypatch.setattr(cli.sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252"))
    cli._emit_json({"quote": "85.1 \u00b1 0.3, \u2212log p"}, None)
    cli.sys.stdout.flush()
    assert json.loads(raw.getvalue().decode("cp1252")) == {"quote": "85.1 \u00b1 0.3, \u2212log p"}
