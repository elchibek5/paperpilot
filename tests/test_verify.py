from __future__ import annotations

from schemas import Citation, Claim, PaperAnalysis
from verify import NOT_FOUND_TEXT, ChunkIndex, normalize, summary_line, verify_citations

CHUNKS = [
    {"chunk_id": "0", "text": "We study the reproduc-\n ibility of deep networks.\nOur “model” uses a 3–4 layer MLP."},
    {"chunk_id": "1", "text": "We train for 100 epochs with Adam and a learning rate of 0.001 on CIFAR-10."},
    {"chunk_id": "2", "text": "Adam"},
]


def _status(chunk_id, quote):
    return ChunkIndex(CHUNKS).classify(chunk_id, quote)


# -- normalize ---------------------------------------------------------------

def test_normalize_joins_end_of_line_hyphenation():
    assert normalize("reproduc-\n ibility") == "reproducibility"
    assert normalize("reproduc-\r\nibility") == "reproducibility"


def test_normalize_unifies_quotes_dashes_case_and_whitespace():
    assert normalize("  “Hello” — it’s\tA  3–4\nTest ") == "\"hello\" - it's a 3-4 test"


def test_normalize_drops_soft_hyphens_and_splits_ligatures():
    assert normalize("repro­ducible") == "reproducible"
    assert normalize("repro­\nducible") == "reproducible"
    assert normalize("ﬁne-tuning") == "fine-tuning"


def test_normalize_keeps_ordinary_hyphens():
    assert normalize("pre-training and fine-tuning") == "pre-training and fine-tuning"


# -- single citations ----------------------------------------------------------

def test_verified_across_line_hyphenation():
    assert _status("0", "the reproducibility of deep networks") == ("verified", [])


def test_verified_with_curly_quotes_and_dashes_in_the_chunk():
    assert _status("0", 'Our "model" uses a 3-4 layer MLP.') == ("verified", [])


def test_verified_ignores_case_and_line_breaks():
    assert _status("1", "TRAIN FOR 100 EPOCHS\nwith adam") == ("verified", [])


def test_wrong_chunk_reports_where_it_was_found():
    assert _status("0", "a learning rate of 0.001") == ("wrong_chunk", ["1"])


def test_unknown_chunk_id_with_real_quote_is_wrong_chunk():
    assert _status("99", "a learning rate of 0.001") == ("wrong_chunk", ["1"])


def test_fabricated_quote_is_not_found():
    assert _status("1", "a learning rate of 0.01 on ImageNet") == ("not_found", [])


def test_truncated_numbers_do_not_verify():
    assert _status("1", "a learning rate of 0.00") == ("not_found", [])
    assert _status("1", "train for 10 epochs with") == ("not_found", [])
    assert _status("1", "with Adam and a learning rate of 0") == ("not_found", [])  # stops at "0." of 0.001


def _one(text, quote):
    return ChunkIndex([{"chunk_id": "0", "text": text}]).classify("0", quote)[0]


def test_numbers_keep_signs_ranges_exponents_and_digit_groups():
    assert _one("The correlation is -0.42 overall.", "The correlation is 0.42 overall") == "not_found"
    assert _one("The correlation is 0.42 overall.", "The correlation is -0.42 overall") == "not_found"
    assert _one("Accuracy changed by \u22123.1% after pruning.", "Accuracy changed by 3.1% after pruning") == "not_found"
    assert _one("Accuracy changed by \u22123.1% after pruning.", "Accuracy changed by -3.1% after pruning") == "verified"
    assert _one("We use a 3\u20134 layer MLP for all runs.", "We use a 34 layer MLP for all runs") == "not_found"
    assert _one("We use a 3\u20134 layer MLP for all runs.", "We use a 3-4 layer MLP for all runs") == "verified"
    assert _one("with a learning rate of 1e-4 and Adam", "with a learning rate of 1e4 and Adam") == "not_found"
    assert _one("we train 3 64-unit layers with dropout", "we train 364-unit layers with dropout") == "not_found"
    assert _one("The dataset has 50,000 training images.", "000 training images in it") == "not_found"
    assert _one("TinyNet reaches 85.1% test accuracy.", "1% test accuracy.") == "not_found"
    assert _one("We train with a batch size of 1,024 on 8 GPUs.", "with a batch size of 1") == "not_found"
    assert _one("results on pages 10-\n12 of the appendix", "results on pages 10-12 of the appendix") == "verified"


def test_negating_prefixes_are_not_cut_off():
    assert _one("Results were non-reproducible across random seeds.", "reproducible across random seeds") == "not_found"
    assert _one("Results were irreproducible across random seeds.", "reproducible across random seeds") == "not_found"
    assert _one("The method is unstable under small batch sizes.", "stable under small batch sizes") == "not_found"
    assert _one("The method is unstable under small batch sizes.", "e method is unstable under") == "not_found"


def test_padding_does_not_dodge_the_length_rule():
    # padded "Adam" is still a 4-letter quote: it only matches chunk 2, which is exactly "Adam"
    assert _status("1", "Adam - - - - - -") == ("wrong_chunk", ["2"])
    assert _status("1", "0.001 - - - - -") == ("not_found", [])


def test_odd_unicode_in_honest_quotes():
    assert _one("It doesn\u00b4t converge in 5\u2033 runs.", "It doesn't converge in 5\" runs.") == "verified"
    assert _one("It doesn\u02bct converge at all.", "It doesn't converge at all.") == "verified"
    assert _one("We apply drop\u200bout to every layer.", "We apply dropout to every layer.") == "verified"


def test_run_together_pdf_text_still_verifies_but_not_cut_numbers():
    index = ChunkIndex([{"chunk_id": "0", "text": "Wetrainfor100epochswithAdam on CIFAR-10."}])
    assert index.classify("0", "We train for 100 epochs with Adam") == ("verified", [])
    assert index.classify("0", "We train for 10") == ("not_found", [])


def test_short_quote_needs_an_exact_chunk_match():
    # "the" and "Adam" appear in chunk 1 but are too short to prove anything there.
    assert _status("1", "the") == ("not_found", [])
    assert _status("1", "Adam") == ("wrong_chunk", ["2"])
    assert _status("2", "Adam") == ("verified", [])


def test_declared_missing_and_blank_quotes_are_empty():
    assert _status("", NOT_FOUND_TEXT) == ("empty", [])
    assert _status("3", "not found in provided text.") == ("empty", [])
    assert _status("0", "   ") == ("empty", [])


def test_chunk_id_spellings_and_int_ids():
    index = ChunkIndex([{"chunk_id": 1, "text": CHUNKS[1]["text"]}])
    assert index.classify("CHUNK 1", "with Adam and a learning rate") == ("verified", [])
    assert index.classify("[chunk 1]", "with Adam and a learning rate") == ("verified", [])
    assert index.classify(1, "with Adam and a learning rate") == ("verified", [])


# -- whole documents -----------------------------------------------------------

def _analysis(claim_citations):
    cit = lambda cid, q: {"chunk_id": cid, "quote": q}
    return {
        "paper_summary": "s",
        "main_claims": [{"claim": "c", "citations": [cit(*x) for x in claim_citations]}],
        "method_overview": {"overview": "o", "citations": [cit("1", "We train for 100 epochs")]},
        "results": [],
        "reproducibility_checklist": [],
        "missing_details": [{"detail": "seed", "impact": "i", "citations": [cit("", NOT_FOUND_TEXT)]}],
        "risk_score": 0,
        "risk_breakdown": [],
        "risk_reasons": None,
        "suggested_experiments": [],
        "citations": [cit("1", "learning rate of 0.001 on CIFAR-10")],
    }


def test_verify_citations_counts_and_locations():
    data = _analysis([("0", "deep networks. Our"), ("0", "on CIFAR-10."), ("1", "invented sentence here")])
    r = verify_citations(data, CHUNKS)
    assert {k: r[k] for k in ("total", "verified", "wrong_chunk", "not_found", "declared_missing")} == {
        "total": 6, "verified": 3, "wrong_chunk": 1, "not_found": 1, "declared_missing": 1,
    }
    assert r["issues"] == [
        {"location": "main_claims[0].citations[1]", "chunk_id": "0", "quote": "on CIFAR-10.",
         "status": "wrong_chunk", "found_in": ["1"]},
        {"location": "main_claims[0].citations[2]", "chunk_id": "1", "quote": "invented sentence here",
         "status": "not_found", "found_in": []},
    ]
    assert summary_line(r) == (
        "3/5 quotes verified against the paper text (1 not found, 1 in a different chunk); "
        "1 marked not found in the paper"
    )


def test_verify_citations_accepts_models_and_nested_dicts():
    model = PaperAnalysis.model_validate(_analysis([("0", "deep networks. Our")]))
    r = verify_citations(model, CHUNKS)
    assert (r["total"], r["verified"], r["issues"]) == (4, 3, [])

    plan_like = {"review": model.model_dump(), "plan": {"citations": [Citation(chunk_id="9", quote="x" * 20).model_dump()]}}
    r = verify_citations(plan_like, CHUNKS)
    assert r["issues"][0]["location"] == "plan.citations[0]"
    assert r["issues"][0]["status"] == "not_found"


def test_claim_model_citations_are_walked():
    claim = Claim(claim="c", citations=[Citation(chunk_id="1", quote="learning rate of 0.001")])
    assert verify_citations(claim, CHUNKS)["verified"] == 1
