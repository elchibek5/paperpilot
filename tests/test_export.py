from __future__ import annotations

from export import combined_md, review_to_md
from schemas import PaperAnalysis, ReplicationPlan

MINIMAL = {
    "paper_summary": "A summary.",
    "main_claims": [{"claim": "It works.", "citations": [{"chunk_id": "0", "quote": "it works"}]}],
    "method_overview": {"overview": "A method.", "citations": []},
    "results": [],
    "reproducibility_checklist": [],
    "missing_details": [],
    "risk_score": 0,
    "risk_breakdown": [],
    "suggested_experiments": [],
    "citations": [],
}


def test_review_markdown_carries_the_citation_check_line():
    a = PaperAnalysis.model_validate(MINIMAL)
    md = review_to_md(a, 40, "1/1 quotes verified against the paper text")
    assert "**Risk score (computed):** 40/100" in md
    assert "**Citation check:** 1/1 quotes verified against the paper text" in md
    assert "Citation check" not in review_to_md(a, 40)


def test_combined_markdown_appends_the_plan():
    a = PaperAnalysis.model_validate(MINIMAL)
    plan = ReplicationPlan(
        prerequisites=["GPU"], data_requirements=[], environment_setup=[], baseline_models=[],
        training_plan=[], evaluation_plan=[], ablation_checklist=[], estimated_compute="1 GPU-hour",
        risks_and_mitigations=[],
    )
    md = combined_md(a, 40, plan, "1/1 quotes verified against the paper text")
    assert md.index("**Citation check:**") < md.index("# Replication Plan")
    assert "1 GPU-hour" in md
