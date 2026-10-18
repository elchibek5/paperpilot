from __future__ import annotations

from paperpilot.schemas import RiskFactor
from paperpilot.scoring import RUBRIC, compute_risk_score_from_breakdown


def _f(factor, status="present"):
    return {"factor": factor, "points": RUBRIC.get(factor, 5), "status": status, "rationale": "", "citations": []}


def test_all_present_subtracts_every_point_and_clamps():
    breakdown = [_f(name) for name in RUBRIC]
    assert sum(RUBRIC.values()) == 100
    assert compute_risk_score_from_breakdown(breakdown) == max(0, 100 - sum(RUBRIC.values())) == 0


def test_only_present_counts():
    breakdown = [_f("dataset defined & access described"), _f("baselines included", "missing"),
                 _f("ablations included", "unclear")]
    assert compute_risk_score_from_breakdown(breakdown) == 85


def test_a_factor_listed_twice_counts_once():
    breakdown = [_f("dataset defined & access described"), _f("dataset defined & access described")]
    assert compute_risk_score_from_breakdown(breakdown) == 85


def test_unknown_factor_is_ignored():
    assert compute_risk_score_from_breakdown([_f("vibes are good"), _f("hyperparameters reported")]) == 90


def test_empty_breakdown_is_maximum_risk_and_models_work():
    assert compute_risk_score_from_breakdown([]) == 100
    model = RiskFactor.model_validate(_f("limitations discussed"))
    assert compute_risk_score_from_breakdown([model]) == 90
