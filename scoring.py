"""
Deterministic reproducibility scoring. Score computed in code from risk_breakdown.
Start at 100 (high risk); subtract when status == "present" per rubric. Clamp 0–100.
"""

from __future__ import annotations

# rubric mapping: if a good thing is PRESENT, subtract points (risk goes down)
RUBRIC: dict[str, int] = {
    "dataset defined & access described": 15,
    "train/val/test split described": 15,
    "preprocessing/cleaning described": 10,
    "hyperparameters reported": 10,
    "compute/hardware reported": 10,
    "baselines included": 10,
    "ablations included": 10,
    "evaluation protocol clear": 10,
    "limitations discussed": 10,
}

# list of rubric factors (basically the allowed/expected checklist)
RUBRIC_FACTORS = list(RUBRIC.keys())


# computes a 0–100 "risk" score using the breakdown output (100 = worst, 0 = best)
def compute_risk_score_from_breakdown(risk_breakdown: list) -> int:
    """
    Compute risk score from Gemini risk_breakdown. Start at 100, subtract
    factor's points when status == "present". Clamp to 0–100.
    """
    score = 100
    for f in risk_breakdown:
        factor = getattr(f, "factor", None) or (f.get("factor") if isinstance(f, dict) else None)
        status = getattr(f, "status", None) or (f.get("status") if isinstance(f, dict) else None)
        pts = RUBRIC.get(str(factor).strip(), 0) if factor else 0
        if status == "present" and pts > 0:
            score -= pts
    return max(0, min(100, score))
