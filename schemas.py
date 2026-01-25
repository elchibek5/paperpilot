"""
Pydantic schemas for PaperPilot Gemini output. All models use extra="forbid".
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


def _forbid_extra() -> ConfigDict:
    return ConfigDict(extra="forbid")


# -----------------------------------------------------------------------------
# Citation (reused across nested types)
# -----------------------------------------------------------------------------


class Citation(BaseModel):
    model_config = _forbid_extra()

    chunk_id: str
    quote: str

    @field_validator("chunk_id", mode="before")
    @classmethod
    def coerce_chunk_id(cls, v: str | int | None) -> str:
        if v is None:
            raise ValueError("chunk_id is required")
        return str(v)


# -----------------------------------------------------------------------------
# Nested types
# -----------------------------------------------------------------------------


class Claim(BaseModel):
    model_config = _forbid_extra()

    claim: str
    citations: list[Citation]


class ResultItem(BaseModel):
    model_config = _forbid_extra()

    metric: str
    value: str | None
    description: str
    citations: list[Citation]


class ChecklistItem(BaseModel):
    model_config = _forbid_extra()

    item: str
    status: Literal["present", "missing", "unclear"]
    citations: list[Citation]


class MissingDetail(BaseModel):
    model_config = _forbid_extra()

    detail: str
    impact: str
    citations: list[Citation]


class RiskFactor(BaseModel):
    model_config = _forbid_extra()

    factor: str
    points: int
    status: Literal["present", "missing", "unclear"]
    rationale: str
    citations: list[Citation]


class Experiment(BaseModel):
    model_config = _forbid_extra()

    title: str
    goal: str
    steps: list[str]


class MethodOverview(BaseModel):
    """method_overview: str with citations."""

    model_config = _forbid_extra()

    overview: str
    citations: list[Citation]


# -----------------------------------------------------------------------------
# Top-level analysis
# -----------------------------------------------------------------------------


class PaperAnalysis(BaseModel):
    model_config = _forbid_extra()

    paper_summary: str
    main_claims: list[Claim]
    method_overview: MethodOverview
    results: list[ResultItem]
    reproducibility_checklist: list[ChecklistItem]
    missing_details: list[MissingDetail]
    risk_score: int  # 0–100 (from Gemini; overwritten by scoring.compute in app)
    risk_breakdown: list[RiskFactor]
    risk_reasons: list[str] | None = None  # optional if breakdown exists
    suggested_experiments: list[Experiment]
    citations: list[Citation]


# -----------------------------------------------------------------------------
# Replication plan (Gemini output)
# -----------------------------------------------------------------------------


class ReplicationPlan(BaseModel):
    model_config = _forbid_extra()

    prerequisites: list[str]
    data_requirements: list[str]
    environment_setup: list[str]
    baseline_models: list[str]
    training_plan: list[str]
    evaluation_plan: list[str]
    ablation_checklist: list[str]
    estimated_compute: str
    risks_and_mitigations: list[str]
    citations: list[Citation] = []
