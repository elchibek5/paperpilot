"""
Export Review + Replication Plan to a single Markdown document.
"""

from __future__ import annotations

from schemas import (
    Citation,
    PaperAnalysis,
    ReplicationPlan,
)


# turns one citation into a nice markdown bullet (chunk id + exact quote)
def _cite_md(c: Citation) -> str:
    return f"- [Chunk {c.chunk_id}] \"{c.quote}\"\n"


# builds the “Evidence” section from a list of citations (or nothing if empty)
def _evidence_md(citations: list[Citation]) -> str:
    if not citations:
        return ""
    return "**Evidence:**\n\n" + "".join(_cite_md(c) for c in citations) + "\n"


# converts the full paper review output into a markdown report
def review_to_md(a: PaperAnalysis, computed_risk: int, citation_check: str | None = None) -> str:
    """Render PaperAnalysis as Markdown. citation_check: one line from verify.summary_line."""
    lines = [
        "# Paper Review",
        "",
        f"**Risk score (computed):** {computed_risk}/100",
        "",
    ]
    if citation_check:
        lines.extend([f"**Citation check:** {citation_check}", ""])
    lines += [
        "## Summary",
        "",
        a.paper_summary,
        "",
        "## Main claims",
        "",
    ]
    for i, c in enumerate(a.main_claims, 1):
        lines.append(f"{i}. {c.claim}")
        lines.append(_evidence_md(c.citations))
    lines.extend(["## Method", "", a.method_overview.overview, "", _evidence_md(a.method_overview.citations)])
    lines.extend(["## Results", ""])
    for r in a.results:
        val = f" = {r.value}" if r.value else ""
        lines.append(f"- **{r.metric}**{val}")
        lines.append(r.description)
        lines.append(_evidence_md(r.citations))
    lines.extend(["## Reproducibility checklist", ""])
    for x in a.reproducibility_checklist:
        lines.append(f"- **{x.item}** — {x.status}")
        lines.append(_evidence_md(x.citations))
    lines.extend(["## Missing details", ""])
    for m in a.missing_details:
        lines.append(f"- **{m.detail}** — impact: {m.impact}")
        lines.append(_evidence_md(m.citations))
    lines.extend(["## Risk breakdown", ""])
    for f in a.risk_breakdown:
        lines.append(f"- **{f.factor}** ({f.points} pts) — {f.status}")
        lines.append(f"  {f.rationale}")
        lines.append(_evidence_md(f.citations))
    if a.risk_reasons:
        lines.append("### Additional reasons")
        for r in a.risk_reasons:
            lines.append(f"- {r}")
    lines.extend(["## Suggested experiments", ""])
    for e in a.suggested_experiments:
        lines.append(f"### {e.title}")
        lines.append(e.goal)
        for s in e.steps:
            lines.append(f"- {s}")
    lines.extend(["## Citations", ""])
    for c in a.citations:
        lines.append(f"- [Chunk {c.chunk_id}] \"{c.quote}\"")
    return "\n".join(lines)


# converts the replication plan output into markdown sections
def replication_plan_to_md(p: ReplicationPlan) -> str:
    """Render ReplicationPlan as Markdown."""
    lines = [
        "# Replication Plan",
        "",
        "## Prerequisites",
        "",
    ]
    for x in p.prerequisites:
        lines.append(f"- {x}")
    lines.extend(["", "## Data requirements", ""])
    for x in p.data_requirements:
        lines.append(f"- {x}")
    lines.extend(["", "## Environment setup", ""])
    for x in p.environment_setup:
        lines.append(f"- {x}")
    lines.extend(["", "## Baseline models", ""])
    for x in p.baseline_models:
        lines.append(f"- {x}")
    lines.extend(["", "## Training plan", ""])
    for x in p.training_plan:
        lines.append(f"- {x}")
    lines.extend(["", "## Evaluation plan", ""])
    for x in p.evaluation_plan:
        lines.append(f"- {x}")
    lines.extend(["", "## Ablation checklist", ""])
    for x in p.ablation_checklist:
        lines.append(f"- {x}")
    lines.extend(["", "## Estimated compute", "", p.estimated_compute, ""])
    lines.extend(["## Risks and mitigations", ""])
    for x in p.risks_and_mitigations:
        lines.append(f"- {x}")
    if p.citations:
        lines.extend(["", "## Citations", ""])
        for c in p.citations:
            lines.append(f"- [Chunk {c.chunk_id}] \"{c.quote}\"")
    return "\n".join(lines)


# combines review + replication plan into one markdown doc (plan optional)
def combined_md(
    a: PaperAnalysis,
    computed_risk: int,
    plan: ReplicationPlan | None,
    citation_check: str | None = None,
) -> str:
    """Single Markdown: Review + Replication Plan."""
    out = review_to_md(a, computed_risk, citation_check)
    if plan:
        out += "\n\n---\n\n" + replication_plan_to_md(plan)
    return out
