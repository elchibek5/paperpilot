"""
Prompt text shared by the Streamlit app and the CLI.
"""

from __future__ import annotations

import json

from .schemas import PaperAnalysis
from .scoring import RUBRIC_FACTORS

# -----------------------------------------------------------------------------
# Analysis schema and prompt (citations everywhere, not-found rule, rubric)
# -----------------------------------------------------------------------------

def _analysis_schema_desc() -> str:
    factors = json.dumps(RUBRIC_FACTORS)
    return f"""{{"paper_summary":"string","main_claims":[{{"claim":"string","citations":[{{"chunk_id":"string","quote":"string"}}]}}],"method_overview":{{"overview":"string","citations":[{{"chunk_id":"string","quote":"string"}}]}},"results":[{{"metric":"string","value":"string or null","description":"string","citations":[{{"chunk_id":"string","quote":"string"}}]}}],"reproducibility_checklist":[{{"item":"string","status":"present|missing|unclear","citations":[{{"chunk_id":"string","quote":"string"}}]}}],"missing_details":[{{"detail":"string","impact":"string","citations":[{{"chunk_id":"string","quote":"string"}}]}}],"risk_score":0,"risk_breakdown":[{{"factor":"string","points":"integer","status":"present|missing|unclear","rationale":"string","citations":[{{"chunk_id":"string","quote":"string"}}]}}],"risk_reasons":["string"] or null,"suggested_experiments":[{{"title":"string","goal":"string","steps":["string"]}}],"citations":[{{"chunk_id":"string","quote":"string"}}]}}
Risk breakdown MUST use exactly these factors (one entry each): {factors}
Use ONLY these keys. chunk_id matches chunks below. CITATIONS: chunk_id + direct quote. If NOT in text: "Not found in provided text" and citations: []."""


def build_analysis_prompt(chunks: list[dict]) -> str:
    chunk_blobs = [f"[CHUNK {c['chunk_id']}]\n{c['text']}" for c in chunks]
    chunks_block = "\n\n---\n\n".join(chunk_blobs)
    return f"""You are analyzing a research paper provided as numbered text chunks. Produce STRICT JSON only, no markdown or extra text.

Required JSON structure:
{_analysis_schema_desc()}

Rules:
- risk_breakdown: one entry per factor; status present|missing|unclear; rationale and citations. risk_score will be computed separately (use 0).
- Every claim, result, checklist item, missing detail, and risk factor: include citations (chunk_id + direct quote). If NOT in the text, use EXACTLY "Not found in provided text" and citations: [].
- Output ONLY valid JSON, no code fences.

Paper chunks:

{chunks_block}

Output (JSON only):"""


# -----------------------------------------------------------------------------
# Replication plan (Gemini JSON)
# -----------------------------------------------------------------------------

_REPLICATION_SCHEMA = """
{"prerequisites":["string"],"data_requirements":["string"],"environment_setup":["string"],"baseline_models":["string"],"training_plan":["string"],"evaluation_plan":["string"],"ablation_checklist":["string"],"estimated_compute":"string","risks_and_mitigations":["string"],"citations":[{"chunk_id":"string","quote":"string"}]}
"""


def build_replication_prompt(
    chunks: list[dict],
    analysis: PaperAnalysis,
) -> str:
    # Use summarized representation if many chunks: first 20 chunk previews
    max_chunks = 20
    parts = []
    for c in chunks[:max_chunks]:
        preview = c["text"][:2000] + ("..." if len(c["text"]) > 2000 else "")
        parts.append(f"[CHUNK {c['chunk_id']}]\n{preview}")
    chunks_block = "\n\n---\n\n".join(parts)
    summary = analysis.paper_summary
    claims = "\n".join(f"- {x.claim}" for x in analysis.main_claims)
    missing = "\n".join(f"- {m.detail} ({m.impact})" for m in analysis.missing_details)
    breakdown = "\n".join(f"- {f.factor}: {f.status} — {f.rationale}" for f in analysis.risk_breakdown)
    return f"""Generate a replication plan for this research paper. Use the review summary, claims, missing details, and risk breakdown below, plus the paper chunks.

JSON structure (use exactly these keys):
{_REPLICATION_SCHEMA}

Paper chunks (excerpts):
{chunks_block}

Review summary:
{summary}

Main claims:
{claims}

Missing details:
{missing}

Risk breakdown:
{breakdown}

Output STRICT JSON only. Where info is not in the text, use "Not found in provided text" and empty citations. Include optional citations with chunk_id + quote where relevant."""


# -----------------------------------------------------------------------------
# Long-paper summarization and RAG-lite Q&A
# -----------------------------------------------------------------------------

def build_summary_prompt(chunk_texts: list[str]) -> str:
    block = "\n\n---\n\n".join(chunk_texts)
    prompt = """Summarize the following sections of a research paper into one coherent summary.
Preserve key claims, methods, results, and reproducibility-relevant details (datasets, splits, hyperparameters, etc.).
Output plain text only, no markdown."""

    prompt += f"\n\nSections:\n\n{block}"
    return prompt


def build_qa_prompt(retrieved: list[dict], question: str) -> str:
    context = "\n\n".join(f"[CHUNK {c['chunk_id']}]\n{c['text']}" for c in retrieved)
    prompt = f"""Answer the question using ONLY the provided paper chunks. Cite each fact with [chunk_id: <id>] and a short direct quote.

Paper context:

{context}

Question: {question}

Rules:
- Answer concisely. For each claim, cite chunk_id and quote.
- If the answer is NOT in the provided chunks, respond EXACTLY: "Not found in provided text."
- Output plain text with inline citations; no JSON."""
    return prompt
