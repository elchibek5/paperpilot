"""
PaperPilot — Extract, chunk, and analyze PDF research papers with Gemini.
"""

from __future__ import annotations

import json

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from export import combined_md
from llm import generate_json, generate_text, _get_api_key
from pdftext import MIN_PDF_CHARS, chunk_text, extract_text_from_pdf
from retrieval import retrieve_chunks_tfidf
from schemas import Citation, PaperAnalysis, ReplicationPlan
from scoring import RUBRIC, RUBRIC_FACTORS, compute_risk_score_from_breakdown

load_dotenv()

MAX_CHUNKS = 40
SUMMARIZE_BATCH_SIZE = 10
DEBUG_CHUNK_PREVIEW = 300

# -----------------------------------------------------------------------------
# Chunk cap + 2-step summarization
# -----------------------------------------------------------------------------

def _summarize_batch(chunk_texts: list[str]) -> str:
    """Summarize a batch of chunk texts into one summary via Gemini."""
    block = "\n\n---\n\n".join(chunk_texts)
    prompt = f"""Summarize the following sections of a research paper into one coherent summary.
Preserve key claims, methods, results, and reproducibility-relevant details (datasets, splits, hyperparameters, etc.).
Output plain text only, no markdown."""

    prompt += f"\n\nSections:\n\n{block}"
    return generate_text(prompt)


def _chunks_to_review_chunks(chunks: list[dict]) -> list[dict]:
    """If len(chunks) > MAX_CHUNKS, batch-summarize into <= MAX_CHUNKS; else return chunks."""
    if len(chunks) <= MAX_CHUNKS:
        return chunks
    batches: list[list[dict]] = []
    for i in range(0, len(chunks), SUMMARIZE_BATCH_SIZE):
        batch = chunks[i : i + SUMMARIZE_BATCH_SIZE]
        batches.append(batch)
    out = []
    for idx, batch in enumerate(batches):
        texts = [c["text"] for c in batch]
        summary = _summarize_batch(texts)
        out.append({"chunk_id": str(idx), "text": summary, "approx_tokens": 0})
    return out


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


def run_analysis(chunks: list[dict]) -> PaperAnalysis:
    prompt = build_analysis_prompt(chunks)
    raw = generate_json(prompt)
    data = json.loads(raw)
    return PaperAnalysis.model_validate(data)


# -----------------------------------------------------------------------------
# RAG-lite Q&A (TF-IDF retrieval + Gemini, citations, not-found rule)
# -----------------------------------------------------------------------------

def answer_rag(
    chunks: list[dict],
    question: str,
    top_k: int = 5,
) -> tuple[str, list[dict]]:
    retrieved = retrieve_chunks_tfidf(chunks, question, top_k=top_k)
    context = "\n\n".join(f"[CHUNK {c['chunk_id']}]\n{c['text']}" for c in retrieved)
    prompt = f"""Answer the question using ONLY the provided paper chunks. Cite each fact with [chunk_id: <id>] and a short direct quote.

Paper context:

{context}

Question: {question}

Rules:
- Answer concisely. For each claim, cite chunk_id and quote.
- If the answer is NOT in the provided chunks, respond EXACTLY: "Not found in provided text."
- Output plain text with inline citations; no JSON."""

    answer = generate_text(prompt)
    return answer, retrieved


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


def run_replication_plan(chunks: list[dict], analysis: PaperAnalysis) -> ReplicationPlan:
    prompt = build_replication_prompt(chunks, analysis)
    raw = generate_json(prompt)
    data = json.loads(raw)
    return ReplicationPlan.model_validate(data)


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def _risk_color(score: int) -> str:
    if score <= 33:
        return "#22c55e"
    if score <= 66:
        return "#eab308"
    return "#ef4444"


def render_ui():
    st.set_page_config(page_title="PaperPilot", page_icon="📄", layout="wide")
    st.title("📄 PaperPilot")
    st.caption("Upload a PDF research paper → extract, chunk, analyze with Gemini → ask follow-up questions.")

    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "replication_plan" not in st.session_state:
        st.session_state.replication_plan = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if not uploaded:
        st.info("Upload a PDF research paper to get started.")
        return

    with st.status("Extracting text…", state="running") as status:
        try:
            text = extract_text_from_pdf(uploaded)
        except Exception as e:
            st.error(f"Could not extract text: {e}")
            return
        if len(text) < MIN_PDF_CHARS:
            st.error(
                f"Extracted only {len(text):,} characters. This PDF may be scanned or image-based. "
                "Try another PDF or use an OCR tool first."
            )
            return
        status.update(label="Extraction done", state="complete")
    st.success(f"Extracted {len(text):,} characters.")

    with st.status("Chunking…", state="running") as ch_status:
        raw_chunks = chunk_text(text)
        ch_status.update(label="Chunking…", state="running")
        review_chunks = _chunks_to_review_chunks(raw_chunks)
        ch_status.update(label="Chunking done", state="complete")
    st.session_state.chunks = review_chunks
    n = len(review_chunks)
    cap_note = f" (capped from {len(raw_chunks)} via summarization)" if len(raw_chunks) > MAX_CHUNKS else ""
    st.caption(f"{n} chunks for review{cap_note}.")

    if st.button("Analyze with Gemini", type="primary", use_container_width=True):
        with st.status("Running LLM review…", state="running") as rev_status:
            try:
                rev_status.update(label="Calling Gemini API…", state="running")
                analysis = run_analysis(review_chunks)
                st.session_state.analysis = analysis
                st.session_state.replication_plan = None
            except ValueError as e:
                if "API" in str(e) or "GEMINI" in str(e).upper() or "GOOGLE" in str(e).upper():
                    rev_status.update(label="Error", state="error")
                    st.error(str(e))
                    return
                raise
            except json.JSONDecodeError as e:
                rev_status.update(label="Error", state="error")
                st.error(f"Gemini returned invalid JSON: {e}")
                return
            except ValidationError as e:
                rev_status.update(label="Error", state="error")
                st.error(f"Schema validation failed: {e}")
                return
            except Exception as e:
                rev_status.update(label="Error", state="error")
                msg = str(e).lower()
                if "rate" in msg or "limit" in msg or "timeout" in msg or "429" in msg or "503" in msg:
                    st.error("Gemini rate limit or timeout. Please retry in a moment.")
                else:
                    st.error(f"Analysis failed: {e}")
                return
            rev_status.update(label="Review complete", state="complete")
        st.rerun()

    analysis: PaperAnalysis = st.session_state.analysis
    if not analysis:
        return

    computed_risk = compute_risk_score_from_breakdown(analysis.risk_breakdown)
    risk = max(0, min(100, computed_risk))
    color = _risk_color(risk)
    st.markdown(
        f'<div style="display:inline-block;padding:0.5rem 1rem;border-radius:8px;'
        f'background:{color}22;color:{color};border:1px solid {color};font-weight:bold;font-size:1.1rem;">'
        f'Risk score: {risk}/100 (computed from rubric)</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    def _evidence_expander(label: str, citations: list) -> None:
        if not citations:
            return
        with st.expander(label, expanded=False):
            for c in citations:
                st.caption(f'[Chunk {c.chunk_id}] "{c.quote}"')

    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
        "Summary", "Main claims", "Method", "Results", "Reproducibility",
        "Missing details", "Risk", "Experiments", "Citations",
    ])
    with t1:
        st.write(analysis.paper_summary)
    with t2:
        for i, c in enumerate(analysis.main_claims, 1):
            st.markdown(f"**{i}.** {c.claim}")
            _evidence_expander("Evidence", c.citations)
    with t3:
        st.write(analysis.method_overview.overview)
        _evidence_expander("Evidence", analysis.method_overview.citations)
    with t4:
        for r in analysis.results:
            val_part = f" = {r.value}" if r.value else ""
            st.markdown(f"**{r.metric}**{val_part}")
            st.write(r.description)
            _evidence_expander("Evidence", r.citations)
    with t5:
        for x in analysis.reproducibility_checklist:
            st.markdown(f"- **{x.item}** — {x.status}")
            _evidence_expander("Evidence", x.citations)
    with t6:
        for m in analysis.missing_details:
            st.markdown(f"- **{m.detail}** — impact: {m.impact}")
            _evidence_expander("Evidence", m.citations)
    with t7:
        st.markdown("**Breakdown (score computed in code from rubric)**")
        for f in analysis.risk_breakdown:
            pts = RUBRIC.get(f.factor.strip(), f.points)
            st.markdown(f"**{f.factor}** — {f.status} ({pts} pts when present)")
            st.write(f.rationale)
            _evidence_expander("Evidence", f.citations)
        if analysis.risk_reasons:
            st.markdown("---")
            st.markdown("**Additional reasons**")
            for r in analysis.risk_reasons:
                st.markdown(f"- {r}")
    with t8:
        for e in analysis.suggested_experiments:
            st.markdown(f"**{e.title}**")
            st.write(e.goal)
            for step in e.steps:
                st.markdown(f"  - {step}")
    with t9:
        for cit in analysis.citations:
            st.markdown(f"**Chunk {cit.chunk_id}:** \"{cit.quote}\"")

    with st.expander("Debug: Show chunks", expanded=False):
        for c in st.session_state.chunks or []:
            preview = (c["text"][:DEBUG_CHUNK_PREVIEW] + "…") if len(c["text"]) > DEBUG_CHUNK_PREVIEW else c["text"]
            st.markdown(f"**Chunk {c['chunk_id']}**")
            st.text(preview)
            st.markdown("---")

    st.markdown("---")
    st.subheader("Replication Plan")
    if st.button("Generate Replication Plan", key="repl_btn") and st.session_state.chunks:
        with st.status("Generating replication plan…", state="running") as repl_status:
            try:
                plan = run_replication_plan(st.session_state.chunks, analysis)
                st.session_state.replication_plan = plan
                repl_status.update(label="Done", state="complete")
            except Exception as e:
                repl_status.update(label="Error", state="error")
                st.error(f"Replication plan failed: {e}")
        st.rerun()

    plan = st.session_state.replication_plan
    if plan:
        st.markdown("### Prerequisites")
        for x in plan.prerequisites:
            st.markdown(f"- {x}")
        st.markdown("### Data requirements")
        for x in plan.data_requirements:
            st.markdown(f"- {x}")
        st.markdown("### Environment setup")
        for x in plan.environment_setup:
            st.markdown(f"- {x}")
        st.markdown("### Baseline models")
        for x in plan.baseline_models:
            st.markdown(f"- {x}")
        st.markdown("### Training plan")
        for x in plan.training_plan:
            st.markdown(f"- {x}")
        st.markdown("### Evaluation plan")
        for x in plan.evaluation_plan:
            st.markdown(f"- {x}")
        st.markdown("### Ablation checklist")
        for x in plan.ablation_checklist:
            st.markdown(f"- {x}")
        st.markdown("### Estimated compute")
        st.write(plan.estimated_compute)
        st.markdown("### Risks and mitigations")
        for x in plan.risks_and_mitigations:
            st.markdown(f"- {x}")
        md = combined_md(analysis, computed_risk, plan)
        st.download_button("Download as Markdown", md, file_name="paperpilot_review_and_plan.md", mime="text/markdown", key="dl_md_full")
    else:
        md = combined_md(analysis, computed_risk, None)
        st.download_button("Download Review as Markdown", md, file_name="paperpilot_review.md", mime="text/markdown", key="dl_md_review")

    st.markdown("---")
    st.subheader("Q&A (RAG)")
    st.caption("Ask about the paper. Answers use TF-IDF retrieval + citations.")
    q = st.text_input("Your question", label_visibility="collapsed", placeholder="e.g. What dataset did the authors use?", key="qa_input")
    if st.button("Ask", key="qa_ask") and q.strip() and st.session_state.chunks:
        with st.status("Retrieving & answering…", state="running") as qa_status:
            try:
                answer, retrieved = answer_rag(st.session_state.chunks, q.strip(), top_k=5)
                st.session_state.qa_history.append({
                    "question": q.strip(),
                    "answer": answer,
                    "retrieved": retrieved,
                })
                qa_status.update(label="Done", state="complete")
            except Exception as e:
                qa_status.update(label="Error", state="error")
                st.error(f"Q&A failed: {e}")
        st.rerun()

    for h in st.session_state.qa_history:
        with st.expander(h["question"], expanded=True):
            st.markdown(h["answer"])
            with st.expander("Evidence (retrieved chunks)", expanded=False):
                for c in h.get("retrieved") or []:
                    pre = (c["text"][:300] + "…") if len(c["text"]) > 300 else c["text"]
                    st.markdown(f"**Chunk {c['chunk_id']}**")
                    st.text(pre)
                    st.markdown("---")


def main():
    try:
        _get_api_key()
    except ValueError as e:
        st.set_page_config(page_title="PaperPilot", page_icon="📄")
        st.title("📄 PaperPilot")
        st.error(str(e))
        st.info("Create a `.env` file with `GEMINI_API_KEY=your_key` or export it in your shell.")
        return
    render_ui()


if __name__ == "__main__":
    main()
