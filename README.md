# PaperPilot

PaperPilot extracts text from PDF research papers, chunks them, and uses the **Gemini API** to produce a structured review (summary, claims, method, results, reproducibility, risk breakdown, experiments, citations). It supports a **replication plan**, **RAG-lite Q&A** over chunks, and **Markdown export**.

## What it does

- **PDF upload** → extract text (**pdfplumber**, fallback **pypdf**)
- **Chunking** → ~800–1200 token chunks with overlap; cap at 40 chunks (with batch summarization when longer)
- **Review** → Gemini returns structured JSON; **risk score** is computed in code from a deterministic rubric (dataset, splits, hyperparameters, etc.)
- **Citations** → Every claim/result/checklist/risk item includes chunk_id + quote, or “Not found in provided text” when absent
- **Replication plan** → Generate prerequisites, data/env, training/eval, ablations, compute, risks
- **Q&A** → TF-IDF retrieval over chunks → Gemini answer with citations; “Not found in provided text” if unsupported
- **Export** → Download Review and/or Review + Replication Plan as a single Markdown file

## Setup

### 1. Virtual environment (recommended)

```bash
cd project
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows:  .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. API key

Secrets are read only from the environment (e.g. via `.env` and `python-dotenv`).

**Option A — `.env`**

```bash
cp .env.example .env
# Edit .env and set:
# GEMINI_API_KEY=your_actual_key_here
```

**Option B — export**

```bash
export GEMINI_API_KEY=your_actual_key_here
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey). `GOOGLE_API_KEY` is also accepted.

## Run

From the `project` directory:

```bash
streamlit run paperpilot.py
```

Open the URL shown (typically `http://localhost:8501`).

## 90-second demo

1. Run `streamlit run paperpilot.py` and open the app.
2. Upload a PDF research paper (e.g. a short ML paper).
3. Wait for extraction and chunking (and summarization if the paper is long).
4. Click **Analyze with Gemini** and wait for the review.
5. Check the **risk score** badge and **Risk** tab (breakdown table).
6. Click **Generate Replication Plan**, then **Download as Markdown**.
7. In **Q&A**, ask e.g. “What dataset did the authors use?” and check the answer and **Evidence** expander.

## Known limitations

- **Scanned PDFs**: If extraction yields &lt; 1500 characters, the app shows an error and suggests trying another PDF or using OCR.
- **Long papers**: Chunks are capped at 40; excess is batch-summarized before review. Very long papers may lose detail.
- **Gemini**: Rate limits or timeouts trigger one automatic retry, then a user-friendly error.

## Requirements

See `requirements.txt`:

- `streamlit` — UI
- `pdfplumber` — PDF extraction (primary)
- `pypdf` — fallback extraction
- `google-generativeai` — Gemini API
- `python-dotenv` — load `.env`
- `pydantic` — schemas
- `scikit-learn` — TF-IDF retrieval

## License

Use and modify as you like.
