# Working on PaperPilot

PaperPilot reviews ML papers for reproducibility. A model (Gemini, or a coding agent
following `skills/paper-review/SKILL.md`) writes a review that quotes the paper;
PaperPilot computes the risk score and verifies every quote against the paper text.

## Module map

The installable code is the `paperpilot/` package (imports inside it are relative).
The Streamlit app is `app.py` at the repo root; it is not installed. Do not add a root
`paperpilot.py`: it would shadow the package.

| Module | Role |
|---|---|
| `paperpilot/__init__.py` | `__version__`, the single source of the version (read by `pyproject.toml`). |
| `paperpilot/__main__.py` | `python -m paperpilot` runs the CLI. |
| `paperpilot/pdftext.py` | PDF text extraction (pdfplumber, then pypdf) and `chunk_text`. Chunk ids are strings. |
| `paperpilot/schemas.py` | Pydantic models: `PaperAnalysis`, `ReplicationPlan`, `Citation`, ... (`extra="forbid"`). |
| `paperpilot/scoring.py` | The rubric (`RUBRIC`) and `compute_risk_score_from_breakdown`. |
| `paperpilot/verify.py` | `normalize`, `ChunkIndex.classify`, `verify_citations`, `summary_line`. |
| `paperpilot/retrieval.py` | TF-IDF ranking (`rank_chunks_tfidf`, `retrieve_chunks_tfidf`). |
| `paperpilot/prompts.py` | All prompt text, shared by the app and the CLI. |
| `paperpilot/llm.py` | Gemini via google-genai: `generate_json`, `generate_text`, `run_analysis`, `run_replication_plan`. Imports the SDK lazily. |
| `paperpilot/export.py` | Markdown export. |
| `paperpilot/cli.py` | The `paperpilot` command (`paperpilot.cli:main`). |
| `app.py` | The Streamlit app. |
| `skills/paper-review/SKILL.md` | Instructions for agents that use PaperPilot without an API key. |
| `.claude-plugin/` | Claude Code plugin and marketplace manifests that ship the skill. |
| `examples/` | A worked review with its clean `check --strict` output. |
| `tests/` | pytest; fixtures in `tests/fixtures/`. |

## Commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[gemini,app]' -r requirements-dev.txt
pytest -q                                   # no key, no network
streamlit run app.py                        # web app (needs GEMINI_API_KEY)
paperpilot extract paper.pdf -o chunks.json # CLI
```

## Rules

- **Schema changes move together.** A change to `paperpilot/schemas.py` must update, in
  the same commit: the field list in `skills/paper-review/SKILL.md`, the hand-written JSON
  description in `prompts.py`, `export.py`, the `check` logic in `cli.py` if it reads
  the field, the fixtures and tests, and `examples/` (re-run `paperpilot check --strict`).
- **Rubric changes move together.** A change to `RUBRIC` in `paperpilot/scoring.py` must update the
  rubric tables in `SKILL.md` and `README.md`, the fixtures, and the example. The
  prompt and `paperpilot schema` read the rubric from `scoring.py`.
- **Prompt text lives in `prompts.py`.** Do not inline prompts in the app or the CLI.
- **The risk score is computed in code.** Never trust or display a model-provided score
  as the result.
- **No test may need an API key or the network.** Mock `google.genai.Client` (see
  `tests/test_llm.py`); build PDFs in the test (see `tests/conftest.py`).
- **The no-key CLI commands must not import `google.genai`.** Keep SDK imports inside
  `paperpilot/llm.py` functions.
- **The default model** is `DEFAULT_MODEL` in `paperpilot/llm.py`. Check Google's models page
  (https://ai.google.dev/gemini-api/docs/models) before changing it, and update
  `.env.example` and the README.
- **`skills/` and `.claude-plugin/` stay in sync with the CLI.** Renaming or changing a
  `paperpilot` command or flag means updating `skills/paper-review/SKILL.md` and, if
  the plugin's name or description changes, `.claude-plugin/plugin.json` and
  `marketplace.json` in the same commit; run `claude plugin validate .` afterwards.
- Never commit `.env`, PDFs, or `chunks.json` dumps. Stage files by name.
- After changing the app, start it (`streamlit run app.py --server.headless true`)
  and check the log for tracebacks.

## Known traps

- pdfplumber can interleave the two columns of a two-column paper line by line, and
  some PDFs lose spaces between words. `paperpilot/verify.py` tolerates spacing and hyphens for
  that reason; do not "fix" the chunk text, quotes are checked against it.
- The loose comparison drops spaces and word hyphens but never touches numbers
  (signs, ranges, exponents, decimals, digit groups), and a quote may not start or
  stop inside a number or a word, except where the PDF text has lost its spaces.
  Quotes under 12 characters or 8 letters and digits only verify against a whole
  chunk. `tests/test_verify.py` pins each case; any loosening needs a test showing
  an altered quote still fails.
- `check --strict` also needs evidence behind each item (`evidence_report` in
  `paperpilot/cli.py`), so a review cannot mark factors present without a verified quote.
- The app summarizes papers over 40 chunks with Gemini once per upload; citations are
  then checked against the summaries. The CLI never summarizes.
