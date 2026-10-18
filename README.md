# PaperPilot

PaperPilot reviews ML research papers for reproducibility: it extracts the PDF text, and a model (Gemini, or the coding agent you already use) writes a structured review in which every claim, result and risk item cites a quote from the paper. PaperPilot then computes the risk score from a fixed rubric and checks mechanically that every quote really appears in the chunk of the paper it cites.

## Use it with your coding agent (no API key)

Claude Code, Codex, Cursor and similar agents can do the reading and writing themselves. PaperPilot supplies the paper text, the schema, the rubric, search, and the citation check.

```bash
pipx install git+https://github.com/elchibek5/PaperPilot-Gemini
```

Then give your agent the skill in [`skills/paper-review/SKILL.md`](skills/paper-review/SKILL.md). For Claude Code, put it in your skills folder (or in `.claude/skills/` inside a project):

```bash
mkdir -p ~/.claude/skills/paper-review
curl -L -o ~/.claude/skills/paper-review/SKILL.md \
  https://raw.githubusercontent.com/elchibek5/PaperPilot-Gemini/main/skills/paper-review/SKILL.md
```

Other agents can be told to follow that file. The skill runs this loop:

1. `paperpilot extract paper.pdf -o chunks.json` turns the PDF into numbered chunks.
2. `paperpilot schema` prints the review format and the rubric.
3. The agent reads the chunks (`paperpilot search chunks.json "learning rate"` for long papers).
4. The agent writes `review.json`, quoting the paper verbatim for every item, or writing exactly "Not found in provided text" when the paper does not say.
5. `paperpilot check review.json --chunks chunks.json --strict` until it exits 0, then `paperpilot export review.json --chunks chunks.json -o review.md`.

A failing check tells the agent exactly what to fix (trimmed):

```text
check: schema ok; rubric ok; 2 item(s) without evidence; 13/15 quotes verified against the paper text (1 not found, 1 in a different chunk); ...
    "issues": [
      {"location": "main_claims[1].citations[0]", "chunk_id": "2",
       "quote": "TinyNet sets a new state of the art on every tabular benchmark",
       "status": "not_found", "found_in": []},
      {"location": "results[0].citations[0]", "chunk_id": "0",
       "quote": "TinyNet reaches 85.1% test accuracy",
       "status": "wrong_chunk", "found_in": ["2"]}
    ]
```

## Use the web app (Gemini key)

```bash
git clone https://github.com/elchibek5/PaperPilot-Gemini
cd PaperPilot-Gemini
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then set GEMINI_API_KEY in .env
streamlit run paperpilot.py
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey); `GOOGLE_API_KEY` also works. The default model is `gemini-3.8-flash`; set `PAPERPILOT_MODEL` to use another. Keys are read from the environment or `.env` only.

In the app: upload a PDF, click **Analyze with Gemini**, and read the risk score and the citation check badges. Each evidence quote is marked verified or not found. **Generate Replication Plan** adds a plan, **Download as Markdown** exports both, and **Q&A** answers questions from the most relevant chunks with citations.

The same Gemini review is available from the command line: `paperpilot review chunks.json -o review.json` and `paperpilot plan chunks.json review.json -o plan.json` (install with `pip install -e '.[gemini]'`, or `pipx inject paperpilot google-genai`).

## CLI reference

| Command | Key | What it does |
|---|---|---|
| `paperpilot extract PAPER.pdf [-o chunks.json]` | no | Extract text and split it into overlapping chunks (~1000 tokens). Exit 2 if the PDF has under 1,500 characters of text. |
| `paperpilot search chunks.json "query" [-k 5]` | no | TF-IDF search; returns `chunk_id`, `score`, `text`. |
| `paperpilot schema [--plan]` | no | JSON Schema of the review (or replication plan), with the rubric in `x-rubric`. |
| `paperpilot check review.json --chunks chunks.json [--strict]` | no | Validate against the schema, recompute the risk score, check the rubric factors, verify every quote, and list items without evidence. Exit 1 if the schema is invalid; with `--strict`, also if any quote is not found or in the wrong chunk, a rubric factor is missing, duplicated, unknown or has the wrong points, or an item lacks the quote it needs (claims and results always; checklist items and risk factors when marked present). |
| `paperpilot export review.json [--plan plan.json] [--chunks chunks.json] [-o review.md]` | no | Markdown with the computed risk score and a citation check line. |
| `paperpilot review chunks.json [-o review.json]` | yes | Gemini writes the review. |
| `paperpilot plan chunks.json review.json [-o plan.json]` | yes | Gemini writes a replication plan. |

Output is JSON on stdout unless `-o` is given (`export` writes Markdown). Messages go to stderr.

## How the risk score works

The score runs from 100 (high risk) to 0. It starts at 100 and drops by a factor's points when the review marks that factor `present`:

| Factor | Points |
|---|---|
| dataset defined & access described | 15 |
| train/val/test split described | 15 |
| preprocessing/cleaning described | 10 |
| hyperparameters reported | 10 |
| compute/hardware reported | 10 |
| baselines included | 10 |
| ablations included | 10 |
| evaluation protocol clear | 10 |
| limitations discussed | 10 |

The model only judges each factor (`present`, `missing` or `unclear`) and quotes its evidence. The arithmetic is done in code (`scoring.py`), so the same breakdown always gives the same score, scores are comparable across papers, and anyone can see why a paper got its number. A score the model invents cannot be checked; a breakdown with quotes can. `paperpilot check` reports the model's number next to the computed one and flags factor names that do not match the rubric, since a misspelled factor would silently stop counting.

## Why citations are verified

Models paraphrase quotes, merge two sentences into one, round or change numbers, and attribute a passage to the wrong section. A review full of plausible quotes looks like evidence even when it is not. PaperPilot makes that visible: each quote is normalized together with the chunk text (case, curly quotes, dashes, ligatures, line-break hyphenation and whitespace, which PDF extraction mangles) and must appear in the chunk it cites.

Each citation ends up `verified`, `wrong_chunk` (found, but in another chunk), `not_found`, or declared missing ("Not found in provided text", which is allowed and counted separately). Numbers are never loosened: signs, ranges, decimals and digit groups must match, and a quote may not start or stop inside a number or a word. So "batch size 6" does not pass for "batch size 64", "0.42" does not pass for "-0.42", and "stable" does not pass for "unstable". Quotes shorter than 12 characters only count if they match a whole chunk.

`--strict` also refuses a review that marks a factor `present`, or states a claim or result, without a verified quote behind it, so a review cannot lower its own risk score by assertion.

The check proves the words are in the paper. It does not prove they support the claim next to them; that is still the reviewer's judgement.

## Example

[`examples/bearing-fault-rl`](examples/bearing-fault-rl) is a full review of arXiv:2506.19929 (CC BY 4.0), a short paper comparing reinforcement learning with supervised networks for bearing fault diagnosis, produced with the skill loop above: 67/67 quotes verified, 11 citations marked missing, risk score 55/100. The main reproducibility gaps it found are an undefined split over 50%-overlapping windows, no validation set for hyperparameters tuned by trial and error, and no seeds, hardware or code.

## Limitations

- **Scanned PDFs**: if extraction yields fewer than 1,500 characters, PaperPilot stops and asks for OCR first.
- **Long papers in the web app**: chunks are capped at 40; longer papers are batch-summarized before review, and quotes are then checked against those summaries rather than the PDF text. The CLI and the skill always work on the full chunks.
- **Retrieval** for search and Q&A is TF-IDF: it matches words, not meaning.
- **PDF layout**: two-column papers can come out with lines from both columns interleaved, and some PDFs lose the spaces between words. The citation check ignores spacing, but quotes from interleaved text should stay within one line.
- **Gemini**: rate limits, server errors and timeouts get one automatic retry, then an error.
- **Install it on its own**: PaperPilot installs flat top-level modules (`cli`, `llm`, `verify`, ...), which can clash with other packages in a shared environment. Use pipx or a dedicated virtualenv.

## Development

```bash
pip install -e '.[gemini,app]' -r requirements-dev.txt
pytest -q
```

Tests need no API key and no network. See [AGENTS.md](AGENTS.md) for the module map and the rules for changing the schema or the rubric.

## License

MIT. See [LICENSE](LICENSE).
