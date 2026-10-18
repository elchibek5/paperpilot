---
name: paper-review
description: Review an ML or scientific paper PDF for reproducibility, backing every claim with a verbatim quote that PaperPilot checks against the paper text. Use when someone asks you to review, audit or judge the reproducibility of a paper PDF.
license: MIT
metadata:
  source: https://github.com/elchibek5/paperpilot
---

# Paper review with verified citations

You are the reviewer. PaperPilot gives you the paper text as numbered chunks, the
review schema, the risk rubric, search, and a checker that confirms every quote you
cite is really in the chunk you cite. No API key is needed.

## Install

```bash
pipx install git+https://github.com/elchibek5/paperpilot
# or, inside a clone of the repo:
pip install -e .
```

Check with `paperpilot --version`.

## The loop

1. **Extract.** `paperpilot extract paper.pdf -o chunks.json`
   If it fails with "extracted only N characters" (exit code 2), the PDF has no
   usable text (scanned or image-only). Tell the user to run OCR first (for example
   `ocrmypdf`) and stop.
2. **Read the schema.** `paperpilot schema` prints the JSON Schema of the review.
   Read it. `x-rubric` lists the risk factors and their points; `x-declared-missing`
   is the exact string for evidence the paper does not contain.
3. **Read the paper.** Read every chunk in `chunks.json` (`chunk_id` and `text`).
   For long papers, read the abstract, method, experiments and limitations chunks in
   full and use search for targeted lookups:
   `paperpilot search chunks.json "learning rate batch size epochs" -k 5`
4. **Write `review.json`** following the schema and the rules below.
5. **Check.** `paperpilot check review.json --chunks chunks.json --strict -o check.json`
   Exit 0 means clean. Otherwise read `check.json` and fix everything it lists:
   - `schema_errors`: fix the field at `loc`.
   - `citations.issues` with `not_found`: open the cited chunk and copy the passage
     again, character for character. If the paper really does not say it, change
     the item instead of keeping an unsupported quote.
   - `citations.issues` with `wrong_chunk`: set `chunk_id` to one of `found_in`.
   - `rubric`: use each factor name exactly once, verbatim, with the rubric points.
   - `evidence`: claims, results and the method overview need a verified quote;
     checklist items and risk factors marked `present` need one too (or change the
     status); an item with no citations needs at least the declared-missing one.
   Run the check again. Repeat until it exits 0. Never weaken a quote into a
   shorter fragment just to make it pass; the quote must still support the item.
6. **Export.** `paperpilot export review.json --chunks chunks.json -o review.md`
   and show the user the Markdown, including the citation check line.

## Writing review.json

Fields (all required, no extra keys):

- `paper_summary`: 3-5 plain sentences: problem, method, data, headline result.
- `main_claims`: 3-6 claims the paper makes, one idea each.
- `method_overview`: `overview` text plus citations.
- `results`: one entry per reported number that matters. `metric`, `value` exactly
  as the paper writes it (or `null`), `description`, citations.
- `reproducibility_checklist`: items such as code release, data access, splits,
  preprocessing, hyperparameters, seeds, hardware, evaluation protocol. `status` is
  `present`, `missing` or `unclear`.
- `missing_details`: what blocks a replication and why (`impact`).
- `risk_breakdown`: exactly one entry per rubric factor (table below).
- `risk_reasons`: optional list of other risks, or `null`.
- `suggested_experiments`: 2-4 experiments that would test the claims (no citations).
- `citations`: the 3-8 quotes that matter most for the review overall.
- `risk_score`: copy `computed_risk_score` from the check output. The score is
  always recomputed in code from `risk_breakdown`; your number is only a label.

Every claim, method overview, result, checklist item, missing detail and risk factor
has at least one citation:

```json
{"chunk_id": "3", "quote": "split into 80% training and 20% testing"}
```

- `chunk_id` is the id of the chunk the quote comes from, as a string.
- `quote` is one contiguous span copied verbatim from that chunk's `text`, about
  5-25 words, containing the fact the item relies on. No ellipses, no paraphrase,
  no merged sentences, no corrected typos.
- Start and end on whole words and whole numbers. The checker rejects a quote that
  cuts into a word or a number: "stable" from "unstable", "0.42" from "-0.42",
  "85" from "85.1%".
- Case, curly quotes, line breaks, spacing and word hyphens do not matter to the
  checker, so you may quote `"reproducibility"` where the PDF split
  `"reproduc-\nibility"`, or add spaces the PDF lost. Inside numbers nothing is
  loosened: signs, ranges, decimals and digit groups must match.
- Chunks overlap. If a passage appears in two chunks, either id passes.
- Two-column PDFs can interleave lines from both columns inside a chunk. Quote a
  span from a single line, or a shorter span, rather than stitching lines together.
- If the paper does not contain the evidence, use exactly
  `{"chunk_id": "", "quote": "Not found in provided text"}` and say so in the item
  (for a risk factor: `status` is `missing`). This is counted as declared missing,
  not as a failure.
- Quotes shorter than 12 characters only pass if they are a whole chunk. Quote the
  sentence around a number, not the number alone.

## Rubric

The risk score starts at 100 and drops by a factor's points when its status is
`present`. Use these factor names verbatim:

| factor | points |
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

`present` means the paper states it clearly enough to act on. A dataset named but
with no way to get it is `unclear`. A split described for one experiment but not
another is `unclear`. Absent is `missing`.

## What good looks like

- Short exact quotes that carry the fact: the number, the dataset name, the setting.
- One idea per claim and one number per result, each with its own quote.
- Mark missing rather than guess. If the paper does not give seeds, hardware or a
  data link, say it is missing; do not infer it from what is typical.
- Never invent numbers. Every number in `results` appears in its quote.
- The review judges reproducibility, not novelty: say what a reader would need
  to rerun the work and whether the paper gives it.
