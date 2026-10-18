# Example: bearing fault diagnosis, RL vs. supervised learning

A full PaperPilot review of one short ML paper, produced with the
[paper-review skill](../../skills/paper-review/SKILL.md) loop and no API key.

- Paper: Efe Çakır and Patrick Dumond, *A Comparative Analysis of Reinforcement
  Learning and Conventional Deep Learning Approaches for Bearing Fault Diagnosis*,
  arXiv:2506.19929 (v2): https://arxiv.org/abs/2506.19929
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on
  the arXiv abstract page. The quotes in these files are short excerpts from the
  paper, attributed by chunk.

## Files

| file | what it is |
|---|---|
| `review.json` | the review, following `paperpilot schema` |
| `check.json` | output of `paperpilot check --strict` on it (exit 0) |
| `review.md` | output of `paperpilot export` |

The PDF and `chunks.json` are not included. To reproduce the check:

```bash
curl -L -o paper.pdf https://arxiv.org/pdf/2506.19929v2
paperpilot extract paper.pdf -o chunks.json
paperpilot check review.json --chunks chunks.json --strict
```

Chunk text depends on the PDF extractor, so a different pdfplumber version can
shift chunk boundaries. These files were made with pdfplumber 0.11.10 (6 chunks,
19,640 characters).

## Result

- Citation check: 67/67 quotes verified against the paper text; 11 citations marked
  not found in the paper.
- Risk score: 55/100, computed from the rubric. Dataset, hyperparameters,
  baselines and ablations are present; the split, preprocessing, evaluation
  protocol and limitations are only partly described; hardware is missing.
- Main findings for a replication: the paper does not say how the 80/20 split was
  drawn even though windows overlap by 50% (possible train/test leakage), there is
  no validation set while hyperparameters and the reward matrix were tuned by trial
  and error, and there are no seeds, repeated runs, hardware details or code.
