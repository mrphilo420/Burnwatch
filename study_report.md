# Historical empirical study report: adaptive within-document screening

_Generated 2026-09-22T14:04:32.894005+00:00 · model gpt2 · calibration m=1600, dev=100 · detectors ll, rank, logrank, slop · n=150 per class._

Preregistered criterion per (corpus, variant, α): ≥20% lower mean online cost than the fixed ll@1024 policy at the same α, with a one-sided 95% lower bound for the paired power difference above −0.02. (Primary criterion is B vs fixed; A vs B is a secondary paired comparison in the manuscript, not part of pass/fail.)

## Report Status

This report is a **historical artifact** from before two implementation fixes:

1. Structured early-decision metrics and metadata corrections (subgroup labels;
   RAID `abstracts` / RealDet `all` were placeholders, not source metadata —
   the current pipeline reports missing metadata as `unknown`).
2. Construction A early-decision denominators now use the **active-set size**
   (`n_active`), not the full registered family of 16 actions. Any early-decision
   rates implied for A in this file may over-count non-alert runs as "early"
   relative to the current code. Regenerate before using those SOP measures.

The reverse-aligned proposal manuscript lives in `paper/main.tex` (single file;
the supplementary proofs appendix is inlined). The designed multi-corpus evaluation study from
the six-month plan is not complete; treat alert, power, token-cost, and futility
values below as historical measurements only.

The current report schema additionally includes:

| Field | Interpretation |
|---|---|
| `early_decision_human_rate` / `early_decision_ai_rate` | Fraction stopping before the construction's full route (A: active-set size; B: full route length) |
| `length` condition breakdown | Alert rates in 40–127, 128–255, 256–511, 512–1023, and 1024+ word bins |
| `output_type` | Explicit source type; current text study records report `plain_text` |
| `source_label` | Dataset domain or generator label, only when actually supplied |
| `timing` | Job-level p50/p95/max latency is available from `/api/audit` |

Regenerate this report with the current implementation:

```bash
python study.py --corpora raid detectrl realdet --out study_report.json
```

For genuine generator-shift labels from the Binoculars paired sources, run:

```bash
python study.py --corpora ccnews cnn pubmed --out study_binoculars.json
```

## raid / clean

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 224 | — |
| 0.01 | A | 0.0% (0/150) | 6.7% (10/150) | 224 | — |
| 0.05 | A | 0.0% (0/150) | 56.0% (84/150) | 224 | — |
| 0.001 | B | 0.0% (0/150) | 6.0% (9/150) | 205 | 73% / 1% |
| 0.01 | B | 0.0% (0/150) | 75.3% (113/150) | 181 | 73% / 1% |
| 0.05 | B | 0.0% (0/150) | 92.7% (139/150) | 157 | 73% / 1% |
| 0.001 | fixed | 0.0% (0/150) | 6.7% (10/150) | 224 | — |
| 0.01 | fixed | 0.0% (0/150) | 76.0% (114/150) | 224 | — |
| 0.05 | fixed | 0.7% (1/150) | 92.7% (139/150) | 224 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff -0.007 (95% lower -0.018), cost ratio 0.99 → **FAIL**
- α=0.01: power diff -0.007 (95% lower -0.026), cost ratio 0.82 → **FAIL**
- α=0.05: power diff +0.000 (95% lower -0.016), cost ratio 0.64 → **PASS**

Human FPR by subgroup (Construction B; legacy label invalid):
- unknown: 75.3% (113/150)

AI power by generator (Construction B; legacy label invalid):
- unknown: 75.3% (113/150)

## raid / mixed

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 201 | — |
| 0.01 | A | 0.0% (0/150) | 0.0% (0/150) | 201 | — |
| 0.05 | A | 0.0% (0/150) | 2.7% (4/150) | 201 | — |
| 0.001 | B | 0.0% (0/150) | 0.0% (0/150) | 146 | 73% / 75% |
| 0.01 | B | 0.0% (0/150) | 0.0% (0/150) | 146 | 73% / 75% |
| 0.05 | B | 0.0% (0/150) | 0.0% (0/150) | 146 | 73% / 75% |
| 0.001 | fixed | 0.0% (0/150) | 0.0% (0/150) | 201 | — |
| 0.01 | fixed | 0.0% (0/150) | 0.0% (0/150) | 201 | — |
| 0.05 | fixed | 0.7% (1/150) | 0.7% (1/150) | 201 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff +0.000 (95% lower +0.000), cost ratio 0.68 → **PASS**
- α=0.01: power diff +0.000 (95% lower +0.000), cost ratio 0.68 → **PASS**
- α=0.05: power diff -0.007 (95% lower -0.018), cost ratio 0.68 → **PASS**

Human FPR by subgroup (Construction B; legacy label invalid):
- unknown: 0.0% (0/150)

AI power by generator (Construction B; legacy label invalid):
- mixed:unknown: 0.0% (0/150)

## detectrl / clean

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 576 | — |
| 0.01 | A | 0.0% (0/150) | 0.0% (0/150) | 576 | — |
| 0.05 | A | 0.0% (0/150) | 4.7% (7/150) | 576 | — |
| 0.001 | B | 0.0% (0/150) | 0.0% (0/150) | 335 | 57% / 50% |
| 0.01 | B | 0.0% (0/150) | 0.0% (0/150) | 335 | 57% / 50% |
| 0.05 | B | 1.3% (2/150) | 2.0% (3/150) | 329 | 57% / 50% |
| 0.001 | fixed | 0.0% (0/150) | 0.0% (0/150) | 576 | — |
| 0.01 | fixed | 0.0% (0/150) | 0.0% (0/150) | 576 | — |
| 0.05 | fixed | 2.0% (3/150) | 1.3% (2/150) | 576 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff +0.000 (95% lower +0.000), cost ratio 0.61 → **PASS**
- α=0.01: power diff +0.000 (95% lower +0.000), cost ratio 0.61 → **PASS**
- α=0.05: power diff +0.007 (95% lower -0.018), cost ratio 0.60 → **PASS**

Human FPR by subgroup (Construction B; metadata unavailable):
- unknown: 0.0% (0/150)

AI power by generator (Construction B; metadata unavailable):
- unknown: 0.0% (0/150)

## detectrl / mixed

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 571 | — |
| 0.01 | A | 0.0% (0/150) | 0.0% (0/150) | 571 | — |
| 0.05 | A | 0.0% (0/150) | 1.3% (2/150) | 571 | — |
| 0.001 | B | 0.0% (0/150) | 0.0% (0/150) | 315 | 57% / 57% |
| 0.01 | B | 0.0% (0/150) | 0.0% (0/150) | 315 | 57% / 57% |
| 0.05 | B | 1.3% (2/150) | 1.3% (2/150) | 311 | 57% / 57% |
| 0.001 | fixed | 0.0% (0/150) | 0.0% (0/150) | 571 | — |
| 0.01 | fixed | 0.0% (0/150) | 0.0% (0/150) | 571 | — |
| 0.05 | fixed | 2.0% (3/150) | 0.0% (0/150) | 571 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff +0.000 (95% lower +0.000), cost ratio 0.55 → **PASS**
- α=0.01: power diff +0.000 (95% lower +0.000), cost ratio 0.55 → **PASS**
- α=0.05: power diff +0.013 (95% lower -0.002), cost ratio 0.54 → **PASS**

Human FPR by subgroup (Construction B; metadata unavailable):
- unknown: 0.0% (0/150)

AI power by generator (Construction B; metadata unavailable):
- mixed:unknown: 0.0% (0/150)

## realdet / clean

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 281 | — |
| 0.01 | A | 0.0% (0/150) | 16.0% (24/150) | 281 | — |
| 0.05 | A | 0.0% (0/150) | 37.3% (56/150) | 281 | — |
| 0.001 | B | 0.0% (0/150) | 5.3% (8/150) | 231 | 49% / 9% |
| 0.01 | B | 0.0% (0/150) | 14.7% (22/150) | 224 | 49% / 9% |
| 0.05 | B | 4.7% (7/150) | 40.0% (60/150) | 205 | 49% / 9% |
| 0.001 | fixed | 0.0% (0/150) | 9.3% (14/150) | 281 | — |
| 0.01 | fixed | 0.7% (1/150) | 17.3% (26/150) | 281 | — |
| 0.05 | fixed | 3.3% (5/150) | 38.0% (57/150) | 281 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff -0.040 (95% lower -0.066), cost ratio 0.97 → **FAIL**
- α=0.01: power diff -0.027 (95% lower -0.053), cost ratio 0.93 → **FAIL**
- α=0.05: power diff +0.020 (95% lower -0.020), cost ratio 0.79 → **PASS**

Human FPR by subgroup (Construction B; legacy label invalid):
- unknown: 14.7% (22/150)

AI power by generator (Construction B; legacy label invalid):
- unknown: 14.7% (22/150)

## realdet / mixed

| α | constr | human FPR | AI power | mean tokens | futility H/A |
|---|---|---|---|---|---|
| 0.001 | A | 0.0% (0/150) | 0.0% (0/150) | 293 | — |
| 0.01 | A | 0.0% (0/150) | 0.7% (1/150) | 293 | — |
| 0.05 | A | 0.0% (0/150) | 8.0% (12/150) | 293 | — |
| 0.001 | B | 0.0% (0/150) | 0.0% (0/150) | 204 | 49% / 53% |
| 0.01 | B | 0.0% (0/150) | 0.0% (0/150) | 204 | 49% / 53% |
| 0.05 | B | 4.7% (7/150) | 2.7% (4/150) | 200 | 49% / 53% |
| 0.001 | fixed | 0.0% (0/150) | 0.0% (0/150) | 293 | — |
| 0.01 | fixed | 0.7% (1/150) | 0.0% (0/150) | 293 | — |
| 0.05 | fixed | 3.3% (5/150) | 0.7% (1/150) | 293 | — |

Paired B−fixed (AI docs):
- α=0.001: power diff +0.000 (95% lower +0.000), cost ratio 0.71 → **PASS**
- α=0.01: power diff +0.000 (95% lower +0.000), cost ratio 0.71 → **PASS**
- α=0.05: power diff +0.020 (95% lower +0.001), cost ratio 0.69 → **PASS**

Human FPR by subgroup (Construction B; legacy label invalid):
- unknown: 0.0% (0/150)

AI power by generator (Construction B; legacy label invalid):
- mixed:unknown: 0.0% (0/150)

## Limitations

- n=150 per class: binomial noise is wide; treat point estimates with their implicit uncertainty.
- Fixed comparator is ll@1024 at full weight (a proxy for the development-selected fixed policy, not a tuned optimum).
- Mixed-authorship splices are synthetic 50/50 word splits, not naturalistic edited text.
- This historical artifact has no early-decision-rate or length/output-type breakdown; regenerate it before using those SOP measures. After regeneration, Construction A early-decision rates use the active-set denominator (`n_active`), not 16 full registered actions.
- The historical RAID and RealDet subgroup labels were placeholders and are not valid domain/generator analyses.
- Paraphrase cells are not valid unless source records carry an explicit paraphrase/attack label; clean-text re-pairing is not paraphrasing.
- Construction A at α=0.001 needs m≥15999 ranks: reported as infeasible by the resolution floor, not as zero power.
