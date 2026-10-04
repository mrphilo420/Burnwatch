# Project Burnwatch

Project Burnwatch is a Flask web platform for adaptive, conformal screening of
AI-generated text. It implements the three inspection paths and the protocol of
the manuscript (`Conformal_v2.pdf`; in-repo draft `paper/main.tex`, single file
with inlined appendix):

- Construction A: a registered detector/action family with a union-bound error budget.
  Default allocations are resolution-aware: the error budget is concentrated on
  the largest equal-weight active subset that can reject at the current
  calibration size m and level α (route actions first), so Construction A
  fulfills its rejection requirement instead of splitting α into unreachable
  equal shares. Early-decision rates for A are counted against that active-set
  size, not the full registered family.
- Simes rule: every registered action is scored, the p-values are sorted, and
  rejection happens at the first ordered rank crossing the corrected level
  α/H_K (H_K the K-th harmonic number — the reading that holds under
  exchangeability alone; plain α Simes is justified under independence/PRDS
  and is reported for comparison). Exhaustive by construction: no weights, no
  route, no early exit, so no token savings. Rank requirement
  m ≥ ⌈H_K/α⌉ − 1, between B's ⌈1/α⌉ − 1 and equal-Bonferroni A's ⌈K/α⌉ − 1.
- Construction B: a development-fixed route whose complete-path maximum is
  calibrated and may stop early. The result also reports the attestation step
  τ* — the first step where the executed prefix attains the complete maximum
  M_π, recomputed with alert stopping disabled. τ* is a diagnostic only; it
  never changes the decision.
- Validity gate: a deterministic pre-scoring predicate (≥ 32 words). Documents
  below it return the explicit **unprocessable** outcome on all three paths —
  no score, no rank, so neither an alert nor an acquittal, just a referral to
  human review. The gate never consumes α and never fires on a scored document.
- Fixed comparison: `ll@1024` for preregistered cost and power studies.
  The primary efficiency criterion is B versus this fixed policy (≥20% lower
  mean online cost at the same α with a one-sided 95% lower bound for the
  paired power difference above −0.02); A versus B is a secondary paired
  comparison (RQ3).
- Human calibration from IMDB, Binoculars, RAID, DetectRL-X, and RealDet caches.
- Calibration diagnostics reported beside the counts: the max share (which
  route action supplies M_π, and how often), the empty-route rate, and the
  per-action failure rate. Bundles built before these fields existed still
  expose the two rates on load; the max share appears after the next
  recalibration.
- Structured benchmark reports for alert rates, power, cost, early decisions, futility, and subgroup conditions.

## Manuscript

`Conformal_v2.pdf` (repo root) is the authoritative manuscript; `paper/main.tex`
is the in-repo reverse-aligned draft (detector family, α grid, resolution floor
numbers, B-primary efficiency criterion, weight-rule clause, Simes rule,
validity gate). Build with `tectonic paper/main.tex` (pdflatex/latexmk also work when
`sn-jnl.cls` is available; the preamble falls back to a preview article class).

## Run the Web App

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python app.py --corpus realdet --m 4000 --n-dev 50
```

The default `--m 4000` is the smallest round size at which every cell of the
counts table fits: at K = 16 the binding count is the corrected Simes rule at
α = 0.001 (m ≥ 3,380), and at K = 24 (binoculars + Fast-DetectGPT) it is
m ≥ 3,775. Calibration needs `m + n_dev` human documents from the corpus, so
the Binoculars pools are built from both generator splits and topped up with a
documented supplement when short (`ccnews` 4,666 · `cnn` 2,271+ ·
`pubmed` 2,211+ · `binoc-all` 9,148 · `realdet` 62,972 · `raid` 13,148 ·
`detectrl` 15,594). A pool that still cannot serve the requested size fails
with the pool size, the maximum feasible `--m`, and the corpora that fit.

Open `http://127.0.0.1:5010`. Calibration is cached under `calibration_cache/`.
The server writes privacy-preserving, append-only decision records to
`audit_log.jsonl`; configure another path with `--audit-log` or
`BURNWATCH_AUDIT_LOG`. Recent records are available at `/api/audit`.

Audit records contain configuration, p-values, steps, verdicts, timings, and
errors. They deliberately exclude document text and uploaded bytes. The schema
is versioned with `schema_version: 1`. `/api/audit` also reports job-level
p50/p95/max latency for completed jobs.

Uploaded files default to the calibrated first-prefix view. The upload panel
also supports a seeded random contiguous window for exploratory inspection;
the selected seed and word position are shown in the result and audit metadata.
Random-window results are not covered by the current prefix calibration bundle,
so they must not be used as calibrated error-rate evidence until calibration is
rebuilt with the same sampling policy.

## Run the Empirical Study

```bash
python study.py --corpora raid detectrl realdet --out study_report.json
```

The committed `study_report.md` is a **historical** run (generated before the
active-set early-decision fix); regenerate it before treating early-decision
or subgroup fields as current. The designed multi-corpus evaluation study
from the six-month plan is not complete.

The study compares A, B, and the fixed comparator across clean and synthetic
mixed-authorship conditions. Paraphrase cells run only when the source cache
contains explicit paraphrase/attack records; clean-text re-pairing is not
treated as paraphrasing. Reports include:

- mean inspected tokens and savings;
- human alert rate and AI detection power;
- early-decision rates and Construction B futility rates;
- paired B-versus-fixed power and cost criteria;
- observed length bins, explicit `plain_text` output type, and available source/domain labels.

For generator-shift analysis, run the study with `ccnews`, `cnn`, or `pubmed`;
their paired records preserve the Binoculars `falcon7` and `llama2_13` source
labels. Recommended RAID/DetectRL/RealDet caches expose only the metadata they
actually contain.

Unknown dataset metadata is reported as `unknown`, not inferred as a generator
or domain. Synthetic 50/50 mixed-authorship splices are labeled as synthetic.

## Tests

```bash
source env/bin/activate
python -m unittest discover -s tests -v
```

The test suite covers conformal rank rules, all three constructions (including
the Simes region and its α/H_K correction), the validity gate and the
unprocessable outcome, Construction B attestation, calibration diagnostics and
bundle persistence (current and legacy caches), fixed-policy comparison, early
stopping, document extraction, data metadata handling, and the audit API.
`tests/test_detection.py` additionally screens real human and AI documents
through the full path with the cached gpt2 calibration — human documents must
stay under the false-alert level while AI documents are flagged far more often —
and is skipped when no sufficiently large calibration is cached.

## CLI Screening

```bash
python conformal.py --screen document.txt --construction B --alpha 0.05
python conformal.py --screen document.txt --construction Simes --alpha 0.05
python conformal.py --screen document.txt --construction both --alpha 0.05
```

`--construction` accepts `A`, `B`, `Simes`, and `both` (all three paths on one
score table). The calibration counts printed by `resolution_table` now carry the
Simes column and the Table 2 reference sizes: equal-weight A (⌈K/α⌉ − 1), B
(⌈1/α⌉ − 1), Simes (⌈H_K/α⌉ − 1), the zero-error audit sizes (59 / 299 / 2,995
for α = 0.05 / 0.01 / 0.001 at γ = 0.05), the 10/α reference tail, and the
uniformity grid ⌈log(2/η)/(2ε²)⌉ = 18,445 at ε = 0.01, η = 0.05.

An alert flags a document for review; it does not establish AI authorship or
misconduct. A non-alert is insufficient evidence, not proof of human authorship.
