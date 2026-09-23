# Project Burnwatch

Project Burnwatch is a Flask web platform for adaptive, conformal screening of
AI-generated text. It implements the constructions and protocol described in
the in-repo manuscript (`paper/main.tex`, single file with inlined appendix):

- Construction A: a registered detector/action family with a union-bound error budget.
  Default allocations are resolution-aware: the error budget is concentrated on
  the largest equal-weight active subset that can reject at the current
  calibration size m and level α (route actions first), so Construction A
  fulfills its rejection requirement instead of splitting α into unreachable
  equal shares. Early-decision rates for A are counted against that active-set
  size, not the full registered family.
- Construction B: a development-fixed route whose complete-path maximum is
  calibrated and may stop early.
- Fixed comparison: `ll@1024` for preregistered cost and power studies.
  The primary efficiency criterion is B versus this fixed policy (≥20% lower
  mean online cost at the same α with a one-sided 95% lower bound for the
  paired power difference above −0.02); A versus B is a secondary paired
  comparison (RQ3).
- Human calibration from IMDB, Binoculars, RAID, DetectRL-X, and RealDet caches.
- Structured benchmark reports for alert rates, power, cost, early decisions, futility, and subgroup conditions.

## Manuscript

`paper/main.tex` is reverse-aligned to this implementation (detector family,
α grid, resolution floor numbers, B-primary efficiency criterion, weight-rule
clause). Build with `tectonic paper/main.tex` (pdflatex/latexmk also work when
`sn-jnl.cls` is available; the preamble falls back to a preview article class).

## Run the Web App

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python app.py --corpus realdet --m 260 --n-dev 50
```

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

The test suite covers conformal rank rules, both constructions, fixed-policy
comparison, early stopping, document extraction, data metadata handling, and
the audit API.

## CLI Screening

```bash
python conformal.py --screen document.txt --construction B --alpha 0.05
```

An alert flags a document for review; it does not establish AI authorship or
misconduct. A non-alert is insufficient evidence, not proof of human authorship.
