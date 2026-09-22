#!/usr/bin/env python3
"""Preregistered empirical study runner (proposal: planned empirical study).

Compares adaptive Constructions A/B against a fixed single-action
comparator (ll@1024 at full weight) on RAID, DetectRL-X, and RealDet:
- nominal levels ALPHAS, matched-population calibration per corpus
- variants: clean, paraphrase (RAID attack rows), mixed authorship (spliced)
- subgroup audits (human FPR by domain; AI power by generator model)
- preregistered criterion per (corpus, variant, alpha):
  >= 20% lower mean online cost than the fixed policy at the same alpha,
  with a one-sided 95% lower bound for the paired power difference above -0.02

Writes study_report.json (raw numbers) and study_report.md (tables).
"""

import argparse
import datetime
import json
import os
import sys
import time

import conformal as C
import data

STUDY_CORPORA = ["raid", "detectrl", "realdet"]
ALPHAS = [0.001, 0.01, 0.05]
CONSTRUCTIONS = ("A", "B", "fixed")
N = 150
M_CAL, N_DEV = 1600, 100
MODEL = "gpt2"


def evaluate_criterion(cell):
    """Preregistered joint criterion per alpha from paired detail records."""
    out = {}
    for a in ALPHAS:
        b_flags, f_flags, b_tok, f_tok = [], [], [], []
        for rec in cell["detail"]:
            b_flags.append(1 if rec["alerts"]["B"][a] else 0)
            f_flags.append(1 if rec["alerts"]["fixed"][a] else 0)
            b_tok.append(rec["tokens"]["B"][a])
            f_tok.append(rec["tokens"]["fixed"][a])
        ci = C.paired_difference_ci(b_flags, f_flags)
        mean_b = sum(b_tok) / len(b_tok)
        mean_f = sum(f_tok) / len(f_tok)
        ratio = mean_b / max(1e-9, mean_f)
        out[str(a)] = {
            "power_diff_B_minus_fixed": round(ci["diff"], 4),
            "power_diff_lower_95": round(ci["lower_95"], 4),
            "cost_ratio_B_over_fixed": round(ratio, 4),
            "cost_gain": round(1 - ratio, 4),
            "pass_cost": (1 - ratio) >= 0.20,
            "pass_power": ci["lower_95"] > -0.02,
        }
        out[str(a)]["pass"] = out[str(a)]["pass_cost"] and out[str(a)]["pass_power"]
    return out


def ensure_study_calibration(corpus):
    C.BASE_MODEL = MODEL
    C.CORPUS = corpus
    C.USE_BINOCULARS = False
    C.USE_FASTDETECT = False
    path = C.cache_path(MODEL.replace("/", "_"), M_CAL, N_DEV, corpus, False, False)
    if os.path.exists(path) and C.cache_compatible(path):
        print(f"[study] loading {path}", flush=True)
        return C.load_bundle(path)
    print(f"[study] building {corpus} calibration (m={M_CAL}, dev={N_DEV})…", flush=True)
    bundle = C.build_calibration(M_CAL, N_DEV)
    C.save_bundle(bundle, path)
    return bundle


def variant_docs(corpus, variant, n):
    recs = data.eval_records(corpus, cache_dir=C.CACHE_DIR)
    humans = [r for r in recs if r["kind"] == "human"]
    if variant == "clean":
        ais = [r for r in recs if r["kind"] == "ai"]
    elif variant == "paraphrase":
        ais = [r for r in recs if r["kind"] == "para"]
    elif variant == "mixed":
        clean_ai = [r for r in recs if r["kind"] == "ai"]
        m = min(len(humans), len(clean_ai), n)
        mixed = []
        for hr, ar in zip(humans[:m], clean_ai[:m]):
            mixed.append({"kind": "ai",
                          "text": C.mix_documents(hr["text"], ar["text"], 0.5),
                          "subgroup": "mixed:" + hr["subgroup"],
                          "model": ar["model"], "attack": "mixed"})
        ais = mixed
    else:
        raise ValueError(f"unknown variant {variant!r}")
    if len(humans) < n or len(ais) < n:
        return None
    h, a = humans[:n], ais[:n]
    return ([r["text"] for r in h], [r["text"] for r in a],
            [r["subgroup"] for r in h], [r["subgroup"] for r in a])


def run_cell(bundle, corpus, variant, progress=None):
    got = variant_docs(corpus, variant, N)
    if got is None:
        return None
    humans, ais, meta_h, meta_a = got
    t0 = time.time()
    rows, paired, detail = _evaluate(bundle, humans, ais, meta_h, meta_a, progress)
    return {"rows": rows, "paired": paired, "detail": detail,
            "elapsed_s": round(time.time() - t0, 1)}


def _evaluate(bundle, humans, ais, meta_h, meta_a, progress):
    rows, paired, detail = C.evaluate_cells(
        bundle, humans, ais, CONSTRUCTIONS, ALPHAS,
        progress=progress, meta_h=meta_h, meta_a=meta_a, detail=True)
    return rows, paired, detail


def summarize_subgroups(detail, alpha, minimum_n=10):
    human_recs = [{"subgroup": r["human_subgroup"], "alert": r["alerts"]["B"][alpha]}
                  for r in detail]
    ai_recs = [{"subgroup": r["ai_subgroup"], "alert": r["alerts"]["B"][alpha]}
               for r in detail]
    return {"human_fpr_by_subgroup": C.aggregate_by_subgroup(human_recs, minimum_n),
             "ai_power_by_model": C.aggregate_by_subgroup(ai_recs, minimum_n)}


def _length_bin(words):
    if words < 128:
        return "40-127"
    if words < 256:
        return "128-255"
    if words < 512:
        return "256-511"
    if words < 1024:
        return "512-1023"
    return "1024+"


def summarize_conditions(detail, alpha, minimum_n=10):
    """Report alert rates by observed length and available output/domain label."""
    out = {"length": {}, "output_type": {}, "source_label": {}}
    for side, word_key, subgroup_key in (
            ("human", "human_words", "human_subgroup"),
            ("ai", "ai_words", "ai_subgroup")):
        for rec in detail:
            length = _length_bin(rec[word_key])
            label = rec[subgroup_key] or "unknown"
            for dimension, value in (("length", length),
                                     ("output_type", "plain_text"),
                                     ("source_label", label)):
                group = out[dimension].setdefault(value, {"human": [], "ai": []})
                group[side].append(bool(rec["alerts"]["B"][alpha]))
    for dimension, groups in out.items():
        for value, sides in groups.items():
            for side, flags in sides.items():
                n = len(flags)
                sides[side] = {"n": n, "alerts": sum(flags),
                               "rate": (sum(flags) / n if n >= minimum_n else None),
                               "suppressed": n < minimum_n}
    return out


def write_markdown(report, path):
    L = []
    L.append("# Empirical study report: adaptive within-document screening\n")
    L.append(f"_Generated {report['generated_utc']} · model {report['model']} · "
             f"calibration m={report['m_cal']}, dev={report['n_dev']} · "
             f"detectors {', '.join(report['detectors'])} · n={report['n']} per class._\n")
    L.append("Preregistered criterion per (corpus, variant, α): ≥20% lower mean online "
             "cost than the fixed ll@1024 policy at the same α, with a one-sided 95% "
             "lower bound for the paired power difference above −0.02.\n")
    for cell in report["cells"]:
        L.append(f"## {cell['corpus']} / {cell['variant']}\n")
        L.append("| α | constr | human FPR | AI power | mean tokens | early H/A | futility H/A |")
        L.append("|---|---|---|---|---|---|---|")
        for row in cell["rows"]:
            fh = row.get("futility_human_rate")
            fa = row.get("futility_ai_rate")
            fut = f"{fh:.0%} / {fa:.0%}" if fh is not None else "—"
            early = f"{row['early_decision_human_rate']:.0%} / {row['early_decision_ai_rate']:.0%}"
            L.append(f"| {row['alpha']} | {row['construction']} | "
                     f"{row['human_rate']:.1%} ({row['human_alerts']}/{row['n']}) | "
                     f"{row['ai_rate']:.1%} ({row['ai_alerts']}/{row['n']}) | "
                     f"{row['mean_tokens_inspected']} | {early} | {fut} |")
        L.append("")
        L.append("Paired B−fixed (AI docs):")
        for a, ev in cell["criterion"].items():
            verdict = "PASS" if ev["pass"] else "FAIL"
            L.append(f"- α={a}: power diff {ev['power_diff_B_minus_fixed']:+.3f} "
                     f"(95% lower {ev['power_diff_lower_95']:+.3f}), "
                     f"cost ratio {ev['cost_ratio_B_over_fixed']:.2f} → **{verdict}**")
        L.append("")
        L.append("Human FPR by subgroup (Construction B):")
        for name, g in cell["subgroups"]["human_fpr_by_subgroup"].items():
            L.append(f"- {name}: " + ("suppressed (n < 10)" if g["suppressed"]
                                      else f"{g['rate']:.1%} ({g['alerts']}/{g['n']})"))
        L.append("")
        L.append("AI power by generator (Construction B):")
        for name, g in cell["subgroups"]["ai_power_by_model"].items():
            L.append(f"- {name}: " + ("suppressed (n < 10)" if g["suppressed"]
                                      else f"{g['rate']:.1%} ({g['alerts']}/{g['n']})"))
        L.append("")
        L.append("Condition breakdown (Construction B, α=0.01):")
        for dimension, groups in cell.get("conditions", {}).items():
            L.append(f"- {dimension}:")
            for name, sides in groups.items():
                h, a = sides["human"], sides["ai"]
                hrate = "suppressed" if h["suppressed"] else f"{h['rate']:.1%}"
                arate = "suppressed" if a["suppressed"] else f"{a['rate']:.1%}"
                L.append(f"  {name}: human {hrate} (n={h['n']}), AI {arate} (n={a['n']})")
        L.append("")
    L.append("## Limitations\n")
    L.append("- n=150 per class: binomial noise is wide; treat point estimates with "
             "their implicit uncertainty.")
    L.append("- Fixed comparator is ll@1024 at full weight (a proxy for the "
             "development-selected fixed policy, not a tuned optimum).")
    L.append("- Mixed-authorship splices are synthetic 50/50 word splits, not "
             "naturalistic edited text.")
    L.append("- Paraphrase cells run only when source records explicitly carry a "
             "paraphrase/attack label; text re-pairing is not treated as paraphrasing.")
    L.append("- Unknown source metadata is reported as unknown and is never presented "
             "as a generator or domain comparison.")
    L.append("- Construction A at α=0.001 needs m≥15999 ranks: reported as "
             "infeasible by the resolution floor, not as zero power.")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Preregistered empirical study runner")
    ap.add_argument("--corpora", nargs="+", default=STUDY_CORPORA)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache"))
    ap.add_argument("--out", default="study_report.json")
    args = ap.parse_args()

    C.DEVICE = C.get_device(args.device)
    C.CACHE_DIR = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = args.cache_dir

    report = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "model": MODEL, "m_cal": M_CAL, "n_dev": N_DEV, "n": N,
              "alphas": ALPHAS, "constructions": list(CONSTRUCTIONS),
              "detectors": None, "cells": []}
    for corpus in args.corpora:
        bundle = ensure_study_calibration(corpus)
        if report["detectors"] is None:
            report["detectors"] = C.active_detectors()
        variants = ["clean", "paraphrase", "mixed"] if corpus == "raid" else ["clean", "mixed"]
        if corpus == "realdet":
            variants = ["clean", "mixed"]
        for variant in variants:
            print(f"[study] {corpus}/{variant}", flush=True)
            got = variant_docs(corpus, variant, N)
            if got is None:
                print(f"[study] {corpus}/{variant}: insufficient paraphrase docs, skipped", flush=True)
                continue
            humans, ais, meta_h, meta_a = got

            def progress(done, total, _tag=f"{corpus}/{variant}"):
                print(f"[study] {_tag}: {done}/{total}", flush=True)

            rows, paired, detail = C.evaluate_cells(
                bundle, humans, ais, CONSTRUCTIONS, ALPHAS,
                progress=progress, meta_h=meta_h, meta_a=meta_a, detail=True)
            cell = {"corpus": corpus, "variant": variant, "rows": rows,
                    "paired": paired, "detail": detail}
            cell["criterion"] = evaluate_criterion(cell)
            cell["subgroups"] = summarize_subgroups(detail, 0.01)
            cell["conditions"] = summarize_conditions(detail, 0.01)
            # detail records are large; keep a compact copy in JSON
            cell["detail"] = [
                {"hs": r["human_subgroup"], "as": r["ai_subgroup"],
                 "alerts": r["alerts"], "futility": r["futility"]}
                for r in detail
            ]
            report["cells"].append(cell)
            with open(args.out, "w") as f:
                json.dump(report, f)
            print(f"[study] {corpus}/{variant} done", flush=True)

    write_markdown(report, args.out.replace(".json", ".md"))
    print(f"[study] wrote {args.out} and {args.out.replace('.json', '.md')}", flush=True)


if __name__ == "__main__":
    main()
