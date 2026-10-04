import unittest

import numpy as np

import conformal
import slop


def _synthetic_bundle(m=99):
    nA = len(conformal.active_detectors()) * len(conformal.BUDGETS)
    cal_scores = np.zeros((m, nA))
    cal_maxima = np.zeros(m)
    g = (np.zeros(nA), np.ones(nA))
    return conformal.CalibrationBundle(
        cal_scores=cal_scores,
        cal_maxima=cal_maxima,
        g_mu=g[0],
        g_sigma=g[1],
        futility_thr=-1.0,
        n_dev=10,
        m=m,
    )


def _all_above_pre(value=10.0):
    return {(det, b): value
            for det in conformal.active_detectors()
            for b in conformal.BUDGETS}


class ScreenAEarlyExitTests(unittest.TestCase):
    def test_stops_at_first_alert(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        res = conformal.screen_a(text, bundle, 0.5, pre=_all_above_pre())
        self.assertTrue(res["alert"])
        self.assertEqual(len(res["steps"]), 1)
        self.assertEqual(res["actions_executed"], 1)

    def test_no_alert_scores_full_family(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        pre = _all_above_pre(value=-10.0)
        res = conformal.screen_a(text, bundle, 0.5, pre=pre)
        self.assertFalse(res["alert"])
        self.assertEqual(len(res["steps"]), bundle.n_actions)


class ScreenAWeightTests(unittest.TestCase):
    def test_concentrated_weight_alerts_where_full_family_equal_cannot(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        pre = _all_above_pre()
        first = conformal._all_actions()[0]
        concentrated = conformal.screen_a(
            text, bundle, 0.02, pre=pre, weights={first: 1.0})
        self.assertTrue(concentrated["alert"])
        # full-family equal 1/16 is infeasible at m=99, alpha=0.02 (needs 799)
        nA = bundle.n_actions
        equal_full = conformal.screen_a(
            text, bundle, 0.02, pre=pre, weights={a: 1.0 / nA for a in conformal._all_actions()})
        self.assertFalse(equal_full["alert"])

    def test_default_weights_fulfill_resolution_requirements(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        pre = _all_above_pre()
        # resolution-aware default concentrates on k=floor(alpha*(m+1)) actions
        res = conformal.screen_a(text, bundle, 0.02, pre=pre)
        self.assertTrue(res["resolution_ok"])
        self.assertLessEqual(res["m_required"], bundle.m)
        self.assertGreater(res["n_active"], 0)
        self.assertTrue(res["alert"])
        # active subset is a proper concentration when full family is infeasible
        nA = bundle.n_actions
        if res["n_active"] < nA:
            self.assertAlmostEqual(res["w"], 1.0 / res["n_active"])
            self.assertGreater(res["w"], 1.0 / nA)

    def test_default_alpha_005_is_resolvable_at_m_260(self):
        # production default: m=260, alpha=0.05 — full equal needs 319
        plan = conformal.a_weight_plan(260, 0.05, 16)
        self.assertTrue(plan["feasible"])
        self.assertTrue(plan["resolution_ok"])
        self.assertEqual(plan["n_active"], 13)
        self.assertAlmostEqual(plan["w"], 1.0 / 13)
        self.assertEqual(plan["m_required"], 259)

    def test_weights_summing_above_one_are_rejected(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        actions = conformal._all_actions()
        with self.assertRaises(ValueError):
            conformal.screen_a(
                text, bundle, 0.05, pre=_all_above_pre(),
                weights={actions[0]: 0.8, actions[1]: 0.8})


class PairedAnalysisTests(unittest.TestCase):
    def test_identical_vectors_give_zero_difference(self):
        out = conformal.paired_difference_ci([1, 0, 1, 0], [1, 0, 1, 0])
        self.assertEqual(out["diff"], 0.0)
        self.assertEqual(out["lower_95"], 0.0)

    def test_full_separation_gives_unit_difference(self):
        out = conformal.paired_difference_ci([1] * 20, [0] * 20)
        self.assertEqual(out["diff"], 1.0)
        self.assertEqual(out["lower_95"], 1.0)

    def test_mixed_case_matches_hand_computation(self):
        a = [1, 1, 1, 0]
        b = [1, 0, 0, 0]
        out = conformal.paired_difference_ci(a, b)
        self.assertAlmostEqual(out["diff"], 0.5)
        # d = [0, 1, 1, 0]: mean 0.5, sd 0.5774, se 0.2887, lower = 0.5 - 1.644854*0.2887
        self.assertAlmostEqual(out["lower_95"], 0.5 - 1.644854 * 0.288675, places=4)

    def test_rejects_empty_or_mismatched_inputs(self):
        with self.assertRaises(ValueError):
            conformal.paired_difference_ci([], [])
        with self.assertRaises(ValueError):
            conformal.paired_difference_ci([1, 0], [1])


if __name__ == "__main__":
    unittest.main()


def _stub_actions(doc):
    kind = doc.split()[0]
    out = {}
    for det in conformal.active_detectors():
        for b in conformal.BUDGETS:
            if kind == "HIGH":
                v = 10.0
            elif kind == "SLOP" and det == "slop":
                v = 10.0
            else:
                v = -10.0
            out[(det, b)] = v
    return out


def _words(*parts):
    return " ".join(parts)


class EvaluateCellsTests(unittest.TestCase):
    def test_paired_diff_equals_marginal_diff(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200) for _ in range(2)]
        ais = [_words("HIGH", *["word"] * 200),
               _words("SLOP", *["word"] * 200)]
        rows, paired, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        a = by_key[("A", 0.5)]
        b = by_key[("B", 0.5)]
        marginal = a["ai_rate"] - b["ai_rate"]
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired[0]["power_diff_A_minus_B"], marginal)
        # doc0 alerts under both, doc1 alerts under A only (B futility-stops)
        self.assertEqual((a["ai_alerts"], b["ai_alerts"]), (2, 1))
        self.assertAlmostEqual(paired[0]["power_diff_A_minus_B"], 0.5)

    def test_futility_rates_count_human_and_ai(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200) for _ in range(2)]
        ais = [_words("HIGH", *["word"] * 200),
               _words("SLOP", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("B",), (0.5,), score_fn=_stub_actions)
        row = rows[0]
        # both LOW humans futility-stop; SLOP ai futility-stops; HIGH ai alerts
        self.assertEqual((row["futility_human"], row["futility_ai"]), (2, 1))
        self.assertAlmostEqual(row["futility_human_rate"], 1.0)
        self.assertAlmostEqual(row["futility_ai_rate"], 0.5)

    def test_mean_actions_reflects_early_exits(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        # per-document means (human + ai docs summed, divided by 2n):
        # LOW human: A scores the full 16-action family; HIGH ai: A stops at 1
        self.assertAlmostEqual(by_key[("A", 0.5)]["mean_actions"], 8.5)
        # B: LOW futility-stops at step 1, HIGH alerts at step 1
        self.assertAlmostEqual(by_key[("B", 0.5)]["mean_actions"], 1.0)

    def test_rows_expose_resolution_fields(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        a = by_key[("A", 0.5)]
        b = by_key[("B", 0.5)]
        # alpha=0.5, m=99: full family (16) is feasible (needs 31)
        self.assertEqual(a["a_n_active"], 16)
        self.assertEqual(a["m_required"], 31)
        self.assertTrue(a["resolution_ok"])
        self.assertEqual(b["m_required"], 1)
        self.assertTrue(b["resolution_ok"])
        for r in rows:
            self.assertEqual(r["resolution_ok"], bundle.m >= r["m_required"])

    def test_rows_flag_unresolvable_when_m_too_small(self):
        # alpha=0.001 needs m>=999 even for a single full-weight action
        bundle = _synthetic_bundle(m=9)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.001,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        a = by_key[("A", 0.001)]
        b = by_key[("B", 0.001)]
        self.assertEqual(a["a_n_active"], 0)
        self.assertFalse(a["resolution_ok"])
        self.assertFalse(b["resolution_ok"])

    def test_concentrated_plan_makes_small_m_resolvable(self):
        # m=19 cannot reject with equal 1/16 at alpha=0.05 (needs 319),
        # but the resolution-aware active subset (k=1, full weight) can.
        bundle = _synthetic_bundle(m=19)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A",), (0.05,), score_fn=_stub_actions)
        a = rows[0]
        self.assertEqual(a["a_n_active"], 1)
        self.assertTrue(a["resolution_ok"])
        self.assertEqual(a["m_required"], 19)
        self.assertTrue(bundle.m >= a["m_required"])


class BenchmarkMeansTests(unittest.TestCase):
    def test_means_are_per_document_not_per_pair(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        # LOW human (201 words): A runs 16 steps, inspects 201 tokens;
        # HIGH ai: A stops at 1 step, inspects 201 tokens
        self.assertAlmostEqual(by_key[("A", 0.5)]["mean_tokens_inspected"], 201.0)
        self.assertAlmostEqual(by_key[("A", 0.5)]["mean_actions"], 8.5)
        self.assertAlmostEqual(by_key[("B", 0.5)]["mean_actions"], 1.0)

    def test_savings_never_exceeds_100_percent(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 1200) for _ in range(3)]
        ais = [_words("HIGH", *["word"] * 1200) for _ in range(3)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        for row in rows:
            self.assertLessEqual(row["mean_savings_pct"], 100.0)
            self.assertLessEqual(row["mean_tokens_inspected"], 1024)

    def test_early_decision_rates_are_reported(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _ = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        self.assertEqual(by_key[("A", 0.5)]["early_decision_ai_rate"], 1.0)
        self.assertEqual(by_key[("B", 0.5)]["early_decision_human_rate"], 1.0)

    def test_early_decision_uses_active_set_for_A(self):
        # m=260, alpha=0.05 -> n_active=13 (not 16). A non-alerting human
        # runs all 13 active actions and must NOT count as early; an AI that
        # alerts at step 1 must count as early.
        bundle = _synthetic_bundle(m=260)
        bundle.corpus = "test-corpus"
        plan = conformal.a_weight_plan(260, 0.05, bundle.n_actions)
        self.assertEqual(plan["n_active"], 13)
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _ = conformal.evaluate_cells(
            bundle, humans, ais, ("A",), (0.05,), score_fn=_stub_actions)
        a = rows[0]
        self.assertEqual(a["a_n_active"], 13)
        self.assertEqual(a["early_decision_human_rate"], 0.0)
        self.assertEqual(a["early_decision_ai_rate"], 1.0)
        # means use the active set: (13 + 1) / 2 = 7
        self.assertAlmostEqual(a["mean_actions"], 7.0)

    def test_early_decision_not_set_when_structurally_inactive(self):
        # m=9, alpha=0.001 -> k=0: A never runs; 0 < 0 is false, not early.
        bundle = _synthetic_bundle(m=9)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _ = conformal.evaluate_cells(
            bundle, humans, ais, ("A",), (0.001,), score_fn=_stub_actions)
        a = rows[0]
        self.assertEqual(a["a_n_active"], 0)
        self.assertEqual(a["early_decision_human_rate"], 0.0)
        self.assertEqual(a["early_decision_ai_rate"], 0.0)
        self.assertAlmostEqual(a["mean_actions"], 0.0)


class ApiValidationTests(unittest.TestCase):
    def test_random_upload_sampling_is_seed_reproducible(self):
        import app as web_app
        text = " ".join(f"w{i}" for i in range(100))
        first, meta = web_app.sample_upload_text(text, "random_window", 7, 20)
        second, meta2 = web_app.sample_upload_text(text, "random_window", 7, 20)
        self.assertEqual(first, second)
        self.assertEqual(meta, meta2)
        self.assertFalse(meta["calibrated"])

    def test_audit_endpoint_returns_versioned_schema(self):
        import app as web_app
        client = web_app.app.test_client()
        resp = client.get("/api/audit?limit=1")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["schema_version"], 1)
        self.assertIn("records", resp.get_json())
        self.assertIn("timing", resp.get_json())

    def test_detect_rejects_non_numeric_alpha(self):
        import app as web_app
        client = web_app.app.test_client()
        text = " ".join(["word"] * 25)
        resp = client.post("/api/detect", json={"text": text, "alpha": "abc"})
        self.assertEqual(resp.status_code, 400)

    def test_detect_rejects_out_of_range_alpha(self):
        import app as web_app
        client = web_app.app.test_client()
        text = " ".join(["word"] * 25)
        resp = client.post("/api/detect", json={"text": text, "alpha": 1.5})
        self.assertEqual(resp.status_code, 400)

    def test_benchmark_rejects_non_numeric_n(self):
        import app as web_app
        client = web_app.app.test_client()
        resp = client.post("/api/benchmark", json={"corpus": "realdet", "n": "abc"})
        self.assertEqual(resp.status_code, 400)

    def test_upload_rejects_non_numeric_alpha(self):
        import app as web_app
        import io
        client = web_app.app.test_client()
        resp = client.post("/api/upload",
                           data={"alpha": "abc",
                                 "file": (io.BytesIO(b"word " * 30), "doc.txt")},
                           content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 400)

    def test_detect_blocks_during_download_phase(self):
        import app as web_app
        web_app.calibration_state["status"] = "downloading"
        web_app.calibration_state["detail"] = "test"
        try:
            client = web_app.app.test_client()
            text = " ".join(["word"] * 25)
            resp = client.post("/api/detect", json={"text": text})
            self.assertEqual(resp.status_code, 409)
        finally:
            web_app.calibration_state["status"] = "missing"
            web_app.calibration_state["detail"] = ""


class SlopMarkAlignmentTests(unittest.TestCase):
    def test_marks_align_with_irregular_whitespace(self):
        marks = slop.per_word_marks("hello  world\n\nfoo bar")
        # only "world" carries a slop marker under this lexicon
        import re
        expected = [0] * 4
        for name, rx in slop._COMPILED:
            for m in rx.finditer("hello  world\n\nfoo bar"):
                s, e = m.start(), m.end()
                spans = [(0, 5), (7, 12), (14, 17), (18, 21)]
                for j, (ws, we) in enumerate(spans):
                    if ws < e and we > s:
                        expected[j] += 1
        self.assertEqual(marks, expected)

    def test_no_overmarking_of_following_word(self):
        marks = slop.per_word_marks("hello world foo")
        import re
        expected = [0] * 3
        for name, rx in slop._COMPILED:
            for m in rx.finditer("hello world foo"):
                s, e = m.start(), m.end()
                spans = [(0, 5), (6, 11), (12, 15)]
                for j, (ws, we) in enumerate(spans):
                    if ws < e and we > s:
                        expected[j] += 1
        self.assertEqual(marks, expected)


class RecommendedCorpusTests(unittest.TestCase):
    def _write_cache(self, root, corpus, kind, texts):
        import json
        import os
        import data
        target = os.path.join(data.cache_dir_for(root), f"{corpus}-{kind}.jsonl")
        with open(target, "w") as f:
            for text in texts:
                f.write(json.dumps({"text": text}) + "\n")
        return root

    def test_recommended_human_docs_enforce_length_floor(self):
        import tempfile
        import data
        root = tempfile.mkdtemp()
        long_doc = " ".join(["word"] * 120)
        self._write_cache(root, "raid", "human", ["too short", long_doc])
        docs = data.get_human_docs("raid", cache_dir=root)
        self.assertEqual(docs, [long_doc])

    def test_recommended_ai_docs_enforce_length_floor(self):
        import tempfile
        import data
        root = tempfile.mkdtemp()
        long_doc = " ".join(["word"] * 50)
        self._write_cache(root, "raid", "ai", ["tiny", long_doc])
        docs = data.get_ai_docs("raid", cache_dir=root)
        self.assertEqual(docs, [long_doc])


class EvaluateCellsFixedTests(unittest.TestCase):
    def test_fixed_comparator_in_evaluate_cells(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, paired, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("fixed",), (0.5,), score_fn=_stub_actions)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["construction"], "fixed")
        self.assertEqual(rows[0]["ai_alerts"], 1)
        self.assertEqual(rows[0]["human_alerts"], 0)
        self.assertAlmostEqual(rows[0]["mean_actions"], 1.0)
        self.assertEqual(paired, [])

    def test_detail_records_carry_subgroups(self):
        bundle = _synthetic_bundle(m=99)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, paired, detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions,
            meta_h=["news"], meta_a=["llama"], detail=True)
        self.assertEqual(len(detail), 1)
        rec = detail[0]
        self.assertEqual(rec["human_subgroup"], "news")
        self.assertEqual(rec["ai_subgroup"], "llama")
        self.assertIn(0.5, rec["alerts"]["A"])
        self.assertIn(0.5, rec["alerts"]["B"])


def _route_pre(values):
    """Action scores whose route entries are given in DEFAULT_ROUTE order."""
    pre = {a: -10.0 for a in conformal._all_actions()}
    for (det, b), v in zip(conformal.DEFAULT_ROUTE, values):
        pre[(det, b)] = v
    return pre


class SimesScreenTests(unittest.TestCase):
    def test_corrected_region_rejects_at_ordered_rank(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        res = conformal.screen_simes(text, bundle, 0.05, pre=_all_above_pre())
        self.assertEqual(res["construction"], "Simes")
        self.assertTrue(res["alert"])
        # every rank is 1/(m+1) = 0.01 here, so the crossing is the first k
        # with level*k/K >= 0.01 (level = alpha/H_16)
        self.assertEqual(res["reject_rank"], 11)
        self.assertEqual(len(res["steps"]), bundle.n_actions)
        ps = [s["p"] for s in res["steps"]]
        self.assertEqual(ps, sorted(ps))
        thr = [s["threshold"] for s in res["steps"]]
        self.assertEqual(thr, sorted(thr))
        self.assertTrue(all(s["threshold"] == res["level"] * s["t"] / bundle.n_actions
                            for s in res["steps"]))

    def test_exhaustive_execution_never_early_exits(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        res = conformal.screen_simes(text, bundle, 0.05, pre=_all_above_pre())
        self.assertEqual(res["actions_executed"], bundle.n_actions)
        self.assertEqual(res["n_evaluated"], bundle.n_actions)
        self.assertEqual(res["tokens_saved_pct"], 0)

    def test_no_alert_when_every_rank_is_one(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        res = conformal.screen_simes(text, bundle, 0.05,
                                     pre=_all_above_pre(value=-10.0))
        self.assertFalse(res["alert"])
        self.assertIsNone(res["reject_rank"])
        self.assertEqual(len(res["steps"]), bundle.n_actions)

    def test_uncorrected_region_fires_earlier_but_is_not_the_guarantee(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        corrected = conformal.screen_simes(text, bundle, 0.05, pre=_all_above_pre())
        uncorrected = conformal.screen_simes(text, bundle, 0.05,
                                             pre=_all_above_pre(), corrected=False)
        self.assertTrue(uncorrected["alert"])
        self.assertEqual(uncorrected["reject_rank"], 4)
        self.assertLess(uncorrected["reject_rank"], corrected["reject_rank"])
        # under exchangeability alone only alpha/H_K carries the guarantee
        self.assertAlmostEqual(corrected["level"], 0.05 / conformal.harmonic(16))
        self.assertAlmostEqual(uncorrected["level"], 0.05)
        self.assertEqual(uncorrected["m_required"], conformal.resolution_b(0, 0.05))

    def test_dispatch_supports_simes(self):
        bundle = _synthetic_bundle(m=99)
        res = conformal._screen_construction(
            "Simes", "word " * 200, bundle, 0.05, _all_above_pre())
        self.assertEqual(res["construction"], "Simes")
        self.assertTrue(res["alert"])


class AttestationTests(unittest.TestCase):
    def test_attestation_is_first_step_attaining_complete_max(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        # route g-values peak at step 3; alpha too small to alert
        res = conformal.screen_b(text, bundle, 0.001,
                                 pre=_route_pre([0.5, 0.6, 3.0, 1.0, 2.0]))
        self.assertFalse(res["alert"])
        self.assertEqual(len(res["steps"]), len(conformal.DEFAULT_ROUTE))
        self.assertEqual(res["attested_at"], 3)
        self.assertTrue(res["attested_executed"])
        self.assertAlmostEqual(res["complete_max"], 3.0)
        self.assertEqual([s["t"] for s in res["steps"] if s["attested"]], [3])

    def test_attestation_may_lie_beyond_an_early_alert(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        # alerts on the first step, but the complete maximum is attained at 4
        res = conformal.screen_b(text, bundle, 0.01,
                                 pre=_route_pre([0.5, 0.6, 0.7, 4.0, 2.0]))
        self.assertTrue(res["alert"])
        self.assertEqual(len(res["steps"]), 1)
        self.assertEqual(res["attested_at"], 4)
        self.assertFalse(res["attested_executed"])
        self.assertAlmostEqual(res["complete_max"], 4.0)

    def test_empty_route_has_no_attestation(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        res = conformal.screen_b(text, bundle, 0.05,
                                 pre=_route_pre([-np.inf] * 5))
        self.assertFalse(res["alert"])
        self.assertIsNone(res["attested_at"])
        self.assertIsNone(res["complete_max"])


class CalibrationDiagnosticTests(unittest.TestCase):
    def _replay(self, rows, futility_thr=-1.0):
        nA = len(conformal._all_actions())
        idx = {a: i for i, a in enumerate(conformal._all_actions())}
        scores = np.full((len(rows), nA), -np.inf)
        for i, row in enumerate(rows):
            for (det, b), v in zip(conformal.DEFAULT_ROUTE, row):
                scores[i, idx[(det, b)]] = v
        g_mu = np.zeros(nA)
        g_sigma = np.ones(nA)
        return conformal._route_replay(scores, conformal.DEFAULT_ROUTE, g_mu,
                                       g_sigma, futility_thr, idx)

    def test_replay_reports_maxima_and_attestation(self):
        maxima, g_values, n_exec, attested = self._replay([
            [-np.inf] * 5,                       # empty route
            [1.0, 2.0, 3.0, 0.5, 0.5],           # attains at step 3
            [1.0, 1.0, 1.0, 1.0, 1.0],           # tie from step 1
            [5.0, 0.1, 0.1, 0.1, 0.1],           # attains at step 1
        ])
        self.assertTrue(np.isneginf(maxima[0]))
        self.assertEqual(attested[0], -1)
        self.assertEqual(n_exec[0], 1)          # empty route futility-stops at step 1
        self.assertAlmostEqual(maxima[1], 3.0)
        self.assertEqual(attested[1], 2)
        self.assertTrue(np.all(n_exec[1:] == 5))

    def test_max_share_sums_to_route_coverage(self):
        maxima, g_values, n_exec, attested = self._replay([
            [-np.inf] * 5,
            [1.0, 2.0, 3.0, 0.5, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [5.0, 0.1, 0.1, 0.1, 0.1],
        ])
        share = conformal.route_max_share(g_values, maxima, n_exec, attested)
        self.assertEqual(len(share), len(conformal.DEFAULT_ROUTE))
        # 4 routes: one empty, one single-step attainment, one five-way tie
        self.assertAlmostEqual(share[2], 0.50)   # step 3 attains once + the tie
        self.assertAlmostEqual(share[0], 0.50)   # step 1 attains once + the tie
        self.assertAlmostEqual(share[1], 0.25)   # tie only
        self.assertAlmostEqual(share[3], 0.25)
        self.assertAlmostEqual(share[4], 0.25)
        self.assertAlmostEqual(float(share.sum()), 1.75)

    def test_bundle_diagnostics_expose_empty_routes_and_bottleneck(self):
        nA = len(conformal._all_actions())
        max_share = np.zeros(len(conformal.DEFAULT_ROUTE))
        max_share[2] = 0.8
        max_share[4] = 0.2
        bundle = conformal.CalibrationBundle(
            cal_scores=np.zeros((99, nA)), cal_maxima=np.zeros(99),
            g_mu=np.zeros(nA), g_sigma=np.ones(nA), futility_thr=-1.0,
            n_dev=10, m=99, max_share=max_share, empty_route_rate=0.05,
            failure_rate=0.01)
        diag = bundle.diagnostics()
        self.assertEqual(diag["max_share_top"]["action"], "rank@512")
        self.assertAlmostEqual(diag["max_share_top"]["share"], 0.8)
        self.assertAlmostEqual(diag["empty_route_rate"], 0.05)
        self.assertAlmostEqual(diag["failure_rate"], 0.01)
        self.assertEqual(len(diag["route"]), len(conformal.DEFAULT_ROUTE))

    def test_legacy_bundle_reports_no_diagnostics(self):
        self.assertIsNone(_synthetic_bundle(m=99).diagnostics())


class ValidityGateTests(unittest.TestCase):
    def test_gate_threshold_is_min_tokens_words(self):
        ok, reason = conformal.validity_gate("word " * (conformal.MIN_TOKENS - 1))
        self.assertFalse(ok)
        self.assertIn("validity gate", reason)
        ok, reason = conformal.validity_gate("word " * conformal.MIN_TOKENS)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_short_document_is_unprocessable_on_every_path(self):
        bundle = _synthetic_bundle(m=99)
        text = "word " * 25
        for res in (conformal.screen_a(text, bundle, 0.05),
                    conformal.screen_b(text, bundle, 0.05),
                    conformal.screen_simes(text, bundle, 0.05)):
            self.assertTrue(res["unprocessable"])
            self.assertFalse(res["alert"])
            self.assertEqual(res["steps"], [])
            self.assertEqual(res["actions_executed"], 0)
            self.assertEqual(res["tokens_inspected"], 0)
            self.assertIn("validity gate", res["gate_reason"])

    def test_verdict_refers_unprocessable_documents_to_review(self):
        import app as web_app
        bundle = _synthetic_bundle(m=99)
        res = conformal.screen_a("word " * 25, bundle, 0.05)
        verdict = web_app.verdict_for(res)
        self.assertEqual(verdict["level"], "referral")
        self.assertIn("Unprocessable", verdict["label"])

    def test_verdict_unchanged_for_scoring_documents(self):
        import app as web_app
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        cleared = web_app.verdict_for(
            conformal.screen_a(text, bundle, 0.05, pre=_all_above_pre(value=-10.0)))
        self.assertEqual(cleared["level"], "clear")
        alerted = web_app.verdict_for(
            conformal.screen_a(text, bundle, 0.05, pre=_all_above_pre()))
        self.assertEqual(alerted["level"], "alert")


class BundlePersistenceTests(unittest.TestCase):
    def _bundle_with_diagnostics(self):
        nA = len(conformal._all_actions())
        share = np.zeros(len(conformal.DEFAULT_ROUTE))
        share[1] = 0.6
        share[3] = 0.4
        return conformal.CalibrationBundle(
            cal_scores=np.zeros((99, nA)), cal_maxima=np.zeros(99),
            g_mu=np.zeros(nA), g_sigma=np.ones(nA), futility_thr=-1.0,
            n_dev=10, m=99, corpus="imdb",
            max_share=share, empty_route_rate=0.02, failure_rate=0.03)

    def test_diagnostics_survive_a_save_load_roundtrip(self):
        import os
        import tempfile
        bundle = self._bundle_with_diagnostics()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "cal.npz")
            bundle.save(path)
            loaded = conformal.CalibrationBundle.load(path)
        diag = loaded.diagnostics()
        self.assertIsNotNone(diag)
        self.assertAlmostEqual(diag["empty_route_rate"], 0.02)
        self.assertAlmostEqual(diag["failure_rate"], 0.03)
        self.assertEqual(diag["max_share_top"]["action"], "ll@256")
        self.assertAlmostEqual(diag["max_share_top"]["share"], 0.6)
        self.assertAlmostEqual(diag["max_share_sum"], 1.0)
        self.assertTrue(loaded.selfcheck())

    def test_legacy_cache_without_diagnostics_still_loads(self):
        import os
        import tempfile
        bundle = self._bundle_with_diagnostics()
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "legacy.npz")
            np.savez_compressed(
                path,
                cal_scores=bundle.cal_scores, cal_maxima=bundle.cal_maxima,
                g_mu=bundle.g_mu, g_sigma=bundle.g_sigma,
                futility_thr=np.array([bundle.futility_thr]),
                n_dev=np.array([bundle.n_dev]), m=np.array([bundle.m]),
            )
            loaded = conformal.CalibrationBundle.load(path)
        # the max share is lost with the legacy cache, but the two rates are
        # recomputed from the stored maxima and scores
        self.assertIsNone(loaded.max_share)
        diag = loaded.diagnostics()
        self.assertIsNotNone(diag)
        self.assertIsNone(diag["max_share"])
        self.assertIsNone(diag["max_share_top"])
        self.assertAlmostEqual(diag["empty_route_rate"], 0.0)
        self.assertAlmostEqual(diag["failure_rate"], 0.0)
        self.assertTrue(loaded.selfcheck())
