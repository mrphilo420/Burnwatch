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
    def test_concentrated_weight_alerts_where_equal_weights_cannot(self):
        bundle = _synthetic_bundle(m=99)
        text = " ".join(["word"] * 200)
        pre = _all_above_pre()
        first = conformal._all_actions()[0]
        concentrated = conformal.screen_a(
            text, bundle, 0.02, pre=pre, weights={first: 1.0})
        self.assertTrue(concentrated["alert"])
        equal = conformal.screen_a(text, bundle, 0.02, pre=pre)
        self.assertFalse(equal["alert"])

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
        self.assertEqual(a["m_required"], 31)
        self.assertTrue(a["resolution_ok"])
        self.assertEqual(b["m_required"], 1)
        self.assertTrue(b["resolution_ok"])
        for r in rows:
            self.assertEqual(r["resolution_ok"], bundle.m >= r["m_required"])

    def test_rows_flag_unresolvable_when_m_too_small(self):
        bundle = _synthetic_bundle(m=9)
        bundle.corpus = "test-corpus"
        humans = [_words("LOW", *["word"] * 200)]
        ais = [_words("HIGH", *["word"] * 200)]
        rows, _, _detail = conformal.evaluate_cells(
            bundle, humans, ais, ("A", "B"), (0.5,), score_fn=_stub_actions)
        by_key = {(r["construction"], r["alpha"]): r for r in rows}
        a = by_key[("A", 0.5)]
        b = by_key[("B", 0.5)]
        self.assertEqual(a["m_required"], 31)
        self.assertFalse(a["resolution_ok"])
        self.assertTrue(b["resolution_ok"])


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
