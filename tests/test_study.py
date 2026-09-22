import unittest

import conformal


class ConstructionDispatchTests(unittest.TestCase):
    def test_fixed_uses_single_ll_action(self):
        bundle = _synthetic_bundle_for_dispatch()
        pre = {a: 10.0 for a in conformal._all_actions()}
        res = conformal._screen_construction(
            "fixed", "word " * 200, bundle, 0.05, pre)
        self.assertEqual(len(res["steps"]), 1)
        self.assertEqual(res["steps"][0]["detector"], "ll")
        self.assertEqual(res["steps"][0]["budget"], 1024)
        self.assertAlmostEqual(res["steps"][0]["threshold"], 0.05)
        self.assertTrue(res["alert"])

    def test_fixed_does_not_alert_on_low_scores(self):
        bundle = _synthetic_bundle_for_dispatch()
        pre = {a: -10.0 for a in conformal._all_actions()}
        res = conformal._screen_construction(
            "fixed", "word " * 200, bundle, 0.05, pre)
        self.assertFalse(res["alert"])
        self.assertEqual(len(res["steps"]), 1)

    def test_unknown_construction_rejected(self):
        bundle = _synthetic_bundle_for_dispatch()
        pre = {a: 0.0 for a in conformal._all_actions()}
        with self.assertRaises(ValueError):
            conformal._screen_construction(
                "mcp", "word " * 200, bundle, 0.05, pre)


def _synthetic_bundle_for_dispatch():
    import numpy as np
    nA = len(conformal.active_detectors()) * len(conformal.BUDGETS)
    return conformal.CalibrationBundle(
        cal_scores=np.zeros((99, nA)),
        cal_maxima=np.zeros(99),
        g_mu=np.zeros(nA),
        g_sigma=np.ones(nA),
        futility_thr=-1.0,
        n_dev=10,
        m=99,
    )


class MixDocumentsTests(unittest.TestCase):
    def test_splice_takes_first_half_human_second_half_ai(self):
        human = "h1 h2 h3 h4"
        ai = "a1 a2 a3 a4"
        self.assertEqual(
            conformal.mix_documents(human, ai, 0.5), "h1 h2 a3 a4")

    def test_fraction_controls_split_point(self):
        human = "h1 h2 h3 h4"
        ai = "a1 a2 a3 a4"
        out = conformal.mix_documents(human, ai, 0.25).split()
        self.assertEqual(out[:1], ["h1"])
        self.assertEqual(out[1:], ["a2", "a3", "a4"])

    def test_rejects_empty_inputs(self):
        with self.assertRaises(ValueError):
            conformal.mix_documents("", "a b c")


class SubgroupAggregationTests(unittest.TestCase):
    def test_rates_grouped_by_subgroup(self):
        records = [
            {"subgroup": "news", "alert": True},
            {"subgroup": "news", "alert": False},
            {"subgroup": "wiki", "alert": False},
        ]
        out = conformal.aggregate_by_subgroup(records, minimum_n=1)
        self.assertAlmostEqual(out["news"]["rate"], 0.5)
        self.assertAlmostEqual(out["wiki"]["rate"], 0.0)
        self.assertEqual(out["news"]["n"], 2)

    def test_small_subgroups_flagged_not_dropped_silently(self):
        records = [{"subgroup": "rare", "alert": True}]
        out = conformal.aggregate_by_subgroup(records, minimum_n=5)
        self.assertTrue(out["rare"]["suppressed"])
        self.assertIsNone(out["rare"]["rate"])


if __name__ == "__main__":
    unittest.main()
