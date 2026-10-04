"""Empirical detection quality: does the shipped detector actually flag AI text?

These tests screen real human and AI documents through the full decision path
(A, B, Simes) with the cached gpt2 calibration for the corpus it was built on,
so the false-alert guarantee and the detection power are both exercised on the
production code, not a fixture. They are skipped when no calibration large
enough for every count to fit is cached — building one costs minutes of model
scoring — so a fresh clone runs the structural suites first.

Measured at m = 4,000 with n = 40 (deterministic pool slices): human alerts
0/40 for every construction and level; AI alerts 0.125/0.20 (A), 0.25/0.50 (B)
and 0.10/0.225 (Simes) at alpha 0.01/0.05 on cpu, higher on mps. The
thresholds below keep roughly a factor of two of headroom.
"""

import glob
import os
import unittest

import conformal

CORPUS = "realdet"
ALPHAS = (0.01, 0.05)
CONSTRUCTIONS = ("A", "B", "Simes")
N = 40
MAX_HUMAN_RATE = 0.10
MIN_AI_RATE = {
    ("A", 0.01): 0.025, ("A", 0.05): 0.10,
    ("B", 0.01): 0.10, ("B", 0.05): 0.35,
    ("Simes", 0.01): 0.025, ("Simes", 0.05): 0.10,
}


def _calibration_cache():
    """Largest cached calibration whose m lets every count fit at alpha 0.001."""
    actions = len(conformal.active_detectors()) * len(conformal.BUDGETS)
    min_m = conformal.resolution_simes(0.001, actions)
    best, best_m = None, 0
    pattern = (f"calibration_cache/cal_{conformal.BASE_MODEL.replace('/', '_')}_"
               f"{CORPUS}_base_*_50.npz")
    for path in glob.glob(pattern):
        try:
            m = int(os.path.basename(path).split("_")[-2])
        except ValueError:
            continue
        if min_m <= m and m > best_m:
            best, best_m = path, m
    return best


CACHE = _calibration_cache()


@unittest.skipUnless(CACHE, f"no calibration with m >= required counts cached for {CORPUS}")
class DetectionQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = conformal.load_bundle(CACHE)
        cls.m = cls.bundle.m
        cls.result = conformal.benchmark(cls.bundle, N, CORPUS, alphas=ALPHAS,
                                         constructions=CONSTRUCTIONS)

    def rows(self):
        return self.result["rows"]

    def test_benchmark_scores_the_calibrated_corpus(self):
        self.assertEqual(self.result["n"], N)
        self.assertTrue(self.result["matched"],
                        "detection power is measured on the calibration corpus")
        self.assertEqual(self.result["m"], self.m)

    def test_every_row_is_resolvable_at_the_current_m(self):
        for row in self.rows():
            with self.subTest(construction=row["construction"], alpha=row["alpha"]):
                self.assertTrue(row["resolution_ok"],
                                f"{row['construction']} at alpha={row['alpha']} needs "
                                f"m >= {row['m_required']}, have {self.m}")

    def test_human_documents_stay_under_the_false_alert_level(self):
        for row in self.rows():
            with self.subTest(construction=row["construction"], alpha=row["alpha"]):
                self.assertLessEqual(
                    row["human_rate"], MAX_HUMAN_RATE,
                    f"{row['human_rate']:.3f} of human documents alerted "
                    f"({row['human_alerts']}/{N}) for {row['construction']} "
                    f"at alpha={row['alpha']}")

    def test_ai_documents_are_flagged_far_more_often_than_human(self):
        for row in self.rows():
            key = (row["construction"], row["alpha"])
            with self.subTest(construction=row["construction"], alpha=row["alpha"]):
                self.assertGreaterEqual(
                    row["ai_rate"], MIN_AI_RATE[key],
                    f"only {row['ai_alerts']}/{N} AI documents alerted for "
                    f"{row['construction']} at alpha={row['alpha']}")
                self.assertGreater(
                    row["ai_rate"], row["human_rate"],
                    f"{row['construction']} at alpha={row['alpha']} does not separate "
                    f"AI from human documents")


if __name__ == "__main__":
    unittest.main()
