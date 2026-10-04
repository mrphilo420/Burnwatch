import math
import tempfile
import unittest
from pathlib import Path

import conformal
import docutils
import slop


class ConformalCoreTests(unittest.TestCase):
    def test_rank_uses_conservative_tie_rule(self):
        self.assertEqual(conformal.conformal_p([1.0, 1.0, 0.2], 1.0), 0.75)
        self.assertEqual(conformal.conformal_p([0.1, 0.2, 0.3], 0.9), 0.25)

    def test_resolution_counts_match_theory(self):
        self.assertEqual(conformal.resolution_b(0, 0.01), 99)
        self.assertEqual(conformal.resolution_a(0, 0.01, 1 / 12), 1199)
        self.assertEqual(conformal.resolution_a(0, 0.001, 1 / 12), 11999)
        self.assertEqual(conformal.resolution_a(260, 0.01, 1 / 16), 1599)

    def test_simes_resolution_matches_the_manuscript_table(self):
        self.assertAlmostEqual(conformal.harmonic(12), 3.103210678, places=8)
        self.assertAlmostEqual(conformal.harmonic(16), 3.380728993, places=8)
        # Table 2, row 3: m >= ceil(H_12/alpha) - 1
        self.assertEqual(conformal.resolution_simes(0.05, 12), 62)
        self.assertEqual(conformal.resolution_simes(0.01, 12), 310)
        self.assertEqual(conformal.resolution_simes(0.001, 12), 3103)
        # perfectly dependent ranks collapse to the single-hypothesis count
        self.assertEqual(conformal.resolution_simes(0.01, 12, corrected=False), 99)
        # row 1 for comparison: equal Bonferroni weights at K = 12
        self.assertEqual(math.ceil(12 / 0.01) - 1, 1199)

    def test_simes_thresholds_and_first_viable_step(self):
        level = conformal.simes_level(0.05, 16, corrected=True)
        self.assertAlmostEqual(level, 0.05 / conformal.harmonic(16))
        self.assertEqual(conformal.simes_threshold(0.05, 16, 16, corrected=False), 0.05)
        # m = 260 at alpha = 0.05: the first threshold reaching 1/(m+1) is k = 5
        self.assertEqual(conformal.simes_first_viable_step(260, 0.05, 16), 5)
        # with m = 9 the grid floor 1/10 blocks every corrected step
        self.assertIsNone(conformal.simes_first_viable_step(9, 0.05, 16))

    def test_audit_and_reference_counts_match_the_manuscript_table(self):
        self.assertEqual([conformal.audit_zero_events(a) for a in (0.05, 0.01, 0.001)],
                         [59, 299, 2995])
        self.assertEqual([conformal.reference_tail_size(a) for a in (0.05, 0.01, 0.001)],
                         [200, 1000, 10000])
        self.assertEqual(conformal.reference_uniformity_size(), 18445)
        # the count really is the smallest N0 pushing the limit below alpha
        for alpha in (0.05, 0.01, 0.001):
            n0 = conformal.audit_zero_events(alpha)
            self.assertLess(1 - 0.05 ** (1.0 / n0), alpha)
            self.assertGreaterEqual(1 - 0.05 ** (1.0 / (n0 - 1)), alpha)

    def test_resolution_table_carries_simes_and_table2(self):
        table = conformal.resolution_table(260)
        self.assertEqual(table["n_actions"], 16)
        self.assertAlmostEqual(table["harmonic"], conformal.harmonic(16))
        by_alpha = {row["alpha"]: row for row in table["rows"]}
        row = by_alpha[0.05]
        self.assertEqual(row["simes_m_req"], conformal.resolution_simes(0.05, 16))
        self.assertTrue(row["simes_ok"])
        self.assertTrue(row["b_ok"])
        self.assertEqual(row["simes_first_step"], 5)
        self.assertFalse(by_alpha[0.01]["simes_ok"])   # needs m >= 338
        t2 = table["table2"]
        self.assertEqual([r["m"] for r in t2["calibration_a"]], [319, 1599, 15999])
        self.assertEqual([r["m"] for r in t2["calibration_b"]], [19, 99, 999])
        self.assertEqual([r["n0"] for r in t2["audit_zero_events"]], [59, 299, 2995])
        self.assertEqual([r["n_dev"] for r in t2["reference_tail"]], [200, 1000, 10000])
        self.assertEqual(t2["reference_uniformity"]["n_dev"], 18445)

    def test_validity_gate_uses_the_minimum_word_count(self):
        ok, reason = conformal.validity_gate("")
        self.assertFalse(ok)
        ok, reason = conformal.validity_gate("word " * conformal.MIN_TOKENS)
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_route_contains_slop_action(self):
        self.assertIn(("slop", 1024), conformal.DEFAULT_ROUTE)

    def test_complete_route_freezes_after_failure_futility_stop(self):
        actions = conformal._all_actions()
        index = {action: i for i, action in enumerate(actions)}
        scores = __import__("numpy").zeros((1, len(actions)))
        scores[0, index[("ll", 128)]] = float("-inf")
        scores[0, index[("ll", 256)]] = 10.0
        maxima = conformal._route_stop_maxima(
            scores,
            conformal.DEFAULT_ROUTE,
            __import__("numpy").zeros(len(actions)),
            __import__("numpy").ones(len(actions)),
            0.0,
            index,
        )
        self.assertTrue(__import__("numpy").isneginf(maxima[0]))


class SlopTests(unittest.TestCase):
    def test_hard_markers_return_spans_and_categories(self):
        result = slop.scan(
            "It cannot be overstated. Let's be honest: the market rewards speed."
        )
        self.assertGreaterEqual(result["total"], 3)
        self.assertTrue(result["spans"])
        self.assertIn("structure", {span["category"] for span in result["spans"]})

    def test_soft_signals_are_reported_separately(self):
        result = slop.scan("Everyone really wants to simply move forward.")
        self.assertEqual(result["total"], 0)
        self.assertGreater(result["soft_total"], 0)

    def test_word_marks_match_text_length(self):
        text = "In today's digital landscape, teams navigate uncertainty."
        self.assertEqual(len(slop.per_word_marks(text)), len(text.split()))


class DocumentValidationTests(unittest.TestCase):
    def test_allowed_plain_text_extracts(self):
        data = b"A plain text document."
        ok, error = docutils.validate("sample.txt", data)
        self.assertTrue(ok)
        self.assertIsNone(error)
        self.assertEqual(docutils.extract_text("sample.txt", data), "A plain text document.")

    def test_unsupported_extension_is_rejected(self):
        ok, error = docutils.validate("sample.exe", b"MZ")
        self.assertFalse(ok)
        self.assertIn("Unsupported file type", error)

    def test_oversized_file_is_rejected(self):
        ok, error = docutils.validate("sample.txt", b"x" * (docutils.MAX_FILE_BYTES + 1))
        self.assertFalse(ok)
        self.assertIn("too large", error)


if __name__ == "__main__":
    unittest.main()
