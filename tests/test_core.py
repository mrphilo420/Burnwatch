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
