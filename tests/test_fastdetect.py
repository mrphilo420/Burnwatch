import unittest

import numpy as np
import torch

import conformal


class FastDetectMathTests(unittest.TestCase):
    def test_prefix_discrepancy_matches_upstream_analytic(self):
        # Reference: uniform distribution (all-zero logits).
        # Scorer: token 0 strongly preferred. Observed labels: token 0.
        ref_logits = torch.zeros(1, 4, 3)
        score_logits = torch.zeros(1, 4, 3)
        score_logits[:, :, 0] = 10.0
        labels = torch.zeros(1, 4, dtype=torch.long)

        out = conformal.fastdetect_prefix_discrepancy(ref_logits, score_logits, labels)

        self.assertEqual(len(out), 4)
        self.assertTrue(all(np.isfinite(out)))
        # every position must show a positive discrepancy: the observed
        # token is far more likely under the scorer than the reference mean
        self.assertTrue(all(v > 0 for v in out))
        # the full-length value equals the upstream analytic formula:
        # (sum(ll - mean_ref)) / sqrt(sum(var_ref))
        lprobs = torch.log_softmax(score_logits, dim=-1)
        probs_ref = torch.softmax(ref_logits, dim=-1)
        ll = lprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mean_ref = (probs_ref * lprobs).sum(-1)
        var_ref = (probs_ref * lprobs.square()).sum(-1) - mean_ref.square()
        expected = float(((ll - mean_ref).sum() / var_ref.sum().sqrt()).item())
        self.assertAlmostEqual(out[-1], expected, places=5)

    def test_fastdetect_family_gate(self):
        prev_flag, prev_model = conformal.USE_FASTDETECT, conformal.BASE_MODEL
        try:
            conformal.USE_FASTDETECT = True
            conformal.BASE_MODEL = "gpt2"
            self.assertIn("fastdetect", conformal.active_detectors())
            conformal.BASE_MODEL = "Qwen/Qwen2.5-1.5B"
            self.assertNotIn("fastdetect", conformal.active_detectors())
            conformal.USE_FASTDETECT = False
            conformal.BASE_MODEL = "gpt2"
            self.assertNotIn("fastdetect", conformal.active_detectors())
        finally:
            conformal.USE_FASTDETECT, conformal.BASE_MODEL = prev_flag, prev_model

    def test_fingerprint_changes_with_fastdetect_flag(self):
        import importlib
        prev_flag, prev_model = conformal.USE_FASTDETECT, conformal.BASE_MODEL
        try:
            conformal.USE_FASTDETECT = False
            conformal.BASE_MODEL = "gpt2"
            off = conformal.config_fingerprint()
            conformal.USE_FASTDETECT = True
            on = conformal.config_fingerprint()
            self.assertNotEqual(off, on)
        finally:
            conformal.USE_FASTDETECT, conformal.BASE_MODEL = prev_flag, prev_model
            importlib.reload(conformal)

    def test_fastdetect_short_document_fails_conservative(self):
        def no_model_load():
            raise AssertionError("must not load models for short documents")

        conformal.load_model = no_model_load
        conformal.load_performer = no_model_load
        try:
            conformal.USE_FASTDETECT = True
            out = conformal.fastdetect_action_scores("only five words here now")
        finally:
            conformal.USE_FASTDETECT = False
            import importlib
            importlib.reload(conformal)
        for budget in conformal.BUDGETS:
            self.assertTrue(np.isneginf(out[("fastdetect", budget)]))


if __name__ == "__main__":
    unittest.main()
