"""Calibration corpus pools: size requirements, pooling, supplements.

The default calibration size must fit every row of the counts table, and the
human pools must actually be able to serve it (both generator splits pooled,
documented supplements for the short Binoculars corpora, and an actionable
error when a corpus still cannot).
"""

import json
import os
import tempfile
import unittest

import numpy as np

import conformal
import data


def _words(n, tag="doc"):
    return f"{tag} " + " ".join(["word"] * n)


def _ai_text(i=0):
    # AI documents must clear data.get_ai_docs' 40-word floor, and the pool is
    # deduplicated, so every record needs its own text
    return f"machine{i} " + " ".join(["word"] * 60)


HUMAN_KEY = {"ccnews": "text", "cnn": "article", "pubmed": "article"}


def _write_raw(root, corpus, source, humans, ai_key="gen_generated_text_wo_prompt"):
    d = os.path.join(root, "binoculars")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{corpus}-{source}.raw.jsonl"), "w") as f:
        for i, human in enumerate(humans):
            rec = {HUMAN_KEY[corpus]: human, ai_key: _ai_text(i)}
            f.write(json.dumps(rec) + "\n")


class DefaultSizeTests(unittest.TestCase):
    """m = 4,000 fits every count the counts table reports."""

    def assertAllFit(self, m, n_actions):
        table = conformal.resolution_table(m)
        self.assertEqual(table["n_actions"], n_actions)
        for row in table["rows"]:
            self.assertTrue(row["a_ok"], f"A fails at α={row['alpha']}")
            self.assertTrue(row["b_ok"], f"B fails at α={row['alpha']}")
            self.assertTrue(row["simes_ok"], f"Simes fails at α={row['alpha']}")

    def test_default_m_fits_every_count_for_the_default_family(self):
        self.assertAllFit(4000, 16)

    def test_default_m_fits_every_count_with_the_full_family(self):
        original = (conformal.USE_BINOCULARS, conformal.USE_FASTDETECT,
                    conformal.BASE_MODEL)
        conformal.USE_BINOCULARS = True
        conformal.USE_FASTDETECT = True
        conformal.BASE_MODEL = "gpt2"
        try:
            n_actions = len(conformal.active_detectors()) * len(conformal.BUDGETS)
            self.assertEqual(n_actions, 24)
            self.assertAllFit(4000, n_actions)
        finally:
            (conformal.USE_BINOCULARS, conformal.USE_FASTDETECT,
             conformal.BASE_MODEL) = original

    def test_binding_count_is_the_corrected_simes_rule_at_0001(self):
        self.assertEqual(conformal.resolution_simes(0.001, 16), 3380)
        self.assertEqual(conformal.resolution_simes(0.001, 24), 3775)
        self.assertGreaterEqual(4000, 3775)
        # B and resolution-aware A are cheaper at every α on the grid
        for alpha in (0.05, 0.01, 0.001):
            self.assertLessEqual(conformal.resolution_b(0, alpha),
                                 conformal.resolution_simes(alpha, 16))

    def test_a_never_binds_when_the_rank_grid_can_reject(self):
        # resolution-aware A concentrates on k = floor(α(m+1)) actions, so its
        # requirement is ≤ m whenever that subset is non-empty
        for m in (1000, 3380, 4000):
            for alpha in (0.05, 0.01, 0.001):
                plan = conformal.a_weight_plan(m, alpha, 16)
                if plan["n_active"] > 0:
                    self.assertTrue(plan["resolution_ok"])
                    self.assertLessEqual(plan["m_required"], m)


class PoolingTests(unittest.TestCase):
    def test_pools_both_generator_splits(self):
        root = tempfile.mkdtemp()
        falcon_doc, llama_doc = _words(120, "falcon"), _words(120, "llama")
        _write_raw(root, "ccnews", "falcon7", [falcon_doc, falcon_doc])
        _write_raw(root, "ccnews", "llama2_13", [llama_doc])
        docs = data.get_human_docs("ccnews", cache_dir=root)
        self.assertEqual(sorted(docs), sorted([falcon_doc, llama_doc]))

    def test_short_documents_are_dropped(self):
        root = tempfile.mkdtemp()
        long_doc, short_doc = _words(150, "long"), "too short"
        _write_raw(root, "cnn", "falcon7", [long_doc, short_doc])
        _write_raw(root, "cnn", "llama2_13", [short_doc])
        docs = data.get_human_docs("cnn", cache_dir=root)
        self.assertEqual(docs, [long_doc])

    def test_pool_sizes_only_report_what_is_cached(self):
        root = tempfile.mkdtemp()
        d = os.path.join(root, "binoculars")
        os.makedirs(d)
        with open(os.path.join(d, "realdet-human.jsonl"), "w") as f:
            f.write(json.dumps({"text": _words(150, "realdet")}) + "\n")
            f.write(json.dumps({"text": "short"}) + "\n")
        self.assertEqual(data.human_pool_sizes(root), {"realdet": 1})


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.calls = []
        self._original = (data.HUMAN_SUPPLEMENT.get("cnn"),
                          data._load_supplement_split)
        rows = [{"text": _words(150, f"supp{i}")} for i in range(10)]

        def fake_load(spec, split):
            self.calls.append((spec["repo"], split))
            return iter(rows)

        data.HUMAN_SUPPLEMENT["cnn"] = {"repo": "fixtures/cnn", "name": None,
                                        "splits": ("validation",), "field": "text"}
        data._load_supplement_split = fake_load

    def tearDown(self):
        spec, load = self._original
        data._load_supplement_split = load
        if spec is None:
            data.HUMAN_SUPPLEMENT.pop("cnn", None)
        else:
            data.HUMAN_SUPPLEMENT["cnn"] = spec

    def test_supplement_is_fetched_only_when_the_pool_is_short(self):
        _write_raw(self.root, "cnn", "falcon7", [_words(150, "base")])
        _write_raw(self.root, "cnn", "llama2_13", [])
        docs = data.get_human_docs("cnn", cache_dir=self.root)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(docs), 1)

        docs = data.get_human_docs("cnn", cache_dir=self.root, need=6)
        self.assertEqual(self.calls, [("fixtures/cnn", "validation")])
        self.assertEqual(len(docs), 6)

    def test_cached_supplement_is_reused_without_a_second_fetch(self):
        _write_raw(self.root, "cnn", "falcon7", [_words(150, "base")])
        _write_raw(self.root, "cnn", "llama2_13", [])
        first = data.get_human_docs("cnn", cache_dir=self.root, need=6)
        self.calls.clear()
        again = data.get_human_docs("cnn", cache_dir=self.root, need=6)
        self.assertEqual(self.calls, [])
        self.assertEqual(again, first)
        # a call without a need keeps the supplemented pool too
        self.assertEqual(data.get_human_docs("cnn", cache_dir=self.root), first)

    def test_corpora_without_a_supplement_are_left_alone(self):
        _write_raw(self.root, "ccnews", "falcon7", [_words(150, "base")])
        _write_raw(self.root, "ccnews", "llama2_13", [])
        docs = data.get_human_docs("ccnews", cache_dir=self.root, need=50)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(docs), 1)


class ShortPoolTests(unittest.TestCase):
    def test_build_calibration_explains_a_short_pool(self):
        original = (conformal.fetch_human_docs, conformal.CACHE_DIR,
                    conformal.CORPUS)
        conformal.fetch_human_docs = lambda n, corpus="imdb", progress=None: [
            _words(150, "only")]
        conformal.CACHE_DIR = tempfile.mkdtemp()
        conformal.CORPUS = "pubmed"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                conformal.build_calibration(4000, 50)
            message = str(ctx.exception)
        finally:
            (conformal.fetch_human_docs, conformal.CACHE_DIR,
             conformal.CORPUS) = original
        self.assertIn("corpus 'pubmed' has 1 eligible human documents", message)
        self.assertIn("need 4,050 = m 4,000 + n_dev 50", message)
        self.assertIn("m >= 3,380", message)
        self.assertIn("lower --m to at most 0", message)


class BenchmarkOffsetTests(unittest.TestCase):
    def _pool(self, root, corpus, count, tag):
        _write_raw(root, corpus, "falcon7", [_words(150, f"{tag}{i}")
                                             for i in range(count)])
        _write_raw(root, corpus, "llama2_13", [])

    def test_offset_counts_pooled_consumption(self):
        root = tempfile.mkdtemp()
        self._pool(root, "ccnews", 3, "cc")
        self._pool(root, "cnn", 4, "cnn")
        self._pool(root, "pubmed", 5, "pm")
        original = conformal.CACHE_DIR
        conformal.CACHE_DIR = root
        try:
            # binoc-all calibration took the first 5 pooled docs: 3 ccnews + 2 cnn
            self.assertEqual(conformal._binoc_consumed("ccnews", 5), 3)
            self.assertEqual(conformal._binoc_consumed("cnn", 5), 2)
            self.assertEqual(conformal._binoc_consumed("pubmed", 5), 0)
            # a pool exhausted before its turn contributes nothing
            self.assertEqual(conformal._binoc_consumed("pubmed", 7), 0)
            # 7 pooled documents precede pubmed, so only 3 of the 10 remain
            self.assertEqual(conformal._binoc_consumed("pubmed", 10), 3)
        finally:
            conformal.CACHE_DIR = original

    def test_benchmark_skips_human_but_never_ai_documents(self):
        root = tempfile.mkdtemp()
        _write_raw(root, "ccnews", "falcon7", [_words(150, f"human{i}")
                                               for i in range(20)])
        _write_raw(root, "ccnews", "llama2_13", [])
        n_actions = len(conformal.active_detectors()) * len(conformal.BUDGETS)
        bundle = conformal.CalibrationBundle(
            cal_scores=np.zeros((5, n_actions)), cal_maxima=np.zeros(5),
            g_mu=np.zeros(n_actions), g_sigma=np.ones(n_actions),
            futility_thr=-1.0, n_dev=2, m=5, corpus="ccnews")
        original = conformal.CACHE_DIR, conformal.CORPUS
        conformal.CACHE_DIR, conformal.CORPUS = root, "ccnews"
        skipped = []
        real_eval = conformal.evaluate_cells

        def fake_eval(b, human, ai, *args, **kwargs):
            skipped.append((len(human), len(ai), human[0], ai[0]))
            return [], [], {}

        conformal.evaluate_cells = fake_eval
        try:
            conformal.benchmark(bundle, 8, "ccnews")
        finally:
            conformal.evaluate_cells = real_eval
            conformal.CACHE_DIR, conformal.CORPUS = original
        human_n, ai_n, first_human, first_ai = skipped[0]
        self.assertEqual((human_n, ai_n), (8, 8))
        # human: skips m + n_dev = 7 calibration documents
        self.assertEqual(first_human, _words(150, "human7"))
        # ai: calibration never consumed AI documents, so nothing is skipped
        self.assertEqual(first_ai, _ai_text(0))


if __name__ == "__main__":
    unittest.main()
