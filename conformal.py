#!/usr/bin/env python3
"""Project Burnwatch — adaptive within-document AI-text screening.

Construction A and B.

Implements the two finite-sample constructions of the accompanying theory
manuscript, per their proofs:

  * per-document detectors over token budgets (likelihood, rank, log-rank,
    entropy) under a GPT-2 class generative model; larger score = more
    suspicious, S_a(X) = -inf on failure;
  * conservative upper-tail conformal ranks (Bates et al. 2023 tie
    convention): p = (1 + #{calibration scores >= test score}) / (m + 1);
  * Construction A: a registered action family with fixed error allocations
    and a union bound over the executed subset (Theorem A);
  * Construction B: a development-fixed route generator whose complete-path
    maximum is calibrated; deployment follows a prefix with arbitrary early
    termination (Theorem B);
  * Corollary calibration-resolution counts, the oracle/audit-style bounds
    (Clopper-Pearson zero-event audit, repeated-look inflation).

No e-process is constructed and no conformal rank products are multiplied.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import data
import slop

# ---------------------------------------------------------------------------
# configuration defaults
# ---------------------------------------------------------------------------

DETECTORS = ["ll", "rank", "logrank", "slop"]      # larger score = more suspicious
BUDGETS = [128, 256, 512, 1024]                    # token budgets b_k (deck route)
MIN_TOKENS = 32                                    # below this a document "fails"
DEFAULT_ROUTE = [                                  # development-fixed route for B (deck: 128→256→512→1024)
    ("ll", 128), ("ll", 256), ("rank", 512), ("logrank", 1024), ("slop", 1024),
]
FUTILITY_QUANTILE = 0.05                            # futility threshold quantile (dev maxima)

DEVICE = "cpu"
BASE_MODEL = "gpt2"
PERFORMER_MODEL = "gpt2-medium"   # contrasting-model detector (Binoculars-style)
BINO_COMPATIBLE = {"gpt2", "gpt2-large"}   # scorers sharing the performer's vocabulary
CORPUS = "imdb"                   # calibration corpus: imdb | ccnews | cnn | pubmed | binoc-all
USE_BINOCULARS = False            # add the contrasting-model detector to the family
USE_FASTDETECT = False            # add Fast-DetectGPT conditional discrepancy to A
FAST_COMPATIBLE = {"gpt2"}        # current adapter uses GPT-2/GPT-2-medium token parity
FAST_SCORING_MODEL = "gpt2-medium"
CACHE_DIR = os.path.expanduser("~/.cache")

_model = None
_tokenizer = None
_performer_model = None


def get_device(prefer):
    if prefer in ("cuda", "mps", "cpu"):
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"[conformal] loading {BASE_MODEL} on {DEVICE}...", file=sys.stderr, flush=True)
        kwargs = {"dtype": torch.float16} if DEVICE == "mps" else {}
        _model = transformers.AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, cache_dir=CACHE_DIR, **kwargs).to(DEVICE)
        _tokenizer = transformers.AutoTokenizer.from_pretrained(
            BASE_MODEL, cache_dir=CACHE_DIR)
        _tokenizer.pad_token_id = _tokenizer.eos_token_id
        _warmup(_model, _tokenizer)
    return _model, _tokenizer


def _warmup(model, tokenizer):
    """Compile MPS kernels at full budget now, so the first real request
    does not pay the 5-10s compile latency."""
    if tokenizer is None:
        return
    try:
        warm = tokenizer("word " * 900, return_tensors="pt",
                         truncation=True, max_length=max(BUDGETS)).to(DEVICE)
        with torch.no_grad():
            model(**warm)
    except Exception:
        pass


def set_model(name):
    """Switch the scoring model; next scoring call reloads weights."""
    global BASE_MODEL, _model, _tokenizer, _performer_model
    if name != BASE_MODEL:
        _model = None
        _tokenizer = None
    _performer_model = None
    BASE_MODEL = name


def load_performer():
    """Second, larger model used only by the Binoculars-style detector."""
    global _performer_model
    if _performer_model is None:
        print(f"[conformal] loading performer model {PERFORMER_MODEL} on {DEVICE}...",
              file=sys.stderr, flush=True)
        kwargs = {"dtype": torch.float16} if DEVICE == "mps" else {}
        _performer_model = transformers.AutoModelForCausalLM.from_pretrained(
            PERFORMER_MODEL, cache_dir=CACHE_DIR, **kwargs).to(DEVICE)
        _warmup(_performer_model, _tokenizer)
    return _performer_model


# ---------------------------------------------------------------------------
# per-token scoring
# ---------------------------------------------------------------------------

def active_detectors():
    dets = DETECTORS + (["binoculars"] if USE_BINOCULARS else [])
    if USE_FASTDETECT and BASE_MODEL in FAST_COMPATIBLE:
        dets.append("fastdetect")
    if "binoculars" in dets and BASE_MODEL not in BINO_COMPATIBLE:
        # the contrast needs observer and performer to share a vocabulary;
        # different-vocab scorers (Qwen, SmolLM2, Pythia) cannot be contrasted
        dets.remove("binoculars")
    return dets


def per_token_scores(text):
    """Return {detector: np.ndarray} of per-token scores over positions
    0..T-2 (token i scores the prediction of token i+1). Larger = suspicious."""
    model, tok = load_model()
    tokenized = tok(text, return_tensors="pt", truncation=True,
                    max_length=max(BUDGETS), padding=True)
    input_ids = tokenized.input_ids.to(DEVICE)
    attn = tokenized.attention_mask.to(DEVICE)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits[0]  # (T, V)
    n = int((input_ids[0] != tok.pad_token_id).sum().item())
    L = logits[: n - 1]                                  # predicts positions 1..n-1
    labels = input_ids[0, 1:n]
    logp = F.log_softmax(L, dim=-1)                      # (n-1, V)
    ll = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    out = {"ll": ll.detach().cpu().numpy()}
    if "rank" in active_detectors() or "logrank" in active_detectors():
        ranks = (logp >= ll.unsqueeze(-1)).sum(-1).float()   # conservative 1-indexed rank
        out["rank"] = -ranks.detach().cpu().numpy()
        out["logrank"] = -torch.log(ranks).detach().cpu().numpy()
    if "entropy" in active_detectors():
        entropy = -(logp * logp.exp()).sum(-1)
        out["entropy"] = -entropy.detach().cpu().numpy()
    if USE_BINOCULARS:
        # Binoculars-style contrasting-model score (Hans et al. 2024):
        # BINOCULARS = ppl_performer / x_ppl, where x_ppl is the cross-entropy
        # of the observer distribution under the performer's softmax. AI text
        # has a small ratio; our per-token score (larger = suspicious) is
        # log p_perf + CE(observer || performer) at each position.
        perf = load_performer()
        with torch.no_grad():
            perf_logits = perf(input_ids=input_ids, attention_mask=attn).logits[0]
        p_logp = F.log_softmax(perf_logits[: n - 1], dim=-1)
        ce = -(p_logp.exp() * logp).sum(-1)              # cross-entropy under performer softmax
        perf_ll = p_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        out["binoculars"] = (perf_ll + ce).detach().cpu().numpy()
    if "slop" in active_detectors():
        out["slop"] = np.asarray(slop.per_word_marks(text), dtype=float)
    return {k: v for k, v in out.items() if k in active_detectors()}


def prefix_scores(per_tok, n_tokens):
    """Aggregate per-token arrays into per-action scores over BUDGETS.
    Larger = suspicious; -inf when the document fails (below MIN_TOKENS)."""
    dets = active_detectors()
    out = {}
    if n_tokens < MIN_TOKENS:
        for det in dets:
            for b in BUDGETS:
                out[(det, b)] = -np.inf
        return out
    for det, arr in per_tok.items():
        k = min(n_tokens - 1, len(arr))
        cs = np.cumsum(arr[:k])
        for b in BUDGETS:
            m = min(b, k)
            out[(det, b)] = float(cs[m - 1] / m)
    return out


def fastdetect_prefix_discrepancy(ref_logits, score_logits, labels):
    """Pure-math core of the upstream analytic sampling discrepancy
    (Fast-DetectGPT get_sampling_discrepancy_analytic), evaluated per prefix
    position: d_k = sum_{t<=k}(ll_t - mu_t) / sqrt(sum_{t<=k} sigma^2_t),
    with the reference expectation under the reference model's softmax.
    The full-length value equals the upstream analytic statistic."""
    labels = labels.unsqueeze(-1) if labels.ndim == score_logits.ndim - 1 else labels
    lprobs_score = F.log_softmax(score_logits, dim=-1)
    probs_ref = F.softmax(ref_logits, dim=-1)
    ll = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * lprobs_score.square()).sum(dim=-1) - mean_ref.square()
    var_ref = var_ref.clamp_min(1e-9)
    cumulative = torch.cumsum(ll - mean_ref, dim=-1)
    cumulative_var = torch.cumsum(var_ref, dim=-1).sqrt()
    return (cumulative / cumulative_var).squeeze(0).detach().cpu().numpy()


def fastdetect_action_scores(text):
    """Fast-DetectGPT analytic conditional-probability-curvature scores.

    This follows the upstream implementation's analytic criterion:
    reference distribution = GPT-2, scoring model = GPT-2-medium. The
    cumulative discrepancy is normalized by cumulative reference variance
    at each budget, yielding one score per registered prefix action.
    """
    if len(text.split()) < MIN_TOKENS:
        # same failure rule as the other detectors: never alert off a stub
        return {("fastdetect", b): -np.inf for b in BUDGETS}
    if not USE_FASTDETECT or BASE_MODEL not in FAST_COMPATIBLE:
        return {}
    ref_model, tok = load_model()
    score_model = load_performer()
    tokenized = tok(text, return_tensors="pt", truncation=True,
                    max_length=max(BUDGETS), padding=True).to(DEVICE)
    with torch.no_grad():
        ref_logits = ref_model(**tokenized).logits[0]
        score_logits = score_model(**tokenized).logits[0]
    n = int((tokenized.input_ids != tok.pad_token_id).sum().item())
    labels = tokenized.input_ids[0, 1:n]
    discrepancy = fastdetect_prefix_discrepancy(
        ref_logits[: n - 1].unsqueeze(0), score_logits[: n - 1].unsqueeze(0),
        labels.unsqueeze(0))
    k = len(discrepancy)
    out = {}
    for b in BUDGETS:
        take = min(b, k)
        out[("fastdetect", b)] = float(discrepancy[take - 1]) if take else -np.inf
    return out


def action_scores_for(text):
    """Canonical document-to-action score path used by calibration, screen,
    and benchmark. Keeping this as one function prevents apples-to-oranges
    score maps between live and evaluation paths."""
    pre = prefix_scores(per_token_scores(text), len(text.split()))
    pre.update(fastdetect_action_scores(text))
    return pre


# ---------------------------------------------------------------------------
# conformal machinery
# ---------------------------------------------------------------------------

def conformal_p(cal_scores, test_score):
    """Conservative upper-tail conformal rank (Lemma 3.1):
    p = (1 + #{cal >= test})/(m+1). The weak inequality is the tie-conservative
    choice the proof requires; cal_scores is the frozen calibration table."""
    scores = np.asarray(cal_scores)
    return (1.0 + float(np.sum(scores >= test_score))) / (len(scores) + 1.0)


def config_fingerprint():
    """Hash of everything that defines the score map and the route. Any
    drift between the loaded calibration and this fingerprint means the
    live query would be compared against a mismatched table (the
    benchmark-vs-screen class of bug), so scoring must refuse to run."""
    import hashlib
    payload = json.dumps({
        "base_model": BASE_MODEL,
        "corpus": CORPUS,
        "detectors": active_detectors(),
        "budgets": BUDGETS,
        "route": DEFAULT_ROUTE,
        "min_tokens": MIN_TOKENS,
        "slop_version": slop.SLOP_VERSION,
        "std_version": 2,
        "performer": PERFORMER_MODEL if USE_BINOCULARS else None,
        "fastdetect": USE_FASTDETECT,
        "fast_scoring": PERFORMER_MODEL if USE_FASTDETECT else None,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def standardize_g(dev_scores):
    """g_a(s) = (s - mu_a)/sigma_a fixed from the development set (D_dev).

    Conservative: the scale uses the larger of the standard deviation and the
    MAD-based robust spread, and is floored at the lower quartile of robust
    spreads across actions. A near-constant action therefore cannot inflate
    its standardized score: with no evidence of spread, its contribution is
    dampened. This is a development-fixed transformation, so Construction B's
    validity is unaffected; only power is traded for conservatism."""
    mu = np.mean(dev_scores, axis=0)
    sd = np.std(dev_scores, axis=0)
    mad = np.median(np.abs(dev_scores - mu), axis=0) * 1.4826
    robust = np.maximum(sd, mad)
    floor = float(np.quantile(robust, 0.25))
    floor = max(floor, 1e-6)
    sd = np.maximum(robust, floor)
    return mu, sd


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def fetch_human_docs(n, corpus="imdb", progress=None):
    """Human calibration documents from the requested corpus."""
    if corpus == "imdb":
        import datasets
        print(f"[conformal] loading stanfordnlp/imdb (human calibration documents, n={n})...",
              file=sys.stderr, flush=True)
        ds = datasets.load_dataset("stanfordnlp/imdb", split="train", cache_dir=CACHE_DIR)
        docs = [d["text"] for d in ds if len(d["text"].split()) >= 100]
    elif corpus == "binoc-all":
        print(f"[conformal] loading Binoculars corpora (cc_news, cnn, pubmed, n={n})...",
              file=sys.stderr, flush=True)
        docs = []
        for c in data.CORPORA:
            docs += data.get_human_docs(c, "falcon7", CACHE_DIR, progress=progress)
    elif corpus in data.RECOMMENDED:
        print(f"[conformal] loading {corpus} human documents (n={n})...",
              file=sys.stderr, flush=True)
        docs = data.get_human_docs(corpus, "falcon7", CACHE_DIR, progress=progress)
    else:
        print(f"[conformal] loading Binoculars corpus {corpus} (n={n})...",
              file=sys.stderr, flush=True)
        docs = data.get_human_docs(corpus, "falcon7", CACHE_DIR)
    docs = list(dict.fromkeys(docs))  # dedupe, deterministic
    print(f"[conformal] {len(docs)} human documents available", file=sys.stderr, flush=True)
    return docs[:n]


def score_docs(docs, progress=None):
    """One forward pass per document; returns action -> np.ndarray over docs."""
    nA = len(active_detectors()) * len(BUDGETS)
    scores = np.full((len(docs), nA), -np.inf)
    action_index = {a: i for i, a in enumerate(_all_actions())}
    for i, doc in enumerate(docs):
        pre = action_scores_for(doc)
        for a, v in pre.items():
            scores[i, action_index[a]] = v
        if progress and (i + 1) % 5 == 0:
            progress(i + 1, len(docs))
    return scores


def _all_actions():
    return [(det, b) for det in active_detectors() for b in BUDGETS]


def _route_actions(route):
    return [(det, b) for det, b in route]


def _route_stop_maxima(scores, route, g_mu, g_sigma, futility_thr, action_index):
    """M_pi(H_i): max over the executed prefix of the route, with alert-based
    stopping disabled and the development-fixed futility rule retained."""
    n = scores.shape[0]
    maxima = np.full(n, -np.inf)
    running = np.full(n, -np.inf)
    active = np.ones(n, dtype=bool)
    for det, b in route:
        col = action_index[(det, b)]
        g = (scores[:, col] - g_mu[col]) / g_sigma[col]
        running = np.where(active, np.maximum(running, g), running)
        stop = active & (running < futility_thr)
        maxima = np.where(stop, running, maxima)
        active = active & ~stop
    maxima = np.where(active, running, maxima)
    return maxima


def build_calibration(m, n_dev, progress=None):
    """Fetch human documents, split dev/calibration, fix g_a from dev,
    fix futility threshold from dev complete maxima, compute calibration
    scores and complete-path maxima. Returns a CalibrationBundle."""
    n_total = m + n_dev
    docs = fetch_human_docs(n_total, CORPUS, progress=progress)
    if len(docs) < n_total:
        raise RuntimeError(f"only {len(docs)} human documents available, need {n_total}")
    dev_docs, cal_docs = docs[:n_dev], docs[n_dev:]
    print(f"[conformal] scoring {n_dev} development documents...", file=sys.stderr, flush=True)
    dev_scores = score_docs(dev_docs, progress)
    print(f"[conformal] scoring {m} calibration documents...", file=sys.stderr, flush=True)
    cal_scores = score_docs(cal_docs, progress)

    g_mu, g_sigma = standardize_g(dev_scores)
    action_index = {a: i for i, a in enumerate(_all_actions())}

    # futility threshold from dev complete maxima (alert-stopping disabled,
    # futility not yet fixed -> full-route maxima)
    dev_full = np.full(n_dev, -np.inf)
    running = np.full(n_dev, -np.inf)
    for det, b in DEFAULT_ROUTE:
        col = action_index[(det, b)]
        running = np.maximum(running, (dev_scores[:, col] - g_mu[col]) / g_sigma[col])
    dev_full = running
    futility_thr = float(np.quantile(dev_full, FUTILITY_QUANTILE))

    cal_maxima = _route_stop_maxima(cal_scores, DEFAULT_ROUTE, g_mu, g_sigma,
                                    futility_thr, action_index)

    bundle = CalibrationBundle(
        cal_scores=cal_scores, cal_maxima=cal_maxima,
        g_mu=g_mu, g_sigma=g_sigma, futility_thr=futility_thr,
        n_dev=n_dev, m=m, corpus=CORPUS)
    return bundle


class CalibrationBundle:
    def __init__(self, cal_scores, cal_maxima, g_mu, g_sigma, futility_thr, n_dev, m, corpus=None):
        self.cal_scores = cal_scores          # (m, nA)
        self.cal_maxima = cal_maxima          # (m,)
        self.g_mu = g_mu                      # (nA,)
        self.g_sigma = g_sigma                # (nA,)
        self.futility_thr = futility_thr
        self.n_dev = n_dev
        self.m = m
        self.corpus = corpus                  # human calibration population
        self.fingerprint = config_fingerprint()
        self.action_index = {a: i for i, a in enumerate(_all_actions())}

    @property
    def n_actions(self):
        return self.cal_scores.shape[1]

    def selfcheck(self):
        """Recompute the complete-path maxima from the stored action scores
        and compare to the stored maxima. A mismatch means the frozen
        reference and the live path disagree — refuse to deploy."""
        try:
            recomputed = _route_stop_maxima(self.cal_scores, DEFAULT_ROUTE,
                                            self.g_mu, self.g_sigma,
                                            self.futility_thr, self.action_index)
            ok = np.allclose(recomputed, self.cal_maxima, equal_nan=True)
        except Exception:
            ok = False
        return ok

    def save(self, path):
        np.savez_compressed(
            path,
            cal_scores=self.cal_scores, cal_maxima=self.cal_maxima,
            g_mu=self.g_mu, g_sigma=self.g_sigma,
            futility_thr=np.array([self.futility_thr]),
            n_dev=np.array([self.n_dev]), m=np.array([self.m]),
        )
        meta = {"base_model": BASE_MODEL, "corpus": CORPUS,
                "detectors": active_detectors(), "budgets": BUDGETS,
                "route": DEFAULT_ROUTE, "min_tokens": MIN_TOKENS,
                "slop_version": slop.SLOP_VERSION,
                "std_version": 2,
                "performer": PERFORMER_MODEL if USE_BINOCULARS else None,
                "fastdetect": USE_FASTDETECT,
                "fast_scoring": PERFORMER_MODEL if USE_FASTDETECT else None}
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        corpus = None
        try:
            with open(path + ".json") as f:
                corpus = json.load(f).get("corpus")
        except (OSError, ValueError):
            corpus = None
        return cls(
            cal_scores=d["cal_scores"], cal_maxima=d["cal_maxima"],
            g_mu=d["g_mu"], g_sigma=d["g_sigma"],
            futility_thr=float(d["futility_thr"][0]),
            n_dev=int(d["n_dev"][0]), m=int(d["m"][0]), corpus=corpus)


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------

def resolution_a(m, alpha, w=1.0):
    """Corollary A: m >= ceil(1/(alpha * w_a)) - 1 for rejection."""
    if w <= 0:
        return math.inf
    return math.ceil(1.0 / (alpha * w)) - 1


def resolution_b(m, alpha):
    """Corollary B: m >= ceil(1/alpha) - 1 for rejection."""
    return math.ceil(1.0 / alpha) - 1


def paired_difference_ci(a_alerts, b_alerts):
    """Matched-document paired power/FPR difference (A minus B) with a
    one-sided 95% lower bound from the paired Wald interval: each document
    contributes d_i = a_i - b_i, lower = mean(d) - 1.644854 * sd(d)/sqrt(n).
    Used for the preregistered paired power comparison on AI documents."""
    a = [int(x) for x in a_alerts]
    b = [int(x) for x in b_alerts]
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("paired inputs must be non-empty and equal length")
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    diff = sum(d) / n
    if n == 1:
        se = 0.0
    else:
        mean = diff
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        se = math.sqrt(var) / math.sqrt(n)
    return {"n": n, "diff": diff, "se": se, "lower_95": diff - 1.644854 * se}


def resolution_table(m):
    """Deck slide 'Calibration counts': A (equal allocations) vs B at
    nominal alpha levels, given current m."""
    nA = len(active_detectors()) * len(BUDGETS)
    w = 1.0 / nA
    rows = []
    for alpha in (0.05, 0.01, 0.001):
        ra = resolution_a(m, alpha, w)
        rb = resolution_b(m, alpha)
        rows.append({
            "alpha": alpha,
            "a_m_req": ra,
            "a_ok": m >= ra,
            "b_m_req": rb,
            "b_ok": m >= rb,
        })
    return {"m": m, "n_actions": nA, "w": w, "rows": rows}


def screen_a(text, bundle, alpha, actions=None, pre=None, weights=None):
    """Construction A: registered family, union bound.

    Equal weights by default; an explicit ``weights`` map concentrates the
    error budget on chosen actions (must sum to <= 1). Execution stops at
    the first crossing — the union bound covers whichever prefix ran, so
    power beyond the first alert costs nothing extra.
    """
    nA = bundle.n_actions
    if actions is None:
        actions = _all_actions()
    if weights is None:
        eff_w = {a: 1.0 / nA for a in actions}
    else:
        if sum(weights.values()) > 1.0 + 1e-9:
            raise ValueError("action weights must sum to <= 1")
        eff_w = {a: weights.get(a, 0.0) for a in actions}
    n_tok = len(text.split())
    if pre is None:
        pre = action_scores_for(text)
    cal = bundle.cal_scores

    steps = []
    alert = False
    for det, b in actions:
        alpha_a = alpha * eff_w[(det, b)]
        col = bundle.action_index[(det, b)]
        s = pre[(det, b)]
        p = 1.0 if np.isneginf(s) else conformal_p(cal[:, col], s)
        hit = p <= alpha_a
        alert = alert or hit
        steps.append({
            "t": len(steps) + 1,
            "detector": det,
            "budget": b,
            "score": None if np.isneginf(s) else float(s),
            "p": p,
            "threshold": alpha_a,
            "alert": bool(hit),
        })
        if hit:
            break
    positive = [w for w in eff_w.values() if w > 0]
    m_req = min([resolution_a(bundle.m, alpha, w) for w in positive]) if positive else math.inf
    return {
        "construction": "A",
        "alpha": alpha,
        "m": bundle.m,
        "m_required": m_req,
        "resolution_ok": bundle.m >= m_req,
        "alert": alert,
        "steps": steps,
        "tokens_inspected": min(max(BUDGETS), n_tok),
        "tokens_full": min(max(BUDGETS), n_tok),
        "tokens_saved_pct": 0,
        "actions_executed": len(steps),
        "n_actions": bundle.n_actions,
    }


def screen_b(text, bundle, alpha, pre=None):
    """Construction B: complete-path calibration, early stopping along route."""
    n_tok = len(text.split())
    if pre is None:
        pre = action_scores_for(text)
    cal = bundle.cal_scores
    cal_max = bundle.cal_maxima

    running = -np.inf
    steps = []
    alert = False
    stopped_futility = False
    for t, (det, b) in enumerate(DEFAULT_ROUTE, 1):
        s = pre[(det, b)]
        if np.isneginf(s):
            g = -np.inf
        else:
            col = bundle.action_index[(det, b)]
            g = (s - bundle.g_mu[col]) / bundle.g_sigma[col]
        running = max(running, g)
        p = 1.0 if np.isneginf(running) else conformal_p(cal_max, running)
        hit = p <= alpha
        steps.append({
            "t": t,
            "detector": det,
            "budget": b,
            "score": None if np.isneginf(s) else float(s),
            "partial_max": None if np.isneginf(running) else float(running),
            "p": p,
            "threshold": alpha,
            "alert": bool(hit),
        })
        if hit:
            alert = True
            break
        if running < bundle.futility_thr:
            stopped_futility = True
            break
    return {
        "construction": "B",
        "alpha": alpha,
        "m": bundle.m,
        "m_required": resolution_b(bundle.m, alpha),
        "resolution_ok": bundle.m >= resolution_b(bundle.m, alpha),
        "alert": alert,
        "stopped_futility": stopped_futility,
        "steps": steps,
        "tokens_inspected": min(steps[-1]["budget"], n_tok) if steps else 0,
        "tokens_full": min(max(BUDGETS), n_tok),
        "tokens_saved_pct": round(100 * (1 - min(steps[-1]["budget"], n_tok) / max(1, min(max(BUDGETS), n_tok)))),
        "actions_executed": len(steps),
        "route_length": len(DEFAULT_ROUTE),
        "n_actions": bundle.n_actions,
    }


# ---------------------------------------------------------------------------
# supporting bounds (manuscript section 6)
# ---------------------------------------------------------------------------

def repeated_look_inflation(alpha, k):
    """P(min U_k <= alpha) = 1 - (1 - alpha)^k for k independent uniform p-values."""
    return 1.0 - (1.0 - alpha) ** k


def clopper_pearson_upper(k, n, gamma):
    """One-sided 1-gamma upper confidence limit for a Binomial(n, q) success
    probability q given k successes (Clopper-Pearson)."""
    if k == 0:
        return 1.0 - gamma ** (1.0 / n)
    if k == n:
        return gamma ** (1.0 / n)
    from scipy.stats import binom
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if binom.cdf(k, n, mid) <= gamma:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# benchmark (empirical alert rates on Binoculars corpora)
# ---------------------------------------------------------------------------

def mix_documents(human_text, ai_text, frac=0.5):
    """Splice a mixed-authorship document: the first frac words from the
    human text, the remainder from the AI text."""
    h = human_text.split()
    a = ai_text.split()
    if not h or not a:
        raise ValueError("mix_documents needs non-empty human and AI texts")
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    k = max(1, int(len(h) * frac))
    j = max(1, int(len(a) * (1 - frac)))
    return " ".join(h[:k] + a[-j:])


def aggregate_by_subgroup(records, minimum_n=5):
    """Group per-document {subgroup, alert} records into alert rates.
    Subgroups below minimum_n are reported with rate None and
    suppressed=True rather than silently dropped."""
    groups = {}
    for r in records:
        g = groups.setdefault(r["subgroup"], {"n": 0, "alerts": 0})
        g["n"] += 1
        g["alerts"] += 1 if r["alert"] else 0
    out = {}
    for name in sorted(groups):
        g = groups[name]
        if g["n"] < minimum_n:
            out[name] = {"n": g["n"], "alerts": g["alerts"],
                         "rate": None, "suppressed": True}
        else:
            out[name] = {"n": g["n"], "alerts": g["alerts"],
                         "rate": g["alerts"] / g["n"], "suppressed": False}
    return out


FIXED_ACTION = ("ll", 1024)


def _screen_construction(c, doc, bundle, alpha, pre):
    """Dispatch one construction, including the fixed single-action
    comparator (ll@1024 at full weight): the matched-error baseline for
    the preregistered efficiency comparison."""
    if c == "A":
        return screen_a(doc, bundle, alpha, pre=pre)
    if c == "B":
        return screen_b(doc, bundle, alpha, pre=pre)
    if c == "fixed":
        return screen_a(doc, bundle, alpha, pre=pre,
                        actions=[FIXED_ACTION], weights={FIXED_ACTION: 1.0})
    raise ValueError(f"unknown construction {c!r}")


def evaluate_cells(bundle, humans, ais, constructions, alphas, progress=None, score_fn=None,
                   meta_h=None, meta_a=None, detail=False):
    """Score paired human/AI documents through both constructions and return
    (rows, paired). The paired comparison uses AI documents only: by linearity
    of expectation the paired mean difference then equals the marginal AI-rate
    difference exactly. score_fn maps a document to an ACTION dict
    {(detector, budget): score} (production default: action_scores_for);
    everything downstream is the real path."""
    if score_fn is None:
        score_fn = action_scores_for
    n = min(len(humans), len(ais))
    counts = {c: {a: [0, 0] for a in alphas} for c in constructions}
    cost = {c: {a: [0, 0] for a in alphas} for c in constructions}  # [inspected, savings_pct]
    acts = {c: {a: 0 for a in alphas} for c in constructions}  # executed actions (A early-exits too)
    futility = {a: [0, 0] for a in alphas}  # [human, ai] B futility stops
    early = {c: {a: [0, 0] for a in alphas} for c in constructions}
    paired_ai = {c: {a: [] for a in alphas} for c in constructions}
    docs_all = humans[:n] + ais[:n]
    pres = [score_fn(doc) for doc in docs_all]
    detail_out = [] if detail else None
    for i in range(n):
        rec = None
        if detail:
            rec = {"human_subgroup": meta_h[i] if meta_h else None,
                   "ai_subgroup": meta_a[i] if meta_a else None,
                   "human_words": len(humans[i].split()),
                   "ai_words": len(ais[i].split()),
                   "alerts": {}, "tokens": {}, "futility": {}}
        for label, doc, pre in (("human", humans[i], pres[i]), ("ai", ais[i], pres[n + i])):
            for c in constructions:
                for a in alphas:
                    res = _screen_construction(c, doc, bundle, a, pre)
                    if res["alert"]:
                        counts[c][a][0 if label == "human" else 1] += 1
                    cost[c][a][0] += res["tokens_inspected"]
                    cost[c][a][1] += res["tokens_saved_pct"]
                    acts[c][a] += res["actions_executed"]
                    full_steps = (len(DEFAULT_ROUTE) if c == "B"
                                  else 1 if c == "fixed" else bundle.n_actions)
                    if res["actions_executed"] < full_steps:
                        early[c][a][0 if label == "human" else 1] += 1
                    if label == "ai":
                        paired_ai[c][a].append(1 if res["alert"] else 0)
                    if c == "B" and res.get("stopped_futility"):
                        futility[a][0 if label == "human" else 1] += 1
                    if detail:
                        rec["alerts"].setdefault(c, {})[a] = bool(res["alert"])
                        rec["tokens"].setdefault(c, {})[a] = res["tokens_inspected"]
                        if c == "B":
                            rec["futility"][a] = bool(res.get("stopped_futility"))
                        rec.setdefault("early", {}).setdefault(c, {})[a] = (
                            res["actions_executed"] < full_steps)
        if detail:
            detail_out.append(rec)
        if progress:
            progress(i + 1, n)
    rows = [{
        "construction": c,
        "alpha": a,
        "n": n,
        "human_alerts": counts[c][a][0],
        "ai_alerts": counts[c][a][1],
        "human_rate": round(counts[c][a][0] / n, 4),
        "ai_rate": round(counts[c][a][1] / n, 4),
        # cost accumulators sum over 2n screened docs (human + ai), so the
        # divisor is 2n: per-document means, never exceeding the budget
        "mean_tokens_inspected": round(cost[c][a][0] / (2 * n)),
        "mean_savings_pct": round(cost[c][a][1] / (2 * n)),
        "mean_actions": round(acts[c][a] / (2 * n), 2),
        "early_decision_human": early[c][a][0],
        "early_decision_ai": early[c][a][1],
        "early_decision_human_rate": round(early[c][a][0] / n, 4),
        "early_decision_ai_rate": round(early[c][a][1] / n, 4),
        "futility_human": futility[a][0] if c == "B" else None,
        "futility_ai": futility[a][1] if c == "B" else None,
        "futility_human_rate": round(futility[a][0] / n, 4) if c == "B" else None,
        "futility_ai_rate": round(futility[a][1] / n, 4) if c == "B" else None,
    } for c in constructions for a in alphas]
    paired = []
    if "A" in constructions and "B" in constructions:
        for a in alphas:
            ci = paired_difference_ci(paired_ai["A"][a], paired_ai["B"][a])
            paired.append({
                "alpha": a,
                "n": n,
                "power_diff_A_minus_B": round(ci["diff"], 4),
                "power_diff_lower_95": round(ci["lower_95"], 4),
                "cost_ratio_B_over_A": (round((cost["B"][a][0] / n) / max(1, cost["A"][a][0] / n), 4)
                                        if "B" in constructions else None),
            })
    return rows, paired, detail_out


def benchmark(bundle, n, corpus, alphas=(0.01, 0.05, 0.1),
              constructions=("A", "B"), progress=None):
    """Evaluate empirical alert rates on human vs AI-generated documents.
    Documents overlapping the calibration corpus are skipped."""
    def sample(corpus_name):
        offset = (bundle.m + bundle.n_dev) if corpus_name == CORPUS else 0
        return (data.get_human_docs(corpus_name, "falcon7", CACHE_DIR, progress=progress)[offset:offset + n],
                data.get_ai_docs(corpus_name, "falcon7", CACHE_DIR, progress=progress)[offset:offset + n])

    if corpus == "binoc-all":
        human, ai = [], []
        for c in data.CORPORA:
            h, a = sample(c)
            human += h
            ai += a
    else:
        human, ai = sample(corpus)
    n = min(len(human), len(ai), n)
    if n < 5:
        raise RuntimeError(f"only {n} usable documents for benchmark")
    rows, paired, _detail = evaluate_cells(bundle, human[:n], ai[:n], constructions, alphas,
                                  progress=progress)
    return {"corpus": corpus, "n": n, "rows": rows, "paired": paired,
            "matched": corpus == bundle.corpus,
            "calibration_corpus": bundle.corpus}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save_bundle(bundle, path):
    bundle.save(path)


def load_bundle(path):
    b = CalibrationBundle.load(path)
    if not b.selfcheck():
        print(f"[conformal] WARNING: {path} fails the complete-path self-check — "
              f"stored maxima disagree with the frozen route; rebuild the calibration",
              file=sys.stderr, flush=True)
    return b


def cache_path(model_slug, m, n_dev, corpus="imdb", binoculars=False, fastdetect=False):
    os.makedirs("calibration_cache", exist_ok=True)
    if corpus == "imdb" and not binoculars and not fastdetect:
        return os.path.join("calibration_cache", f"cal_{model_slug}_{m}_{n_dev}.npz")
    parts = []
    if binoculars:
        parts.append("bino")
    if fastdetect:
        parts.append("fast")
    tag = "_".join(parts) if parts else "base"
    return os.path.join("calibration_cache",
                        f"cal_{model_slug}_{corpus}_{tag}_{m}_{n_dev}.npz")


def cache_compatible(path):
    """True when the cached calibration matches the current configuration.
    Legacy caches (predating the corpus/detector metadata) are accepted for
    the default imdb configuration."""
    meta_path = path + ".json"
    if not os.path.exists(meta_path):
        return True  # legacy cache; accept it
    with open(meta_path) as f:
        meta = json.load(f)
    return (meta.get("base_model") == BASE_MODEL
            and meta.get("corpus", "imdb") == CORPUS
            and meta.get("detectors", DETECTORS) == active_detectors()
            and meta.get("slop_version", 0) == slop.SLOP_VERSION
            and meta.get("std_version", 0) == 2)


def main():
    global BASE_MODEL, DEVICE, CACHE_DIR, CORPUS, USE_BINOCULARS, USE_FASTDETECT
    ap = argparse.ArgumentParser(description="Project Burnwatch — adaptive within-document AI-text screening")
    ap.add_argument("--calibrate", action="store_true", help="build calibration cache")
    ap.add_argument("--screen", type=str, default=None, help="text file to screen")
    ap.add_argument("--replay", type=str, default=None,
                    help="text file: verify the two constructions are deterministic and "
                         "consistent with the frozen calibration (replay harness)")
    ap.add_argument("--construction", choices=["A", "B"], default="B")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--m", type=int, default=200, help="calibration documents")
    ap.add_argument("--n-dev", type=int, default=50, help="development documents")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--corpus", default=CORPUS,
                    help="calibration corpus: imdb | ccnews | cnn | pubmed | binoc-all")
    ap.add_argument("--binoculars", action="store_true",
                    help="add the contrasting-model detector to the family")
    ap.add_argument("--fastdetect", action="store_true",
                    help="add the Fast-DetectGPT conditional-curvature detector to the family")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    args = ap.parse_args()

    BASE_MODEL, DEVICE, CACHE_DIR = args.base_model, get_device(args.device), args.cache_dir
    CORPUS = args.corpus
    USE_BINOCULARS = args.binoculars
    USE_FASTDETECT = args.fastdetect
    os.environ["XDG_CACHE_HOME"] = CACHE_DIR
    print(f"[conformal] model={BASE_MODEL} corpus={CORPUS} device={DEVICE} "
          f"binoculars={USE_BINOCULARS} fastdetect={USE_FASTDETECT}", file=sys.stderr, flush=True)

    path = cache_path(BASE_MODEL.replace("/", "_"), args.m, args.n_dev,
                      args.corpus, args.binoculars, args.fastdetect)
    if args.calibrate or not os.path.exists(path) or not cache_compatible(path):
        print("[conformal] building calibration...", file=sys.stderr, flush=True)
        bundle = build_calibration(args.m, args.n_dev)
        save_bundle(bundle, path)
        print(f"[conformal] saved {path}", file=sys.stderr, flush=True)
    else:
        bundle = load_bundle(path)
        print(f"[conformal] loaded {path}", file=sys.stderr, flush=True)

    if args.screen:
        text = open(args.screen).read()
        if args.construction == "A":
            res = screen_a(text, bundle, args.alpha)
        else:
            res = screen_b(text, bundle, args.alpha)
        print(json.dumps(res))

    if args.replay:
        text = open(args.replay).read()
        checks = {"fingerprint": bundle.fingerprint, "selfcheck": bundle.selfcheck()}
        # determinism: identical p-values on two independent runs
        pre1 = action_scores_for(text)
        pre2 = action_scores_for(text)
        r1 = (screen_a(text, bundle, args.alpha, pre=pre1), screen_b(text, bundle, args.alpha, pre=pre1))
        r2 = (screen_a(text, bundle, args.alpha, pre=pre2), screen_b(text, bundle, args.alpha, pre=pre2))
        pa1 = [s["p"] for s in r1[0]["steps"]]
        pa2 = [s["p"] for s in r2[0]["steps"]]
        pb1 = [s["p"] for s in r1[1]["steps"]]
        pb2 = [s["p"] for s in r2[1]["steps"]]
        checks["deterministic"] = (pa1 == pa2) and (pb1 == pb2)
        # benchmark-path scoring is byte-identical to the screen path by construction:
        # every path funnels through action_scores_for; verify equality
        checks["same_score_path"] = all(np.array_equal(v, w) for v, w in zip(pre1.values(), pre2.values()))
        checks["alert_A"] = r1[0]["alert"]
        checks["alert_B"] = r1[1]["alert"]
        checks["tokens_inspected_B"] = r1[1]["tokens_inspected"]
        checks["tokens_saved_pct_B"] = r1[1]["tokens_saved_pct"]
        checks["m"] = bundle.m
        checks["route"] = DEFAULT_ROUTE
        print(json.dumps(checks))


if __name__ == "__main__":
    main()
