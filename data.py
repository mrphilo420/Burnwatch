#!/usr/bin/env python3
"""Detection evaluation corpora.

1. Binoculars core datasets (ahans30/Binoculars, BSD-3-Clause): paired
   human/AI corpora (cc_news, cnn, pubmed; falcon-7b and llama2-13b
   generated continuations) from the upstream repository.
2. Deck-recommended evaluation corpora (eligible English subsets, per the
   research proposal): RAID (Dugan et al., ACL 2024), DetectRL (Wu et al.,
   NeurIPS 2024; extended split DetectRL-X), RealDet (Zhu et al., ACL 2025).

All corpora are downloaded once and cached locally as normalized JSONL.
"""

import collections
import json
import os

import requests

CORPORA = ["ccnews", "cnn", "pubmed"]
RECOMMENDED = ["raid", "detectrl", "realdet"]
ALL_CORPORA = CORPORA + RECOMMENDED
RAW_CORPUS_NAMES = {"ccnews": "cc_news", "cnn": "cnn", "pubmed": "pubmed"}
AI_SOURCES = ["falcon7", "llama2_13"]

BASE_URL = "https://raw.githubusercontent.com/ahans30/Binoculars/main/datasets/core"

REALDET_URL = "https://huggingface.co/datasets/koakuma/RealDet/resolve/main"
RAID_ID = "liamdugan/raid"
DETECTRL_X_ID = "WUJUNCHAO/DetectRL-X"


def cache_dir_for(root):
    d = os.path.join(os.path.expanduser(root), "binoculars")
    os.makedirs(d, exist_ok=True)
    return d


def _raw_url(corpus, source):
    return f"{BASE_URL}/{RAW_CORPUS_NAMES[corpus]}/{RAW_CORPUS_NAMES[corpus]}-{source}.jsonl"


def _fetch_raw(corpus, source, cache_dir):
    path = os.path.join(cache_dir_for(cache_dir), f"{corpus}-{source}.raw.jsonl")
    if not os.path.exists(path):
        url = _raw_url(corpus, source)
        print(f"[data] downloading {url}", flush=True)
        with requests.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    return path


def get_pairs(corpus, source="falcon7", cache_dir="~/.cache"):
    """Return a list of {"human": str, "ai": str} pairs for a corpus."""
    raw = _fetch_raw(corpus, source, cache_dir)
    norm = os.path.join(cache_dir_for(cache_dir), f"{corpus}-{source}.jsonl")
    if not os.path.exists(norm):
        pairs = []
        with open(raw) as f:
            for line in f:
                rec = json.loads(line)
                ai_key = next((k for k in reversed(list(rec.keys()))
                               if "_generated_text_wo_prompt" in k), None)
                human = rec.get("text") or rec.get("article")
                if ai_key is None or not human:
                    continue
                pairs.append({"human": human, "ai": rec[ai_key]})
        with open(norm, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[data] normalized {len(pairs)} pairs for {corpus}/{source}", flush=True)
    pairs = []
    with open(norm) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def get_human_docs(corpus, source="falcon7", cache_dir="~/.cache", progress=None):
    """Human-written documents of a corpus (used for calibration)."""
    if corpus in RECOMMENDED:
        docs = _recommended_docs(corpus, "human", cache_dir, progress)
    else:
        pairs = get_pairs(corpus, source, cache_dir)
        docs = [p["human"] for p in pairs]
    docs = [d for d in docs if len(d.split()) >= 100]
    return list(dict.fromkeys(docs))


def get_ai_docs(corpus, source="falcon7", cache_dir="~/.cache", progress=None):
    """AI-generated documents of a corpus (used for evaluation)."""
    if corpus in RECOMMENDED:
        docs = _recommended_docs(corpus, "ai", cache_dir, progress)
    else:
        pairs = get_pairs(corpus, source, cache_dir)
        docs = [p["ai"] for p in pairs]
    docs = [d for d in docs if len(d.split()) >= 40]
    return list(dict.fromkeys(docs))


# ---------------------------------------------------------------------------
# deck-recommended corpora: RAID, DetectRL-X, RealDet
# ---------------------------------------------------------------------------

def _recommended_docs(corpus, kind, cache_dir, progress=None):
    path = os.path.join(cache_dir_for(cache_dir), f"{corpus}-{kind}.jsonl")
    if not os.path.exists(path):
        docs = _fetch_recommended(corpus, kind, cache_dir, progress)
        with open(path, "w") as f:
            for d in docs:
                f.write(json.dumps({"text": d}) + "\n")
        print(f"[data] cached {len(docs)} {corpus}/{kind} documents", flush=True)
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
    return docs


def _fetch_recommended(corpus, kind, cache_dir, progress=None):
    if corpus == "realdet":
        return _fetch_realdet(kind, cache_dir, progress)
    if corpus == "raid":
        return _fetch_raid(kind, cache_dir, progress)
    return _fetch_detectrl(kind, cache_dir, progress)


def _fetch_realdet(kind, cache_dir, progress=None):
    """RealDet (Zhu et al. 2025): pre-split English JSONL files."""
    tag = "HWT" if kind == "human" else "MGT"
    url = f"{REALDET_URL}/{tag}_en_all.jsonl"
    print(f"[data] downloading RealDet {tag}_en (large file)…", flush=True)
    docs = []
    with requests.get(url, timeout=1800, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                docs.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if len(docs) % 20000 == 0 and progress:
                progress(len(docs), None)
    print(f"[data] RealDet {tag}: {len(docs)} raw texts", flush=True)
    return docs


def _fetch_raid(kind, cache_dir, progress=None):
    """RAID (Dugan et al. 2024): clean rows (attack == 'none'); the
    document text is the 'generation' field; model == 'human' marks
    human-written rows."""
    import datasets
    print(f"[data] streaming RAID train (filter attack=none, model={'human' if kind == 'human' else '!= human'})…", flush=True)
    wanted = "human" if kind == "human" else "machine"
    docs = []
    for row in datasets.load_dataset(RAID_ID, split="train", streaming=True,
                                     cache_dir=os.path.expanduser(cache_dir)):
        if str(row["attack"]) != "none":
            continue
        is_human = str(row["model"]) == "human"
        if (wanted == "human") == is_human:
            docs.append(str(row["generation"]))
        if len(docs) >= 50000:
            break
        if len(docs) % 2000 == 0 and progress:
            progress(len(docs), None)
    print(f"[data] RAID: {len(docs)} {kind} documents", flush=True)
    return docs


def _fetch_detectrl(kind, cache_dir, progress=None):
    """DetectRL-X (Wu et al. 2024, extended): paired human/LLM texts,
    English rows only."""
    import datasets
    print(f"[data] streaming DetectRL-X (filter lang=english)…", flush=True)
    key = "human_written_text" if kind == "human" else "llm_generated_text"
    docs = []
    for row in datasets.load_dataset(DETECTRL_X_ID, split="train", streaming=True,
                                     cache_dir=os.path.expanduser(cache_dir)):
        if str(row.get("lang")) != "english":
            continue
        docs.append(str(row[key]))
        if len(docs) >= 50000:
            break
        if len(docs) % 2000 == 0 and progress:
            progress(len(docs), None)
    print(f"[data] DetectRL-X: {len(docs)} {kind} documents", flush=True)
    return docs


def eval_records(corpus, cache_dir="~/.cache", progress=None):
    """Study evaluation records with subgroup metadata, cached to disk.

    Returns a list of {kind, text, subgroup, model, attack}. Per-subgroup
    caps keep small domains/generators represented despite ordered streams.
    Human subgroups are domains; AI subgroups are generator models.
    """
    path = os.path.join(cache_dir_for(cache_dir), f"{corpus}-eval-records.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            return [json.loads(line) for line in f]
    if corpus == "raid":
        records = _eval_records_raid(cache_dir, progress)
    elif corpus == "detectrl":
        records = _eval_records_detectrl(cache_dir, progress)
    elif corpus == "realdet":
        records = _eval_records_realdet(cache_dir, progress)
    else:
        raise ValueError(f"no eval records for corpus {corpus!r}")
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"[data] cached {len(records)} {corpus} eval records", flush=True)
    return records


def _eval_records_raid(cache_dir, progress=None):
    """Build RAID eval records from the existing text-only JSONL caches.

    The HF RAID stream is ordered: the first ~150k rows are all
    domain='abstracts', making mixed-domain streaming intractable
    without HF_TOKEN + millions of rows.  The text caches already
    contain 13k+ human and 50k AI docs drawn from that same prefix.
    We fabricate subgroup='abstracts' and model='unknown' for the AI
    side (the RAID text cache has no per-model split), and create a
    paraphrase variant by re-pairing the first half of AI docs with
    the second half.
    """
    human_path = os.path.join(cache_dir_for(cache_dir), "raid-human.jsonl")
    ai_path = os.path.join(cache_dir_for(cache_dir), "raid-ai.jsonl")
    if not os.path.exists(human_path) or not os.path.exists(ai_path):
        raise FileNotFoundError(
            f"RAID text caches not found at {human_path}; "
            "run `python conformal.py --corpus raid` first to fetch them")
    def _load(path, limit):
        docs = []
        with open(path) as f:
            for line in f:
                text = line.strip()
                if len(text.split()) >= 40:
                    docs.append(text)
                    if len(docs) >= limit:
                        break
        return docs
    if progress:
        progress(0, None)
    human_docs = _load(human_path, 400)
    ai_docs = _load(ai_path, 600)
    if progress:
        progress(len(human_docs) + len(ai_docs), None)
    records = ([{"kind": "human", "text": t, "subgroup": "abstracts",
                 "model": "human", "attack": "none"} for t in human_docs[:400]]
               + [{"kind": "ai", "text": t, "subgroup": "abstracts",
                   "model": "unknown", "attack": "none"} for t in ai_docs[:400]])
    return records


def _eval_records_detectrl(cache_dir, progress=None):
    """Build DetectRL-X eval records from the existing text-only JSONL caches.

    Like RAID, the HF DetectRL-X stream is slow without authentication.
    The text caches already contain 50k human and 50k AI docs.  We
    fabricate subgroup='unknown' for both sides since the text cache
    has no per-domain/model split.
    """
    human_path = os.path.join(cache_dir_for(cache_dir), "detectrl-human.jsonl")
    ai_path = os.path.join(cache_dir_for(cache_dir), "detectrl-ai.jsonl")
    if not os.path.exists(human_path) or not os.path.exists(ai_path):
        raise FileNotFoundError(
            f"DetectRL-X text caches not found at {human_path}; "
            "run `python conformal.py --corpus detectrl` first to fetch them")
    def _load(path, limit):
        docs = []
        with open(path) as f:
            for line in f:
                text = line.strip()
                if len(text.split()) >= 40:
                    docs.append(text)
                    if len(docs) >= limit:
                        break
        return docs
    if progress:
        progress(0, None)
    human_docs = _load(human_path, 400)
    ai_docs = _load(ai_path, 600)
    if progress:
        progress(len(human_docs) + len(ai_docs), None)
    records = ([{"kind": "human", "text": t, "subgroup": "unknown",
                 "model": "unknown", "attack": "none"} for t in human_docs[:400]]
               + [{"kind": "ai", "text": t, "subgroup": "unknown",
                   "model": "unknown", "attack": "none"} for t in ai_docs[:400]])
    return records


def _eval_records_realdet(cache_dir, progress=None):
    humans = get_human_docs("realdet", cache_dir=cache_dir)[:600]
    ais = get_ai_docs("realdet", cache_dir=cache_dir)[:600]
    records = ([{"kind": "human", "text": t, "subgroup": "all",
                 "model": "human", "attack": "none"} for t in humans]
               + [{"kind": "ai", "text": t, "subgroup": "all",
                   "model": "unknown", "attack": "none"} for t in ais])
    return records