#!/usr/bin/env python3
"""Conformal within-document AI-text screening web interface.

Implements the two conformal constructions (registered family + union bound;
complete-path calibration with early stopping) per the accompanying theory
manuscript. Calibration uses human-written documents and is cached to disk.
"""

import argparse
import os
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

import conformal as C
import data
import docutils

app = Flask(__name__)

CFG = {}
bundles = {}                      # {(model, corpus, binoculars): CalibrationBundle}
bundle_lock = threading.Lock()
calibration_state = {"status": "missing", "detail": "", "done": 0, "total": 0,
                     "model": None, "corpus": None}


def _parse_alpha(value):
    try:
        alpha = float(value)
    except (TypeError, ValueError):
        raise ValueError("alpha must be a number in (0, 1).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    return alpha


def _parse_count(value, name="n", lo=5, hi=50):
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer between {lo} and {hi}.")
    if not lo <= n <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}.")
    return n

MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "Qwen/Qwen2.5-1.5B",
    "EleutherAI/pythia-1.4b",
    "HuggingFaceTB/SmolLM2-1.7B",
]

job_queue = []
queue_cond = threading.Condition()
jobs = {}

MIN_TEXT_WORDS = 20
MAX_TEXT_WORDS = 2000
JOB_TIMEOUT = 3600


def active_job_detail():
    for j in jobs.values():
        if j["status"] not in ("done", "error"):
            return f"{j.get('kind', 'job')}: {j.get('detail') or j['status']}"
    return None


def effective_binoculars(model, requested):
    """The contrasting detector needs observer and performer to share a
    vocabulary (GPT-2 family), and observer != performer."""
    return bool(requested) and model in C.BINO_COMPATIBLE and model != C.PERFORMER_MODEL


def effective_fastdetect(model, requested):
    """Fast-DetectGPT needs the scorer to share the reference model's
    tokenizer (same constraint class as the contrasting detector)."""
    return bool(requested) and model in C.FAST_COMPATIBLE


def build_calibration_bg(model, corpus, progress_target):
    calibration_state.update(status="loading", model=model, corpus=corpus,
                             detail=f"Loading {model}…", done=0, total=0)

    def progress(done, total):
        if total is None:
            calibration_state.update(status="downloading", done=done, total=0,
                                     detail=f"Fetching {corpus} documents: {done} so far…")
        else:
            calibration_state.update(status="scoring", done=done, total=total,
                                     detail=f"Scoring documents {done}/{total}")

    try:
        b = C.build_calibration(CFG["m"], CFG["n_dev"], progress=progress)
        path = C.cache_path(model.replace("/", "_"), CFG["m"], CFG["n_dev"],
                            corpus, C.USE_BINOCULARS, C.USE_FASTDETECT)
        C.save_bundle(b, path)
        with bundle_lock:
            bundles[(model, corpus, C.USE_BINOCULARS, C.USE_FASTDETECT)] = b
        calibration_state.update(status="ready",
                                 detail=f"Ready: {CFG['m']} calibration + {CFG['n_dev']} development documents")
        print(f"[app] calibration ready: {path}", flush=True)
        return b
    except Exception as exc:
        calibration_state.update(status="error", detail=str(exc))
        print(f"[app] calibration FAILED\n{traceback.format_exc()}", flush=True)
        raise


def ensure_bundle(model, corpus, binoculars, fastdetect, build=True):
    global bundles
    with bundle_lock:
        if (model, corpus, binoculars, fastdetect) in bundles:
            return bundles[(model, corpus, binoculars, fastdetect)]
    path = C.cache_path(model.replace("/", "_"), CFG["m"], CFG["n_dev"], corpus, binoculars, fastdetect)
    if os.path.exists(path) and C.cache_compatible(path):
        b = C.load_bundle(path)
        with bundle_lock:
            bundles[(model, corpus, binoculars, fastdetect)] = b
        calibration_state.update(status="ready", model=model, corpus=corpus,
                                 detail=f"Loaded from cache: {b.m} calibration documents")
        return b
    if build:
        return build_calibration_bg(model, corpus, None)
    raise ConnectionError(
        f"Calibration for {model}/{corpus} is not built yet. Select the model in the Screen tab "
        f"to build it (first use takes a few minutes).")


def verdict_for(res):
    if res["alert"]:
        return {
            "label": "Alert: flag for review",
            "level": "alert",
            "note": "The document departed from the human reference distribution at the requested false-alert level. This flags the document for review; it does not by itself establish AI authorship or misconduct.",
        }
    return {
        "label": "No alert: insufficient evidence",
        "level": "clear",
        "note": "No executed action crossed its conformal boundary. The document retains an insufficient-evidence status; non-rejection does not certify human authorship.",
    }


def run_file_screen(job):
    """Extract text from an uploaded file, then run the standard screen."""
    ok, err = docutils.validate(job["file_name"], job["file_bytes"])
    if not ok:
        raise ValueError(err)
    job["detail"] = f"Extracting text from {job['file_name']}…"
    text = docutils.extract_text(job["file_name"], job["file_bytes"])
    text = " ".join(text.split())
    n_words = len(text.split())
    truncated = n_words > MAX_TEXT_WORDS
    if truncated:
        text = " ".join(text.split()[:MAX_TEXT_WORDS])
        n_words = MAX_TEXT_WORDS
    if n_words < MIN_TEXT_WORDS:
        raise ValueError(
            f"'{job['file_name']}' yielded only {n_words} words of text "
            f"(need at least {MIN_TEXT_WORDS}). It may be a scanned image or a non-text layout.")

    b = verified_bundle(build=False)
    if C._model is None:
        job["detail"] = f"Loading the scoring model ({C.BASE_MODEL}) — first request after a switch, ~10s…"
    else:
        job["detail"] = f"Scoring {job['file_name']} ({n_words} words)…"

    slop_report = C.slop.scan(text)
    pre = C.action_scores_for(text)
    ra = C.screen_a(text, b, job["alpha"], pre=pre)
    ra["verdict"] = verdict_for(ra)
    rb = C.screen_b(text, b, job["alpha"], pre=pre)
    rb["verdict"] = verdict_for(rb)
    res = {
        "construction": "both",
        "alpha": job["alpha"],
        "m": b.m,
        "A": ra,
        "B": rb,
        "slop": slop_report,
        "text": text,
        "file": {"name": job["file_name"], "n_words": n_words, "truncated": truncated},
        "elapsed_seconds": round(time.time() - job["created"], 1),
    }
    job["result"] = res


def verified_bundle(build=False):
    """Load the bundle for the current config and refuse to score against a
    mismatched calibration (the benchmark-vs-screen class of bug)."""
    b = ensure_bundle(C.BASE_MODEL, CFG["corpus"], C.USE_BINOCULARS, C.USE_FASTDETECT, build=build)
    if b.fingerprint != C.config_fingerprint():
        raise ConnectionError(
            "Calibration bundle does not match the current configuration "
            "(score map or route changed after it was built). Reselect the model/corpus "
            "to rebuild it before screening.")
    return b


def run_screen(job):
    b = verified_bundle(build=False)
    if C._model is None:
        job["detail"] = f"Loading the scoring model ({C.BASE_MODEL}) — first request after a switch, ~10s…"
    else:
        job["detail"] = "Scoring document under the generative model…"
    slop_report = C.slop.scan(job["text"])
    if job["construction"] == "both":
        pre = C.action_scores_for(job["text"])
        ra = C.screen_a(job["text"], b, job["alpha"], pre=pre)
        ra["verdict"] = verdict_for(ra)
        rb = C.screen_b(job["text"], b, job["alpha"], pre=pre)
        rb["verdict"] = verdict_for(rb)
        res = {"construction": "both", "alpha": job["alpha"], "m": b.m, "A": ra, "B": rb,
               "slop": slop_report}
    elif job["construction"] == "A":
        res = C.screen_a(job["text"], b, job["alpha"])
        res["verdict"] = verdict_for(res)
        res["slop"] = slop_report
    else:
        res = C.screen_b(job["text"], b, job["alpha"])
        res["verdict"] = verdict_for(res)
        res["slop"] = slop_report
    if job["construction"] == "both":
        res["text"] = job["text"]
    res["elapsed_seconds"] = round(time.time() - job["created"], 1)
    job["result"] = res


def run_benchmark(job):
    b = verified_bundle(build=False)

    def progress(done, total):
        if total is None:
            job["detail"] = f"Fetching {job['corpus']} documents: {done} so far…"
        else:
            job["detail"] = f"Benchmarking documents {done}/{total}"

    res = C.benchmark(b, job["n"], job["corpus"], progress=progress)
    res["verdict"] = "benchmark"
    res["elapsed_seconds"] = round(time.time() - job["created"], 1)
    res["model"] = C.BASE_MODEL
    job["result"] = res


def run_model_switch(job):
    model = job["model"]
    requested_bino = job["binoculars"]
    requested_fast = job.get("fastdetect", False)
    bino = effective_binoculars(model, requested_bino)
    fast = effective_fastdetect(model, requested_fast)
    prev_model, prev_bino, prev_fast = C.BASE_MODEL, C.USE_BINOCULARS, C.USE_FASTDETECT
    C.set_model(model)
    C.USE_BINOCULARS = bino
    C.USE_FASTDETECT = fast
    job["detail"] = f"Switching scoring model to {model}…"
    try:
        b = ensure_bundle(model, CFG["corpus"], bino, fast, build=True)
    except Exception:
        # roll back globals so a failed switch cannot corrupt the runtime state
        C.set_model(prev_model)
        C.USE_BINOCULARS = prev_bino
        C.USE_FASTDETECT = prev_fast
        raise
    job["result"] = {
        "model": model,
        "corpus": CFG["corpus"],
        "binoculars": bino,
        "fastdetect": fast,
        "m": b.m,
        "n_actions": b.n_actions,
        "resolution": C.resolution_table(b.m),
        "elapsed_seconds": round(time.time() - job["created"], 1),
    }


def worker():
    while True:
        with queue_cond:
            while not job_queue:
                queue_cond.wait()
            job = job_queue.pop(0)
        jobs[job["id"]] = job
        try:
            if job["kind"] == "screen":
                run_screen(job)
            elif job["kind"] == "file":
                run_file_screen(job)
            elif job["kind"] == "benchmark":
                run_benchmark(job)
            elif job["kind"] == "model":
                run_model_switch(job)
            job["status"] = "done"
            print(f"[job:{job['id']}] {job['kind']} done in {job['result']['elapsed_seconds']}s", flush=True)
        except Exception as exc:
            job["status"] = "error"
            job["error"] = str(exc)
            print(f"[job:{job['id']}] FAILED\n{traceback.format_exc()}", flush=True)
        with queue_cond:
            if len(jobs) > 20:
                stale = [jid for jid, j in jobs.items()
                         if j["status"] in ("done", "error") and jid != job["id"]]
                for jid in stale[: len(jobs) - 20]:
                    del jobs[jid]


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    cal = dict(calibration_state)
    cal["m"] = CFG["m"]
    cal["n_dev"] = CFG["n_dev"]
    cal["model"] = C.BASE_MODEL
    cal["corpus"] = C.CORPUS
    cal["budgets"] = C.BUDGETS
    cal["detectors"] = C.active_detectors()
    cal["route"] = C.DEFAULT_ROUTE
    cal["binoculars"] = C.USE_BINOCULARS
    cal["fastdetect"] = C.USE_FASTDETECT
    cal["n_actions"] = len(C.active_detectors()) * len(C.BUDGETS)
    cal["resolution"] = C.resolution_table(CFG["m"])
    cal["models"] = MODELS
    cal["models_ready"] = {
        m: os.path.exists(C.cache_path(m.replace("/", "_"), CFG["m"], CFG["n_dev"],
                                       CFG["corpus"], effective_binoculars(m, True),
                                       effective_fastdetect(m, True)))
        for m in MODELS
    }
    cal["corpora"] = data.ALL_CORPORA
    _active = bundles.get((C.BASE_MODEL, CFG["corpus"], C.USE_BINOCULARS, C.USE_FASTDETECT))
    cal["fingerprint"] = _active.fingerprint if _active else None
    cal["corpora_ready"] = {
        c: os.path.exists(C.cache_path(C.BASE_MODEL.replace("/", "_"), CFG["m"], CFG["n_dev"],
                                       c, C.USE_BINOCULARS, C.USE_FASTDETECT))
        for c in data.ALL_CORPORA + ["imdb"]
    }
    return jsonify({
        "engine": "conformal",
        "device": C.DEVICE,
        "model": C.BASE_MODEL,
        "corpus": C.CORPUS,
        "calibration": cal,
        "busy": any(j["status"] not in ("done", "error") for j in jobs.values()),
    })


@app.post("/api/detect")
def api_detect():
    payload = request.get_json(force=True)
    text = (payload.get("text") or "").strip()
    n_words = len(text.split())
    if n_words < MIN_TEXT_WORDS:
        return jsonify({"error": f"Text too short: please provide at least {MIN_TEXT_WORDS} words (currently {n_words})."}), 400
    if n_words > MAX_TEXT_WORDS:
        return jsonify({"error": f"Text too long: maximum {MAX_TEXT_WORDS} words (currently {n_words})."}), 400

    construction = payload.get("construction") or "both"
    if construction not in ("A", "B", "both"):
        return jsonify({"error": "construction must be 'A', 'B', or 'both'."}), 400
    try:
        alpha = _parse_alpha(payload.get("alpha", 0.05))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with queue_cond:
        if any(j["status"] not in ("done", "error") for j in jobs.values()):
            return jsonify({"error": "Another screening is already running. Please wait."}), 409

    if calibration_state["status"] in ("loading", "scoring", "downloading"):
        return jsonify({"error": f"Calibration in progress: {calibration_state['detail']}"}), 409
    if calibration_state["status"] == "error":
        return jsonify({"error": f"Calibration failed: {calibration_state['detail']}"}), 503
    if calibration_state["status"] != "ready":
        return jsonify({"error": "Calibration not ready."}), 409

    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": "screen",
        "text": text,
        "construction": construction,
        "alpha": alpha,
        "status": "queued",
        "detail": "Queued…",
        "created": time.time(),
    }
    with queue_cond:
        job_queue.append(job)
        queue_cond.notify()
    return jsonify({"job_id": job["id"]})


@app.post("/api/benchmark")
def api_benchmark():
    payload = request.get_json(force=True)
    corpus = (payload.get("corpus") or "").strip()
    if corpus not in data.ALL_CORPORA + ["binoc-all"]:
        return jsonify({"error": f"corpus must be one of {', '.join(data.ALL_CORPORA + ['binoc-all'])}."}), 400
    try:
        n = _parse_count(payload.get("n", 20))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with queue_cond:
        if any(j["status"] not in ("done", "error") for j in jobs.values()):
            return jsonify({"error": "Another job is already running. Please wait.",
                            "active_job": active_job_detail()}), 409

    if calibration_state["status"] in ("loading", "scoring", "downloading"):
        return jsonify({"error": f"Calibration in progress: {calibration_state['detail']}"}), 409
    if calibration_state["status"] == "error":
        return jsonify({"error": f"Calibration failed: {calibration_state['detail']}"}), 503
    if calibration_state["status"] != "ready":
        return jsonify({"error": "Calibration not ready."}), 409

    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": "benchmark",
        "corpus": corpus,
        "n": n,
        "status": "queued",
        "detail": "Queued…",
        "created": time.time(),
    }
    with queue_cond:
        job_queue.append(job)
        queue_cond.notify()
    return jsonify({"job_id": job["id"]})


@app.post("/api/model")
def api_model():
    payload = request.get_json(force=True)
    model = (payload.get("model") or "").strip()
    if model not in MODELS:
        return jsonify({"error": f"Unknown model '{model}'."}), 400
    binoculars = bool(payload.get("binoculars", False))
    fastdetect = bool(payload.get("fastdetect", False))
    if (model == C.BASE_MODEL and binoculars == C.USE_BINOCULARS
            and fastdetect == C.USE_FASTDETECT):
        return jsonify({"job_id": "noop", "status": "done", "result": {
            "model": model, "corpus": CFG["corpus"], "binoculars": C.USE_BINOCULARS,
            "fastdetect": C.USE_FASTDETECT,
            "m": CFG["m"], "n_actions": len(C.active_detectors()) * len(C.BUDGETS)}})

    with queue_cond:
        if any(j["status"] not in ("done", "error") for j in jobs.values()):
            return jsonify({"error": "Another job is already running. Please wait.",
                            "active_job": active_job_detail()}), 409

    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": "model",
        "model": model,
        "binoculars": binoculars,
        "fastdetect": fastdetect,
        "status": "queued",
        "detail": "Queued…",
        "created": time.time(),
    }
    with queue_cond:
        job_queue.append(job)
        queue_cond.notify()
    return jsonify({"job_id": job["id"]})


@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file received — the field must be named 'file'."}), 400
    f = request.files["file"]
    name = (f.filename or "").strip()
    data = f.read()
    ok, err = docutils.validate(name, data)
    if not ok:
        return jsonify({"error": err}), 400
    try:
        alpha = _parse_alpha(request.form.get("alpha", 0.05))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    with queue_cond:
        if any(j["status"] not in ("done", "error") for j in jobs.values()):
            return jsonify({"error": "Another job is already running — please wait.",
                            "active_job": active_job_detail()}), 409
    if calibration_state["status"] in ("loading", "scoring", "downloading"):
        return jsonify({"error": f"Calibration in progress: {calibration_state['detail']}"}), 409
    if calibration_state["status"] == "error":
        return jsonify({"error": f"Calibration failed: {calibration_state['detail']}"}), 503
    if calibration_state["status"] != "ready":
        return jsonify({"error": "Calibration not ready."}), 409

    job = {
        "id": uuid.uuid4().hex[:8],
        "kind": "file",
        "file_name": name,
        "file_bytes": data,
        "alpha": alpha,
        "status": "queued",
        "detail": "Queued…",
        "created": time.time(),
    }
    with queue_cond:
        job_queue.append(job)
        queue_cond.notify()
    return jsonify({"job_id": job["id"]})


@app.get("/api/job/<job_id>")
def api_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    resp = {"status": job["status"], "detail": job.get("detail"), "created": job.get("created")}
    resp["elapsed"] = round(time.time() - job.get("created", time.time()))
    if job["status"] not in ("done", "error") and resp["elapsed"] > JOB_TIMEOUT:
        resp["status"] = "error"
        resp["error"] = (f"Job timed out after {resp['elapsed']}s. The worker is still running in the "
                         f"background; check /tmp/screen-web.log or retry after it finishes.")
        return jsonify(resp)
    if job["status"] == "done":
        resp["result"] = job["result"]
    if job["status"] == "error":
        resp["error"] = job.get("error")
    return jsonify(resp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conformal AI-text screening web interface")
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--m", type=int, default=260, help="calibration documents")
    parser.add_argument("--n-dev", type=int, default=50, help="development documents")
    parser.add_argument("--base-model", default="gpt2", help="initial scoring model")
    parser.add_argument("--corpus", default="realdet",
                        help="calibration corpus: imdb | ccnews | cnn | pubmed | binoc-all | raid | detectrl | realdet")
    parser.add_argument("--binoculars", action="store_true",
                        help="add the contrasting-model detector to the family")
    parser.add_argument("--fastdetect", action="store_true",
                        help="add the Fast-DetectGPT conditional-curvature detector to the family")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache"))
    args = parser.parse_args()

    CFG.update(vars(args))
    C.BASE_MODEL = args.base_model
    C.DEVICE = C.get_device(args.device)
    C.CACHE_DIR = args.cache_dir
    C.CORPUS = args.corpus
    C.USE_BINOCULARS = effective_binoculars(args.base_model, args.binoculars)
    C.USE_FASTDETECT = effective_fastdetect(args.base_model, args.fastdetect)
    os.environ["XDG_CACHE_HOME"] = args.cache_dir

    print(f"[app] conformal engine · {C.BASE_MODEL} on {C.DEVICE} · corpus={C.CORPUS} "
          f"m={args.m} dev={args.n_dev} binoculars={C.USE_BINOCULARS} fastdetect={C.USE_FASTDETECT}", flush=True)
    print("[app] models: " + ", ".join(MODELS), flush=True)
    try:
        ensure_bundle(C.BASE_MODEL, C.CORPUS, C.USE_BINOCULARS, C.USE_FASTDETECT, build=False)
        print("[app] calibration loaded from cache", flush=True)
    except ConnectionError:
        print("[app] no compatible calibration cache; building at startup", flush=True)
        startup_job = {
            "id": uuid.uuid4().hex[:8],
            "kind": "model",
            "model": C.BASE_MODEL,
            "binoculars": C.USE_BINOCULARS,
            "fastdetect": C.USE_FASTDETECT,
            "status": "queued",
            "detail": "Building calibration at startup…",
            "created": time.time(),
        }
        with queue_cond:
            job_queue.append(startup_job)
            queue_cond.notify()

    threading.Thread(target=worker, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=False)
