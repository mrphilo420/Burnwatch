"""Append-only, privacy-preserving audit records for Burnwatch jobs."""

import datetime as dt
import json
import os
import threading
import time


_lock = threading.Lock()
DEFAULT_PATH = os.environ.get("BURNWATCH_AUDIT_LOG", "audit_log.jsonl")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def record(job, *, status, error=None, path=None):
    """Write one structured event, excluding source text and uploaded bytes."""
    path = path or DEFAULT_PATH
    result = job.get("result") or {}
    record = {
        "schema_version": 1,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "job_id": job.get("id"),
        "job_kind": job.get("kind"),
        "status": status,
        "elapsed_seconds": round(max(0.0, time.time() - job.get("created", time.time())), 3),
        "configuration": {
            "model": result.get("model"),
            "corpus": result.get("corpus"),
            "alpha": job.get("alpha", result.get("alpha")),
            "construction": job.get("construction", result.get("construction")),
            "calibration_m": result.get("m"),
        },
        "result": _summary(result),
    }
    if error:
        record["error"] = str(error)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with _lock:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _summary(result):
    """Keep decision evidence while deliberately omitting document content."""
    allowed = {
        "construction", "alpha", "m", "model", "corpus", "matched", "calibration_corpus",
        "verdict", "elapsed_seconds", "file", "sampling", "A", "B", "rows", "paired",
    }
    out = {key: _jsonable(value) for key, value in result.items() if key in allowed}
    if "file" in out:
        out["file"].pop("bytes", None)
    for key in ("A", "B"):
        if isinstance(out.get(key), dict):
            out[key] = {k: _jsonable(v) for k, v in out[key].items() if k != "text"}
    return out


def read_recent(limit=100, path=None):
    path = path or DEFAULT_PATH
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows[-max(0, int(limit)):]


def timing_summary(records):
    """Return job-level latency percentiles for ISO/IEC 25010 monitoring."""
    values = sorted(float(r["elapsed_seconds"]) for r in records
                    if r.get("status") == "done" and r.get("elapsed_seconds") is not None)
    if not values:
        return {"n": 0, "p50_seconds": None, "p95_seconds": None, "max_seconds": None}

    def percentile(q):
        index = (len(values) - 1) * q
        lo, hi = int(index), min(len(values) - 1, int(index) + 1)
        fraction = index - lo
        return round(values[lo] + (values[hi] - values[lo]) * fraction, 3)

    return {"n": len(values), "p50_seconds": percentile(0.50),
            "p95_seconds": percentile(0.95), "max_seconds": round(values[-1], 3)}
