"""Eval runner v1 — score the incident-copilot against evals/eval_set.jsonl.

Loops over the JSONL eval set, calls POST /incidents/ask per question, scores
the response using evals.metrics, and writes both a structured JSON report and
a human-readable summary table.

v1 scope (per the approved plan):
    Scored: lookup, multi-hop, no-answer (non-date — no explicit start/end fields)
    Deferred: temporal, ambiguous, no-answer with explicit start_time/end_time

Errors during an individual API call are recorded (status="errored") and do
NOT kill the run — the loop always completes and writes results.

Usage:
    python -m evals.runner
    python -m evals.runner --base-url http://localhost:8000 --output /tmp/run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from evals.metrics import mrr, normalize_event_id, recall_at_k, refusal_correct

# ── Configuration ──────────────────────────────────────────────────────────────

# Use 127.0.0.1 (IPv4) explicitly rather than 'localhost' — on machines with
# IPv6 enabled, 'localhost' may resolve to ::1 first, which can route to a
# different process if anything else is bound to [::]:8000 (e.g. a stale
# docker-compose api container — this bit us once, took an hour to diagnose).
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "eval_results.json"

# Categories that v1 actually scores. Everything else is recorded as "deferred".
SCORED_CATEGORIES = {"lookup", "multi-hop", "no-answer"}

# Within no-answer, defer questions that carry explicit start_time/end_time —
# those overlap with temporal date-window logic and are batched with temporal
# in v1.5.
NO_ANSWER_DATE_KEYS = ("start_time", "end_time")

# Timeout for /incidents/ask. Accommodates cold reranker/embedding model loads
# on the first hit. Subsequent calls are fast.
REQUEST_TIMEOUT_S = 60.0

# Default time window applied to lookup/multi-hop questions that don't already
# carry explicit start_time/end_time. These categories test FACT retrieval, not
# time-bounded retrieval — but the natural-language `extract_time_window`
# heuristic can't parse phrases like "March 15, 2026" or "v2.3.1 deployment",
# so without an explicit window the API defaults to "last 1 hour from now" and
# returns 0 sources for the (older) seed corpus. We work around that here so
# v1 scores measure retrieval/ranking correctness, not the (separately known)
# window-parser gap. Wide window covers the full eval corpus span 2026 with margin.
WIDE_WINDOW_START = "2026-01-01T00:00:00Z"
WIDE_WINDOW_END   = "2026-12-31T23:59:59Z"
WIDE_WINDOW_CATEGORIES = {"lookup", "multi-hop"}


# ── Routing ────────────────────────────────────────────────────────────────────


def route(question: dict) -> str:
    """Decide what to do with a question. Returns 'score' or 'defer'."""
    cat = question["category"]
    if cat not in SCORED_CATEGORIES:
        return "defer"
    if cat == "no-answer" and any(k in question for k in NO_ANSWER_DATE_KEYS):
        return "defer"
    return "score"


# ── Single-question execution ──────────────────────────────────────────────────


def call_api(client: httpx.Client, base_url: str, question: dict) -> tuple[dict, float]:
    """Call POST /incidents/ask with timing.

    Forwards start_time / end_time from the JSONL when present so temporal /
    date-window questions can bypass the natural-language window parser.

    Returns:
        (response_json, latency_ms)

    Raises:
        httpx.HTTPError / httpx.TimeoutException — caller must handle.
    """
    body: dict[str, Any] = {"query": question["question"]}
    if "start_time" in question:
        body["start_time"] = question["start_time"]
    if "end_time" in question:
        body["end_time"] = question["end_time"]
    # For fact-retrieval categories without explicit windows, force a wide one
    # so the broken extract_time_window heuristic doesn't silently return 0
    # hits. See WIDE_WINDOW_* docstring above for why this is the right call
    # for v1.
    if question["category"] in WIDE_WINDOW_CATEGORIES and "start_time" not in body:
        body["start_time"] = WIDE_WINDOW_START
        body["end_time"]   = WIDE_WINDOW_END

    t0 = time.perf_counter()
    resp = client.post(f"{base_url}/incidents/ask", json=body, timeout=REQUEST_TIMEOUT_S)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    return resp.json(), latency_ms


def score_question(question: dict, retrieved: list[str], confidence: str | None) -> dict:
    """Compute the right metrics for this question's category. Both lists must
    already be normalized (12-char hex SHA prefixes)."""
    gt = [normalize_event_id(e) for e in question["ground_truth_event_ids"]]
    cat = question["category"]

    if cat in ("lookup", "multi-hop"):
        return {
            "recall@5": recall_at_k(retrieved, gt, k=5),
            "mrr": mrr(retrieved, gt),
        }
    if cat == "no-answer":
        return {"refusal_correct": refusal_correct(retrieved, confidence)}

    # Defensive — shouldn't happen given the routing filter.
    return {}


def process_question(
    client: httpx.Client, base_url: str, question: dict
) -> dict:
    """Run one question end-to-end. Always returns a result dict (never raises)."""
    base_result = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "ground_truth": [normalize_event_id(e) for e in question["ground_truth_event_ids"]],
    }

    if route(question) == "defer":
        return {**base_result, "status": "deferred"}

    try:
        response, latency_ms = call_api(client, base_url, question)
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        return {
            **base_result,
            "status": "errored",
            "error": f"{type(exc).__name__}: {exc}",
            "retrieved": [],
            "confidence": None,
            "scores": {},
            "latency_ms": None,
            "answer": None,
        }

    try:
        sources = response.get("sources", [])
        retrieved = [normalize_event_id(s["event_id"]) for s in sources]
        confidence = response.get("confidence")
        scores = score_question(question, retrieved, confidence)
    except (KeyError, ValueError) as exc:
        return {
            **base_result,
            "status": "errored",
            "error": f"post-response {type(exc).__name__}: {exc} | sources_sample={response.get('sources', [])[:1]}",
            "retrieved": [],
            "confidence": response.get("confidence"),
            "scores": {},
            "latency_ms": round(latency_ms, 1),
            "answer": response.get("answer"),
        }

    return {
        **base_result,
        "status": "scored",
        "retrieved": retrieved,
        "confidence": confidence,
        "scores": scores,
        "latency_ms": round(latency_ms, 1),
        "answer": response.get("answer"),
        "error": None,
    }


# ── Aggregation ────────────────────────────────────────────────────────────────


def _safe_mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    """Return the pct-th percentile (0..100). None if values is empty."""
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    # statistics.quantiles with n=100 gives us 1..99th percentile cut points.
    qs = statistics.quantiles(sorted(values), n=100)
    idx = max(0, min(98, int(pct) - 1))
    return round(qs[idx], 1)


def aggregate(results: list[dict]) -> dict:
    """Compute per-category and overall aggregates."""
    scored = [r for r in results if r["status"] == "scored"]
    deferred = [r for r in results if r["status"] == "deferred"]
    errored = [r for r in results if r["status"] == "errored"]

    per_category: dict[str, dict] = {}
    for cat in sorted({r["category"] for r in results}):
        cat_rows = [r for r in results if r["category"] == cat]
        cat_scored = [r for r in cat_rows if r["status"] == "scored"]

        recalls = [r["scores"]["recall@5"] for r in cat_scored if "recall@5" in r["scores"]]
        mrrs    = [r["scores"]["mrr"]      for r in cat_scored if "mrr"      in r["scores"]]
        refusals = [r["scores"]["refusal_correct"] for r in cat_scored if "refusal_correct" in r["scores"]]

        per_category[cat] = {
            "total": len(cat_rows),
            "scored": len(cat_scored),
            "deferred": sum(1 for r in cat_rows if r["status"] == "deferred"),
            "errored": sum(1 for r in cat_rows if r["status"] == "errored"),
            "mean_recall_at_5": _safe_mean(recalls),
            "mean_mrr": _safe_mean(mrrs),
            "refusal_accuracy": _safe_mean([1.0 if r else 0.0 for r in refusals]),
        }

    all_recalls = [r["scores"]["recall@5"] for r in scored if "recall@5" in r["scores"]]
    all_mrrs    = [r["scores"]["mrr"]      for r in scored if "mrr"      in r["scores"]]
    all_refusals = [r["scores"]["refusal_correct"] for r in scored if "refusal_correct" in r["scores"]]
    latencies = [r["latency_ms"] for r in scored if r["latency_ms"] is not None]

    return {
        "total": len(results),
        "scored": len(scored),
        "deferred": len(deferred),
        "errored": len(errored),
        "overall_mean_recall_at_5": _safe_mean(all_recalls),
        "overall_mean_mrr": _safe_mean(all_mrrs),
        "overall_refusal_accuracy": _safe_mean([1.0 if r else 0.0 for r in all_refusals]),
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "per_category": per_category,
    }


# ── Output ─────────────────────────────────────────────────────────────────────


def _fmt(val: float | None, places: int = 3) -> str:
    return f"{val:.{places}f}" if val is not None else "  —  "


def print_summary(aggregates: dict) -> None:
    """Print a human-readable summary table to stdout."""
    print()
    print(
        f"Eval results — {aggregates['total']} questions  "
        f"(scored: {aggregates['scored']}, deferred: {aggregates['deferred']}, "
        f"errored: {aggregates['errored']})"
    )
    print()
    print(f"{'Category':<14}│{'n':>5} │{'recall@5':>10} │{'MRR':>8} │{'refusal':>9}")
    print(f"{'─' * 14}┼{'─' * 6}┼{'─' * 11}┼{'─' * 9}┼{'─' * 10}")
    for cat, m in aggregates["per_category"].items():
        recall = _fmt(m["mean_recall_at_5"])
        mrr_val = _fmt(m["mean_mrr"])
        refusal = _fmt(m["refusal_accuracy"], places=2)
        print(f"{cat:<14}│{m['total']:>5} │{recall:>10} │{mrr_val:>8} │{refusal:>9}")
    print(f"{'─' * 14}┼{'─' * 6}┼{'─' * 11}┼{'─' * 9}┼{'─' * 10}")
    overall_recall = _fmt(aggregates["overall_mean_recall_at_5"])
    overall_mrr = _fmt(aggregates["overall_mean_mrr"])
    overall_refusal = _fmt(aggregates["overall_refusal_accuracy"], places=2)
    print(f"{'overall':<14}│{aggregates['scored']:>5} │{overall_recall:>10} │{overall_mrr:>8} │{overall_refusal:>9}")
    print()
    p50 = aggregates["latency_ms_p50"]
    p95 = aggregates["latency_ms_p95"]
    p50_s = f"{p50:.0f} ms" if p50 is not None else "—"
    p95_s = f"{p95:.0f} ms" if p95 is not None else "—"
    print(f"Latency (scored only):  p50 = {p50_s}   p95 = {p95_s}")
    if aggregates["errored"] > 0:
        print(f"⚠️  {aggregates['errored']} question(s) errored — see eval_results.json for details")


# ── Entry point ────────────────────────────────────────────────────────────────


def load_eval_set(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL (default: %(default)s)")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET, help="JSONL eval-set path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    args = parser.parse_args()

    questions = load_eval_set(args.eval_set)
    print(f"Loaded {len(questions)} questions from {args.eval_set}")
    print(f"Hitting {args.base_url}/incidents/ask  (timeout={REQUEST_TIMEOUT_S}s per call)\n")

    results: list[dict] = []
    with httpx.Client() as client:
        for i, q in enumerate(questions, start=1):
            print(f"  [{i:>2}/{len(questions)}] {q['id']:<6} {q['category']:<10} ... ", end="", flush=True)
            r = process_question(client, args.base_url, q)
            results.append(r)
            if r["status"] == "scored":
                # Compact one-line score recap.
                scores = r["scores"]
                if "recall@5" in scores:
                    print(f"recall={scores['recall@5']:.2f} mrr={scores['mrr']:.2f}  ({r['latency_ms']:.0f}ms)")
                else:
                    print(f"refusal={'OK' if scores['refusal_correct'] else 'FAIL'}  ({r['latency_ms']:.0f}ms)")
            elif r["status"] == "deferred":
                print("deferred")
            else:
                print(f"ERROR — {r['error']}")

    aggregates = aggregate(results)

    payload = {
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "eval_set": str(args.eval_set),
            "total_questions": len(questions),
        },
        "aggregates": aggregates,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))

    print_summary(aggregates)
    print(f"\nFull per-question results written to {args.output}")

    # Exit non-zero if any question errored — gives CI a signal.
    return 1 if aggregates["errored"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
