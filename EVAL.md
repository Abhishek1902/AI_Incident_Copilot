# Incident Copilot — Evaluation Baseline

The signed scoreboard of retrieval and refusal quality across the synthetic
incident corpus. Per-run details live in `evals/eval_results.json` (gitignored
because they vary per run); this file is the committed reference you diff
against when you change the pipeline.

- **Eval set**: [`evals/eval_set.jsonl`](evals/eval_set.jsonl) (31 questions, 5 categories)
- **Metrics**: [`evals/metrics.py`](evals/metrics.py) — `recall_at_k`, `mrr`, `refusal_correct`
- **Runner**: `python -m evals.runner` ([`evals/runner.py`](evals/runner.py))
- **Corpus**: 41 synthetic incident events across 3 incidents (Mar 15, Apr 10, May 22 2026)

---

## Baseline — 2026-06-24 (v1 runner, first signed run)

**31 questions: 16 scored · 15 deferred · 0 errored**

| Category    | n  | recall@5   | MRR        | refusal |
|-------------|----|------------|------------|---------|
| lookup      | 7  | **1.000**  | **1.000**  | —       |
| multi-hop   | 6  | 0.333      | 0.783      | —       |
| no-answer\* | 3  | —          | —          | **1.00** |
| temporal    | 6  | *deferred* | *deferred* | —       |
| ambiguous   | 6  | *deferred* | *deferred* | —       |
| **overall** | 16 | **0.692**  | **0.900**  | **1.00** |

\* **3 of 6 no-answer questions scored.** Scored (non-windowed refusals): q004, q022, q025.
Deferred: q023, q024, q026 — see "Deferred categories" below.

**Latency (scored only):** p50 = 1927 ms · p95 = 9506 ms.
First call cold-loads the cross-encoder reranker (~15s) and dominates p95;
subsequent calls run in the 1–4s range.

### Findings

- **Lookup is solved.** Every single-fact retrieval lands at rank 1 across all 7 questions.
- **Multi-hop is the headline weakness.** Recall@5 = 0.333 means ~1.3 of 4 chain events
  surface; MRR = 0.783 means when *something* matches, it's usually at rank 1 — i.e. the
  reranker grabs the closest semantic match and stops, doesn't traverse the causal chain.
  Worst: **q014** (checkout root-cause: recall=0.25, MRR=0.20).
- **Refusal works on retrieval-level cases.** All 3 scored no-answer questions returned
  empty `retrieved` + `confidence="none"`. The LLM-output-refusal case (q024) is the
  remaining unknown — see below.

---

## Deferred categories (deliberate)

- **temporal (6)** — needs a precision metric (exact-window vs retrieved set) and a
  cross-window negative-test rubric (March query ≠ April result). `recall_at_k` alone
  doesn't score the negative case.
- **ambiguous (6)** — needs per-interpretation scoring per the `_ambiguity_note` field.
  Full-union recall@5 caps too low (q005's 8 ground-truth events vs k=5); the right
  rubric splits ground truth by interpretation and credits any-hit-per-interpretation.
- **no-answer with explicit window (3)** — q023, q024, q026. Two sub-cases:
  - **q023, q026**: empty time windows (June 2026, January 2026 — pre/post corpus).
    Could be scored by `refusal_correct` once the runner stops deferring them — the
    retrieval-level emptiness signal is correct. Bundled here for v1 consistency only.
  - **q024**: *topic* refusal in a populated window (no Kubernetes events on March 15
    even though 18 other events exist). Retrieval WILL return non-empty sources; the
    refusal happens at the LLM-output level (the LLM should say "no Kubernetes events
    recorded" instead of fabricating from surfaced logs). `refusal_correct` measures
    retrieval-level emptiness, so it's the wrong metric for q024 — needs an
    answer-text refusal check (e.g. detect "no data" / "I don't have" phrasing).

---

## Known runner quirks worth knowing

1. **Wide window forced for lookup/multi-hop.** These categories test fact retrieval,
   not time-bounded retrieval, but the natural-language `extract_time_window` heuristic
   can't parse phrases like "March 15, 2026" or "v2.3.1 deployment". Without an
   explicit window the API defaults to "last 1 hour from now" → 0 hits on the seed
   corpus. The runner forwards `2026-01-01 → 2026-12-31` for these categories to
   isolate retrieval correctness from the (separately known) window-parser gap. See
   `WIDE_WINDOW_*` in [evals/runner.py](evals/runner.py).
2. **`127.0.0.1`, not `localhost`.** On IPv6-enabled machines, `localhost` resolves
   to `::1` first. If a stale `docker-compose` API container is bound to `[::]:8000`,
   `httpx` hits the stale code path. Use `127.0.0.1` to force IPv4. This bit us once.

---

## v2 targets (what to fix before the next baseline)

| Target | Effort | Impact |
|---|---|---|
| Per-interpretation scoring for ambiguous queries | medium | unlocks 6 questions, exposes "wrongly collapses to one root" failure mode |
| Precision metric + negative-test rubric for temporal | medium | unlocks 6 questions, exposes cross-month bleeding |
| `answer_refuses_topic()` metric for LLM-level refusal | small | unlocks q024 (and future topic-refusal questions) |
| Multi-hop recall lift (q014, q012, q013, q015 — all 0.25–0.50) | high | the headline weakness; likely needs iterative retrieval or chain-aware reranking |

---

## How to reproduce this baseline

```bash
docker compose up -d db
alembic upgrade head
python scripts/seed.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning &
python -m evals.runner
```

Compare the resulting `evals/eval_results.json` `aggregates` block against the
numbers above. Any drift larger than ~0.02 in recall or MRR is worth investigating.
