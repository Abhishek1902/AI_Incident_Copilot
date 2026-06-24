"""Per-question scoring metrics for the incident-copilot eval set.

The metric functions are intentionally pure string-comparison routines. The
caller is responsible for putting both retrieved IDs and ground-truth IDs into
the same namespace before invoking — use `normalize_event_id()` for the two
forms in current use:
  - Eval set: 'evt_<12-hex>'  (eval_set.jsonl convention)
  - API:      full 64-char SHA-256  (/incidents/ask `sources[i].event_id`)

Both normalize to a lowercase 12-char hex prefix.

For aggregate scores across the eval set, call these per question and
aggregate (mean) at the caller.
"""

from __future__ import annotations


def normalize_event_id(event_id: str) -> str:
    """Normalize either form of event identifier to a lowercase 12-char hex SHA prefix.

    Accepts:
      - 'evt_36a8ef80ec9a'  (eval-set form)               → '36a8ef80ec9a'
      - '36a8ef80ec9aa074...' (full 64-char API form)     → '36a8ef80ec9a'
      - already-normalized '36a8ef80ec9a'                 → '36a8ef80ec9a'

    Case is forced lowercase so callers don't have to think about it.
    """
    return event_id.lower().removeprefix("evt_")[:12]


def recall_at_k(
    retrieved: list[str],
    ground_truth: list[str],
    k: int = 5,
) -> float:
    """Fraction of ground-truth IDs present in the top-k retrieved.

    Args:
        retrieved:    Retrieved event IDs in ranked order (best first), normalized.
        ground_truth: Ground-truth event IDs, normalized.
        k:            Cutoff for retrieved list. Default 5 matches FINAL_TOP_K.

    Returns:
        Float in [0, 1].  1.0 = every ground-truth ID was retrieved within top-k.

    Raises:
        ValueError: if `ground_truth` is empty. No-answer eval questions should
            be scored via `refusal_correct()`, not here — empty ground_truth
            here indicates a caller bug (likely forgot to branch on category).

    Note on ceiling:
        When `len(ground_truth) > k`, recall caps at `k / len(ground_truth)`.
        Example: q021 has 18 ground-truth IDs and top-k=5 → max possible recall
        is 5/18 ≈ 0.278. This is structural, not a retriever failure. If you
        need a per-question 'normalized recall' that handles this, divide
        recall_at_k by min(1, k/len(ground_truth)) at the caller.
    """
    if not ground_truth:
        raise ValueError(
            "recall_at_k requires non-empty ground_truth. "
            "No-answer questions should be evaluated via refusal_correct()."
        )
    top_k = set(retrieved[:k])
    gt = set(ground_truth)
    return len(top_k & gt) / len(gt)


def mrr(retrieved: list[str], ground_truth: list[str]) -> float:
    """Reciprocal rank of the FIRST retrieved item that appears in ground_truth.

    Args:
        retrieved:    Retrieved event IDs in ranked order (best first), normalized.
        ground_truth: Ground-truth event IDs, normalized.

    Returns:
        1.0   if `retrieved[0]` is in ground_truth.
        0.5   if `retrieved[1]` is the first match.
        1/3   if `retrieved[2]` is the first match. ...
        0.0   if no retrieved item is in ground_truth (or retrieved is empty).

    Raises:
        ValueError: if `ground_truth` is empty (same reason as recall_at_k —
            no-answer questions route through refusal_correct).

    First-match wins: if relevant items appear at ranks 1 and 3, the rank-3
    hit does NOT pull the score down. This is the standard MRR definition and
    intentionally rewards getting the first answer right.
    """
    if not ground_truth:
        raise ValueError(
            "mrr requires non-empty ground_truth. "
            "No-answer questions should be evaluated via refusal_correct()."
        )
    gt = set(ground_truth)
    for rank, evt in enumerate(retrieved, start=1):
        if evt in gt:
            return 1.0 / rank
    return 0.0


def refusal_correct(retrieved: list[str], confidence: str | None) -> bool:
    """True when the system correctly refused to answer.

    Used to score no-answer eval questions (`ground_truth_event_ids == []`),
    where a 'good' response surfaces no sources and signals no confidence.

    Confidence interpretation:
        "none"     → DELIBERATE refusal (pipeline found zero relevant events). Counts.
        "unknown"  → DEGRADED state (reranker unavailable). Does NOT count — the
                     system isn't saying 'I have no answer', it's saying 'my
                     pipeline broke'. Rewarding this would mask infrastructure
                     failure as success. Flip this branch if you disagree.
        "low" / "medium" / "high" → system DID answer. Never a refusal.
        None       → confidence not supplied by caller. Permissive: trust the
                     emptiness of `retrieved` alone.

    A non-empty `retrieved` ALWAYS means False — if any source was surfaced,
    the system did not refuse, regardless of confidence label.

    Args:
        retrieved:  Retrieved event IDs (any form; only emptiness is checked).
        confidence: The `confidence` field from /incidents/ask, or None.

    Returns:
        True iff the system cleanly refused to answer.
    """
    if retrieved:
        return False
    # retrieved is empty — refusal is "clean" unless confidence flags degradation
    if confidence == "unknown":
        return False
    return True
