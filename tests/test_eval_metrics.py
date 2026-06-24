"""Tests for evals/metrics.py — per-question scoring functions."""

import pytest

from evals.metrics import (
    mrr,
    normalize_event_id,
    recall_at_k,
    refusal_correct,
)


# ── normalize_event_id ─────────────────────────────────────────────────────────

class TestNormalizeEventId:
    def test_strips_evt_prefix(self):
        assert normalize_event_id("evt_36a8ef80ec9a") == "36a8ef80ec9a"

    def test_truncates_full_sha_to_12_chars(self):
        full = "36a8ef80ec9aa074196b80d4cd298c909a277174bbe999f4658e097f24dd4583"
        assert normalize_event_id(full) == "36a8ef80ec9a"

    def test_lowercases(self):
        assert normalize_event_id("EVT_36A8EF80EC9A") == "36a8ef80ec9a"

    def test_already_normalized_passes_through(self):
        assert normalize_event_id("36a8ef80ec9a") == "36a8ef80ec9a"

    def test_eval_set_and_api_forms_map_to_same_value(self):
        # The whole point of normalization: both forms produce the same key.
        eval_form = "evt_36a8ef80ec9a"
        api_form = "36a8ef80ec9aa074196b80d4cd298c909a277174bbe999f4658e097f24dd4583"
        assert normalize_event_id(eval_form) == normalize_event_id(api_form)


# ── recall_at_k ────────────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall_all_ground_truth_in_top_k(self):
        # 3 ground-truth events, all surfaced within top-5 → 3/3 = 1.0
        retrieved = ["a", "b", "c", "x", "y"]
        gt = ["a", "b", "c"]
        assert recall_at_k(retrieved, gt, k=5) == 1.0

    def test_partial_recall(self):
        # 4 ground-truth events, 2 within top-5 → 2/4 = 0.5
        retrieved = ["a", "b", "x", "y", "z"]
        gt = ["a", "b", "c", "d"]
        assert recall_at_k(retrieved, gt, k=5) == 0.5

    def test_zero_recall_no_overlap(self):
        retrieved = ["x", "y", "z"]
        gt = ["a", "b", "c"]
        assert recall_at_k(retrieved, gt, k=5) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], ["a", "b"], k=5) == 0.0

    def test_ground_truth_larger_than_k_caps_at_k_over_gt(self):
        # The q021 ceiling case: 18 ground-truth events but top-5 retrieval.
        # Best possible recall is 5/18 ≈ 0.2778. This is a structural limit,
        # not a retriever failure.
        retrieved = ["e1", "e2", "e3", "e4", "e5"]  # 5 hits, all in ground truth
        gt = [f"e{i}" for i in range(1, 19)]         # 18 ground-truth events
        result = recall_at_k(retrieved, gt, k=5)
        assert result == pytest.approx(5 / 18)
        assert result == pytest.approx(0.2778, abs=0.001)

    def test_relevant_item_past_position_k_does_not_count(self):
        # 'b' is in ground_truth but ranked at position 6 (past k=5).
        retrieved = ["a", "x", "y", "z", "w", "b"]
        gt = ["a", "b"]
        # only 'a' is within top-5 → 1/2
        assert recall_at_k(retrieved, gt, k=5) == 0.5

    def test_k_equals_one(self):
        retrieved = ["a", "b", "c"]
        gt = ["b"]
        # 'b' is at rank 2, k=1 excludes it → 0/1
        assert recall_at_k(retrieved, gt, k=1) == 0.0

    def test_k_larger_than_retrieved_uses_whole_list(self):
        retrieved = ["a", "b"]
        gt = ["a", "b", "c"]
        # k=10 but only 2 retrieved; both hit gt → 2/3
        assert recall_at_k(retrieved, gt, k=10) == pytest.approx(2 / 3)

    def test_duplicates_in_retrieved_do_not_inflate(self):
        # If the same event somehow appears twice in retrieved, it counts once.
        retrieved = ["a", "a", "a", "x", "y"]
        gt = ["a", "b"]
        # only 'a' hits → 1/2 = 0.5
        assert recall_at_k(retrieved, gt, k=5) == 0.5

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError, match="non-empty ground_truth"):
            recall_at_k(["a", "b"], [], k=5)

    def test_empty_ground_truth_message_mentions_refusal_correct(self):
        with pytest.raises(ValueError, match="refusal_correct"):
            recall_at_k(["a"], [], k=5)


# ── mrr ────────────────────────────────────────────────────────────────────────

class TestMRR:
    def test_first_position_match_returns_one(self):
        retrieved = ["a", "x", "y"]
        gt = ["a"]
        assert mrr(retrieved, gt) == 1.0

    def test_second_position_match_returns_half(self):
        retrieved = ["x", "a", "y"]
        gt = ["a"]
        assert mrr(retrieved, gt) == 0.5

    def test_third_position_match_returns_one_third(self):
        retrieved = ["x", "y", "a", "z"]
        gt = ["a"]
        assert mrr(retrieved, gt) == pytest.approx(1 / 3)

    def test_no_match_returns_zero(self):
        retrieved = ["x", "y", "z"]
        gt = ["a", "b"]
        assert mrr(retrieved, gt) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert mrr([], ["a"]) == 0.0

    def test_first_match_wins_when_relevant_at_rank_1_and_3(self):
        # Critical edge case: rank-1 hit AND rank-3 hit. Rank-3 must NOT pull
        # score down. First-match-only semantics — this is real MRR, not a
        # multi-relevant average.
        retrieved = ["a", "x", "b"]
        gt = ["a", "b"]
        assert mrr(retrieved, gt) == 1.0

    def test_first_match_at_3_when_only_relevant_at_3(self):
        retrieved = ["x", "y", "b"]
        gt = ["a", "b"]
        assert mrr(retrieved, gt) == pytest.approx(1 / 3)

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError, match="non-empty ground_truth"):
            mrr(["a"], [])


# ── refusal_correct ────────────────────────────────────────────────────────────

class TestRefusalCorrect:
    def test_empty_retrieved_and_confidence_none_is_correct_refusal(self):
        assert refusal_correct([], "none") is True

    def test_empty_retrieved_and_confidence_unknown_is_not_a_clean_refusal(self):
        # "unknown" = reranker unavailable = degraded state, NOT a deliberate
        # refusal. This is the judgment-call branch documented in metrics.py.
        assert refusal_correct([], "unknown") is False

    def test_empty_retrieved_and_confidence_none_value_is_permissive(self):
        # Confidence not supplied at all → trust the empty-retrieved signal.
        assert refusal_correct([], None) is True

    def test_empty_retrieved_and_confidence_low_still_counts_as_refusal(self):
        # Weird state (current pipeline produces "none" for empty), but if it
        # ever happens, empty retrieved IS the refusal.
        assert refusal_correct([], "low") is True

    def test_nonempty_retrieved_is_never_a_refusal(self):
        assert refusal_correct(["evt_a"], "none") is False
        assert refusal_correct(["evt_a"], "unknown") is False
        assert refusal_correct(["evt_a"], "low") is False
        assert refusal_correct(["evt_a"], "medium") is False
        assert refusal_correct(["evt_a"], "high") is False
        assert refusal_correct(["evt_a"], None) is False

    def test_nonempty_retrieved_with_one_source_is_not_refusal(self):
        # Even a single surfaced source means the system didn't refuse.
        assert refusal_correct(["evt_a"], "none") is False


# ── Format-bridge integration ──────────────────────────────────────────────────
#
# These tests exercise normalize_event_id + a metric together, using real
# two-format inputs (eval-set 'evt_<12-hex>' on one side, full 64-char API SHA
# on the other). They're the safeguard against a regression in
# normalize_event_id's output shape that would silently score every question 0.
#
# SHAs below are real event_ids from the seeded incident_events table:
#   evt_36a8ef80ec9a → payment-service v2.3.1 deploy started
#   evt_103e6964090d → order-service downstream call to inventory (attempt 1/3)
#   evt_9cefd43d6dbd → payment-service CRITICAL pool exhausted

# Real two-format pairs used across the bridge tests below.
_EVENT_DEPLOY_EVAL = "evt_36a8ef80ec9a"
_EVENT_DEPLOY_API  = "36a8ef80ec9aa074196b80d4cd298c909a277174bbe999f4658e097f24dd4583"
_EVENT_ORDER_EVAL  = "evt_103e6964090d"
_EVENT_ORDER_API   = "103e6964090d07668318d7da2a5933c384b4dd48194d2fc46bcac7cb4ed0c5e1"
_EVENT_POOL_EVAL   = "evt_9cefd43d6dbd"
_EVENT_POOL_API    = "9cefd43d6dbd95645a0b99330e4cdd13e1f0ab8fec9b37ee0ae6b2de16cb2113"

# Decoy: 64-char hex that doesn't match any seeded event.
_DECOY_API = "f" * 64


class TestFormatBridge:
    """End-to-end: normalize_event_id + metric on REAL two-format inputs.

    If any of these tests fail, the bridge between eval-set IDs and API IDs
    is broken. A passing run proves the normalizer's output is compatible with
    the metric's set-intersection logic — the invariant the runner depends on.
    """

    def test_recall_at_k_perfect_overlap_across_formats(self):
        # Ground truth in eval-set form, retrieved in full 64-char API form,
        # both events overlap. After normalization → recall must be 1.0.
        gt = [_EVENT_DEPLOY_EVAL, _EVENT_ORDER_EVAL]
        retrieved = [_EVENT_DEPLOY_API, _EVENT_ORDER_API, _DECOY_API]

        gt_n = [normalize_event_id(e) for e in gt]
        retrieved_n = [normalize_event_id(e) for e in retrieved]

        assert recall_at_k(retrieved_n, gt_n, k=5) == 1.0

    def test_recall_at_k_partial_overlap_across_formats(self):
        # Same shape, but retrieved only contains 1 of 2 ground-truth events.
        # Bridge must preserve the partial-match fraction (0.5), not collapse
        # to 0 or 1.
        gt = [_EVENT_DEPLOY_EVAL, _EVENT_ORDER_EVAL]
        retrieved = [_EVENT_DEPLOY_API, _DECOY_API]  # only deploy event matches

        gt_n = [normalize_event_id(e) for e in gt]
        retrieved_n = [normalize_event_id(e) for e in retrieved]

        assert recall_at_k(retrieved_n, gt_n, k=5) == 0.5

    def test_mrr_bridges_formats(self):
        # Single ground-truth event in eval form, matching SHA at rank 1 of
        # retrieved (API form). After normalization → MRR must be 1.0.
        gt_n = [normalize_event_id(_EVENT_POOL_EVAL)]
        retrieved_n = [
            normalize_event_id(_EVENT_POOL_API),       # rank 1 — match
            normalize_event_id(_DECOY_API),            # rank 2 — noise
        ]

        assert mrr(retrieved_n, gt_n) == 1.0

    def test_zero_overlap_across_formats_does_not_falsely_match(self):
        # Negative test: gt and retrieved are real two-format strings, but
        # they reference DIFFERENT events. Normalization must not accidentally
        # collapse distinct event_ids — recall must be 0.0, not >0.
        gt = [_EVENT_DEPLOY_EVAL]                       # one event (eval form)
        retrieved = [_EVENT_ORDER_API, _EVENT_POOL_API]  # two other events (API form)

        gt_n = [normalize_event_id(e) for e in gt]
        retrieved_n = [normalize_event_id(e) for e in retrieved]

        assert recall_at_k(retrieved_n, gt_n, k=5) == 0.0
