"""Tests for app/services/query_processor.py

Covers:
  - normalize_query: lowercasing, hyphen normalisation, punctuation stripping,
    whitespace collapsing.
  - classify_query: analytical / incident / unknown, priority order.
  - extract_time_window: numeric regex (minutes, hours, days, months, years),
    literal patterns (yesterday, today, last week/month/year/hour), default.
  - extract_service: normalised matching (hyphen vs space, case insensitive),
    no-match returns None, static fallback when db is None.
  - process_query: integration — all fields populated, correct types.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.services.query_processor import (
    QueryContext,
    classify_query,
    extract_service,
    extract_time_window,
    normalize_query,
    process_query,
)


# ── normalize_query ────────────────────────────────────────────────────────────

class TestNormalizeQuery:
    def test_lowercases(self):
        assert normalize_query("PAYMENT-SERVICE ERRORS") == "payment service errors"

    def test_hyphen_to_space(self):
        assert normalize_query("payment-service") == "payment service"

    def test_strips_question_mark(self):
        assert "?" not in normalize_query("why did it fail?")

    def test_strips_exclamation(self):
        assert "!" not in normalize_query("error!")

    def test_strips_comma(self):
        assert "," not in normalize_query("errors, alerts, and outages")

    def test_strips_period(self):
        assert "." not in normalize_query("service.down")

    def test_strips_colon(self):
        assert ":" not in normalize_query("error: connection refused")

    def test_collapses_multiple_spaces(self):
        result = normalize_query("payment  service   failed")
        assert "  " not in result

    def test_strips_leading_trailing_spaces(self):
        result = normalize_query("  payment service  ")
        assert result == result.strip()

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_preserves_numbers(self):
        result = normalize_query("last 24 months")
        assert "24" in result

    def test_hyphen_service_matches_space_form(self):
        """Core requirement: both forms produce the same token stream."""
        assert normalize_query("payment-service") == normalize_query("payment service")


# ── classify_query ─────────────────────────────────────────────────────────────

class TestClassifyQuery:
    # ── analytical ────────────────────────────────────────────────────────────
    def test_which_is_analytical(self):
        assert classify_query("which payment service failed?") == "analytical"

    def test_how_many_is_analytical(self):
        assert classify_query("how many errors occurred last week?") == "analytical"

    def test_count_is_analytical(self):
        assert classify_query("count of failures in payment-service") == "analytical"

    def test_most_is_analytical(self):
        assert classify_query("which service failed most often?") == "analytical"

    def test_top_is_analytical(self):
        assert classify_query("top errors in the last month") == "analytical"

    def test_trend_is_analytical(self):
        assert classify_query("show error trend for payment service") == "analytical"

    def test_how_often_is_analytical(self):
        assert classify_query("how often does payment-service fail?") == "analytical"

    def test_list_all_is_analytical(self):
        assert classify_query("list all services with outages") == "analytical"

    # ── incident ──────────────────────────────────────────────────────────────
    def test_why_is_incident(self):
        assert classify_query("why did payment-service fail yesterday?") == "incident"

    def test_failed_is_incident(self):
        assert classify_query("payment service failed in the last hour") == "incident"

    def test_error_is_incident(self):
        assert classify_query("connection error in order-service") == "incident"

    def test_outage_is_incident(self):
        assert classify_query("outage in api-gateway last 2 hours") == "incident"

    def test_timeout_is_incident(self):
        assert classify_query("timeout errors in payment pipeline") == "incident"

    def test_what_happened_is_incident(self):
        assert classify_query("what happened in the last hour?") == "incident"

    # ── unknown ───────────────────────────────────────────────────────────────
    def test_generic_query_is_unknown(self):
        assert classify_query("show me the logs") == "unknown"

    def test_empty_string_is_unknown(self):
        assert classify_query("") == "unknown"

    def test_unrelated_query_is_unknown(self):
        assert classify_query("hello world") == "unknown"

    # ── priority ──────────────────────────────────────────────────────────────
    def test_analytical_takes_priority_over_incident(self):
        """A query with both 'which' and 'failed' is analytical, not incident."""
        assert classify_query("which payment service failed most?") == "analytical"

    # ── normalisation ─────────────────────────────────────────────────────────
    def test_case_insensitive(self):
        assert classify_query("WHICH service had errors?") == "analytical"

    def test_hyphenated_service_in_incident_query(self):
        assert classify_query("payment-service errors in the last hour") == "incident"


# ── extract_time_window ────────────────────────────────────────────────────────

class TestExtractTimeWindow:
    """Tests for query_processor.extract_time_window (superset of incidents.py version)."""

    _TOL_S = 5  # tolerance in seconds for datetime comparisons

    def _approx(self, actual: datetime, expected: datetime) -> bool:
        return abs((actual - expected).total_seconds()) < self._TOL_S

    # ── numeric: minutes ──────────────────────────────────────────────────────
    def test_last_30_minutes(self):
        start, end = extract_time_window("alerts in the last 30 minutes")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(minutes=30))

    def test_past_5_minutes(self):
        start, _ = extract_time_window("errors in the past 5 minutes")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(minutes=5))

    # ── numeric: hours ────────────────────────────────────────────────────────
    def test_last_2_hours(self):
        start, _ = extract_time_window("payment-service alerts last 2 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=2))

    def test_last_6_hours(self):
        start, _ = extract_time_window("errors in the last 6 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=6))

    def test_last_24_hours(self):
        start, _ = extract_time_window("incidents in the last 24 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=24))

    # ── numeric: days ─────────────────────────────────────────────────────────
    def test_last_3_days(self):
        start, _ = extract_time_window("failures in the last 3 days")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=3))

    def test_past_7_days(self):
        start, _ = extract_time_window("events past 7 days")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=7))

    # ── numeric: months ───────────────────────────────────────────────────────
    def test_last_24_months(self):
        """The original failing query — 24 months → 24*30 days."""
        start, _ = extract_time_window("which payment service failed in last 24 months")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=24 * 30))

    def test_last_3_months(self):
        start, _ = extract_time_window("show failures in the past 3 months")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=3 * 30))

    # ── numeric: years ────────────────────────────────────────────────────────
    def test_last_2_years(self):
        start, _ = extract_time_window("errors in last 2 years")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=2 * 365))

    def test_last_1_year(self):
        start, _ = extract_time_window("what happened in the last 1 year?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=365))

    # ── literal: yesterday / today ────────────────────────────────────────────
    def test_yesterday(self):
        start, _ = extract_time_window("what failed yesterday?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=24))

    def test_today_start_is_midnight(self):
        start, end = extract_time_window("show me today's errors")
        now = datetime.now(timezone.utc)
        # start should be midnight UTC today
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        assert self._approx(start, midnight)

    def test_today_end_is_now(self):
        _, end = extract_time_window("show me today's errors")
        now = datetime.now(timezone.utc)
        assert self._approx(end, now)

    # ── literal: last hour / week / month / year ──────────────────────────────
    def test_last_hour_literal(self):
        start, _ = extract_time_window("what happened in the last hour?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=1))

    def test_past_hour_literal(self):
        start, _ = extract_time_window("errors from the past hour")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=1))

    def test_last_week_literal(self):
        start, _ = extract_time_window("deployment failures last week")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=7))

    def test_last_month_literal(self):
        start, _ = extract_time_window("incidents from last month")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=30))

    def test_last_year_literal(self):
        start, _ = extract_time_window("failures last year")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=365))

    # ── default ───────────────────────────────────────────────────────────────
    def test_no_time_hint_defaults_to_1_hour(self):
        start, _ = extract_time_window("payment-service is down")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=1))

    # ── invariants ────────────────────────────────────────────────────────────
    def test_returns_timezone_aware(self):
        start, end = extract_time_window("any query")
        assert start.tzinfo is not None
        assert end.tzinfo is not None

    def test_start_before_end(self):
        start, end = extract_time_window("anything")
        assert start < end

    def test_end_is_approximately_now(self):
        _, end = extract_time_window("what happened today?")
        now = datetime.now(timezone.utc)
        assert self._approx(end, now)


# ── extract_service ────────────────────────────────────────────────────────────

class TestExtractService:
    """Tests for query_processor.extract_service (normalised matching)."""

    # ── exact stored form ─────────────────────────────────────────────────────
    def test_hyphenated_form_matches(self):
        assert extract_service("payment-service errors") == "payment-service"

    def test_order_service_hyphenated(self):
        assert extract_service("why is order-service failing?") == "order-service"

    def test_payment_pipeline_hyphenated(self):
        assert extract_service("payment-pipeline job failed") == "payment-pipeline"

    # ── natural language (space) form ─────────────────────────────────────────
    def test_payment_service_with_space(self):
        """Core fix: 'payment service' must match stored 'payment-service'."""
        assert extract_service("which payment service failed") == "payment-service"

    def test_order_service_with_space(self):
        assert extract_service("order service is responding slowly") == "order-service"

    def test_payment_pipeline_with_space(self):
        assert extract_service("payment pipeline job failed") == "payment-pipeline"

    # ── case insensitivity ────────────────────────────────────────────────────
    def test_uppercase_query(self):
        assert extract_service("PAYMENT-SERVICE is down") == "payment-service"

    def test_mixed_case(self):
        assert extract_service("Why did Payment Service fail?") == "payment-service"

    # ── no match ──────────────────────────────────────────────────────────────
    def test_unknown_service_returns_none(self):
        assert extract_service("database connection pool exhausted") is None

    def test_empty_query_returns_none(self):
        assert extract_service("") is None

    def test_generic_query_returns_none(self):
        assert extract_service("what happened in the last hour?") is None

    # ── db fallback ───────────────────────────────────────────────────────────
    def test_works_without_db(self):
        """db=None must fall back to static list gracefully."""
        result = extract_service("payment-service errors", db=None)
        assert result == "payment-service"

    def test_mock_db_falls_back_to_static_list(self):
        """When db.execute() returns an un-iterable mock, falls back to static list."""
        mock_db = MagicMock()
        result = extract_service("payment-service down", db=mock_db)
        assert result == "payment-service"


# ── process_query ──────────────────────────────────────────────────────────────

class TestProcessQuery:
    def test_returns_query_context(self):
        result = process_query("why did payment-service fail yesterday?")
        assert isinstance(result, QueryContext)

    def test_normalized_query_populated(self):
        result = process_query("Payment-Service ERRORS?")
        assert result.normalized_query == "payment service errors"

    def test_query_type_incident(self):
        result = process_query("why did payment-service fail yesterday?")
        assert result.query_type == "incident"

    def test_query_type_analytical(self):
        result = process_query("which payment service failed most?")
        assert result.query_type == "analytical"

    def test_query_type_unknown(self):
        result = process_query("show me the logs")
        assert result.query_type == "unknown"

    def test_time_window_is_tuple_of_datetimes(self):
        result = process_query("what happened in the last 2 hours?")
        assert isinstance(result.time_window, tuple)
        assert len(result.time_window) == 2
        assert all(isinstance(t, datetime) for t in result.time_window)

    def test_time_window_start_before_end(self):
        result = process_query("errors in the last 6 hours")
        assert result.time_window[0] < result.time_window[1]

    def test_time_window_timezone_aware(self):
        result = process_query("what failed yesterday?")
        assert result.time_window[0].tzinfo is not None
        assert result.time_window[1].tzinfo is not None

    def test_service_detected(self):
        result = process_query("why did payment service fail?")
        assert result.service == "payment-service"

    def test_service_none_when_not_found(self):
        result = process_query("what happened in the last hour?")
        assert result.service is None

    def test_service_detected_with_hyphen_form(self):
        result = process_query("payment-service errors in the last 2 hours")
        assert result.service == "payment-service"

    def test_analytical_query_still_extracts_time_window(self):
        """process_query always populates all fields, even for analytical queries."""
        result = process_query("which payment service failed in last 24 months")
        assert result.query_type == "analytical"
        now = datetime.now(timezone.utc)
        expected_start = now - timedelta(days=24 * 30)
        delta = abs((result.time_window[0] - expected_start).total_seconds())
        assert delta < 10

    def test_analytical_query_still_extracts_service(self):
        result = process_query("which payment service failed in last 24 months")
        assert result.service == "payment-service"
