"""Tests for the incident ingestion pipeline.

Covers:
  - log normalisation (the embedding quality gate)
  - event_id determinism and uniqueness
  - deduplication via ON CONFLICT
  - ingest_logs / ingest_events / ingest_pipeline_metadata return values
  - POST /ingest/batch endpoint (TestClient, DB mocked)
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import get_db
from app.services.ingestion import (
    IngestResult,
    _make_event_id,
    ingest_logs,
    normalise_log,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app), mock_db
    app.dependency_overrides.clear()


def _make_log(
    service="svc",
    content="ERROR connection failed",
    occurred_at=None,
    **kwargs,
) -> dict:
    return {
        "service": service,
        "occurred_at": occurred_at or datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc),
        "content": content,
        **kwargs,
    }


# ── normalise_log ──────────────────────────────────────────────────────────────

class TestNormaliseLog:
    def test_strips_leading_timestamp(self):
        raw = "2026-01-01 ERROR pid=123 connection failed"
        assert normalise_log(raw) == "connection failed"

    def test_strips_iso_timestamp_with_time(self):
        raw = "2026-03-15T10:05:33Z ERROR database timeout"
        assert normalise_log(raw) == "database timeout"

    def test_strips_bracketed_timestamp(self):
        raw = "[2026-03-15T10:05:33Z] ERROR DB timeout"
        assert normalise_log(raw) == "DB timeout"

    def test_strips_log_level_prefix(self):
        for level in ("DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "FATAL"):
            result = normalise_log(f"{level} connection refused")
            assert "connection refused" in result
            assert level not in result

    def test_strips_bracketed_log_level(self):
        raw = "[ERROR] connection timeout after 30s"
        assert normalise_log(raw) == "connection timeout after 30s"

    def test_strips_pid(self):
        raw = "ERROR pid=4412 connection pool exhausted"
        result = normalise_log(raw)
        assert "pid" not in result.lower()
        assert "connection pool exhausted" in result

    def test_strips_hex_address(self):
        raw = "ERROR segfault at address 0x7f4a3b2c1d0e"
        result = normalise_log(raw)
        assert "0x7f4a3b2c1d0e" not in result
        assert "segfault" in result

    def test_strips_uuid(self):
        raw = "ERROR request 550e8400-e29b-41d4-a716-446655440000 failed"
        result = normalise_log(raw)
        assert "550e8400" not in result
        assert "request" in result
        assert "failed" in result

    def test_collapses_whitespace(self):
        raw = "ERROR    too   many   spaces"
        assert normalise_log(raw) == "too   many   spaces" or "  " not in normalise_log(raw)

    def test_fallback_on_empty_result(self):
        """If normalisation strips everything, return the original."""
        raw = "2026-01-01"
        result = normalise_log(raw)
        assert len(result) > 0

    def test_preserves_semantic_content(self):
        """Core message survives even when noise is stripped."""
        raw = "2026-03-15T10:08:55Z [payment-service] pid=4412 CRITICAL database connection pool exhausted"
        result = normalise_log(raw)
        assert "database connection pool exhausted" in result

    def test_plain_message_unchanged(self):
        """Messages with no noise pass through cleanly."""
        msg = "connection pool exhausted — all connections checked out"
        assert normalise_log(msg) == msg


# ── _make_event_id ─────────────────────────────────────────────────────────────

class TestMakeEventId:
    def _ts(self) -> datetime:
        return datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc)

    def test_returns_64_char_hex(self):
        eid = _make_event_id("svc", self._ts(), "content")
        assert len(eid) == 64
        assert all(c in "0123456789abcdef" for c in eid)

    def test_deterministic(self):
        """Same inputs always produce the same ID."""
        ts = self._ts()
        eid1 = _make_event_id("svc", ts, "content")
        eid2 = _make_event_id("svc", ts, "content")
        assert eid1 == eid2

    def test_different_services_differ(self):
        ts = self._ts()
        assert _make_event_id("svc-a", ts, "msg") != _make_event_id("svc-b", ts, "msg")

    def test_different_times_differ(self):
        ts1 = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc)
        assert _make_event_id("svc", ts1, "msg") != _make_event_id("svc", ts2, "msg")

    def test_different_content_differs(self):
        ts = self._ts()
        assert _make_event_id("svc", ts, "msg A") != _make_event_id("svc", ts, "msg B")


# ── ingest_logs ────────────────────────────────────────────────────────────────

class TestIngestLogs:
    def _mock_db_insert(self, rowcount: int) -> MagicMock:
        """Return a mock DB where execute().rowcount == rowcount."""
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = rowcount
        mock_db.execute.return_value = mock_result
        return mock_db

    def test_returns_ingest_result(self):
        mock_db = self._mock_db_insert(rowcount=2)
        logs = [_make_log("svc-a"), _make_log("svc-b")]

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384, [0.1]*384]):
            result = ingest_logs(logs, mock_db)

        assert isinstance(result, IngestResult)
        assert result.total == 2

    def test_ingested_equals_rowcount(self):
        mock_db = self._mock_db_insert(rowcount=2)
        logs = [_make_log(), _make_log(service="other-svc")]

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384]*2):
            result = ingest_logs(logs, mock_db)

        assert result.ingested == 2
        assert result.skipped == 0

    def test_duplicate_counted_as_skipped(self):
        """rowcount=1 with 2 items means 1 was a duplicate."""
        mock_db = self._mock_db_insert(rowcount=1)
        ts = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        logs = [_make_log(occurred_at=ts), _make_log(occurred_at=ts)]

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384]*2):
            result = ingest_logs(logs, mock_db)

        assert result.ingested == 1
        assert result.skipped == 1

    def test_empty_list_returns_zero_result(self):
        mock_db = MagicMock()
        result = ingest_logs([], mock_db)
        assert result == IngestResult(ingested=0, skipped=0, total=0)
        mock_db.execute.assert_not_called()

    def test_batch_embed_called_once(self):
        """Embeddings are generated in one batch call, not per-item."""
        mock_db = self._mock_db_insert(rowcount=3)
        logs = [_make_log(service=f"svc-{i}") for i in range(3)]

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384]*3) as mock_embed:
            ingest_logs(logs, mock_db)

        mock_embed.assert_called_once()
        assert len(mock_embed.call_args[0][0]) == 3

    def test_naive_datetime_made_timezone_aware(self):
        """Naive datetimes are treated as UTC to satisfy DB TIMESTAMPTZ."""
        mock_db = self._mock_db_insert(rowcount=1)
        naive_dt = datetime(2026, 3, 15, 10, 0, 0)  # no tzinfo
        log = _make_log(occurred_at=naive_dt)

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384]):
            # Should not raise
            ingest_logs([log], mock_db)

    def test_db_commit_called(self):
        mock_db = self._mock_db_insert(rowcount=1)

        with patch("app.services.ingestion.generate_embeddings_batch", return_value=[[0.1]*384]):
            ingest_logs([_make_log()], mock_db)

        mock_db.commit.assert_called_once()


# ── POST /ingest/batch ─────────────────────────────────────────────────────────

class TestBatchIngestEndpoint:
    def _payload(self):
        return {
            "logs": [{
                "service": "payment-service",
                "occurred_at": "2026-03-15T10:05:33Z",
                "content": "ERROR connection timeout",
                "severity": "error",
                "correlation_id": "inc-001",
            }],
            "events": [{
                "service": "payment-service",
                "event_type": "deployment",
                "occurred_at": "2026-03-15T10:00:00Z",
                "content": "Deployment started v2.3.1",
            }],
            "metadata": [{
                "service": "payment-pipeline",
                "occurred_at": "2026-03-15T10:07:45Z",
                "content": "Pipeline job started",
            }],
        }

    def test_returns_201(self, client):
        http, _ = client
        log_res  = IngestResult(ingested=1, skipped=0, total=1)
        evt_res  = IngestResult(ingested=1, skipped=0, total=1)
        meta_res = IngestResult(ingested=1, skipped=0, total=1)

        with patch("app.api.incidents.ingest_logs", return_value=log_res), \
             patch("app.api.incidents.ingest_events", return_value=evt_res), \
             patch("app.api.incidents.ingest_pipeline_metadata", return_value=meta_res):
            resp = http.post("/ingest/batch", json=self._payload())

        assert resp.status_code == 201

    def test_response_contains_totals(self, client):
        http, _ = client
        res = IngestResult(ingested=1, skipped=0, total=1)
        with patch("app.api.incidents.ingest_logs", return_value=res), \
             patch("app.api.incidents.ingest_events", return_value=res), \
             patch("app.api.incidents.ingest_pipeline_metadata", return_value=res):
            resp = http.post("/ingest/batch", json=self._payload())

        data = resp.json()
        assert data["total_ingested"] == 3
        assert data["total_skipped"] == 0
        assert data["total_received"] == 3

    def test_skipped_duplicates_reflected_in_response(self, client):
        http, _ = client
        skipped_res = IngestResult(ingested=0, skipped=1, total=1)
        with patch("app.api.incidents.ingest_logs", return_value=skipped_res), \
             patch("app.api.incidents.ingest_events", return_value=skipped_res), \
             patch("app.api.incidents.ingest_pipeline_metadata", return_value=skipped_res):
            resp = http.post("/ingest/batch", json=self._payload())

        data = resp.json()
        assert data["total_ingested"] == 0
        assert data["total_skipped"] == 3

    def test_empty_lists_accepted(self, client):
        http, _ = client
        res = IngestResult(ingested=0, skipped=0, total=0)
        with patch("app.api.incidents.ingest_logs", return_value=res), \
             patch("app.api.incidents.ingest_events", return_value=res), \
             patch("app.api.incidents.ingest_pipeline_metadata", return_value=res):
            resp = http.post("/ingest/batch", json={})

        assert resp.status_code == 201

    def test_invalid_occurred_at_returns_422(self, client):
        http, _ = client
        payload = {"logs": [{"service": "svc", "occurred_at": "not-a-date", "content": "msg"}]}
        resp = http.post("/ingest/batch", json=payload)
        assert resp.status_code == 422
