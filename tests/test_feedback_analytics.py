"""Tests for POST /feedback and GET /analytics/summary."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import get_db
from app.db.models import Feedback, QueryLog


@pytest.fixture
def client():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app), mock_db
    app.dependency_overrides.clear()


# ── POST /feedback ─────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    def test_good_rating_returns_201(self, client):
        http, mock_db = client
        entry = Feedback(query="q", answer="a", rating="good")
        entry.id = 1
        mock_db.refresh.side_effect = lambda obj: setattr(obj, "id", 1)

        resp = http.post("/feedback", json={
            "query": "What is ML?",
            "answer": "Machine learning is...",
            "rating": "good",
        })

        assert resp.status_code == 201
        assert resp.json()["id"] == 1
        assert "Thank" in resp.json()["message"]

    def test_bad_rating_accepted(self, client):
        http, mock_db = client
        mock_db.refresh.side_effect = lambda obj: setattr(obj, "id", 2)

        resp = http.post("/feedback", json={
            "query": "q",
            "answer": "a",
            "rating": "bad",
            "comment": "Not helpful at all.",
        })

        assert resp.status_code == 201

    def test_invalid_rating_returns_422(self, client):
        http, _ = client
        resp = http.post("/feedback", json={
            "query": "q",
            "answer": "a",
            "rating": "meh",   # not in Literal["good", "bad"]
        })
        assert resp.status_code == 422

    def test_feedback_added_to_db(self, client):
        """Verifies db.add is called with a Feedback instance."""
        http, mock_db = client
        mock_db.refresh.side_effect = lambda obj: setattr(obj, "id", 5)

        http.post("/feedback", json={"query": "q", "answer": "a", "rating": "good"})

        mock_db.add.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert isinstance(added, Feedback)
        assert added.rating == "good"

    def test_comment_is_optional(self, client):
        http, mock_db = client
        mock_db.refresh.side_effect = lambda obj: setattr(obj, "id", 3)

        resp = http.post("/feedback", json={
            "query": "q", "answer": "a", "rating": "good"
            # no comment field
        })
        assert resp.status_code == 201


# ── GET /analytics/summary ─────────────────────────────────────────────────────

class TestAnalyticsEndpoint:
    def _configure_db(self, mock_db, total=10, avg_lat=2.5, low_ground=2, bad_fb=3):
        """Wire the mock DB query chain to return fixed aggregate values."""
        # The analytics endpoint makes four separate db.query() calls.
        # Each call goes through .filter() (optionally) then .scalar().
        # We use side_effect to return values in order.
        mock_db.query.return_value.scalar.side_effect = [
            total, avg_lat, low_ground, bad_fb,
        ]
        mock_db.query.return_value.filter.return_value.scalar.side_effect = [
            low_ground, bad_fb,
        ]

    def test_returns_200_with_all_fields(self, client):
        http, mock_db = client
        # Use a simpler mock: each scalar() call in order
        scalars = iter([10, 2.5, 2, 3])
        mock_db.query.return_value.scalar.side_effect = lambda: next(scalars)
        mock_db.query.return_value.filter.return_value.scalar.side_effect = lambda: next(scalars)

        resp = http.get("/analytics/summary")

        assert resp.status_code == 200
        data = resp.json()
        for key in ("total_queries", "avg_latency", "low_groundedness_count", "bad_feedback_count"):
            assert key in data

    def test_empty_db_returns_zeros(self, client):
        """When there are no rows, all metrics should be 0."""
        http, mock_db = client
        scalars = iter([0, None, 0, 0])
        mock_db.query.return_value.scalar.side_effect = lambda: next(scalars)
        mock_db.query.return_value.filter.return_value.scalar.side_effect = lambda: next(scalars)

        resp = http.get("/analytics/summary")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_queries"] == 0
        assert data["avg_latency"] == 0.0


# ── query_logger ───────────────────────────────────────────────────────────────

class TestQueryLogger:
    def test_log_query_writes_to_db(self):
        from app.services.query_logger import log_query

        mock_db = MagicMock()
        log_query(mock_db, query="q", answer="a", latency=1.23,
                  groundedness_score=0.9, relevance_score=0.8)

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        added = mock_db.add.call_args[0][0]
        assert isinstance(added, QueryLog)
        assert added.latency == 1.23
        assert added.groundedness_score == 0.9

    def test_log_query_without_scores(self):
        from app.services.query_logger import log_query

        mock_db = MagicMock()
        log_query(mock_db, query="q", answer="a", latency=0.5)

        added = mock_db.add.call_args[0][0]
        assert added.groundedness_score is None
        assert added.relevance_score is None

    def test_log_query_db_failure_does_not_raise(self):
        """A DB error must never propagate out of log_query."""
        from app.services.query_logger import log_query

        mock_db = MagicMock()
        mock_db.commit.side_effect = Exception("connection lost")

        # Should not raise
        log_query(mock_db, query="q", answer="a", latency=1.0)
        mock_db.rollback.assert_called_once()
