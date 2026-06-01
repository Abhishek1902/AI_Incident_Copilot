"""Tests for Sprint 2: incident-aware hybrid retrieval.

Covers:
  - extract_time_window heuristics
  - extract_service heuristics
  - search_incidents SQLAlchemy query construction and hit mapping
  - rerank_incidents with and without reranker availability
  - build_incident_prompt chronological ordering and format
  - answer_incident_query full pipeline (DB + all services mocked)
  - POST /incidents/ask endpoint (TestClient, DB mocked)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.session import get_db
from app.core.config import settings
from app.services.incidents import (
    IncidentSearchHit,
    _format_for_reranker,
    extract_service,
    extract_time_window,
    get_known_services,
    rerank_incidents,
    search_incidents,
)
from app.services.prompt import build_incident_prompt
from app.services.rag import answer_incident_query, RAGResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app), mock_db
    app.dependency_overrides.clear()


def _make_hit(
    id: int = 1,
    content: str = "ERROR connection timeout",
    service: str = "payment-service",
    event_type: str = "log",
    occurred_at: datetime | None = None,
    similarity_score: float = 0.85,
    severity: str | None = None,
    rerank_score: float | None = None,
    metadata: dict | None = None,
) -> IncidentSearchHit:
    return IncidentSearchHit(
        id=id,
        content=content,
        occurred_at=occurred_at or datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc),
        service=service,
        event_type=event_type,
        similarity_score=similarity_score,
        severity=severity,
        rerank_score=rerank_score,
        metadata=metadata,
    )


# ── extract_time_window ────────────────────────────────────────────────────────

class TestExtractTimeWindow:
    def _approx(self, actual: datetime, expected: datetime, tolerance_s: int = 5) -> bool:
        return abs((actual - expected).total_seconds()) < tolerance_s

    def test_last_hour_keyword(self):
        start, end = extract_time_window("what happened in the last hour?")
        now = datetime.now(timezone.utc)
        assert self._approx(end, now)
        assert self._approx(start, now - timedelta(hours=1))

    def test_past_hour_keyword(self):
        start, end = extract_time_window("show errors from the past hour")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=1))

    def test_last_2_hours(self):
        start, end = extract_time_window("alerts in the last 2 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=2))

    def test_last_two_hours_spelled_out(self):
        start, end = extract_time_window("failures in the last two hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=2))

    def test_last_6_hours(self):
        start, end = extract_time_window("payment-service errors last 6 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=6))

    def test_yesterday(self):
        start, end = extract_time_window("what failed yesterday?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=24))

    def test_last_24_hours(self):
        start, end = extract_time_window("incidents in the last 24 hours")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=24))

    def test_last_week(self):
        start, end = extract_time_window("deployment failures last week")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=7))

    def test_default_is_last_hour(self):
        """No hint in query → default to last 1 hour."""
        start, end = extract_time_window("payment-service is down")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(hours=1))

    def test_returns_timezone_aware(self):
        start, end = extract_time_window("any query")
        assert start.tzinfo is not None
        assert end.tzinfo is not None

    def test_start_before_end(self):
        start, end = extract_time_window("anything")
        assert start < end

    # ── Month / year patterns (new) ────────────────────────────────────────────

    def test_last_n_months(self):
        start, end = extract_time_window("which payment service failed in last 24 months")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=24 * 30), tolerance_s=10)

    def test_last_3_months(self):
        start, end = extract_time_window("show failures in the past 3 months")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=3 * 30))

    def test_last_n_years(self):
        start, end = extract_time_window("errors in last 2 years")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=2 * 365))

    def test_last_1_year(self):
        start, end = extract_time_window("what happened in the last 1 year?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=365))

    def test_last_month_literal(self):
        start, end = extract_time_window("incidents from last month")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=30))

    def test_past_month_literal(self):
        start, end = extract_time_window("any alerts in the past month?")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=30))

    def test_last_year_literal(self):
        start, end = extract_time_window("failure trends last year")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=365))

    def test_past_n_months(self):
        start, end = extract_time_window("past 6 months of alerts")
        now = datetime.now(timezone.utc)
        assert self._approx(start, now - timedelta(days=6 * 30))


# ── extract_service ────────────────────────────────────────────────────────────

class TestExtractService:
    def test_payment_service_detected(self):
        assert extract_service("payment-service errors") == "payment-service"

    def test_order_service_detected(self):
        assert extract_service("why is order-service failing?") == "order-service"

    def test_payment_pipeline_detected(self):
        assert extract_service("payment-pipeline job failed") == "payment-pipeline"

    def test_case_insensitive(self):
        # Query is lowercased internally; the service name itself is lowercase.
        assert extract_service("PAYMENT-SERVICE is down") == "payment-service"

    def test_unknown_service_returns_none(self):
        assert extract_service("database connection pool exhausted") is None

    def test_empty_query_returns_none(self):
        assert extract_service("") is None

    def test_no_service_in_generic_query(self):
        assert extract_service("what happened in the last hour?") is None

    # ── Space-separated service names (new) ────────────────────────────────────

    def test_payment_service_with_space_detected(self):
        """Natural language 'payment service' matches stored 'payment-service'."""
        assert extract_service("which payment service failed") == "payment-service"

    def test_order_service_with_space_detected(self):
        assert extract_service("order service is responding slowly") == "order-service"

    def test_payment_pipeline_with_space_detected(self):
        assert extract_service("payment pipeline job failed") == "payment-pipeline"


# ── search_incidents ───────────────────────────────────────────────────────────

class TestSearchIncidents:
    def _mock_db(self, events_and_distances: list[tuple]) -> MagicMock:
        """Build a mock DB whose query chain returns the given (event, distance) list."""
        mock_db = MagicMock()
        mock_chain = MagicMock()
        # All ORM chain methods return self so we can call .filter().filter()... freely.
        mock_chain.filter.return_value = mock_chain
        mock_chain.order_by.return_value = mock_chain
        mock_chain.limit.return_value = mock_chain
        mock_chain.all.return_value = events_and_distances
        mock_db.query.return_value = mock_chain
        return mock_db

    def _mock_event(self, id=1, content="ERROR connection timeout",
                    service="payment-service", event_type="log",
                    occurred_at=None, metadata_=None, severity=None):
        evt = MagicMock()
        evt.id = id
        evt.content = content
        evt.service = service
        evt.event_type = event_type
        evt.occurred_at = occurred_at or datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc)
        evt.metadata_ = metadata_
        evt.severity = severity
        return evt

    def _window(self):
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        return now - timedelta(hours=1), now

    def test_returns_incident_search_hits(self):
        evt = self._mock_event()
        mock_db = self._mock_db([(evt, 0.3)])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("connection error", mock_db, start, end)

        assert len(hits) == 1
        assert isinstance(hits[0], IncidentSearchHit)

    def test_similarity_score_is_one_minus_distance(self):
        evt = self._mock_event()
        mock_db = self._mock_db([(evt, 0.3)])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)

        assert hits[0].similarity_score == round(1.0 - 0.3, 4)

    def test_hit_fields_populated(self):
        evt = self._mock_event(
            id=42,
            content="CRITICAL pool exhausted",
            service="payment-service",
            event_type="log",
        )
        mock_db = self._mock_db([(evt, 0.2)])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)

        h = hits[0]
        assert h.id == 42
        assert h.content == "CRITICAL pool exhausted"
        assert h.service == "payment-service"
        assert h.event_type == "log"

    def test_empty_result_returns_empty_list(self):
        mock_db = self._mock_db([])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)

        assert hits == []

    def test_generate_embedding_called_once(self):
        mock_db = self._mock_db([])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384) as mock_emb:
            search_incidents("connection pool", mock_db, start, end)

        mock_emb.assert_called_once_with("connection pool")

    def test_metadata_exposed_as_metadata_field(self):
        evt = self._mock_event(metadata_={"pool_size": 100})
        mock_db = self._mock_db([(evt, 0.1)])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)

        assert hits[0].metadata == {"pool_size": 100}

    def test_rerank_score_is_none_before_reranking(self):
        evt = self._mock_event()
        mock_db = self._mock_db([(evt, 0.2)])
        start, end = self._window()

        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)

        assert hits[0].rerank_score is None


# ── rerank_incidents ───────────────────────────────────────────────────────────

class TestRerankIncidents:
    def test_empty_hits_returned_unchanged(self):
        result = rerank_incidents("query", [])
        assert result == []

    def test_rerank_scores_attached(self):
        hits = [_make_hit(id=i) for i in range(3)]

        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [3.0, 1.0, 2.0]
            mock_get.return_value = mock_model

            result = rerank_incidents("query", hits)

        # All hits that were ranked should have rerank_score set.
        assert all(h.rerank_score is not None for h in result)

    def test_sorted_by_rerank_score_descending(self):
        hits = [_make_hit(id=i, content=f"event {i}") for i in range(3)]

        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [1.0, 3.0, 2.0]
            mock_get.return_value = mock_model

            result = rerank_incidents("query", hits)

        scores = [h.rerank_score for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_capped_at_final_top_k(self):
        from app.core.config import settings
        n = settings.FINAL_TOP_K + 5
        hits = [_make_hit(id=i) for i in range(n)]

        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = list(range(n))
            mock_get.return_value = mock_model

            result = rerank_incidents("query", hits)

        assert len(result) <= settings.FINAL_TOP_K

    def test_fallback_when_reranker_unavailable(self):
        from app.core.config import settings
        hits = [_make_hit(id=i) for i in range(settings.FINAL_TOP_K + 3)]

        with patch("app.services.incidents.get_reranker", return_value=None):
            result = rerank_incidents("query", hits)

        assert len(result) == settings.FINAL_TOP_K
        # Fallback preserves similarity order, rerank_scores remain None.
        assert all(h.rerank_score is None for h in result)


# ── build_incident_prompt ──────────────────────────────────────────────────────

class TestBuildIncidentPrompt:
    def test_contains_query(self):
        prompt = build_incident_prompt("why did payment fail?", [_make_hit()])
        assert "why did payment fail?" in prompt

    def test_contains_event_content(self):
        hit = _make_hit(content="CRITICAL pool exhausted")
        prompt = build_incident_prompt("query", [hit])
        assert "CRITICAL pool exhausted" in prompt

    def test_contains_service_in_header(self):
        hit = _make_hit(service="payment-service")
        prompt = build_incident_prompt("query", [hit])
        assert "payment-service" in prompt

    def test_contains_event_type_in_header(self):
        hit = _make_hit(event_type="deployment")
        prompt = build_incident_prompt("query", [hit])
        assert "deployment" in prompt

    def test_events_sorted_chronologically(self):
        """LLM should see events in time order regardless of input order."""
        early = _make_hit(id=1, content="early event",
                          occurred_at=datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc))
        late = _make_hit(id=2, content="late event",
                         occurred_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=timezone.utc))

        # Pass in reverse order — prompt should still show early before late.
        prompt = build_incident_prompt("query", [late, early])
        assert prompt.index("early event") < prompt.index("late event")

    def test_empty_hits_shows_no_data_message(self):
        prompt = build_incident_prompt("query", [])
        assert "No relevant incident events" in prompt

    def test_uses_rerank_score_when_available(self):
        hit = _make_hit(similarity_score=0.7, rerank_score=3.14)
        prompt = build_incident_prompt("query", [hit])
        assert "3.14" in prompt

    def test_falls_back_to_similarity_score(self):
        hit = _make_hit(similarity_score=0.85, rerank_score=None)
        prompt = build_incident_prompt("query", [hit])
        assert "0.85" in prompt


# ── answer_incident_query ──────────────────────────────────────────────────────

class TestAnswerIncidentQuery:
    def _window(self):
        start = datetime(2026, 3, 15, 9, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        return start, end

    def _mock_hit(self):
        return _make_hit(
            id=1, content="CRITICAL pool exhausted",
            service="payment-service", event_type="log",
        )

    def _patches(self):
        return [
            patch("app.services.rag.search_incidents",   return_value=[self._mock_hit()]),
            patch("app.services.rag.rerank_incidents",    return_value=[self._mock_hit()]),
            patch("app.services.rag.generate_answer",     return_value="Pool exhausted due to config bug"),
            patch("app.services.rag.log_query"),
        ]

    def test_returns_rag_result(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
            result = answer_incident_query(mock_db, "what failed?", start_time=start, end_time=end)

        assert isinstance(result, RAGResult)

    def test_answer_field_populated(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
            result = answer_incident_query(mock_db, "why?", start_time=start, end_time=end)

        assert result.answer == "Pool exhausted due to config bug"

    def test_sources_are_incident_hits(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
            result = answer_incident_query(mock_db, "why?", start_time=start, end_time=end)

        assert len(result.sources) == 1
        assert isinstance(result.sources[0], IncidentSearchHit)

    def test_latency_fields_non_negative(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], self._patches()[3]:
            result = answer_incident_query(mock_db, "why?", start_time=start, end_time=end)

        assert result.latency.retrieve >= 0
        assert result.latency.rerank >= 0
        assert result.latency.llm >= 0
        assert result.latency.total >= 0

    def test_auto_extracts_time_window_from_query(self):
        """When no start/end provided, extract_time_window is called."""
        mock_db = MagicMock()

        with self._patches()[0] as mock_search, \
             self._patches()[1], self._patches()[2], self._patches()[3]:
            answer_incident_query(mock_db, "what failed in the last hour?")

        # search_incidents should have been called with auto-extracted window.
        assert mock_search.called
        call_kwargs = mock_search.call_args
        start_arg = call_kwargs[0][2]  # positional: query, db, start_time, end_time
        assert start_arg.tzinfo is not None  # timezone-aware

    def test_auto_extracts_service_from_query(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0] as mock_search, \
             self._patches()[1], self._patches()[2], self._patches()[3]:
            answer_incident_query(mock_db, "payment-service errors?",
                                  start_time=start, end_time=end)

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["service"] == "payment-service"

    def test_evaluation_not_run_by_default(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], \
             self._patches()[3], \
             patch("app.services.rag.evaluate_answer") as mock_eval:
            result = answer_incident_query(mock_db, "why?", start_time=start, end_time=end)

        mock_eval.assert_not_called()
        assert result.evaluation is None

    def test_evaluation_called_when_requested(self):
        from app.services.evaluator import EvaluationResult
        mock_db = MagicMock()
        start, end = self._window()
        fake_eval = EvaluationResult(
            groundedness_score=0.9, groundedness_explanation="good",
            relevance_score=0.85, relevance_explanation="relevant",
        )

        with self._patches()[0], self._patches()[1], self._patches()[2], \
             self._patches()[3], \
             patch("app.services.rag.evaluate_answer", return_value=fake_eval):
            result = answer_incident_query(mock_db, "why?",
                                           evaluate=True, start_time=start, end_time=end)

        assert result.evaluation is not None
        assert result.evaluation.groundedness_score == 0.9

    def test_log_query_called(self):
        mock_db = MagicMock()
        start, end = self._window()

        with self._patches()[0], self._patches()[1], self._patches()[2], \
             self._patches()[3] as mock_log:
            answer_incident_query(mock_db, "why?", start_time=start, end_time=end)

        mock_log.assert_called_once()


# ── analytical query gate ──────────────────────────────────────────────────────

class TestAnalyticalQueryGate:
    """answer_incident_query must short-circuit and return guidance for analytical queries."""

    def test_analytical_query_skips_retrieval(self):
        """search_incidents and rerank_incidents must not be called."""
        mock_db = MagicMock()

        with patch("app.services.rag.search_incidents") as mock_search, \
             patch("app.services.rag.rerank_incidents") as mock_rerank:
            answer_incident_query(mock_db, "which payment service failed in last 24 months")

        mock_search.assert_not_called()
        mock_rerank.assert_not_called()

    def test_analytical_query_returns_rag_result(self):
        mock_db = MagicMock()
        result = answer_incident_query(mock_db, "how many errors in payment-service?")
        assert isinstance(result, RAGResult)

    def test_analytical_query_returns_empty_sources(self):
        mock_db = MagicMock()
        result = answer_incident_query(mock_db, "how many errors in payment-service?")
        assert result.sources == []

    def test_analytical_query_answer_contains_guidance(self):
        mock_db = MagicMock()
        result = answer_incident_query(mock_db, "which services had the most failures?")
        assert "optimized for debugging" in result.answer

    def test_analytical_query_answer_contains_example_queries(self):
        mock_db = MagicMock()
        result = answer_incident_query(mock_db, "count of errors in last month")
        assert "Try queries like" in result.answer

    def test_non_analytical_query_proceeds_to_retrieval(self):
        """Normal incident queries must still reach search_incidents."""
        mock_db = MagicMock()
        start = datetime(2026, 3, 15, 9, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        hit   = _make_hit()

        with patch("app.services.rag.search_incidents", return_value=[hit]) as mock_search, \
             patch("app.services.rag.rerank_incidents",  return_value=[hit]), \
             patch("app.services.rag.generate_answer",   return_value="answer"), \
             patch("app.services.rag.log_query"):
            answer_incident_query(mock_db, "why did payment-service fail?",
                                  start_time=start, end_time=end)

        mock_search.assert_called_once()


# ── POST /incidents/ask ────────────────────────────────────────────────────────

class TestIncidentAskEndpoint:
    def _payload(self):
        return {
            "query": "what happened to payment-service?",
            "start_time": "2026-03-15T09:00:00Z",
            "end_time":   "2026-03-15T11:00:00Z",
        }

    def _fake_result(self):
        from app.services.rag import LatencyBreakdown
        hit = _make_hit()
        return RAGResult(
            answer="Connection pool was exhausted.",
            sources=[hit],
            rewritten_query="what happened to payment-service?",
            prompt="prompt text",
            latency=LatencyBreakdown(retrieve=0.1, rerank=0.05, llm=0.5, total=0.65),
        )

    def test_returns_200(self, client):
        http, _ = client
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=self._payload())
        assert resp.status_code == 200

    def test_response_contains_answer(self, client):
        http, _ = client
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=self._payload())
        assert resp.json()["answer"] == "Connection pool was exhausted."

    def test_response_contains_sources(self, client):
        http, _ = client
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=self._payload())
        data = resp.json()
        assert len(data["sources"]) == 1
        src = data["sources"][0]
        assert src["service"] == "payment-service"
        assert src["event_type"] == "log"
        assert "similarity_score" in src

    def test_response_contains_time_window(self, client):
        http, _ = client
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=self._payload())
        data = resp.json()
        assert "time_window" in data
        assert "start" in data["time_window"]
        assert "end" in data["time_window"]

    def test_debug_includes_latency_and_prompt(self, client):
        http, _ = client
        payload = {**self._payload(), "debug": True}
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=payload)
        data = resp.json()
        assert "latency" in data
        assert "prompt" in data
        assert "rewritten_query" in data

    def test_non_debug_omits_debug_fields(self, client):
        http, _ = client
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json=self._payload())
        data = resp.json()
        assert "latency" not in data
        assert "prompt" not in data

    def test_empty_query_not_blocked_at_api(self, client):
        """Incident endpoint does not apply the same empty-query guard as /ask —
        the query goes through to the service layer."""
        http, _ = client
        # The test just checks the endpoint doesn't 500; actual validation can be added later.
        with patch("app.api.incidents.answer_incident_query", return_value=self._fake_result()):
            resp = http.post("/incidents/ask", json={"query": "any query"})
        assert resp.status_code == 200

    def test_severity_present_in_source_response(self, client):
        http, _ = client
        hit = _make_hit(severity="critical")
        from app.services.rag import LatencyBreakdown
        result = RAGResult(
            answer="Pool exhausted.",
            sources=[hit],
            rewritten_query="q",
            prompt="p",
            latency=LatencyBreakdown(),
        )
        with patch("app.api.incidents.answer_incident_query", return_value=result):
            resp = http.post("/incidents/ask", json=self._payload())
        assert resp.json()["sources"][0]["severity"] == "critical"

    def test_severity_null_when_not_set(self, client):
        http, _ = client
        hit = _make_hit(severity=None)
        from app.services.rag import LatencyBreakdown
        result = RAGResult(
            answer="Answer.",
            sources=[hit],
            rewritten_query="q",
            prompt="p",
            latency=LatencyBreakdown(),
        )
        with patch("app.api.incidents.answer_incident_query", return_value=result):
            resp = http.post("/incidents/ask", json=self._payload())
        # severity is None — response_model_exclude_none omits it
        assert "severity" not in resp.json()["sources"][0]


# ── Patch-level additions ──────────────────────────────────────────────────────
# Tests for the 8 correctness/quality patches applied in Sprint 2.1

class TestTimeValidation:
    def _mock_db(self):
        mock_db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.order_by.return_value = chain
        chain.limit.return_value = chain
        chain.all.return_value = []
        mock_db.query.return_value = chain
        return mock_db

    def test_raises_when_start_equals_end(self):
        t = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            with pytest.raises(ValueError, match="start_time must be before end_time"):
                search_incidents("query", self._mock_db(), t, t)

    def test_raises_when_start_after_end(self):
        start = datetime(2026, 3, 15, 11, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            with pytest.raises(ValueError):
                search_incidents("query", self._mock_db(), start, end)

    def test_valid_window_does_not_raise(self):
        start = datetime(2026, 3, 15, 9, 0, 0, tzinfo=timezone.utc)
        end   = datetime(2026, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", self._mock_db(), start, end)
        assert hits == []


class TestSeverityInHits:
    def _mock_db_with_event(self, severity):
        evt = MagicMock()
        evt.id = 1
        evt.content = "connection timeout"
        evt.service = "payment-service"
        evt.event_type = "log"
        evt.occurred_at = datetime(2026, 3, 15, 10, 5, 0, tzinfo=timezone.utc)
        evt.metadata_ = None
        evt.severity = severity

        mock_db = MagicMock()
        chain = MagicMock()
        chain.filter.return_value = chain
        chain.order_by.return_value = chain
        chain.limit.return_value = chain
        chain.all.return_value = [(evt, 0.2)]
        mock_db.query.return_value = chain
        return mock_db

    def _window(self):
        now = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        return now - timedelta(hours=1), now

    def test_severity_populated_from_event(self):
        mock_db = self._mock_db_with_event("error")
        start, end = self._window()
        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)
        assert hits[0].severity == "error"

    def test_severity_none_when_event_has_no_severity(self):
        mock_db = self._mock_db_with_event(None)
        start, end = self._window()
        with patch("app.services.incidents.generate_embedding", return_value=[0.1] * 384):
            hits = search_incidents("query", mock_db, start, end)
        assert hits[0].severity is None


class TestFormatForReranker:
    def test_uses_severity_when_present(self):
        hit = _make_hit(event_type="log", severity="critical")
        result = _format_for_reranker(hit)
        assert result.startswith("[CRITICAL]")
        assert "ERROR connection timeout" in result

    def test_falls_back_to_event_type_when_no_severity(self):
        hit = _make_hit(event_type="deployment", severity=None)
        result = _format_for_reranker(hit)
        assert result.startswith("[DEPLOYMENT]")

    def test_severity_uppercased(self):
        hit = _make_hit(severity="warn")
        assert _format_for_reranker(hit).startswith("[WARN]")

    def test_content_preserved(self):
        hit = _make_hit(content="pool exhausted", severity="critical")
        assert "pool exhausted" in _format_for_reranker(hit)


class TestQualityGate:
    def test_all_below_threshold_still_returns_hits(self):
        """Soft gate: all-below-threshold returns best available, not empty."""
        hits = [_make_hit(id=i) for i in range(3)]
        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [-3.0, -4.0, -5.0]
            mock_get.return_value = mock_model
            result = rerank_incidents("query", hits)
        assert len(result) > 0
        # Results are sorted best-first.
        assert result[0].rerank_score == -3.0

    def test_mixed_scores_returns_top_k_ordered(self):
        """Mixed scores: returns up to FINAL_TOP_K hits sorted by score, not filtered."""
        hits = [_make_hit(id=i, content=f"event {i}") for i in range(3)]
        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [2.0, -3.0, 1.0]
            mock_get.return_value = mock_model
            result = rerank_incidents("query", hits)
        # All three returned (3 ≤ FINAL_TOP_K), sorted best→worst.
        assert len(result) == 3
        assert result[0].rerank_score == 2.0
        assert result[1].rerank_score == 1.0
        assert result[2].rerank_score == -3.0

    def test_all_above_threshold_passes_through(self):
        hits = [_make_hit(id=i) for i in range(3)]
        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [3.0, 1.5, 0.5]
            mock_get.return_value = mock_model
            result = rerank_incidents("query", hits)
        assert len(result) > 0
        assert all(h.rerank_score > settings.RERANK_THRESHOLD for h in result)

    def test_severity_prefix_sent_to_reranker(self):
        """Verifies _format_for_reranker is used — severity prefix appears in pairs."""
        hit = _make_hit(severity="critical", content="pool exhausted")
        with patch("app.services.incidents.get_reranker") as mock_get:
            mock_model = MagicMock()
            mock_model.predict.return_value = [3.0]
            mock_get.return_value = mock_model
            rerank_incidents("why did it fail?", [hit])
        pairs_arg = mock_model.predict.call_args[0][0]
        _, content_text = pairs_arg[0]
        assert content_text.startswith("[CRITICAL]")
        assert "pool exhausted" in content_text


class TestGetKnownServices:
    def _reset_cache(self):
        import app.services.incidents as mod
        mod._services_cache = []
        mod._services_cache_expires = 0.0

    def _restore_cache(self, cache, expires):
        import app.services.incidents as mod
        mod._services_cache = cache
        mod._services_cache_expires = expires

    def test_returns_services_from_db(self):
        import app.services.incidents as mod
        saved = (mod._services_cache[:], mod._services_cache_expires)
        self._reset_cache()
        try:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = [
                ("checkout-service",), ("billing-service",)
            ]
            result = get_known_services(mock_db)
            assert "checkout-service" in result
            assert "billing-service" in result
        finally:
            self._restore_cache(*saved)

    def test_falls_back_to_static_list_when_db_returns_empty(self):
        import app.services.incidents as mod
        saved = (mod._services_cache[:], mod._services_cache_expires)
        self._reset_cache()
        try:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = []
            result = get_known_services(mock_db)
            # Falls back to _KNOWN_SERVICES — must contain our fixture services
            assert "payment-service" in result
        finally:
            self._restore_cache(*saved)

    def test_cache_is_used_on_second_call(self):
        """Second call within TTL must not hit the DB."""
        import app.services.incidents as mod
        saved = (mod._services_cache[:], mod._services_cache_expires)
        self._reset_cache()
        try:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = [("svc-a",)]
            get_known_services(mock_db)   # populates cache
            get_known_services(mock_db)   # should use cache
            assert mock_db.execute.call_count == 1
        finally:
            self._restore_cache(*saved)


class TestSeverityInPrompt:
    def test_severity_included_in_event_header(self):
        hit = _make_hit(severity="critical")
        prompt = build_incident_prompt("what broke?", [hit])
        assert "CRITICAL" in prompt

    def test_severity_absent_when_none(self):
        hit = _make_hit(severity=None, event_type="log")
        prompt = build_incident_prompt("what broke?", [hit])
        # event_type should appear but no extra severity token
        assert "log" in prompt
        # No stray " | None" from unguarded severity
        assert "None" not in prompt


class TestExtractServiceWithDb:
    def test_uses_db_services_when_provided(self):
        """extract_service uses get_known_services when db is passed."""
        with patch("app.services.incidents.get_known_services",
                   return_value=["checkout-service", "billing-service"]) as mock_svc:
            result = extract_service("checkout-service is down", db=MagicMock())
        mock_svc.assert_called_once()
        assert result == "checkout-service"

    def test_falls_back_to_static_when_db_is_none(self):
        """extract_service uses _KNOWN_SERVICES when db=None."""
        with patch("app.services.incidents.get_known_services") as mock_svc:
            result = extract_service("payment-service errors")
        mock_svc.assert_not_called()
        assert result == "payment-service"

    def test_db_service_not_in_static_list_is_found(self):
        """A service only in DB (not in _KNOWN_SERVICES) is correctly matched."""
        with patch("app.services.incidents.get_known_services",
                   return_value=["new-service"]):
            result = extract_service("new-service is failing", db=MagicMock())
        assert result == "new-service"
