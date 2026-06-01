"""Incident API: batch ingestion and incident-aware RAG query."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.ingestion import (
    IngestResult,
    ingest_events,
    ingest_logs,
    ingest_pipeline_metadata,
)
from app.services.incidents import extract_time_window, extract_service
from app.services.rag import answer_incident_query

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────────

class LogEntry(BaseModel):
    """A single application log line."""
    service: str
    occurred_at: datetime
    content: str
    severity: str | None = None
    correlation_id: str | None = None
    metadata: dict | None = None


class EventEntry(BaseModel):
    """A deployment, alert, or config-change event."""
    service: str
    occurred_at: datetime
    content: str
    event_type: str = "deployment"    # deployment | alert | config_change
    severity: str | None = None
    correlation_id: str | None = None
    metadata: dict | None = None


class MetadataEntry(BaseModel):
    """A pipeline job run record."""
    service: str
    occurred_at: datetime
    content: str
    correlation_id: str | None = None
    metadata: dict | None = None


class BatchIngestRequest(BaseModel):
    """At least one of logs / events / metadata must be non-empty."""
    logs:     list[LogEntry]      = []
    events:   list[EventEntry]    = []
    metadata: list[MetadataEntry] = []

    @field_validator("logs", "events", "metadata", mode="before")
    @classmethod
    def _default_empty(cls, v):
        return v or []


# ── Response schema ────────────────────────────────────────────────────────────

class BatchIngestResponse(BaseModel):
    logs:     IngestResult
    events:   IngestResult
    metadata: IngestResult
    # Aggregate totals across all three signal types.
    total_ingested: int
    total_skipped:  int
    total_received: int


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/ingest/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def batch_ingest(payload: BatchIngestRequest, db: Session = Depends(get_db)):
    """Ingest logs, events, and pipeline metadata in a single request.

    All three signal types are stored in the incident_events table with
    full deduplication: re-submitting the same payload is safe and returns
    zero ingested / all-skipped counts.

    Each list is processed independently so a failure in one type does not
    roll back the others.  Results per type are returned in the response.

    Args:
        payload: BatchIngestRequest containing logs, events, and metadata lists.

    Returns:
        BatchIngestResponse with per-type and aggregate ingestion counts.
    """
    log_result  = ingest_logs(
        [e.model_dump() for e in payload.logs], db
    )
    evt_result  = ingest_events(
        [e.model_dump() for e in payload.events], db
    )
    meta_result = ingest_pipeline_metadata(
        [e.model_dump() for e in payload.metadata], db
    )

    total_ingested = log_result.ingested + evt_result.ingested + meta_result.ingested
    total_skipped  = log_result.skipped  + evt_result.skipped  + meta_result.skipped
    total_received = log_result.total    + evt_result.total     + meta_result.total

    logger.info(
        "POST /ingest/batch  received=%d  ingested=%d  skipped=%d",
        total_received, total_ingested, total_skipped,
    )

    return BatchIngestResponse(
        logs=log_result,
        events=evt_result,
        metadata=meta_result,
        total_ingested=total_ingested,
        total_skipped=total_skipped,
        total_received=total_received,
    )


# ── Incident query schemas ─────────────────────────────────────────────────────

class IncidentAskRequest(BaseModel):
    """Query the incident knowledge base with temporal + semantic search."""

    query: str
    debug: bool = False
    evaluate: bool = False
    expected_answer: str | None = None

    # Optional explicit overrides — auto-extracted from query when omitted.
    start_time: datetime | None = None
    end_time: datetime | None = None
    service: str | None = None
    event_types: list[str] | None = None


class IncidentSource(BaseModel):
    """A single incident event returned as a source in the answer."""

    id: int
    content: str
    occurred_at: datetime
    service: str
    event_type: str
    similarity_score: float
    severity: str | None = None
    rerank_score: float | None = None
    metadata: dict | None = None


class IncidentAskResponse(BaseModel):
    """Response from POST /incidents/ask."""

    answer: str
    sources: list[IncidentSource]
    # Always present so callers can see which window was searched.
    time_window: dict     # {"start": ISO str, "end": ISO str}

    # Cross-encoder confidence tier — always present.
    # "high" | "medium" | "low" | "none" (no hits) | "unknown" (reranker unavailable)
    confidence: str = "unknown"

    # Evaluation scores (present when evaluate=True or debug=True).
    evaluation: dict | None = None

    # Debug-only fields (present when debug=True).
    rewritten_query: str | None = None
    prompt: str | None = None
    latency: dict | None = None


# ── Incident ask endpoint ──────────────────────────────────────────────────────

@router.post(
    "/incidents/ask",
    response_model=IncidentAskResponse,
    response_model_exclude_none=True,
)
def incident_ask(payload: IncidentAskRequest, db: Session = Depends(get_db)):
    """Answer a question about production incidents using temporal + semantic search.

    Automatically extracts a time window and service name from the query when
    not provided explicitly.  Uses the full RAG pipeline (retrieve → rerank →
    LLM) over the incident_events table rather than the generic documents table.

    Example queries:
        "What happened to payment-service in the last hour?"
        "Why did the payment pipeline fail yesterday?"
        "Show me critical alerts from order-service in the last 6 hours"

    Args:
        payload: IncidentAskRequest with query and optional filters.

    Returns:
        IncidentAskResponse with answer, incident sources, and time window used.
    """
    # Resolve time window — explicit payload values take precedence over auto-extraction.
    start_time = payload.start_time
    end_time = payload.end_time
    if start_time is None or end_time is None:
        auto_start, auto_end = extract_time_window(payload.query)
        start_time = start_time or auto_start
        end_time = end_time or auto_end

    service = payload.service or extract_service(payload.query, db)
    should_evaluate = payload.debug or payload.evaluate

    logger.info(
        "POST /incidents/ask  query=%r  window=[%s → %s]  service=%r",
        payload.query,
        start_time.isoformat(),
        end_time.isoformat(),
        service,
    )

    result = answer_incident_query(
        db,
        payload.query,
        evaluate=should_evaluate,
        expected_answer=payload.expected_answer,
        start_time=start_time,
        end_time=end_time,
        service=service,
        event_types=payload.event_types,
    )

    sources = [
        IncidentSource(
            id=hit.id,
            content=hit.content,
            occurred_at=hit.occurred_at,
            service=hit.service,
            event_type=hit.event_type,
            similarity_score=hit.similarity_score,
            severity=hit.severity,
            rerank_score=hit.rerank_score,
            metadata=hit.metadata,
        )
        for hit in result.sources
    ]

    response = IncidentAskResponse(
        answer=result.answer,
        sources=sources,
        time_window={
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        },
        confidence=result.confidence,
    )

    if should_evaluate and result.evaluation is not None:
        ev = result.evaluation
        eval_dict: dict = {
            "groundedness_score": ev.groundedness_score,
            "groundedness_explanation": ev.groundedness_explanation,
            "relevance_score": ev.relevance_score,
            "relevance_explanation": ev.relevance_explanation,
        }
        if ev.correctness_score is not None:
            eval_dict["correctness_score"] = ev.correctness_score
            eval_dict["correctness_explanation"] = ev.correctness_explanation
        response.evaluation = eval_dict

    if payload.debug:
        response.rewritten_query = result.rewritten_query
        response.prompt = result.prompt
        response.latency = {
            "retrieve": result.latency.retrieve,
            "rerank": result.latency.rerank,
            "llm": result.latency.llm,
            "eval": result.latency.eval,
            "total": result.latency.total,
        }

    return response
