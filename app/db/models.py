from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector

from app.core.config import settings


class Base(DeclarativeBase):
    pass


class QueryLog(Base):
    """One row per completed /ask request — used for analytics and quality monitoring."""

    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    # LLM-as-judge scores — None when evaluate=False for that request.
    groundedness_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    # Total wall-clock latency for the full RAG pipeline (seconds).
    latency = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Feedback(Base):
    """User-submitted thumbs-up / thumbs-down on a query–answer pair."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    # "good" or "bad" — enforced at the API layer via Pydantic Literal.
    rating = Column(String(10), nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class IncidentEvent(Base):
    """A single timestamped signal from any production data source.

    Stores raw logs, deployment events, alerts, and pipeline metadata
    in one unified table so temporal and semantic queries can span sources.

    The embedding is generated from *normalised* content (timestamps, PIDs
    and noise stripped) so semantically identical messages cluster together
    regardless of their exact raw format.

    Deduplication is enforced at the DB level via the UNIQUE constraint on
    event_id — callers must generate event_id as sha256(service|occurred_at|content[:200]).
    """

    __tablename__ = "incident_events"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Content hash — prevents duplicate ingestion on replay or retry.
    event_id = Column(String(64), nullable=False, unique=True)

    # Signal classification — drives filtering in temporal queries.
    event_type = Column(String(32), nullable=False)   # log | deployment | alert | metadata

    # Source identifier — always present, used in index (service, occurred_at).
    service = Column(String(128), nullable=False)

    # Optional severity; not all event types carry one.
    severity = Column(String(16), nullable=True)      # error | warn | info | debug | critical

    # THE critical field for incident analysis — enables temporal correlation.
    occurred_at = Column(DateTime(timezone=True), nullable=False, index=True)

    # Original raw content — stored verbatim so the LLM sees real log lines.
    content = Column(Text, nullable=False)

    # Embedding generated from *normalised* content (noise stripped).
    embedding = Column(Vector(settings.EMBEDDING_DIMENSION), nullable=False)

    # Optional group key linking causally related events (e.g. one incident).
    correlation_id = Column(String(64), nullable=True, index=True)

    # Arbitrary structured fields (job ID, version, region, etc.).
    metadata_ = Column("metadata", JSONB, nullable=True)
