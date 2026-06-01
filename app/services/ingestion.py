"""Ingestion pipeline for incident signals.

Handles three distinct signal types — logs, deployment/alert events, and
pipeline metadata — all normalised into the unified incident_events table.

Design decisions:
  - Raw content is stored verbatim; normalisation affects only the embedding.
    This means the LLM always sees real log lines, not stripped versions.
  - event_id is a SHA-256 hash of (service, occurred_at, raw content[:200]).
    Deduplication is enforced at the DB level (UNIQUE constraint) so replays
    and retries are safe.
  - Embeddings are generated in one batch call per ingest invocation, not one
    call per item.  This is ~10× faster for batches of 50+ items.
  - Normalisation strips timing noise (timestamps, PIDs, thread IDs, hex
    addresses) so semantically identical log lines cluster together in the
    embedding space even when they carry different metadata.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.models import IncidentEvent
from app.services.embedding import generate_embeddings_batch

logger = logging.getLogger(__name__)


# ── Log normalisation ──────────────────────────────────────────────────────────
#
# Applied only to generate embeddings — the raw content is always stored.
# Order matters: strip timestamps first so level-prefix stripping works on
# the remaining string.

_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"(?:[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"\s*",
)
_BRACKETED_TIMESTAMP_RE = re.compile(
    r"\[\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\]]*\]\s*"
)
_LEVEL_PREFIX_RE = re.compile(
    r"^\[?(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL)\]?\s*",
    re.IGNORECASE,
)
_PID_RE = re.compile(r"\[?pid[=:\s]\d+\]?", re.IGNORECASE)
_THREAD_RE = re.compile(r"\[[\w.\-]+-\d+\]|\bthread[=:\s]\S+", re.IGNORECASE)
_HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]{4,}\b")
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalise_log(raw: str) -> str:
    """Strip timing noise from a log line to improve embedding quality.

    Removes: leading timestamps, log-level prefixes, PIDs, thread IDs,
    hex addresses, and UUIDs.  Collapses multiple spaces.  Falls back to
    the original string if the result would be empty.

    Examples:
        "2024-01-01 ERROR pid=123 connection failed"  → "connection failed"
        "[2024-01-15T10:05:33Z] [ERROR] DB timeout"   → "DB timeout"
        "WARN thread=io-worker-3 pool exhausted"       → "pool exhausted"

    Args:
        raw: The original log line or event description.

    Returns:
        Normalised string suitable for embedding.
    """
    text = raw

    # 1. Strip bracketed ISO timestamps: [2024-01-15T10:05:33Z]
    text = _BRACKETED_TIMESTAMP_RE.sub("", text)
    # 2. Strip leading bare ISO timestamps: 2024-01-15T10:05:33Z
    text = _TIMESTAMP_RE.sub("", text, count=1)
    # 3. Strip leading log level: ERROR | [WARN] | INFO
    text = _LEVEL_PREFIX_RE.sub("", text)
    # 4. Strip PID tokens: pid=123 | [pid:18432]
    text = _PID_RE.sub("", text)
    # 5. Strip thread identifiers: [main-thread-7] | thread=io-worker-3
    text = _THREAD_RE.sub("", text)
    # 6. Strip hex memory addresses: 0x7f4a3b2c1d0e
    text = _HEX_ADDR_RE.sub("", text)
    # 7. Strip UUIDs: 550e8400-e29b-41d4-a716-446655440000
    text = _UUID_RE.sub("", text)
    # 8. Collapse runs of whitespace and strip edges
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text if text else raw


# ── Deduplication ──────────────────────────────────────────────────────────────

def _make_event_id(service: str, occurred_at: datetime, content: str) -> str:
    """Compute a 64-char SHA-256 deduplication key.

    Including occurred_at ensures two identical log lines at different times
    produce different IDs (they are distinct events).  Using only the first
    200 chars of content is deliberate — the tail of a log line is often
    dynamic noise that differs between duplicates.
    """
    key = f"{service}|{occurred_at.isoformat()}|{content[:200]}"
    return hashlib.sha256(key.encode()).hexdigest()


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    """Outcome of one batch ingest call."""
    ingested: int   # rows actually written to the DB
    skipped: int    # rows that already existed (duplicate event_id)
    total: int      # len(input list)


# ── Internal engine ────────────────────────────────────────────────────────────

def _ingest_batch(
    items: list[dict],
    default_event_type: str,
    db: Session,
) -> IngestResult:
    """Core ingest logic shared by all three public functions.

    Each item dict must contain:
        service     (str)
        occurred_at (datetime, timezone-aware)
        content     (str)

    Optional keys:
        event_type    — overrides default_event_type if present
        severity      (str | None)
        correlation_id (str | None)
        metadata      (dict | None)

    The function:
      1. Normalises content for embedding.
      2. Computes event_id (dedup hash) from raw content.
      3. Generates embeddings in a single batch call.
      4. Inserts all rows with ON CONFLICT (event_id) DO NOTHING.
      5. Returns IngestResult with accurate ingested / skipped counts.

    Args:
        items:              List of item dicts (see above for required keys).
        default_event_type: Fallback event_type when the item dict lacks one.
        db:                 Active SQLAlchemy session.

    Returns:
        IngestResult with ingested, skipped, and total counts.
    """
    if not items:
        return IngestResult(ingested=0, skipped=0, total=0)

    # ── Step 1: Normalise + compute event IDs ─────────────────────────────────
    normalised: list[str] = []
    rows: list[dict] = []

    for item in items:
        service     = item["service"]
        occurred_at = item["occurred_at"]
        raw_content = item["content"]

        if not isinstance(occurred_at, datetime):
            raise ValueError(
                f"occurred_at must be a datetime, got {type(occurred_at)} "
                f"for service={service!r}"
            )

        # Ensure timezone-awareness so PostgreSQL TIMESTAMPTZ is satisfied.
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)

        event_id = _make_event_id(service, occurred_at, raw_content)
        normalised.append(normalise_log(raw_content))
        rows.append({
            "event_id":       event_id,
            "event_type":     item.get("event_type", default_event_type),
            "service":        service,
            "severity":       item.get("severity"),
            "occurred_at":    occurred_at,
            "content":        raw_content,       # raw stored verbatim
            "correlation_id": item.get("correlation_id"),
            "metadata":       item.get("metadata"),
            # embedding filled below after batch encode
        })

    # ── Step 2: Batch embed normalised content ────────────────────────────────
    embeddings = generate_embeddings_batch(normalised)
    for row, emb in zip(rows, embeddings):
        row["embedding"] = emb

    # ── Step 3: Insert with ON CONFLICT DO NOTHING ────────────────────────────
    # The UNIQUE constraint on event_id handles duplicates atomically.
    # result.rowcount returns the number of rows actually inserted (conflicts
    # are excluded from the count by PostgreSQL).
    stmt = pg_insert(IncidentEvent.__table__).values(rows)
    stmt = stmt.on_conflict_do_nothing(index_elements=["event_id"])
    result = db.execute(stmt)
    db.commit()

    ingested = result.rowcount
    skipped  = len(rows) - ingested

    logger.info(
        "Ingested %d/%d %s events (skipped %d duplicates)",
        ingested, len(rows), default_event_type, skipped,
    )
    return IngestResult(ingested=ingested, skipped=skipped, total=len(rows))


# ── Public interface ───────────────────────────────────────────────────────────

def ingest_logs(logs: list[dict], db: Session) -> IngestResult:
    """Ingest application log lines.

    Expected keys per item: service, occurred_at, content.
    Optional keys: severity, correlation_id, metadata.

    Args:
        logs: List of log item dicts.
        db:   Active SQLAlchemy session.

    Returns:
        IngestResult with ingested, skipped, total counts.
    """
    return _ingest_batch(logs, default_event_type="log", db=db)


def ingest_events(events: list[dict], db: Session) -> IngestResult:
    """Ingest deployment and alert events.

    Expected keys: service, occurred_at, content, event_type
    (typically "deployment" or "alert").
    Optional keys: severity, correlation_id, metadata.

    Args:
        events: List of event item dicts.
        db:     Active SQLAlchemy session.

    Returns:
        IngestResult with ingested, skipped, total counts.
    """
    return _ingest_batch(events, default_event_type="deployment", db=db)


def ingest_pipeline_metadata(metadata: list[dict], db: Session) -> IngestResult:
    """Ingest pipeline job run records.

    Expected keys: service, occurred_at, content.
    Optional keys: correlation_id, metadata (job ID, run status, duration, etc.).

    Args:
        metadata: List of pipeline metadata dicts.
        db:       Active SQLAlchemy session.

    Returns:
        IngestResult with ingested, skipped, total counts.
    """
    return _ingest_batch(metadata, default_event_type="metadata", db=db)
