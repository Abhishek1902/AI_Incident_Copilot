"""Incident-aware hybrid retrieval: time-bounded + semantic search.

Extends the base vector search with temporal filtering against the
incident_events table.  The primary use case is queries like
"what happened in payment-service in the last hour?".

Design:
  - Time window is always required — bounded queries are far faster than
    full-table scans and are the correct mental model for incident analysis.
  - Vector similarity is computed *inside* the time window, not across all events.
  - The cross-encoder reranker is shared with the document pipeline via
    get_reranker() so only one model is loaded in memory.
  - extract_time_window / extract_service are simple heuristics — fast,
    deterministic, zero LLM cost.  Replace with an LLM call if higher
    accuracy is needed later.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import IncidentEvent
from app.services.embedding import generate_embedding
from app.services.reranker import get_reranker

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class IncidentSearchHit:
    """A single incident event retrieval result.

    Attributes:
        id:               Primary key of the incident_events row.
        event_id:         SHA-256 dedup hash. Stable across reseeds; preferred
                          identifier for external systems (e.g. eval ground-truth).
        content:          Raw log/event text as stored (verbatim).
        occurred_at:      Timezone-aware UTC timestamp of the event.
        service:          Service that emitted this event.
        event_type:       Signal class: "log" | "deployment" | "alert" | "metadata".
        similarity_score: 1 - cosine_distance in [0, 1]; higher = more similar to query.
        metadata:         Optional JSONB payload (job IDs, versions, alert thresholds…).
        rerank_score:     Cross-encoder logit, set by rerank_incidents().  None until
                          reranking has run.  Higher is better; scale is unbounded.
    """

    id: int
    event_id: str
    content: str
    occurred_at: datetime
    service: str
    event_type: str
    similarity_score: float
    severity: str | None = None
    metadata: dict | None = None
    rerank_score: float | None = None


# ── Service registry ───────────────────────────────────────────────────────────

# Static fallback — used when the DB is empty or unreachable.
_KNOWN_SERVICES = [
    "payment-service",
    "order-service",
    "payment-pipeline",
    "auth-service",
    "api-gateway",
    "notification-service",
    "inventory-service",
]

# Simple TTL cache: avoids a DB round-trip on every query while staying current.
_services_cache: list[str] = []
_services_cache_expires: float = 0.0
_SERVICES_CACHE_TTL: float = 300.0   # seconds


def get_known_services(db: Session) -> list[str]:
    """Return distinct service names from incident_events, cached for 5 minutes.

    Falls back to the static _KNOWN_SERVICES list when the DB is empty or the
    query fails (e.g. during tests with a mock Session).

    Args:
        db: Active SQLAlchemy session.

    Returns:
        List of known service name strings.
    """
    global _services_cache, _services_cache_expires
    if time.monotonic() < _services_cache_expires and _services_cache:
        return _services_cache

    try:
        rows = db.execute(
            text("SELECT DISTINCT service FROM incident_events ORDER BY service")
        ).fetchall()
        fresh = [r[0] for r in rows if isinstance(r[0], str)]
    except Exception:
        fresh = []

    # If the DB returned nothing (empty table or mock), use the static list.
    result = fresh if fresh else _KNOWN_SERVICES
    _services_cache = result
    _services_cache_expires = time.monotonic() + _SERVICES_CACHE_TTL
    return result


# ── Time window extraction ─────────────────────────────────────────────────────

# Matches "last/past N months" or "last/past N years".
# Group 1 = numeric count, group 2 = unit ("month", "months", "year", "years").
_TIME_NUMERIC_RE = re.compile(
    r"(?:last|past)\s+(\d+)\s+(months?|years?)", re.IGNORECASE
)


def extract_time_window(query: str) -> tuple[datetime, datetime]:
    """Parse a natural-language query for a time-window hint.

    Heuristics checked in order (most specific first):

        "last/past N months"               → past N*30 days
        "last/past N years"                → past N*365 days
        "last month"  / "past month"       → past 30 days
        "last year"   / "past year"        → past 365 days
        "last week"   / "past week"        → past 7 days
        "last 24 hours" / "yesterday"      → past 24 hours
        "last 6 hours"  / "past 6 hours"   → past 6 hours
        "last 2 hours"  / "last two hours" → past 2 hours
        (default)                          → past 1 hour

    All times are UTC.  Returns (start_time, end_time) as timezone-aware
    datetimes where end_time is the current moment.

    Args:
        query: Raw user query string.

    Returns:
        (start_time, end_time) tuple — both timezone-aware UTC datetimes.
    """
    now = datetime.now(timezone.utc)
    q = query.lower()

    # Numeric month/year expressions — checked first so "last 24 months" is not
    # accidentally parsed as a partial match against any literal pattern below.
    m = _TIME_NUMERIC_RE.search(q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * (365 if unit.startswith("year") else 30)
        return now - timedelta(days=days), now

    if any(p in q for p in ("last month", "past month")):
        return now - timedelta(days=30), now
    if any(p in q for p in ("last year", "past year")):
        return now - timedelta(days=365), now
    if any(p in q for p in ("last week", "past week")):
        return now - timedelta(days=7), now
    if any(p in q for p in ("last 24 hours", "past 24 hours", "yesterday")):
        return now - timedelta(hours=24), now
    if any(p in q for p in ("last 6 hours", "past 6 hours")):
        return now - timedelta(hours=6), now
    if any(p in q for p in ("last 2 hours", "past 2 hours", "last two hours")):
        return now - timedelta(hours=2), now

    # Default: last 1 hour — conservative window for active incident investigation.
    logger.warning(
        "extract_time_window: no pattern matched query=%r — defaulting to 1-hour window",
        query,
    )
    return now - timedelta(hours=1), now


# ── Service extraction ─────────────────────────────────────────────────────────

def extract_service(query: str, db: Session | None = None) -> str | None:
    """Try to identify a service name in the query via substring matching.

    Checks the lowercased query against the live service registry from the DB
    (via get_known_services) when a session is provided.  Falls back to the
    static _KNOWN_SERVICES list when db is None (e.g. in unit tests).

    Args:
        query: Raw user query string.
        db:    Optional active session.  When provided, uses the dynamic
               service list queried from incident_events.

    Returns:
        Matched service name, or None if no known service is identified.
    """
    services = get_known_services(db) if db is not None else _KNOWN_SERVICES
    q = query.lower()
    for service in services:
        service_lower = service.lower()
        # Match both "payment-service" (stored form) and "payment service" (natural language).
        if service_lower in q or service_lower.replace("-", " ") in q:
            return service
    return None


# ── Hybrid search ──────────────────────────────────────────────────────────────

def search_incidents(
    query: str,
    db: Session,
    start_time: datetime,
    end_time: datetime,
    service: str | None = None,
    event_types: list[str] | None = None,
    limit: int = 20,
) -> list[IncidentSearchHit]:
    """Time-bounded semantic search over incident_events.

    Steps:
        1. Generate a query embedding.
        2. Filter incident_events to the [start_time, end_time] window.
        3. Apply optional service and event_type filters.
        4. Rank by cosine distance (ascending — most similar first).
        5. Return up to *limit* results as IncidentSearchHit objects.

    The query uses ORM (``db.query``) rather than Core INSERT, so column
    names resolve correctly through the Python attribute names — the
    metadata_ / "metadata" alias does not cause issues here.

    Args:
        query:       Natural-language search query.
        db:          Active SQLAlchemy session.
        start_time:  Inclusive lower bound on occurred_at (must be timezone-aware).
        end_time:    Inclusive upper bound on occurred_at (must be timezone-aware).
        service:     Optional single-service filter (exact match).
        event_types: Optional whitelist, e.g. ["log", "alert"].
        limit:       Maximum candidate rows to retrieve before reranking.

    Returns:
        List of IncidentSearchHit ordered by ascending cosine distance
        (most semantically similar first).
    """
    if start_time >= end_time:
        raise ValueError(
            f"start_time must be before end_time: {start_time.isoformat()} >= {end_time.isoformat()}"
        )

    query_embedding = generate_embedding(query)

    # pgvector cosine_distance returns values in [0, 2]; 0 = identical, 2 = opposite.
    # We convert to similarity_score = 1 - distance when building hits.
    distance_col = IncidentEvent.embedding.cosine_distance(query_embedding).label("distance")

    stmt = (
        db.query(IncidentEvent, distance_col)
        .filter(
            IncidentEvent.occurred_at >= start_time,
            IncidentEvent.occurred_at <= end_time,
        )
    )

    if service:
        stmt = stmt.filter(IncidentEvent.service == service)
    if event_types:
        stmt = stmt.filter(IncidentEvent.event_type.in_(event_types))

    rows = stmt.order_by(distance_col).limit(limit).all()

    hits = [
        IncidentSearchHit(
            id=event.id,
            event_id=event.event_id,
            content=event.content,
            occurred_at=event.occurred_at,
            service=event.service,
            event_type=event.event_type,
            similarity_score=round(1.0 - float(distance), 4),
            severity=event.severity,
            metadata=event.metadata_,
        )
        for event, distance in rows
    ]

    logger.info(
        "search_incidents: window=[%s → %s]  service=%r  event_types=%r  hits=%d",
        start_time.isoformat(),
        end_time.isoformat(),
        service,
        event_types,
        len(hits),
    )
    return hits


# ── Reranking ──────────────────────────────────────────────────────────────────

def _format_for_reranker(hit: IncidentSearchHit) -> str:
    """Prefix hit content with severity (or event_type) for cross-encoder scoring.

    The cross-encoder scores text similarity — it can't see structured fields.
    Prepending the severity level gives the model signal to distinguish
    CRITICAL from INFO events that share similar vocabulary.

    Args:
        hit: IncidentSearchHit with optional severity field.

    Returns:
        Formatted string: "[CRITICAL] connection pool exhausted…"
    """
    prefix = f"[{hit.severity.upper()}]" if hit.severity else f"[{hit.event_type.upper()}]"
    return f"{prefix} {hit.content}"


def rerank_incidents(
    query: str,
    hits: list[IncidentSearchHit],
) -> list[IncidentSearchHit]:
    """Re-score incident hits with the cross-encoder and return top FINAL_TOP_K.

    Uses the same cached CrossEncoder model as the document pipeline via
    get_reranker() — only one model is ever loaded regardless of which
    pipeline called first.

    Falls back to similarity order (ANN) if the reranker is unavailable
    (e.g. Rosetta + PyTorch crash on Apple Silicon).

    Args:
        query: User query (possibly rewritten).
        hits:  Candidates from search_incidents(), in similarity order.

    Returns:
        Up to FINAL_TOP_K IncidentSearchHit objects, ordered by rerank_score
        descending.  rerank_score is None on each hit when falling back.
    """
    if not hits:
        return hits

    reranker = get_reranker()

    if reranker is None:
        logger.warning(
            "Reranker unavailable — returning similarity order "
            "(top %d of %d incident candidates)",
            settings.FINAL_TOP_K,
            len(hits),
        )
        return hits[: settings.FINAL_TOP_K]

    pairs = [(query, _format_for_reranker(hit)) for hit in hits]
    raw_scores = reranker.predict(pairs)

    for hit, raw_score in zip(hits, raw_scores):
        hit.rerank_score = round(float(raw_score), 4)

    reranked = sorted(hits, key=lambda h: h.rerank_score, reverse=True)  # type: ignore[arg-type]

    top = reranked[: settings.FINAL_TOP_K]

    # Soft quality gate: warn when ALL top hits are below threshold but still
    # return them — callers use rerank_score to compute confidence and decide
    # how to present results rather than seeing an empty list.
    all_below = all(h.rerank_score <= settings.RERANK_THRESHOLD for h in top)  # type: ignore[operator]
    if all_below:
        logger.warning(
            "incident_retrieval: all_below_threshold "
            "candidates=%d threshold=%.1f top_score=%.4f — returning best available matches",
            len(reranked),
            settings.RERANK_THRESHOLD,
            top[0].rerank_score if top else float("nan"),
        )

    return top
