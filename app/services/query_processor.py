"""Unified query processing layer for the incident investigation pipeline.

All parsing, classification, and metadata extraction from raw user queries
lives here.  The RAG pipeline calls process_query() once and uses the
returned QueryContext for the rest of its logic — it never needs to parse
raw text directly.

Design principles:
  - Pure functions where possible; all state flows through QueryContext.
  - No ML / NLP libraries.  Regex + keyword matching is fast, deterministic,
    and requires zero model-load time.
  - extract_service() reads from the live service registry (incident_events)
    when a DB session is provided, falling back to a static list otherwise.
  - extract_time_window() is a strict superset of the version in incidents.py:
    it adds minute and day granularity plus "today" / "yesterday" handling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Normalisation ──────────────────────────────────────────────────────────────

# Punctuation to strip (technical chars such as _ @ # are preserved).
_PUNCT_RE = re.compile(r"""[?!.,;:()\[\]{}'\"\\]""")
_SPACES_RE = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Return a cleaned, lowercase version of *query* suitable for matching.

    Steps applied in order:
        1. Lowercase.
        2. Replace hyphens with spaces — "payment-service" and "payment service"
           produce identical tokens after this step.
        3. Strip common punctuation (?, !, ., ,, ;, :, brackets, quotes).
        4. Collapse runs of whitespace and trim edges.

    The normalised form is used internally for classification and service
    matching.  The original query is always preserved for embedding and LLM
    calls — normalisation is never applied to those paths.

    Args:
        query: Raw user input string.

    Returns:
        Normalised string.
    """
    text = query.lower()
    text = text.replace("-", " ")       # "payment-service" → "payment service"
    text = _PUNCT_RE.sub("", text)      # strip punctuation
    text = _SPACES_RE.sub(" ", text).strip()
    return text


# ── Classification ─────────────────────────────────────────────────────────────

# Analytical terms are checked first — they take priority when both sets match.
# (e.g. "which service failed most" has both "which" and "failed".)
_ANALYTICAL_TERMS: frozenset[str] = frozenset({
    "which",
    "how many",
    "count",
    "most",
    "top",
    "trend",
    "how often",
    "list all",
})

_INCIDENT_TERMS: frozenset[str] = frozenset({
    "why",
    "failed",
    "fail",
    "error",
    "errors",
    "issue",
    "problem",
    "outage",
    "alert",
    "crash",
    "down",
    "timeout",
    "slow",
    "latency",
    "what happened",
    "what caused",
    "happened",
    "caused",
})


def classify_query(query: str) -> str:
    """Classify a query as 'incident', 'analytical', or 'unknown'.

    Classification rules (first match wins):
        1. Contains any _ANALYTICAL_TERM  → "analytical"
           Aggregation / summary queries; the RAG pipeline cannot answer these
           correctly — it retrieves a fixed number of semantically similar
           events, not a count or ranking across all events.
        2. Contains any _INCIDENT_TERM    → "incident"
           Specific failure or event queries — the primary pipeline use case.
        3. Otherwise                      → "unknown"
           Treated as incident by the pipeline; retrieval still runs so the
           system degrades gracefully rather than refusing the query.

    Matching is performed on the normalised form of the query so
    "payment-service" and "payment service" trigger the same patterns.

    Args:
        query: Raw user query string.

    Returns:
        "incident" | "analytical" | "unknown"
    """
    q = normalize_query(query)
    if any(term in q for term in _ANALYTICAL_TERMS):
        return "analytical"
    if any(term in q for term in _INCIDENT_TERMS):
        return "incident"
    return "unknown"


# ── Time window extraction ─────────────────────────────────────────────────────

# Matches "last/past N <unit>" for any time unit from minute to year.
# Group 1 = numeric count (digits), Group 2 = unit string (possibly plural).
_TIME_NUMERIC_RE = re.compile(
    r"(?:last|past)\s+(\d+)\s+(minutes?|hours?|days?|months?|years?)",
    re.IGNORECASE,
)

# Maps normalised singular unit → timedelta factory callable.
_UNIT_TO_DELTA: dict[str, Callable[[int], timedelta]] = {
    "minute": lambda n: timedelta(minutes=n),
    "hour":   lambda n: timedelta(hours=n),
    "day":    lambda n: timedelta(days=n),
    "month":  lambda n: timedelta(days=n * 30),
    "year":   lambda n: timedelta(days=n * 365),
}


def extract_time_window(query: str) -> tuple[datetime, datetime]:
    """Parse a natural-language query and return a (start_time, end_time) pair.

    Pattern priority (most specific first):

        Numeric expressions:
            "last/past N minutes"  → past N minutes
            "last/past N hours"    → past N hours
            "last/past N days"     → past N days
            "last/past N months"   → past N*30 days
            "last/past N years"    → past N*365 days

        Literal expressions:
            "yesterday"            → past 24 hours
            "today"                → from midnight UTC today to now
            "last/past hour"       → past 1 hour
            "last/past week"       → past 7 days
            "last/past month"      → past 30 days
            "last/past year"       → past 365 days

        Default (+ warning log):
            (none of the above)    → past 1 hour

    All times are UTC-aware.

    Args:
        query: Raw user query string.

    Returns:
        (start_time, end_time) tuple — both timezone-aware UTC datetimes,
        with end_time always equal to the current moment.
    """
    now = datetime.now(timezone.utc)
    q = query.lower()

    # ── Numeric expressions: "last N <unit>" ──────────────────────────────────
    m = _TIME_NUMERIC_RE.search(q)
    if m:
        n = int(m.group(1))
        # Normalise plural to singular: "hours" → "hour", "months" → "month".
        unit = m.group(2).rstrip("s")
        delta_fn = _UNIT_TO_DELTA.get(unit)
        if delta_fn:
            return now - delta_fn(n), now

    # ── Literal expressions ────────────────────────────────────────────────────
    if "yesterday" in q:
        return now - timedelta(hours=24), now
    if "today" in q:
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight, now
    if any(p in q for p in ("last hour", "past hour")):
        return now - timedelta(hours=1), now
    if any(p in q for p in ("last week", "past week")):
        return now - timedelta(days=7), now
    if any(p in q for p in ("last month", "past month")):
        return now - timedelta(days=30), now
    if any(p in q for p in ("last year", "past year")):
        return now - timedelta(days=365), now

    # ── Default ────────────────────────────────────────────────────────────────
    logger.warning(
        "extract_time_window: no pattern matched query=%r — defaulting to 1-hour window",
        query,
    )
    return now - timedelta(hours=1), now


# ── Service extraction ─────────────────────────────────────────────────────────

# Static fallback used when db is None (mirrors the list in incidents.py).
_STATIC_SERVICES: list[str] = [
    "payment-service",
    "order-service",
    "payment-pipeline",
    "auth-service",
    "api-gateway",
    "notification-service",
    "inventory-service",
]


def extract_service(query: str, db: Session | None = None) -> str | None:
    """Identify a service name in *query* via normalised substring matching.

    Both the query and each candidate service name are normalised before
    comparison — hyphens are converted to spaces — so "payment-service" and
    "payment service" match each other regardless of which form appears in
    the query or the registry.

    Uses the live service registry from incident_events when *db* is
    provided, falling back to _STATIC_SERVICES when db is None (e.g. in
    unit tests or when the DB is unavailable).

    Args:
        query: Raw user query string.
        db:    Optional active SQLAlchemy session.

    Returns:
        Matched service name in its stored form (e.g. "payment-service"),
        or None when no known service is identified.
    """
    # Deferred import to avoid a module-level circular import:
    # query_processor → incidents (for get_known_services only).
    from app.services.incidents import get_known_services  # noqa: PLC0415

    services = get_known_services(db) if db is not None else _STATIC_SERVICES
    # Normalise the query once; reuse across all service comparisons.
    q_norm = normalize_query(query)
    for service in services:
        # Normalise the service name the same way so hyphen / space variants
        # always match (e.g. "payment-service" → "payment service").
        if service.lower().replace("-", " ") in q_norm:
            return service
    return None


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class QueryContext:
    """Structured output of process_query — consumed by the RAG pipeline.

    Attributes:
        normalized_query: Cleaned, lowercase query with punctuation stripped
                          and hyphens replaced by spaces.  Useful for logging
                          and for callers that want a canonical form.
        query_type:       "incident" | "analytical" | "unknown"
        time_window:      (start_time, end_time) — both timezone-aware UTC.
                          Derived from the query when not overridden by the
                          caller; defaults to (now-1h, now).
        service:          Matched service name in stored form, or None.
    """

    normalized_query: str
    query_type: str
    time_window: tuple[datetime, datetime]
    service: str | None


# ── Unified entry point ────────────────────────────────────────────────────────

def process_query(query: str, db: Session | None = None) -> QueryContext:
    """Normalise, classify, and extract metadata from a raw user query.

    This is the single entry point for the incident RAG pipeline.  All
    parsing logic is encapsulated here — the pipeline receives a typed
    QueryContext and never has to parse raw text directly.

    All extraction functions operate on the *original* query (not the
    normalised form) so that patterns relying on original spacing or
    capitalisation are not disrupted.  Normalisation is applied only
    internally by each function where needed.

    Args:
        query: The user's raw natural-language query.
        db:    Optional active SQLAlchemy session (used for service lookup).

    Returns:
        Populated QueryContext.
    """
    normalized = normalize_query(query)
    query_type = classify_query(query)
    time_window = extract_time_window(query)
    service = extract_service(query, db)

    return QueryContext(
        normalized_query=normalized,
        query_type=query_type,
        time_window=time_window,
        service=service,
    )
