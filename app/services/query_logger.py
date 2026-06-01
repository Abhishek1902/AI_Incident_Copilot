"""Write completed RAG requests to the query_logs table.

Kept as a standalone module so rag.py doesn't need to know about the ORM
model directly, and so the logging logic is testable in isolation.
"""

import logging

from sqlalchemy.orm import Session

from app.db.models import QueryLog

logger = logging.getLogger(__name__)


def log_query(
    db: Session,
    query: str,
    answer: str,
    latency: float,
    groundedness_score: float | None = None,
    relevance_score: float | None = None,
) -> None:
    """Persist a query–answer pair to the query_logs table.

    This is fire-and-forget: if the write fails (e.g. DB briefly unreachable)
    the failure is logged as a warning and the caller is not interrupted.

    Args:
        db:                  Active SQLAlchemy session.
        query:               The original user query (before rewriting).
        answer:              The LLM-generated answer.
        latency:             Total pipeline latency in seconds.
        groundedness_score:  LLM-as-judge groundedness score, or None.
        relevance_score:     LLM-as-judge relevance score, or None.
    """
    try:
        entry = QueryLog(
            query=query,
            answer=answer,
            latency=latency,
            groundedness_score=groundedness_score,
            relevance_score=relevance_score,
        )
        db.add(entry)
        db.commit()
        logger.debug("Query logged (latency=%.3fs)", latency)
    except Exception:
        logger.warning("Failed to write query log — skipping", exc_info=True)
        db.rollback()
