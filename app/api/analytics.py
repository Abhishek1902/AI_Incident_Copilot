"""GET /analytics/summary — aggregated metrics for monitoring answer quality."""

import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Feedback, QueryLog
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

# Queries with groundedness below this threshold are flagged as potential hallucinations.
_LOW_GROUNDEDNESS_THRESHOLD = 0.5


class AnalyticsSummary(BaseModel):
    total_queries: int
    avg_latency: float
    # Count of queries where the LLM judge scored groundedness < 0.5.
    # Only queries run with evaluate=True contribute to this count.
    low_groundedness_count: int
    bad_feedback_count: int


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(db: Session = Depends(get_db)):
    """Return aggregated quality metrics across all logged queries and feedback.

    Metrics:
        total_queries         — total rows in query_logs.
        avg_latency           — mean total pipeline latency across all queries (seconds).
        low_groundedness_count — queries where groundedness_score < 0.5 (potential hallucinations).
        bad_feedback_count     — feedback rows where rating == "bad".

    Returns:
        AnalyticsSummary with the four aggregated values.
    """
    total: int = db.query(func.count(QueryLog.id)).scalar() or 0

    avg_lat_raw = db.query(func.avg(QueryLog.latency)).scalar()
    avg_latency: float = round(float(avg_lat_raw), 3) if avg_lat_raw is not None else 0.0

    low_groundedness: int = (
        db.query(func.count(QueryLog.id))
        .filter(
            QueryLog.groundedness_score.isnot(None),
            QueryLog.groundedness_score < _LOW_GROUNDEDNESS_THRESHOLD,
        )
        .scalar()
        or 0
    )

    bad_feedback: int = (
        db.query(func.count(Feedback.id))
        .filter(Feedback.rating == "bad")
        .scalar()
        or 0
    )

    logger.debug(
        "Analytics: total=%d  avg_latency=%.3f  low_ground=%d  bad_fb=%d",
        total, avg_latency, low_groundedness, bad_feedback,
    )

    return AnalyticsSummary(
        total_queries=total,
        avg_latency=avg_latency,
        low_groundedness_count=low_groundedness,
        bad_feedback_count=bad_feedback,
    )
