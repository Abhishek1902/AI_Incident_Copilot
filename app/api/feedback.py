"""POST /feedback — store user thumbs-up / thumbs-down on a query–answer pair."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import Feedback
from app.db.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    # Constrained to "good" or "bad" — validated by Pydantic before hitting the DB.
    rating: Literal["good", "bad"]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: int
    message: str


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)):
    """Store user feedback for a query–answer pair.

    Called by the UI when the user clicks 👍 or 👎.  The rating is used in the
    analytics summary to track answer quality over time.

    Args:
        payload: Query, answer text, rating ("good"/"bad"), and optional comment.

    Returns:
        The ID of the created feedback row and a confirmation message.
    """
    entry = Feedback(
        query=payload.query,
        answer=payload.answer,
        rating=payload.rating,
        comment=payload.comment,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    logger.info(
        "Feedback recorded: id=%d  rating=%s  query=%r",
        entry.id,
        entry.rating,
        payload.query[:80],
    )
    return FeedbackResponse(id=entry.id, message="Feedback recorded. Thank you!")
