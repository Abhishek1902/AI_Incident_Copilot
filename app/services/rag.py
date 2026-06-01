import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from app.services.prompt import build_incident_prompt
from app.services.llm import generate_answer
from app.services.evaluator import evaluate_answer, EvaluationResult
from app.services.query_logger import log_query
from app.services.incidents import (
    IncidentSearchHit,
    search_incidents,
    rerank_incidents,
)
from app.services.query_processor import process_query
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Per-step wall-clock latency (seconds) for a single RAG call."""

    retrieve: float = 0.0
    rerank: float = 0.0
    llm: float = 0.0
    # Populated only when evaluate=True; zero otherwise.
    eval: float = 0.0
    total: float = 0.0


@dataclass
class RAGResult:
    """Full output of the RAG pipeline, including debug artefacts."""

    answer: str
    # Top FINAL_TOP_K events after reranking — sent to the LLM and returned as sources.
    sources: list[IncidentSearchHit]
    # The query that was actually used for retrieval (may differ from the original).
    rewritten_query: str
    # The complete prompt that was sent to the LLM.
    prompt: str
    # Latency breakdown for each pipeline stage.
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    # LLM-as-judge evaluation scores.  None when evaluate=False.
    evaluation: EvaluationResult | None = None
    # Cross-encoder confidence tier for the incident pipeline.
    # "high" | "medium" | "low" | "none" (no hits) | "unknown" (reranker unavailable).
    confidence: str = "unknown"


# ── Query rewriting ────────────────────────────────────────────────────────────

def rewrite_query(query: str) -> str:
    """Expand short or keyword-like queries into fuller descriptive questions.

    Short queries (≤ 4 words, no question mark) are often noun phrases rather
    than questions ("deep learning", "pgvector index").  Expanding them gives
    the bi-encoder more signal to work with and improves recall at the retrieval
    stage.

    This is a heuristic rewriter — fast, deterministic, free.  More sophisticated
    systems can replace this with an LLM call for rephrasing/disambiguation.

    Args:
        query: The raw user input.

    Returns:
        Either the original query (if it already looks well-formed) or an
        expanded version.
    """
    stripped = query.strip()
    word_count = len(stripped.split())
    has_verb_marker = "?" in stripped or any(
        stripped.lower().startswith(w)
        for w in ("what", "how", "why", "when", "where", "who", "explain", "describe", "is", "are", "does", "do")
    )

    # Only rewrite short keyword phrases — leave questions and sentences alone.
    if word_count <= 4 and not has_verb_marker:
        expanded = f"Explain {stripped} in detail, including its key concepts and use cases."
        logger.info("Query rewrite: %r → %r", stripped, expanded)
        return expanded

    return stripped


# ── Incident RAG pipeline ──────────────────────────────────────────────────────

def answer_incident_query(
    db: Session,
    query: str,
    evaluate: bool = False,
    expected_answer: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    service: str | None = None,
    event_types: list[str] | None = None,
) -> RAGResult:
    """Run the RAG pipeline over incident_events with temporal + semantic search.

    Pipeline stages:
        1. Rewrite       — same keyword-expansion heuristic as the document pipeline.
        2. Time window   — extracted from query when not explicitly provided.
        3. Service       — extracted from query when not explicitly provided.
        4. Retrieve      — search_incidents(): time-bounded ANN, top RERANK_TOP_K.
        5. Rerank        — cross-encoder over incident content, keeps FINAL_TOP_K.
        6. Prompt        — chronological incident context with temporal headers.
        7. Generate      — LLM answer grounded in incident events.
        8. Evaluate      — (optional) LLM-as-judge groundedness + relevance.
        9. Log           — writes to query_logs for analytics.

    Args:
        db:             Active SQLAlchemy session.
        query:          User's natural-language question about an incident.
        evaluate:       When True, run the LLM-as-judge evaluation stage.
        expected_answer: Reference answer for the correctness metric.
        start_time:     Override for window start; auto-extracted from query if None.
        end_time:       Override for window end; auto-extracted from query if None.
        service:        Override service filter; auto-extracted from query if None.
        event_types:    Explicit event_type whitelist, e.g. ["log", "alert"].

    Returns:
        RAGResult with answer, incident sources, prompt, latency, and optional
        evaluation.  sources contains IncidentSearchHit objects.
    """
    pipeline_start = time.perf_counter()
    latency = LatencyBreakdown()

    # ── Query processing ───────────────────────────────────────────────────────
    # Single call normalises, classifies, and extracts time window + service.
    qp = process_query(query, db)
    logger.info(
        "query_processing: type=%s window=%s→%s service=%r",
        qp.query_type,
        qp.time_window[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        qp.time_window[1].strftime("%Y-%m-%dT%H:%M:%SZ"),
        qp.service,
    )

    # ── Analytical query gate ──────────────────────────────────────────────────
    # Short-circuit before any retrieval: this pipeline cannot answer aggregation
    # or summary questions.  Return guidance so the user reformulates the query.
    if qp.query_type == "analytical":
        logger.info(
            "incident_query: analytical query detected — skipping pipeline  query=%r",
            query,
        )
        return RAGResult(
            answer=(
                "This system is optimized for debugging specific incidents.\n"
                "Try queries like:\n"
                '  - "Why did payment-service fail yesterday?"\n'
                '  - "What caused the payment pipeline failure in the last 6 hours?"\n'
                '  - "Show me CRITICAL errors in order-service in the last 2 hours."'
            ),
            sources=[],
            rewritten_query=qp.normalized_query,
            prompt="",
            latency=LatencyBreakdown(),
        )

    # ── Stage 1: Query rewriting ───────────────────────────────────────────────
    rewritten = rewrite_query(query)

    # ── Stage 2: Resolve time window ───────────────────────────────────────────
    if start_time is None or end_time is None:
        auto_start, auto_end = qp.time_window
        start_time = start_time or auto_start
        end_time = end_time or auto_end

    # ── Stage 3: Resolve service ───────────────────────────────────────────────
    if service is None:
        service = qp.service

    logger.info(
        "Incident RAG: window=[%s → %s]  service=%r  event_types=%r",
        start_time.isoformat(),
        end_time.isoformat(),
        service,
        event_types,
    )

    # ── Stage 4: Retrieve ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        candidates: list[IncidentSearchHit] = search_incidents(
            rewritten,
            db,
            start_time,
            end_time,
            service=service,
            event_types=event_types,
            limit=settings.RERANK_TOP_K,
        )
    except Exception:
        logger.error(
            "incident_retrieval: search failed query=%r service=%r",
            query, service, exc_info=True,
        )
        latency.retrieve = round(time.perf_counter() - t0, 3)
        latency.total = round(time.perf_counter() - pipeline_start, 3)
        return RAGResult(
            answer="Unable to process incident query at this time.",
            sources=[],
            rewritten_query=rewritten,
            prompt="",
            latency=latency,
        )
    latency.retrieve = round(time.perf_counter() - t0, 3)

    avg_sim = (
        round(sum(h.similarity_score for h in candidates) / len(candidates), 4)
        if candidates else 0.0
    )
    if not candidates:
        logger.warning(
            "incident_retrieval: no_signal reason=no_data window=%s→%s",
            start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
    logger.info(
        "Incident retrieve: %d candidates  avg_sim=%.4f  latency=%.3fs",
        len(candidates),
        avg_sim,
        latency.retrieve,
    )

    # ── Stage 5: Rerank ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    final_hits: list[IncidentSearchHit] = rerank_incidents(rewritten, candidates)
    latency.rerank = round(time.perf_counter() - t0, 3)

    # ── Rerank stats ───────────────────────────────────────────────────────────
    scores: list[float] = []
    if final_hits:
        scores = [h.rerank_score for h in final_hits if h.rerank_score is not None]
        avg_rerank = round(sum(scores) / len(scores), 4) if scores else float("nan")
        min_rerank = round(min(scores), 4) if scores else float("nan")
        max_rerank = round(max(scores), 4) if scores else float("nan")
    else:
        avg_rerank = min_rerank = max_rerank = float("nan")

    logger.info(
        "rerank_stats: count=%d avg_score=%.4f min=%.4f max=%.4f",
        len(final_hits),
        avg_rerank,
        min_rerank,
        max_rerank,
    )
    logger.info(
        "incident_retrieval: window=%s→%s retrieved=%d reranked=%d threshold=%.1f avg_sim=%.4f",
        start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        len(candidates),
        len(final_hits),
        settings.RERANK_THRESHOLD,
        avg_sim,
    )

    # ── Confidence metric ──────────────────────────────────────────────────────
    # Rules (first match wins):
    #   none    → no hits at all
    #   unknown → reranker unavailable (all rerank_scores are None)
    #   high    → avg cross-encoder score > -1.0
    #   medium  → avg score > -2.0  AND  at least 3 supporting events
    #   low     → everything else (weak signal or too few events)
    has_scores = bool(scores)  # False when reranker was unavailable
    if not final_hits:
        confidence = "none"
    elif not has_scores:
        confidence = "unknown"
    elif avg_rerank > -1.0:
        confidence = "high"
    elif avg_rerank > -2.0 and len(final_hits) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    logger.info(
        "incident_confidence: avg_rerank_score=%.4f confidence=%s",
        avg_rerank,
        confidence,
    )

    # ── Investigation metrics ──────────────────────────────────────────────────
    error_count = sum(
        1 for h in final_hits if h.severity and h.severity.upper() in ("ERROR", "CRITICAL")
    )
    deployment_count = sum(1 for h in final_hits if h.event_type == "deployment")
    logger.info(
        "investigation: timeline=%d errors=%d deployments=%d",
        len(final_hits),
        error_count,
        deployment_count,
    )

    # ── Stage 6: Build prompt ──────────────────────────────────────────────────
    prompt = build_incident_prompt(rewritten, final_hits, confidence=confidence)

    # ── Stage 7: Generate ──────────────────────────────────────────────────────
    # Guardrail: if hits exist the prompt must not contain the "no data" escape
    # phrase — that would allow the LLM to bail out even when events are present.
    if final_hits and "I don't have enough incident data" in prompt:
        logger.error(
            "guardrail: prompt contains 'insufficient data' instruction with %d hits"
            " — check build_incident_prompt",
            len(final_hits),
        )

    t0 = time.perf_counter()
    if not final_hits:
        # No events found in the time window — skip LLM to avoid hallucination.
        answer = "No incident data found for the specified time window and service."
        logger.info("Skipping LLM call — no grounding data (final_hits empty)")
    elif confidence == "low":
        # Events exist but the cross-encoder rated them poorly.  Still call the
        # LLM — it may extract useful signal — but prepend a confidence caveat.
        llm_answer = generate_answer(prompt)
        answer = f"Low confidence results — limited or weak signal detected. Showing best available matches.\n\n{llm_answer}"
    else:
        answer = generate_answer(prompt)
    latency.llm = round(time.perf_counter() - t0, 3)

    # ── Stage 8: Evaluate (optional) ───────────────────────────────────────────
    evaluation: EvaluationResult | None = None
    if evaluate:
        t0 = time.perf_counter()
        try:
            evaluation = evaluate_answer(
                query=query,
                answer=answer,
                context=[hit.content for hit in final_hits],
                expected_answer=expected_answer,
            )
        except Exception:
            logger.exception("Incident evaluation stage failed — skipping")
        latency.eval = round(time.perf_counter() - t0, 3)

    latency.total = round(time.perf_counter() - pipeline_start, 3)

    # ── Stage 9: Log ───────────────────────────────────────────────────────────
    log_query(
        db,
        query=query,
        answer=answer,
        latency=latency.total,
        groundedness_score=evaluation.groundedness_score if evaluation else None,
        relevance_score=evaluation.relevance_score if evaluation else None,
    )

    logger.info(
        "incident_latency: retrieval=%.3fs rerank=%.3fs llm=%.3fs eval=%.3fs total=%.3fs answer_chars=%d",
        latency.retrieve,
        latency.rerank,
        latency.llm,
        latency.eval,
        latency.total,
        len(answer),
    )

    return RAGResult(
        answer=answer,
        sources=final_hits,          # list[IncidentSearchHit]
        rewritten_query=rewritten,
        prompt=prompt,
        latency=latency,
        evaluation=evaluation,
        confidence=confidence,
    )
