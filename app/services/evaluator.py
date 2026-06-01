"""LLM-as-judge evaluation service.

Evaluates RAG system outputs on three axes:
  - Groundedness: hallucination detection — are all claims in the answer supported by context?
  - Relevance:    does the answer correctly address the question?
  - Correctness:  does the answer match a reference answer? (optional, only when expected_answer is given)

A single OpenAI call with JSON mode returns all metrics at once, keeping
latency and cost low while maintaining clear scoring guidance per criterion.
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI, APIError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Reuse a module-level client — same pattern as llm.py.
# max_retries=1 (lower than the answer client) because eval failures are non-critical.
_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=30.0,
    max_retries=1,
)

# System prompt tells the model its role and response contract up front.
# Keeping instructions here (rather than in the user message) gives the
# model clear role context before it reads the content to evaluate.
_SYSTEM_PROMPT = (
    "You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems. "
    "Your job is to score the quality of a generated answer. "
    "Be strict: partial support counts as partial score, not full score. "
    "Respond ONLY with a valid JSON object — no markdown fences, no prose outside JSON."
)


@dataclass
class EvaluationResult:
    """Structured output of one LLM-as-judge evaluation run.

    Attributes:
        groundedness_score:       0–1.  Is every claim traceable to the provided context?
                                  1.0 = fully grounded, 0.5 = partially, 0.0 = hallucinated.
        groundedness_explanation: One-sentence justification for the groundedness score.
        relevance_score:          0–1.  Does the answer correctly address the question?
                                  1.0 = complete and accurate, 0.5 = partial, 0.0 = irrelevant.
        relevance_explanation:    One-sentence justification for the relevance score.
        correctness_score:        0–1 or None.  Semantic similarity to expected_answer.
                                  Only populated when expected_answer is passed in.
        correctness_explanation:  Justification for the correctness score, or None.
    """

    groundedness_score: float
    groundedness_explanation: str
    relevance_score: float
    relevance_explanation: str
    correctness_score: float | None = None
    correctness_explanation: str | None = None


def evaluate_answer(
    query: str,
    answer: str,
    context: list[str],
    expected_answer: str | None = None,
) -> EvaluationResult:
    """Run LLM-as-judge evaluation on a single RAG system output.

    Makes one OpenAI call using JSON mode so the response is guaranteed to be
    parseable.  All three metrics are requested in a single prompt to minimise
    latency and token cost.

    Groundedness is the most critical metric for production RAG: a fluent,
    on-topic answer can still be dangerously wrong if the LLM introduced
    claims not present in the retrieved context.

    Args:
        query:           The original user question.
        answer:          The RAG system's generated answer to evaluate.
        context:         The documents that were passed to the LLM prompt.
        expected_answer: Optional reference answer for correctness comparison.
                         When omitted, the correctness metric is skipped.

    Returns:
        EvaluationResult with scores (clamped to [0, 1]) and explanations.

    Raises:
        APIError: If the OpenAI call fails after retries.
    """
    user_message = _build_eval_prompt(query, answer, context, expected_answer)

    try:
        response = _client.chat.completions.create(
            model=settings.EVAL_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            # temperature=0 makes the judge deterministic — same input → same scores.
            temperature=0.0,
            # JSON mode guarantees a parseable response, eliminating defensive parsing.
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        result = _parse_eval_response(raw, include_correctness=expected_answer is not None)

    except APIError as e:
        logger.error("Evaluator API error (model=%s): %s", settings.EVAL_MODEL, e)
        raise

    # Log scores for every evaluation so ops can monitor quality over time.
    correctness_log = (
        f"  correctness={result.correctness_score:.2f}"
        if result.correctness_score is not None
        else ""
    )
    logger.info(
        "Evaluation: groundedness=%.2f  relevance=%.2f%s",
        result.groundedness_score,
        result.relevance_score,
        correctness_log,
    )

    # Surface low-groundedness answers as warnings — these are likely hallucinations.
    if result.groundedness_score < 0.5:
        logger.warning(
            "Low groundedness (%.2f) — possible hallucination: %s",
            result.groundedness_score,
            result.groundedness_explanation,
        )

    return result


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_eval_prompt(
    query: str,
    answer: str,
    context: list[str],
    expected_answer: str | None,
) -> str:
    """Assemble the user-turn evaluation prompt.

    Structures the content clearly so the judge LLM can locate each piece
    without ambiguity.  The schema section tells the model exactly what JSON
    to return, reducing parse failures.
    """
    context_block = "\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(context))

    # Build the optional correctness section only when a reference answer exists.
    expected_block = f"\n## Expected Answer:\n{expected_answer}\n" if expected_answer else ""

    correctness_criterion = (
        '\n3. "correctness" — Does the generated answer convey the same information '
        "as the expected answer?\n"
        "   1.0 = semantically equivalent\n"
        "   0.5 = partially correct or incomplete\n"
        "   0.0 = incorrect or contradicts the expected answer\n"
    ) if expected_answer else ""

    correctness_schema = (
        ',\n  "correctness": {"score": <float 0.0–1.0>, "explanation": "<1-2 sentences>"}'
    ) if expected_answer else ""

    return (
        f"## Context (retrieved documents):\n{context_block}\n\n"
        f"## Question:\n{query}\n\n"
        f"## Generated Answer:\n{answer}\n"
        f"{expected_block}\n"
        "Evaluate the generated answer on these criteria:\n\n"
        '1. "groundedness" — Is every claim in the answer directly supported by the context?\n'
        "   1.0 = fully grounded (all claims traceable to the context)\n"
        "   0.5 = partially grounded (some claims lack context support)\n"
        "   0.0 = not grounded (hallucinated or contradicts context)\n\n"
        '2. "relevance" — Does the answer correctly and completely address the question?\n'
        "   1.0 = complete, accurate answer to the question\n"
        "   0.5 = partial or tangential answer\n"
        "   0.0 = irrelevant or wrong\n"
        f"{correctness_criterion}\n"
        "Return JSON:\n"
        "{\n"
        '  "groundedness": {"score": <float 0.0–1.0>, "explanation": "<1-2 sentences>"},\n'
        '  "relevance": {"score": <float 0.0–1.0>, "explanation": "<1-2 sentences>"}'
        f"{correctness_schema}\n"
        "}"
    )


def _parse_eval_response(raw: str, include_correctness: bool) -> EvaluationResult:
    """Parse the judge LLM's JSON response into an EvaluationResult.

    Returns a safe fallback (all-zero scores, error message) if the response
    cannot be parsed, so callers are never broken by a malformed LLM output.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Evaluator returned invalid JSON (%s): %r", exc, raw[:200])
        return EvaluationResult(
            groundedness_score=0.0,
            groundedness_explanation="Evaluation failed: could not parse LLM response.",
            relevance_score=0.0,
            relevance_explanation="Evaluation failed: could not parse LLM response.",
        )

    def _clamp(value: object) -> float:
        """Ensure scores stay in [0, 1] regardless of what the LLM returns."""
        return round(min(1.0, max(0.0, float(value))), 4)  # type: ignore[arg-type]

    g = data.get("groundedness", {})
    r = data.get("relevance", {})

    result = EvaluationResult(
        groundedness_score=_clamp(g.get("score", 0.0)),
        groundedness_explanation=str(g.get("explanation", "No explanation provided.")),
        relevance_score=_clamp(r.get("score", 0.0)),
        relevance_explanation=str(r.get("explanation", "No explanation provided.")),
    )

    if include_correctness and "correctness" in data:
        c = data["correctness"]
        result.correctness_score = _clamp(c.get("score", 0.0))
        result.correctness_explanation = str(c.get("explanation", "No explanation provided."))

    return result
