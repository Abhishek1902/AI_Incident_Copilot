from __future__ import annotations

from app.core.config import settings
from app.services.incidents import IncidentSearchHit


def build_incident_prompt(
    query: str,
    hits: list[IncidentSearchHit],
    confidence: str = "unknown",
) -> str:
    """Assemble a grounding prompt for incident analysis queries.

    Formats incident events with their full temporal context (timestamp,
    service, event_type) and sorts them chronologically so the LLM can
    reason about the sequence of events.

    Args:
        query:      The user's question (possibly rewritten).
        hits:       Reranked IncidentSearchHit objects, best-relevance first.
        confidence: Cross-encoder confidence tier ("high"/"medium"/"low"/
                    "none"/"unknown") — included in the signal header so the
                    LLM is calibrated before reading the events.

    Returns:
        A fully formatted prompt string ready for the LLM.
    """
    if not hits:
        context_block = "(No relevant incident events found in selected time window.)"
    else:
        signal_header = (
            f"[Signal context: {len(hits)} event(s) | confidence: {confidence}]"
        )
        key_signals = _extract_key_signals(hits)
        context_section = _format_incident_context(hits)
        parts = [signal_header]
        if key_signals:
            parts.append(key_signals)
        parts.append(context_section)
        context_block = "\n\n".join(parts)

    return (
        "You are an AI assistant that analyses system incidents and operational events.\n"
        "IMPORTANT RULES:\n"
        "  1. Answer ONLY from the incident events provided below.\n"
        "  2. You MUST use the provided events to formulate an answer — never say\n"
        "     'no data' or 'insufficient data' when events are listed above.\n"
        "  3. If the signal is weak or ambiguous, provide your best analysis and\n"
        "     explicitly state uncertainty (e.g. 'Based on limited evidence…').\n"
        "  4. Pay close attention to timestamps and sequence to identify root causes,\n"
        "     cascading failures, and recovery patterns.\n"
        "  5. Respond that data is unavailable ONLY when the Incident events section\n"
        "     below is explicitly empty.\n\n"
        "---\n"
        f"Incident events:\n\n{context_block}\n"
        "---\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# Severity ordering for Key Signals summary.  Lower rank = higher priority.
_SEVERITY_RANK: dict[str, int] = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3}


def _extract_key_signals(hits: list[IncidentSearchHit]) -> str:
    """Build a compact Key Signals header for the top of the incident prompt.

    Only emitted when at least one CRITICAL or ERROR event is present.
    Shows the top 1–2 highest-severity events plus the top semantic match
    (when not already included in the severity list).

    Args:
        hits: IncidentSearchHit objects returned by rerank_incidents.

    Returns:
        Formatted "### Key Signals" block, or empty string when no high-severity
        events are found.
    """
    high_sev = [
        h for h in hits
        if h.severity and h.severity.upper() in ("CRITICAL", "ERROR")
    ]
    if not high_sev:
        return ""

    top_severity = sorted(
        high_sev,
        key=lambda h: _SEVERITY_RANK.get(h.severity.upper(), 99),  # type: ignore[union-attr]
    )[:2]

    top_semantic = max(hits, key=lambda h: h.similarity_score)

    lines = ["### Key Signals"]
    seen_ids: set[int] = set()

    for hit in top_severity:
        sev = hit.severity.upper()  # type: ignore[union-attr]
        ts = hit.occurred_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(f"- [{sev}] {hit.service} @ {ts}: {hit.content[:120]}")
        seen_ids.add(hit.id)

    if top_semantic.id not in seen_ids:
        ts = top_semantic.occurred_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(
            f"- [TOP MATCH] {top_semantic.service} @ {ts}: {top_semantic.content[:120]}"
        )

    return "\n".join(lines)


def _format_incident_context(hits: list[IncidentSearchHit]) -> str:
    """Format incident hits into labelled sections grouped by event category.

    Groups: Errors/Alerts (CRITICAL + ERROR) → Deployments/Changes → Timeline
    (all remaining events).  Within each section events are sorted chronologically.
    Each event appears exactly once — no duplication across sections.

    Args:
        hits: IncidentSearchHit objects (any order).

    Returns:
        Multi-line string of labelled, sectioned event blocks.
    """
    chars_per_event = max(100, settings.MAX_CONTEXT_CHARS // len(hits))

    errors = [
        h for h in hits
        if h.severity and h.severity.upper() in ("CRITICAL", "ERROR")
    ]
    deployments = [h for h in hits if h.event_type == "deployment"]
    other = [
        h for h in hits
        if not (h.severity and h.severity.upper() in ("CRITICAL", "ERROR"))
        and h.event_type != "deployment"
    ]

    def _render_section(label: str, section_hits: list[IncidentSearchHit]) -> str:
        sorted_h = sorted(section_hits, key=lambda h: h.occurred_at)
        blocks: list[str] = []
        for i, hit in enumerate(sorted_h, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.similarity_score
            ts = hit.occurred_at.strftime("%Y-%m-%dT%H:%M:%SZ")
            sev = f" | {hit.severity.upper()}" if hit.severity else ""
            header = f"[{label} {i} | {ts} | {hit.service} | {hit.event_type}{sev} | score: {score}]"
            content = _truncate(hit.content, chars_per_event)
            blocks.append(f"{header}\n{content}")
        return "\n\n".join(blocks)

    sections: list[str] = []
    if errors:
        sections.append(f"## Errors / Alerts\n\n{_render_section('Error', errors)}")
    if deployments:
        sections.append(f"## Deployments / Changes\n\n{_render_section('Deploy', deployments)}")
    if other:
        sections.append(f"## Timeline\n\n{_render_section('Event', other)}")

    return "\n\n".join(sections)


def _truncate(text: str, max_chars: int) -> str:
    """Truncate *text* to *max_chars*, breaking on a word boundary.

    Appends '...' when truncation occurs so the model knows the text is partial.
    """
    if len(text) <= max_chars:
        return text

    # Step back from the hard limit to find the last space, avoiding mid-word cuts.
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + "..."
