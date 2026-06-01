import logging

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Returned whenever the LLM is unreachable — keeps the API responsive.
_LLM_FALLBACK = "LLM unavailable. Unable to analyze incident at this time."

# Module-level client — instantiated once, reused across requests.
# timeout=30 s for the full response; max_retries=2 handles transient errors.
_client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=30.0,
    max_retries=2,  # retry transient network errors automatically
)


def generate_answer(prompt: str) -> str:
    """Send *prompt* to the configured OpenAI chat model and return the response text.

    On timeout, connection error, or API error: logs the failure and returns
    _LLM_FALLBACK so the caller always gets a usable string (never raises).

    Args:
        prompt: The fully-formatted prompt string (built by prompt.build_incident_prompt).

    Returns:
        The model's answer, or _LLM_FALLBACK if the LLM is unreachable.
    """
    try:
        response = _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            # Low temperature keeps answers grounded and deterministic.
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        if response.usage:
            logger.info(
                "llm_usage: model=%s prompt=%d completion=%d total=%d",
                settings.OPENAI_MODEL,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
            )
        return answer

    except APITimeoutError:
        logger.error("llm: request timed out model=%s", settings.OPENAI_MODEL)
        return _LLM_FALLBACK
    except APIConnectionError as e:
        logger.error("llm: connection error model=%s error=%s", settings.OPENAI_MODEL, e)
        return _LLM_FALLBACK
    except APIError as e:
        logger.error(
            "llm: api error model=%s status=%s message=%s",
            settings.OPENAI_MODEL, e.status_code, e.message,
        )
        return _LLM_FALLBACK
