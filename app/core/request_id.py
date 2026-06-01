"""Per-request ID: UUID tracing propagated via ContextVar + logging filter.

Usage:
    - RequestIDMiddleware reads X-Request-ID header (or generates a UUID)
      and stores it in request_id_var for the lifetime of the request.
    - RequestIDFilter injects request_id into every log record so it appears
      automatically in the format string without touching individual loggers.
    - Downstream code can call request_id_var.get() to read the current ID
      (e.g. in exception handlers for structured error responses).
"""

import logging
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Holds the request ID for the current request.  Default is "-" outside a request
# context (startup, background tasks, tests without middleware).
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(logging.Filter):
    """Inject request_id into every log record emitted during a request.

    Attach to root handlers after basicConfig so all loggers pick it up:

        filter = RequestIDFilter()
        for handler in logging.root.handlers:
            handler.addFilter(filter)

    The format string must include %(request_id)s to display the value.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()  # type: ignore[attr-defined]
        return True


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a unique request_id to every HTTP request.

    Reads X-Request-ID from the incoming headers when present; otherwise
    generates a 12-character hex UUID.  Stores the ID in request_id_var
    so it propagates to all log records emitted during the request.
    Echoes the ID back in the response X-Request-ID header for client tracing.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(req_id)
        try:
            response: Response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = req_id
        return response
