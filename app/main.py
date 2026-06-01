import logging
import platform
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text

from app.api.feedback import router as feedback_router
from app.api.analytics import router as analytics_router
from app.api.incidents import router as incidents_router
from app.core.request_id import RequestIDFilter, RequestIDMiddleware, request_id_var

# ── Logging configuration ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(request_id)s]  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Inject request_id into every log record via a filter on all root handlers.
_request_id_filter = RequestIDFilter()
for _handler in logging.root.handlers:
    _handler.addFilter(_request_id_filter)
logger = logging.getLogger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Run startup diagnostics before the server begins accepting requests."""
    logger.info("AI Incident Copilot v2.0.0 starting")
    _log_environment_info()
    _check_db_connection()
    logger.info("AI Incident Copilot v2.0.0 ready")
    yield
    # Nothing to tear down — DB engine pool closes automatically on process exit.


# ── Application ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Incident Copilot",
    description=(
        "Semantic search and Retrieval-Augmented Generation (RAG) over documents "
        "using pgvector + sentence-transformers + OpenAI."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)

app.include_router(feedback_router)
app.include_router(analytics_router)
app.include_router(incidents_router)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: FastAPIRequest, exc: Exception) -> JSONResponse:
    """Return structured JSON with request_id instead of a raw 500 traceback."""
    req_id = request_id_var.get()
    logger.error(
        "unhandled_exception: %s request_id=%s path=%s",
        type(exc).__name__,
        req_id,
        request.url.path,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": req_id},
    )

_FRONTEND = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def ui():
    """Serve the minimal incident copilot UI."""
    return _FRONTEND.read_text()


@app.get("/health")
def health():
    """Liveness + readiness probe — reports DB connectivity alongside server status."""
    from app.db.session import engine  # noqa: PLC0415 — deferred to avoid circular import
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "failed"
    return {"status": "ok", "db": db_status}


# ── Environment diagnostics ────────────────────────────────────────────────────

def _log_environment_info() -> None:
    """Log Python/PyTorch architecture info and warn about unsafe configurations.

    The primary risk on Apple Silicon Macs is running x86_64 Python via Rosetta 2
    with a PyTorch build that uses OpenBLAS.  Under multi-threaded BLAS workloads
    (e.g. the cross-encoder reranker) this can cause a hard crash:
      EXC_CRASH (SIGABRT) — Namespace ROSETTA — libtorch_cpu.dylib

    Logging this on every startup makes the issue immediately visible in
    production logs rather than appearing only at crash time.
    """
    py_arch = platform.machine()
    py_version = platform.python_version()
    os_name = platform.system()

    logger.info("Python %s  arch=%s  os=%s", py_version, py_arch, os_name)

    # ── Rosetta detection (macOS only) ─────────────────────────────────────────
    if os_name == "Darwin":
        rosetta = _is_rosetta()
        if rosetta:
            logger.warning(
                "ENVIRONMENT WARNING: x86_64 Python running under Rosetta 2 on Apple Silicon. "
                "PyTorch operations may crash (SIGABRT in libtorch_cpu.dylib). "
                "Action required: reinstall using ARM64 Python from /opt/homebrew. "
                "See README — 'Apple Silicon Setup' section."
            )
        else:
            logger.info("Architecture OK: native %s (Rosetta not active)", py_arch)

    # ── PyTorch diagnostics ────────────────────────────────────────────────────
    try:
        import torch  # noqa: PLC0415
        logger.info("PyTorch %s available", torch.__version__)

        cpu_cap = torch.backends.cpu.get_cpu_capability()
        logger.info("CPU capability: %s", cpu_cap)
        if cpu_cap == "NO AVX":
            # AVX is missing either because this is a non-AVX CPU or because
            # Rosetta is not exposing AVX to the translated process.
            logger.warning(
                "CPU has NO AVX capability — likely running under Rosetta. "
                "This is the known crash trigger for libtorch_cpu.dylib."
            )

        if os_name == "Darwin":
            mps_available = torch.backends.mps.is_available()
            logger.info("Apple MPS backend available: %s", mps_available)
            if mps_available and py_arch != "arm64":
                # MPS is available but Python is x86_64 — MPS won't actually be
                # used because the process is running under Rosetta.
                logger.warning(
                    "MPS is available but Python is x86_64. "
                    "ARM64 Python is required to use native GPU acceleration."
                )

    except ImportError:
        logger.warning(
            "PyTorch is not installed. "
            "The reranker will be disabled. "
            "Install with: pip install torch"
        )


def _check_db_connection() -> None:
    """Verify the database is reachable at startup.

    Logs a clear warning if the connection fails but never raises — the API
    process starts regardless so liveness/readiness probes keep working and
    the issue is visible in logs without a hard crash.
    """
    from app.db.session import engine  # noqa: PLC0415 — deferred to avoid circular import
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection: OK")
    except Exception as exc:
        logger.warning(
            "Database connection check failed at startup: %s — "
            "the API will start but database operations will fail until the DB is reachable.",
            exc,
        )


def _is_rosetta() -> bool:
    """Return True if this process is running under Rosetta 2."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return out == "1"
    except Exception:
        return False
