"""Cross-encoder reranker with graceful degradation.

On Apple Silicon Macs running x86_64 Python under Rosetta 2, PyTorch's
libtorch_cpu.dylib can crash with SIGABRT due to OpenBLAS thread conflicts
in the Rosetta translation layer.  This module detects that scenario at
import time, warns clearly, and falls back to ANN ordering if the model
load fails — so the RAG pipeline continues working without reranking.
"""

import logging
import platform
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Flipped to False the first time _load_reranker() catches an exception.
_reranker_available: bool = True


# ── Environment safety check ───────────────────────────────────────────────────

def _is_rosetta() -> bool:
    """Return True if this process is running under Rosetta 2 translation.

    sysctl.proc_translated is 1 when the kernel is running an x86_64 binary
    on an ARM64 chip via Rosetta 2, and absent or 0 otherwise.
    """
    try:
        out = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        return out == "1"
    except Exception:
        return False


def _warn_if_rosetta() -> None:
    """Emit a startup warning when the unsafe Rosetta + PyTorch combination is detected.

    Root cause: x86_64 PyTorch on ARM64 Mac
    - PyTorch's x86_64 wheel uses OpenBLAS compiled with AVX/AVX2 instructions.
    - Rosetta 2 does not emulate AVX reliably under multi-threaded workloads.
    - When the cross-encoder calls BLAS routines across threads, the process
      receives SIGABRT from libtorch_cpu.dylib (crash namespace: ROSETTA).

    Fix: use ARM64-native Python from /opt/homebrew.
    See README — "Apple Silicon Setup" section.
    """
    if platform.system() != "Darwin":
        return
    if platform.machine() == "arm64":
        return  # native ARM64 — no issue
    if _is_rosetta():
        logger.warning(
            "ROSETTA DETECTED: running x86_64 Python on ARM64 Mac. "
            "PyTorch may crash (SIGABRT in libtorch_cpu.dylib / OpenBLAS). "
            "Reranker will be disabled if model load fails. "
            "Fix: reinstall with ARM64 Python — see README Apple Silicon Setup."
        )


# Run the check once at module import so the warning appears in startup logs.
_warn_if_rosetta()


# ── Model loading ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_reranker():
    """Load and cache the CrossEncoder model.

    Returns the loaded model on success, or None if loading fails.
    The lru_cache ensures this runs only once: either the model is cached
    on success, or None is cached on failure — no repeated retries.
    """
    global _reranker_available
    try:
        # Defer the import to here so that an import-time crash in torch
        # (which can happen under Rosetta) is caught at the call site rather
        # than at module import, keeping the rest of the application usable.
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading reranker model: %s", _RERANK_MODEL)
        model = CrossEncoder(_RERANK_MODEL)
        logger.info("Reranker loaded successfully")
        return model

    except Exception as exc:
        _reranker_available = False
        logger.error(
            "Reranker model load failed (%s: %s). "
            "Pipeline will continue without reranking (ANN order preserved).",
            type(exc).__name__,
            exc,
        )
        return None


# ── Public interface ───────────────────────────────────────────────────────────

def get_reranker():
    """Return the cached CrossEncoder model, or None if unavailable.

    The model is loaded at most once via the lru_cache on _load_reranker.

    Returns:
        CrossEncoder instance, or None if the model failed to load.
    """
    return _load_reranker()
