#!/usr/bin/env python3
"""Environment diagnostic script.

Run this before starting the server to verify the Python and PyTorch
architecture is correct for your machine.

Usage:
    python scripts/check_env.py

Expected output on a correctly configured Apple Silicon Mac:
    [OK]  Python 3.12.x  arch=arm64
    [OK]  Not running under Rosetta
    [OK]  PyTorch 2.x.x  arch=arm64 wheel
    [OK]  MPS available (Apple GPU acceleration)
    [OK]  NumPy < 2 installed
"""

import platform
import struct
import subprocess
import sys


def _header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print('─' * 50)


def _ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def _warn(msg: str) -> None:
    print(f"  [!!]  {msg}")


def _fail(msg: str) -> None:
    print(f"  [XX]  {msg}")


def check_python() -> bool:
    _header("Python")
    arch = platform.machine()
    version = platform.python_version()
    bits = struct.calcsize("P") * 8
    exe = sys.executable

    print(f"  Executable : {exe}")
    print(f"  Version    : {version}")
    print(f"  Arch       : {arch}  ({bits}-bit)")

    ok = True
    if arch == "arm64":
        _ok(f"Native ARM64 Python {version}")
    elif arch == "x86_64":
        _fail(
            "x86_64 Python detected. On Apple Silicon this runs under Rosetta 2 "
            "and will cause PyTorch crashes."
        )
        ok = False
    else:
        _ok(f"arch={arch}")
    return ok


def check_rosetta() -> bool:
    _header("Rosetta 2")
    try:
        result = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"],
            capture_output=True, text=True, timeout=2,
        )
        translated = result.stdout.strip()
    except FileNotFoundError:
        _ok("sysctl not available (not macOS) — skipping")
        return True

    if translated == "1":
        _fail(
            "This process IS running under Rosetta 2.\n"
            "  PyTorch (x86_64 wheel) will crash on multi-threaded BLAS calls.\n"
            "  Fix: install ARM64 Python via /opt/homebrew."
        )
        return False
    else:
        _ok("Not running under Rosetta (native execution)")
        return True


def check_torch() -> bool:
    _header("PyTorch")
    try:
        import torch
    except ImportError:
        _fail("PyTorch is not installed.")
        return False

    print(f"  Version    : {torch.__version__}")

    # Check torch wheel architecture by inspecting the compiled C extension.
    # torch._C is a .so/.dylib — `file` on it reveals the true binary arch.
    try:
        import subprocess as sp
        import os
        torch_c = getattr(torch, "_C", None)
        torch_lib = getattr(torch_c, "__file__", None) if torch_c else None
        if not torch_lib or not os.path.isfile(torch_lib):
            # Fallback: find any .so in the torch package directory
            import glob
            candidates = glob.glob(
                os.path.join(os.path.dirname(torch.__file__), "**/*.so"),
                recursive=True,
            )
            torch_lib = candidates[0] if candidates else None
        if torch_lib:
            result = sp.run(["file", torch_lib], capture_output=True, text=True)
            file_out = result.stdout.strip()
            if "arm64" in file_out:
                _ok("PyTorch wheel is ARM64 native")
                arch_ok = True
            elif "x86_64" in file_out:
                _fail("PyTorch wheel is x86_64 — will crash under Rosetta on Apple Silicon")
                arch_ok = False
            else:
                _warn(f"Could not determine wheel arch from: {os.path.basename(torch_lib)}")
                arch_ok = True
        else:
            _warn("Could not locate a compiled torch extension to check arch")
            arch_ok = True
    except Exception:
        arch_ok = True  # can't determine, don't fail

    cpu_cap = torch.backends.cpu.get_cpu_capability()
    print(f"  CPU cap    : {cpu_cap}")
    if cpu_cap == "NO AVX":
        _fail(
            "NO AVX — either running under Rosetta or on a non-AVX CPU. "
            "This is the known PyTorch crash trigger."
        )
        arch_ok = False
    else:
        _ok(f"CPU capability: {cpu_cap}")

    if platform.system() == "Darwin":
        mps = torch.backends.mps.is_available()
        print(f"  MPS avail  : {mps}")
        if mps:
            if platform.machine() == "arm64":
                _ok("MPS (Apple GPU) available and usable")
            else:
                _warn("MPS detected but Python is x86_64 — MPS won't be used under Rosetta")
        else:
            _warn("MPS not available")

    return arch_ok


def check_numpy() -> bool:
    _header("NumPy")
    try:
        import numpy as np
        version = np.__version__
        print(f"  Version    : {version}")
        major = int(version.split(".")[0])
        if major >= 2:
            _fail(
                f"NumPy {version} is >= 2. "
                "torch 2.2.x requires numpy < 2. "
                "Fix: pip install 'numpy<2'"
            )
            return False
        else:
            _ok(f"NumPy {version} (compatible with torch 2.2.x)")
            return True
    except ImportError:
        _fail("NumPy not installed")
        return False


def check_sentence_transformers() -> bool:
    _header("sentence-transformers")
    try:
        import sentence_transformers
        _ok(f"sentence-transformers {sentence_transformers.__version__}")
        return True
    except ImportError:
        _fail("sentence-transformers not installed")
        return False


def main() -> None:
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   AI Incident Copilot — Environment Check       ║")
    print("╚══════════════════════════════════════════════════╝")

    results = {
        "python":               check_python(),
        "rosetta":              check_rosetta(),
        "torch":                check_torch(),
        "numpy":                check_numpy(),
        "sentence_transformers": check_sentence_transformers(),
    }

    _header("Summary")
    all_ok = True
    for name, ok in results.items():
        status = "[OK]" if ok else "[XX]"
        print(f"  {status}  {name}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  ✓  Environment looks good. You can start the server.")
    else:
        print("  ✗  Issues detected. See above for fixes.")
        print("  Refer to README — 'Apple Silicon Setup' section.")
        sys.exit(1)


if __name__ == "__main__":
    main()
