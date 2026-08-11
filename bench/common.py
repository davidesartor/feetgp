"""Bits every bench needs: device naming, timing, and a lock-safe jsonl sink."""

import fcntl
import json
import platform
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp


def eps_multiple(residual, dtype) -> float:
    """Residuals bottom out a few eps above zero, so report them in eps units."""
    return float(residual / jnp.finfo(dtype).eps)


def chip_name(device) -> str:
    """Device kind, except on cpu where every host reports the useless "cpu"."""
    if device.platform != "cpu":
        return device.device_kind
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "cpu"


def time_call(call: Callable[[], Any]) -> tuple[float, Any]:
    """Wall time of one blocking call, the first one also carrying the compile."""
    start = time.perf_counter()
    result = jax.block_until_ready(call())
    return time.perf_counter() - start, result


def write_row(results_path: str, row: dict) -> None:
    """Append under an exclusive lock, so parallel array tasks share one file."""
    with open(results_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
