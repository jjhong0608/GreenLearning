from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


CPU_THREAD_ENV_VARS: tuple[str, ...] = (
    "ACCELERATE_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)

_INTEROP_THREADS_APPLIED: int | None = None


@dataclass(frozen=True)
class RuntimeCPUConfig:
    """CPU-only process settings applied before model construction."""

    p_core_policy: Literal["auto"] | int = "auto"
    interop_threads: int = 1
    flush_denormal: bool = True

    def __post_init__(self) -> None:
        if self.p_core_policy != "auto" and (
            not isinstance(self.p_core_policy, int)
            or isinstance(self.p_core_policy, bool)
            or self.p_core_policy <= 0
        ):
            raise ValueError("p_core_policy must be 'auto' or a positive integer.")
        if (
            not isinstance(self.interop_threads, int)
            or isinstance(self.interop_threads, bool)
            or self.interop_threads <= 0
        ):
            raise ValueError("interop_threads must be a positive integer.")
        if not isinstance(self.flush_denormal, bool):
            raise TypeError("flush_denormal must be a boolean.")


@dataclass(frozen=True)
class RuntimeCPUState:
    """Observed CPU runtime settings for logging and reproducibility."""

    p_core_count: int
    env_values: dict[str, str]
    torch_num_threads_before: int
    torch_num_threads_after: int
    torch_interop_threads_before: int
    torch_interop_threads_after: int
    interop_status: str
    flush_denormal_supported: bool


def _sysctl_int(name: str) -> int | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    try:
        value = int(completed.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def detect_performance_core_count() -> int:
    """Return the macOS performance-core count or a safe torch fallback."""

    detected = _sysctl_int("hw.perflevel0.physicalcpu")
    if detected is not None:
        return detected
    return max(1, int(torch.get_num_threads()))


def apply_runtime_cpu_settings(
    config: RuntimeCPUConfig | None = None,
) -> RuntimeCPUState:
    """Apply CPU thread settings and return the observed process state."""

    global _INTEROP_THREADS_APPLIED

    resolved = config or RuntimeCPUConfig()
    p_core_count = (
        detect_performance_core_count()
        if resolved.p_core_policy == "auto"
        else resolved.p_core_policy
    )
    env_values = {name: str(p_core_count) for name in CPU_THREAD_ENV_VARS}
    os.environ.update(env_values)

    num_threads_before = int(torch.get_num_threads())
    torch.set_num_threads(p_core_count)
    num_threads_after = int(torch.get_num_threads())

    interop_before = int(torch.get_num_interop_threads())
    if interop_before == resolved.interop_threads:
        interop_status = "already_set"
    elif _INTEROP_THREADS_APPLIED is not None:
        interop_status = f"already_applied:{_INTEROP_THREADS_APPLIED}"
    else:
        try:
            torch.set_num_interop_threads(resolved.interop_threads)
        except RuntimeError as exc:
            interop_status = f"unavailable:{exc}"
        else:
            _INTEROP_THREADS_APPLIED = resolved.interop_threads
            interop_status = "applied"
    interop_after = int(torch.get_num_interop_threads())

    flush_denormal_supported = False
    set_flush_denormal = getattr(torch, "set_flush_denormal", None)
    if resolved.flush_denormal and callable(set_flush_denormal):
        flush_denormal_supported = bool(set_flush_denormal(True))

    return RuntimeCPUState(
        p_core_count=p_core_count,
        env_values=env_values,
        torch_num_threads_before=num_threads_before,
        torch_num_threads_after=num_threads_after,
        torch_interop_threads_before=interop_before,
        torch_interop_threads_after=interop_after,
        interop_status=interop_status,
        flush_denormal_supported=flush_denormal_supported,
    )


def write_runtime_cpu_summary(work_dir: Path, state: RuntimeCPUState) -> Path:
    """Append one reproducibility-oriented runtime line to ``training.log``."""

    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "training.log"
    env_summary = ",".join(
        f"{name}={state.env_values[name]}" for name in CPU_THREAD_ENV_VARS
    )
    line = (
        "runtime_cpu - "
        f"p_core_count={state.p_core_count} "
        f"env={env_summary} "
        "torch_num_threads="
        f"{state.torch_num_threads_before}->{state.torch_num_threads_after} "
        "torch_interop_threads="
        f"{state.torch_interop_threads_before}->{state.torch_interop_threads_after} "
        f"interop_status={state.interop_status} "
        f"flush_denormal_supported={state.flush_denormal_supported}\n"
    )
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(line)
    return log_path
