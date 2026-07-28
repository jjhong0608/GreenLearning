from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import optim

from greenonet.config import SoapOptimizerConfig
from greenonet.optimizers import SOAP

SOAP_UPSTREAM_REPOSITORY = "https://github.com/nikhilvyas/SOAP"
SOAP_UPSTREAM_COMMIT = "a1e553530fde97d0e6b307d7c82ac6d38b072340"


@dataclass(frozen=True)
class OptimizerProvenance:
    """Resolved optimizer metadata stored with training artifacts."""

    name: str
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    profile_step_time: bool
    soap: dict[str, object] | None
    implementation: str
    upstream_commit: str | None
    checkpoint_policy: str = "model_only_no_optimizer_resume"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_adamw_or_soap(
    parameters: Iterable[torch.nn.Parameter],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    soap: SoapOptimizerConfig,
) -> optim.Optimizer:
    """Build a project-supported first-stage optimizer."""

    if name == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    if name != "soap":
        raise ValueError(f"Unsupported optimizer name: {name!r}.")
    return SOAP(
        parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        shampoo_beta=soap.shampoo_beta,
        precondition_frequency=soap.precondition_frequency,
        max_precond_dim=soap.max_precondition_dim,
        merge_dims=soap.merge_dims,
        precondition_1d=soap.precondition_1d,
        normalize_grads=soap.normalize_grads,
        correct_bias=soap.correct_bias,
    )


def build_optimizer_provenance(
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    profile_step_time: bool,
    soap: SoapOptimizerConfig,
) -> OptimizerProvenance:
    """Materialize optimizer provenance without constructing optimizer state."""

    if name == "soap":
        return OptimizerProvenance(
            name=name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            profile_step_time=profile_step_time,
            soap=asdict(soap),
            implementation=SOAP_UPSTREAM_REPOSITORY,
            upstream_commit=SOAP_UPSTREAM_COMMIT,
        )
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer name: {name!r}.")
    return OptimizerProvenance(
        name=name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        profile_step_time=profile_step_time,
        soap=None,
        implementation="torch.optim.AdamW",
        upstream_commit=None,
    )


class OptimizerStepProfiler:
    """Optional optimizer-step timing and SOAP refresh telemetry."""

    _MIB = 1024.0 * 1024.0

    def __init__(
        self,
        *,
        optimizer: optim.Optimizer,
        enabled: bool,
        device: torch.device,
    ) -> None:
        self.optimizer = optimizer
        self.enabled = enabled
        self.device = device
        self._step_times_ms: list[float] = []
        self._basis_refresh_start = 0
        self._steps_this_epoch = 0

    def begin_epoch(self) -> None:
        if not self.enabled:
            return
        self._step_times_ms.clear()
        self._basis_refresh_start = self._counter("basis_refresh_count")
        self._steps_this_epoch = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def step(self) -> None:
        if not self.enabled:
            self.optimizer.step()
            return
        self._synchronize()
        start = time.perf_counter()
        self.optimizer.step()
        self._synchronize()
        self._step_times_ms.append((time.perf_counter() - start) * 1000.0)
        self._steps_this_epoch += 1

    def finish_epoch(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        if not self._step_times_ms:
            raise ValueError("Cannot profile an epoch with zero optimizer steps.")
        peak_memory_mib = 0.0
        if self.device.type == "cuda":
            peak_memory_mib = (
                float(torch.cuda.max_memory_allocated(self.device)) / self._MIB
            )
        return {
            "optimizer_step_time_mean_ms": statistics.fmean(self._step_times_ms),
            "optimizer_step_time_p95_ms": self._percentile(
                self._step_times_ms,
                0.95,
            ),
            "optimizer_step_time_max_ms": max(self._step_times_ms),
            "optimizer_step_count": float(self._steps_this_epoch),
            "optimizer_basis_refresh_count": float(
                self._counter("basis_refresh_count") - self._basis_refresh_start
            ),
            "optimizer_peak_memory_mib": peak_memory_mib,
        }

    def _counter(self, name: str) -> int:
        value = getattr(self.optimizer, name, 0)
        if not isinstance(value, int):
            raise TypeError(f"Optimizer telemetry counter {name!r} must be an integer.")
        return value

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = quantile * (len(ordered) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = position - lower_index
        return ordered[lower_index] + fraction * (
            ordered[upper_index] - ordered[lower_index]
        )
