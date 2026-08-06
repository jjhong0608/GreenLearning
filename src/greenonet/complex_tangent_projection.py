from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import torch

from greenonet.complex_axial_response_operator import (
    FrozenAxialResponseOperatorBuilder,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import (
    ComplexPostLineSearchStationarityConfig,
    SymmetricTangentGreenResponseProjectionConfig,
)
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule


@dataclass(frozen=True)
class SymmetricTangentEtaCapSchedule:
    """Half-cosine eta cap that reuses CouplingNet's LR warmup duration."""

    eta_strategy: Literal["fixed", "closed_loop_exact_line_search"]
    final_eta: float
    enabled: bool
    configured_warmup_epochs: int
    effective_warmup_epochs: int
    total_epochs: int

    @classmethod
    def from_learning_rate_schedule(
        cls,
        *,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
        learning_rate_schedule: CouplingLearningRateSchedule,
    ) -> SymmetricTangentEtaCapSchedule:
        resolved = SymmetricTangentGreenResponseProjectionConfig.from_raw(config)
        adaptive = resolved.eta_strategy == "closed_loop_exact_line_search"
        enabled = adaptive and learning_rate_schedule.enabled
        return cls(
            eta_strategy=resolved.eta_strategy,
            final_eta=float(resolved.eta),
            enabled=enabled,
            configured_warmup_epochs=(
                learning_rate_schedule.configured_warmup_epochs if enabled else 0
            ),
            effective_warmup_epochs=(
                learning_rate_schedule.effective_warmup_epochs if enabled else 0
            ),
            total_epochs=learning_rate_schedule.total_epochs,
        )

    @property
    def kind(self) -> str:
        if self.eta_strategy == "fixed":
            return "fixed_eta"
        if not self.enabled or self.effective_warmup_epochs == 0:
            return "closed_loop_final_cap"
        return "closed_loop_half_cosine_warmup_hold"

    def cap_for_epoch_index(self, epoch_index: int) -> float:
        if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
            raise TypeError("epoch_index must be an integer.")
        if epoch_index < 0 or epoch_index >= self.total_epochs:
            raise ValueError(f"epoch_index must be in [0, {self.total_epochs - 1}].")
        warmup = self.effective_warmup_epochs
        if not self.enabled or warmup == 0 or epoch_index >= warmup:
            return self.final_eta
        progress = float(epoch_index + 1) / float(warmup)
        return self.final_eta * 0.5 * (1.0 - math.cos(math.pi * progress))


@dataclass(frozen=True)
class SymmetricTangentStepResult:
    """One fixed or closed-loop Jacobi-preconditioned tangent step."""

    delta: torch.Tensor
    eta_applied: torch.Tensor
    eta_cap: float
    eta_star: torch.Tensor | None = None
    eta_capped: torch.Tensor | None = None
    line_search_numerator: torch.Tensor | None = None
    line_search_denominator: torch.Tensor | None = None
    response_direction: torch.Tensor | None = None
    directional_response: torch.Tensor | None = None


@dataclass(frozen=True)
class NormalizedPostLineSearchStationarityResult:
    """Full stationarity residual after the uncapped scalar line search."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    hessian_direction: torch.Tensor
    stationarity_residual: torch.Tensor
    initial_preconditioned_energy_per_sample: torch.Tensor
    residual_preconditioned_energy_per_sample: torch.Tensor


@dataclass(frozen=True)
class SymmetricTangentGreenResponseContext:
    """Frozen response operator and fixed Jacobi tangent preconditioner."""

    response_operator: FrozenBidirectionalResponseOperator
    gamma_x_squared: torch.Tensor
    gamma_y_squared: torch.Tensor
    preconditioner_base: torch.Tensor
    gain_scale: torch.Tensor
    denominator: torch.Tensor
    point_mass: torch.Tensor
    eta: float
    eta_strategy: Literal["fixed", "closed_loop_exact_line_search"]
    line_search_relative_eps: float
    relative_lambda: float
    denominator_relative_eps: float

    @classmethod
    def from_response_operator(
        cls,
        *,
        response_operator: FrozenBidirectionalResponseOperator,
        point_mass: torch.Tensor | float,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> SymmetricTangentGreenResponseContext:
        resolved = SymmetricTangentGreenResponseProjectionConfig.from_raw(config)
        mass = torch.as_tensor(
            point_mass,
            dtype=response_operator.x.dtype,
            device=response_operator.x.device,
        )
        if mass.numel() != 1 or not torch.isfinite(mass):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((mass <= 0.0).item()):
            raise ValueError("point_mass must be positive.")
        gamma_x_squared, gamma_y_squared = response_operator.column_gain_squared(
            point_mass=mass
        )
        preconditioner_base = gamma_x_squared + gamma_y_squared
        gain_scale = preconditioner_base.mean()
        if not torch.isfinite(gain_scale) or bool((gain_scale <= 0.0).item()):
            raise ValueError(
                "The tangent Green-response preconditioner gain scale must be "
                "finite and positive."
            )
        denominator = (
            preconditioner_base
            + (
                float(resolved.relative_lambda)
                + float(resolved.denominator_relative_eps)
            )
            * gain_scale
        )
        if not torch.all(torch.isfinite(denominator)) or torch.any(denominator <= 0.0):
            raise ValueError(
                "The tangent Green-response denominator must be finite and positive."
            )
        return cls(
            response_operator=response_operator,
            gamma_x_squared=gamma_x_squared.detach(),
            gamma_y_squared=gamma_y_squared.detach(),
            preconditioner_base=preconditioner_base.detach(),
            gain_scale=gain_scale.detach(),
            denominator=denominator.detach(),
            point_mass=mass.detach(),
            eta=float(resolved.eta),
            eta_strategy=resolved.eta_strategy,
            line_search_relative_eps=float(resolved.line_search_relative_eps),
            relative_lambda=float(resolved.relative_lambda),
            denominator_relative_eps=float(resolved.denominator_relative_eps),
        )

    @property
    def num_points(self) -> int:
        return self.response_operator.point_count

    def validate_for(self, reference: torch.Tensor) -> None:
        if reference.shape[-1] != self.num_points:
            raise ValueError(
                "Tangent Green-response point count does not match projection input."
            )
        if reference.device != self.denominator.device:
            raise ValueError(
                "Tangent Green-response context and projection input must share a "
                "device."
            )
        if reference.dtype != self.denominator.dtype:
            raise ValueError(
                "Tangent Green-response context and projection input must share a "
                "dtype."
            )

    def tangent_gradient(self, mismatch: torch.Tensor) -> torch.Tensor:
        self.validate_for(mismatch)
        return self.response_operator.tangent_gradient(
            mismatch,
            point_mass=self.point_mass,
        )

    def tangent_delta(self, gradient: torch.Tensor) -> torch.Tensor:
        """Preserve the original fixed-eta tangent update exactly."""

        self.validate_for(gradient)
        if self.eta == 0.0:
            return torch.zeros_like(gradient)
        return -self.eta * gradient / self.denominator.unsqueeze(0)

    def tangent_step(
        self,
        *,
        mismatch: torch.Tensor,
        gradient: torch.Tensor,
        eta_cap: float | None = None,
    ) -> SymmetricTangentStepResult:
        self.validate_for(mismatch)
        self.validate_for(gradient)
        resolved_cap = self._resolve_eta_cap(eta_cap)
        if self.eta_strategy == "fixed":
            delta = self.tangent_delta(gradient)
            applied = gradient.new_full((gradient.shape[0],), self.eta)
            return SymmetricTangentStepResult(
                delta=delta,
                eta_applied=applied,
                eta_cap=self.eta,
            )
        direction = gradient / self.denominator.unsqueeze(0)
        directional_response = self.response_operator.forward_pair(
            torch.stack((direction, direction), dim=1)
        )
        response_direction = directional_response[:, 0] + directional_response[:, 1]
        mismatch_energy = self.point_mass * mismatch.square().sum(dim=1)
        response_energy = self.point_mass * response_direction.square().sum(dim=1)
        scale = torch.maximum(mismatch_energy, response_energy)
        numerical_eps = (
            self.line_search_relative_eps * scale + torch.finfo(mismatch.dtype).tiny
        )
        numerator = (gradient * direction).sum(dim=1).clamp_min(0.0)
        denominator = response_energy + numerical_eps
        eta_star = numerator / denominator
        cap = eta_star.new_full(eta_star.shape, resolved_cap)
        eta_applied = torch.minimum(eta_star, cap)
        delta = -eta_applied.unsqueeze(1) * direction
        return SymmetricTangentStepResult(
            delta=delta,
            eta_applied=eta_applied,
            eta_cap=resolved_cap,
            eta_star=eta_star,
            eta_capped=eta_star > cap,
            line_search_numerator=numerator,
            line_search_denominator=denominator,
            response_direction=response_direction,
            directional_response=directional_response,
        )

    def _resolve_eta_cap(self, eta_cap: float | None) -> float:
        if eta_cap is None:
            return self.eta
        if not isinstance(eta_cap, (int, float)) or isinstance(eta_cap, bool):
            raise TypeError("eta_cap must be numeric.")
        resolved = float(eta_cap)
        if not math.isfinite(resolved) or resolved < 0.0:
            raise ValueError("eta_cap must be finite and non-negative.")
        if resolved > self.eta:
            raise ValueError("eta_cap must not exceed the configured final eta.")
        return resolved

    def statistics(self) -> dict[str, float | int | bool | str]:
        x_stats = self.response_operator.x.statistics()
        y_stats = self.response_operator.y.statistics()
        return {
            "eta": self.eta,
            "eta_strategy": self.eta_strategy,
            "line_search_relative_eps": self.line_search_relative_eps,
            "relative_lambda": self.relative_lambda,
            "denominator_relative_eps": self.denominator_relative_eps,
            "gain_scale": float(self.gain_scale.item()),
            "gamma_x_squared_min": float(self.gamma_x_squared.min().item()),
            "gamma_x_squared_max": float(self.gamma_x_squared.max().item()),
            "gamma_y_squared_min": float(self.gamma_y_squared.min().item()),
            "gamma_y_squared_max": float(self.gamma_y_squared.max().item()),
            "preconditioner_base_min": float(self.preconditioner_base.min().item()),
            "preconditioner_base_max": float(self.preconditioner_base.max().item()),
            "denominator_min": float(self.denominator.min().item()),
            "denominator_max": float(self.denominator.max().item()),
            "x_segment_block_count": int(x_stats["segment_block_count"]),
            "y_segment_block_count": int(y_stats["segment_block_count"]),
            "x_local_matrix_entry_count": int(x_stats["local_matrix_entry_count"]),
            "y_local_matrix_entry_count": int(y_stats["local_matrix_entry_count"]),
            "global_matrix_materialized": False,
            "full_gram_solve": False,
            "row_norm_used": False,
        }


def normalized_post_line_search_stationarity_loss(
    *,
    context: SymmetricTangentGreenResponseContext,
    gradient: torch.Tensor,
    response_direction: torch.Tensor,
    eta_star: torch.Tensor,
    config: (ComplexPostLineSearchStationarityConfig | dict[str, object]),
) -> NormalizedPostLineSearchStationarityResult:
    """Measure full stationarity after the uncapped line-optimal tangent step."""

    resolved = ComplexPostLineSearchStationarityConfig.from_raw(config)
    if not resolved.enabled:
        raise ValueError(
            "Normalized post-line-search stationarity must be enabled before "
            "computing its loss."
        )
    context.validate_for(gradient)
    context.validate_for(response_direction)
    if gradient.shape != response_direction.shape:
        raise ValueError("gradient and response_direction must share a shape.")
    if eta_star.dim() != 1 or eta_star.shape[0] != gradient.shape[0]:
        raise ValueError("eta_star must have shape (B,).")
    if eta_star.dtype != gradient.dtype or eta_star.device != gradient.device:
        raise ValueError("eta_star must share dtype and device with gradient.")
    if not torch.all(torch.isfinite(eta_star)) or torch.any(eta_star < 0.0):
        raise ValueError("eta_star must be finite and non-negative.")

    # A z = S^T M S z uses one cached segment-local adjoint action.
    hessian_direction = context.tangent_gradient(response_direction)
    stationarity_residual = gradient - eta_star.unsqueeze(1) * hessian_direction
    inverse_denominator = context.denominator.reciprocal().unsqueeze(0)
    initial_energy = (gradient.square() * inverse_denominator).sum(dim=1)
    residual_energy = (stationarity_residual.square() * inverse_denominator).sum(dim=1)
    loss_per_sample = residual_energy / (initial_energy + float(resolved.eps))
    if not torch.all(torch.isfinite(loss_per_sample)):
        raise ValueError("Post-line-search stationarity loss is non-finite.")
    return NormalizedPostLineSearchStationarityResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        hessian_direction=hessian_direction,
        stationarity_residual=stationarity_residual,
        initial_preconditioned_energy_per_sample=initial_energy,
        residual_preconditioned_energy_per_sample=residual_energy,
    )


class SymmetricTangentGreenResponseContextBuilder:
    """Build one frozen tangent response context from the production Green kernel."""

    def __init__(
        self,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.config = SymmetricTangentGreenResponseProjectionConfig.from_raw(config)

    @torch.no_grad()
    def build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> SymmetricTangentGreenResponseContext:
        response_operator = FrozenAxialResponseOperatorBuilder.build(
            green_model=green_model,
            geometry=geometry,
            x_green_branch=x_green_branch,
            y_green_branch=y_green_branch,
        )
        point_mass = (geometry.hx * geometry.hy).to(
            device=x_green_branch.device,
            dtype=x_green_branch.dtype,
        )
        return SymmetricTangentGreenResponseContext.from_response_operator(
            response_operator=response_operator,
            point_mass=point_mass,
            config=self.config,
        )


class SymmetricTangentGreenResponseContextCache:
    """Lazily build and reuse one tangent response context per runtime."""

    def __init__(
        self,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.builder = SymmetricTangentGreenResponseContextBuilder(config)
        self.context: SymmetricTangentGreenResponseContext | None = None
        self.build_count = 0
        self.build_seconds = 0.0

    def get_or_build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> SymmetricTangentGreenResponseContext:
        if self.context is None:
            start = time.perf_counter()
            self.context = self.builder.build(
                green_model=green_model,
                geometry=geometry,
                x_green_branch=x_green_branch,
                y_green_branch=y_green_branch,
            )
            self.build_seconds = time.perf_counter() - start
            self.build_count += 1
        self.context.validate_for(x_green_branch.new_empty(geometry.num_points))
        return self.context
