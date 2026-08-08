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
    ComplexResponseTrustConfig,
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
        if resolved.subspace_dimension != 1:
            raise ValueError(
                "Tangent eta-cap scheduling is available only for subspace_dimension=1."
            )
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
    """One matrix-free balance-preserving tangent subspace step."""

    delta: torch.Tensor
    subspace_dimension: int = 1
    eta_applied: torch.Tensor | None = None
    eta_cap: float | None = None
    eta_star: torch.Tensor | None = None
    eta_capped: torch.Tensor | None = None
    line_search_numerator: torch.Tensor | None = None
    line_search_denominator: torch.Tensor | None = None
    response_direction: torch.Tensor | None = None
    directional_response: torch.Tensor | None = None
    direction_0: torch.Tensor | None = None
    direction_1: torch.Tensor | None = None
    response_direction_0: torch.Tensor | None = None
    response_direction_1: torch.Tensor | None = None
    directional_response_0: torch.Tensor | None = None
    directional_response_1: torch.Tensor | None = None
    coefficient_0: torch.Tensor | None = None
    coefficient_1: torch.Tensor | None = None
    second_direction_active: torch.Tensor | None = None
    mismatch_k1: torch.Tensor | None = None
    mismatch_k2: torch.Tensor | None = None
    cost_k1: torch.Tensor | None = None
    cost_k2: torch.Tensor | None = None
    residual_gradient_post: torch.Tensor | None = None


@dataclass(frozen=True)
class KrylovK2StepResult:
    """Two preconditioned directions and their exact subspace coefficients."""

    delta_k1: torch.Tensor
    delta_k2: torch.Tensor
    direction_0: torch.Tensor
    direction_1: torch.Tensor
    response_direction_0: torch.Tensor
    response_direction_1: torch.Tensor
    directional_response_0: torch.Tensor
    directional_response_1: torch.Tensor
    coefficient_0: torch.Tensor
    coefficient_1: torch.Tensor
    line_search_numerator_0: torch.Tensor
    line_search_denominator_0: torch.Tensor
    second_direction_active: torch.Tensor
    mismatch_k1: torch.Tensor
    mismatch_k2: torch.Tensor
    cost_k1: torch.Tensor
    cost_k2: torch.Tensor
    residual_gradient_post: torch.Tensor


@dataclass(frozen=True)
class NormalizedPostLineSearchStationarityResult:
    """Source-normalized stationarity after the selected tangent correction."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    relative_ratio: torch.Tensor
    relative_ratio_per_sample: torch.Tensor
    initial_source_ratio: torch.Tensor
    initial_source_ratio_per_sample: torch.Tensor
    hessian_direction: torch.Tensor
    stationarity_residual: torch.Tensor
    initial_preconditioned_energy_per_sample: torch.Tensor
    residual_preconditioned_energy_per_sample: torch.Tensor
    source_response_energy: torch.Tensor
    source_response_energy_per_sample: torch.Tensor
    source_response: torch.Tensor


@dataclass(frozen=True)
class TangentSourceResponseNormalization:
    """Shared Hx(f/2), Hy(f/2) response and its per-sample energy."""

    source_response: torch.Tensor
    energy: torch.Tensor
    energy_per_sample: torch.Tensor


@dataclass(frozen=True)
class TangentResponseTrustResult:
    """Final tangent mismatch and correction response normalized by source."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    post_mismatch_ratio: torch.Tensor
    post_mismatch_ratio_per_sample: torch.Tensor
    correction_ratio: torch.Tensor
    correction_ratio_per_sample: torch.Tensor
    source_response_energy: torch.Tensor
    source_response_energy_per_sample: torch.Tensor
    post_mismatch_energy_per_sample: torch.Tensor
    correction_energy_per_sample: torch.Tensor
    source_response: torch.Tensor
    correction_response: torch.Tensor
    trust_weight: float


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
    subspace_dimension: Literal[1, 2]
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
            subspace_dimension=resolved.subspace_dimension,
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
        if self.subspace_dimension == 2:
            if eta_cap is not None:
                raise ValueError("eta_cap is not applicable to subspace_dimension=2.")
            result = matrix_free_krylov_k2_step(
                context=self,
                mismatch=mismatch,
                gradient=gradient,
                relative_eps=self.line_search_relative_eps,
            )
            return SymmetricTangentStepResult(
                delta=result.delta_k2,
                subspace_dimension=2,
                direction_0=result.direction_0,
                direction_1=result.direction_1,
                response_direction_0=result.response_direction_0,
                response_direction_1=result.response_direction_1,
                directional_response_0=result.directional_response_0,
                directional_response_1=result.directional_response_1,
                coefficient_0=result.coefficient_0,
                coefficient_1=result.coefficient_1,
                second_direction_active=result.second_direction_active,
                mismatch_k1=result.mismatch_k1,
                mismatch_k2=result.mismatch_k2,
                cost_k1=result.cost_k1,
                cost_k2=result.cost_k2,
                residual_gradient_post=result.residual_gradient_post,
            )
        resolved_cap = self._resolve_eta_cap(eta_cap)
        if self.eta_strategy == "fixed":
            delta = self.tangent_delta(gradient)
            applied = gradient.new_full((gradient.shape[0],), self.eta)
            return SymmetricTangentStepResult(
                delta=delta,
                subspace_dimension=1,
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
            subspace_dimension=1,
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
            "subspace_dimension": self.subspace_dimension,
            "eta": self.eta,
            "eta_applicability": (
                "k1_only_not_applied" if self.subspace_dimension == 2 else "applied"
            ),
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


def matrix_free_krylov_k2_step(
    *,
    context: SymmetricTangentGreenResponseContext,
    mismatch: torch.Tensor,
    gradient: torch.Tensor,
    relative_eps: float,
    monotonicity_relative_tol: float | None = None,
) -> KrylovK2StepResult:
    """Minimize response mismatch in a two-direction preconditioned subspace."""

    context.validate_for(mismatch)
    context.validate_for(gradient)
    if mismatch.shape != gradient.shape:
        raise ValueError("mismatch and gradient must share shape (B, P).")
    if not math.isfinite(relative_eps) or relative_eps <= 0.0:
        raise ValueError("relative_eps must be finite and positive.")
    if monotonicity_relative_tol is not None and (
        not math.isfinite(monotonicity_relative_tol) or monotonicity_relative_tol <= 0.0
    ):
        raise ValueError(
            "monotonicity_relative_tol must be finite and positive when provided."
        )

    inverse_denominator = context.denominator.reciprocal().unsqueeze(0)
    direction_0 = inverse_denominator * gradient
    directional_response_0 = context.response_operator.forward_pair(
        torch.stack((direction_0, direction_0), dim=1)
    )
    response_0 = directional_response_0[:, 0] + directional_response_0[:, 1]
    mass = context.point_mass
    mismatch_energy = mass * mismatch.square().sum(dim=1)
    response_0_energy = mass * response_0.square().sum(dim=1)
    scale_0 = torch.maximum(mismatch_energy, response_0_energy)
    eps_0 = relative_eps * scale_0 + torch.finfo(mismatch.dtype).tiny
    numerator_0 = (gradient * direction_0).sum(dim=1).clamp_min(0.0)
    denominator_0 = response_0_energy + eps_0
    coefficient_0 = numerator_0 / denominator_0

    hessian_direction_0 = context.tangent_gradient(response_0)
    residual_gradient_1 = gradient - coefficient_0.unsqueeze(1) * hessian_direction_0
    direction_1_raw = inverse_denominator * residual_gradient_1
    directional_response_1_raw = context.response_operator.forward_pair(
        torch.stack((direction_1_raw, direction_1_raw), dim=1)
    )
    response_1_raw = directional_response_1_raw[:, 0] + directional_response_1_raw[:, 1]

    cross = mass * (response_0 * response_1_raw).sum(dim=1)
    orthogonal_scale = cross / denominator_0
    direction_1 = direction_1_raw - orthogonal_scale.unsqueeze(1) * direction_0
    directional_response_1 = (
        directional_response_1_raw
        - orthogonal_scale.view(-1, 1, 1) * directional_response_0
    )
    response_1 = directional_response_1[:, 0] + directional_response_1[:, 1]
    response_1_energy = mass * response_1.square().sum(dim=1)
    eps_1 = (
        relative_eps * torch.maximum(mismatch_energy, response_1_energy)
        + torch.finfo(mismatch.dtype).tiny
    )
    numerator_1 = (gradient * direction_1).sum(dim=1)
    second_direction_active = response_1_energy > eps_1
    coefficient_1 = torch.where(
        second_direction_active,
        numerator_1 / (response_1_energy + eps_1),
        torch.zeros_like(numerator_1),
    )

    delta_k1 = -coefficient_0.unsqueeze(1) * direction_0
    delta_k2 = delta_k1 - coefficient_1.unsqueeze(1) * direction_1
    mismatch_k1 = mismatch - coefficient_0.unsqueeze(1) * response_0
    mismatch_k2 = mismatch_k1 - coefficient_1.unsqueeze(1) * response_1
    cost_k1 = mass * mismatch_k1.square().sum(dim=1)
    cost_k2 = mass * mismatch_k2.square().sum(dim=1)
    residual_gradient_post = context.tangent_gradient(mismatch_k2)

    if monotonicity_relative_tol is not None:
        tolerance = (
            monotonicity_relative_tol * torch.maximum(cost_k1, mismatch_energy)
            + torch.finfo(mismatch.dtype).tiny
        )
        if torch.any(cost_k2 > cost_k1 + tolerance):
            maximum = float((cost_k2 - cost_k1).max().item())
            raise RuntimeError(
                "K=2 response mismatch exceeded K=1 beyond tolerance: "
                f"max_increase={maximum:.6e}."
            )
    outputs = (
        delta_k1,
        delta_k2,
        direction_0,
        direction_1,
        directional_response_0,
        directional_response_1,
        coefficient_0,
        coefficient_1,
        mismatch_k1,
        mismatch_k2,
        cost_k1,
        cost_k2,
        residual_gradient_post,
    )
    if any(not torch.all(torch.isfinite(value)) for value in outputs):
        raise RuntimeError(
            "K=2 tangent subspace calculation produced non-finite values."
        )
    return KrylovK2StepResult(
        delta_k1=delta_k1,
        delta_k2=delta_k2,
        direction_0=direction_0,
        direction_1=direction_1,
        response_direction_0=response_0,
        response_direction_1=response_1,
        directional_response_0=directional_response_0,
        directional_response_1=directional_response_1,
        coefficient_0=coefficient_0,
        coefficient_1=coefficient_1,
        line_search_numerator_0=numerator_0,
        line_search_denominator_0=denominator_0,
        second_direction_active=second_direction_active,
        mismatch_k1=mismatch_k1,
        mismatch_k2=mismatch_k2,
        cost_k1=cost_k1,
        cost_k2=cost_k2,
        residual_gradient_post=residual_gradient_post,
    )


def normalized_post_line_search_stationarity_loss(
    *,
    context: SymmetricTangentGreenResponseContext,
    gradient: torch.Tensor,
    source_normalization: TangentSourceResponseNormalization,
    config: (ComplexPostLineSearchStationarityConfig | dict[str, object]),
    response_direction: torch.Tensor | None = None,
    eta_star: torch.Tensor | None = None,
    stationarity_residual: torch.Tensor | None = None,
) -> NormalizedPostLineSearchStationarityResult:
    """Measure source-normalized stationarity after a tangent subspace step."""

    resolved = ComplexPostLineSearchStationarityConfig.from_raw(config)
    if not resolved.enabled:
        raise ValueError(
            "Normalized post-line-search stationarity must be enabled before "
            "computing its loss."
        )
    context.validate_for(gradient)
    _validate_source_normalization(
        context=context,
        reference=gradient,
        source_normalization=source_normalization,
    )

    if stationarity_residual is None:
        if response_direction is None or eta_star is None:
            raise ValueError(
                "K=1 stationarity requires response_direction and eta_star."
            )
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
    else:
        context.validate_for(stationarity_residual)
        if stationarity_residual.shape != gradient.shape:
            raise ValueError(
                "stationarity_residual and gradient must share shape (B, P)."
            )
        if not torch.all(torch.isfinite(stationarity_residual)):
            raise ValueError("stationarity_residual must be finite.")
        hessian_direction = gradient - stationarity_residual
    inverse_denominator = context.denominator.reciprocal().unsqueeze(0)
    initial_energy = (gradient.square() * inverse_denominator).sum(dim=1)
    residual_energy = (stationarity_residual.square() * inverse_denominator).sum(dim=1)
    source_denominator = source_normalization.energy_per_sample + float(resolved.eps)
    loss_per_sample = residual_energy / source_denominator
    relative_ratio_per_sample = residual_energy / (initial_energy + float(resolved.eps))
    initial_source_ratio_per_sample = initial_energy / source_denominator
    outputs = (
        loss_per_sample,
        relative_ratio_per_sample,
        initial_source_ratio_per_sample,
    )
    if any(not torch.all(torch.isfinite(value)) for value in outputs):
        raise ValueError("Post-line-search stationarity loss is non-finite.")
    return NormalizedPostLineSearchStationarityResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        relative_ratio=relative_ratio_per_sample.mean(),
        relative_ratio_per_sample=relative_ratio_per_sample,
        initial_source_ratio=initial_source_ratio_per_sample.mean(),
        initial_source_ratio_per_sample=initial_source_ratio_per_sample,
        hessian_direction=hessian_direction,
        stationarity_residual=stationarity_residual,
        initial_preconditioned_energy_per_sample=initial_energy,
        residual_preconditioned_energy_per_sample=residual_energy,
        source_response_energy=source_normalization.energy,
        source_response_energy_per_sample=source_normalization.energy_per_sample,
        source_response=source_normalization.source_response,
    )


def tangent_source_response_normalization(
    *,
    context: SymmetricTangentGreenResponseContext,
    rhs_phys: torch.Tensor,
) -> TangentSourceResponseNormalization:
    """Evaluate the shared source-response denominator exactly once per batch."""

    context.validate_for(rhs_phys)
    if rhs_phys.dim() != 2:
        raise ValueError("rhs_phys must have shape (B, P).")
    half_rhs = 0.5 * rhs_phys
    source_response = context.response_operator.forward_pair(
        torch.stack((half_rhs, half_rhs), dim=1)
    )
    energy_per_sample = context.point_mass * source_response.square().sum(dim=(1, 2))
    if not torch.all(torch.isfinite(source_response)) or not torch.all(
        torch.isfinite(energy_per_sample)
    ):
        raise ValueError("Tangent source response contains non-finite values.")
    if torch.any(energy_per_sample < 0.0):
        raise ValueError("Tangent source-response energy must be non-negative.")
    return TangentSourceResponseNormalization(
        source_response=source_response,
        energy=energy_per_sample.mean(),
        energy_per_sample=energy_per_sample,
    )


def tangent_response_trust_loss(
    *,
    context: SymmetricTangentGreenResponseContext,
    mismatch_pre: torch.Tensor,
    mismatch_post: torch.Tensor,
    source_normalization: TangentSourceResponseNormalization,
    config: ComplexResponseTrustConfig | dict[str, object],
) -> TangentResponseTrustResult:
    """Measure the final response mismatch and correction magnitude."""

    resolved = ComplexResponseTrustConfig.from_raw(config)
    if not resolved.enabled:
        raise ValueError("Response-trust must be enabled before computing its loss.")
    context.validate_for(mismatch_pre)
    context.validate_for(mismatch_post)
    if mismatch_pre.shape != mismatch_post.shape:
        raise ValueError("mismatch_pre and mismatch_post must share shape (B, P).")
    _validate_source_normalization(
        context=context,
        reference=mismatch_pre,
        source_normalization=source_normalization,
    )

    correction_response = mismatch_post - mismatch_pre
    source_energy = source_normalization.energy_per_sample
    post_energy = context.point_mass * mismatch_post.square().sum(dim=1)
    correction_energy = context.point_mass * correction_response.square().sum(dim=1)
    denominator = source_energy + float(resolved.eps)
    post_ratio = post_energy / denominator
    correction_ratio = correction_energy / denominator
    loss_per_sample = post_ratio + float(resolved.trust_weight) * correction_ratio
    tensors = (
        source_normalization.source_response,
        correction_response,
        source_energy,
        post_energy,
        correction_energy,
        post_ratio,
        correction_ratio,
        loss_per_sample,
    )
    if any(not torch.all(torch.isfinite(value)) for value in tensors):
        raise ValueError("Response-trust loss contains non-finite values.")
    return TangentResponseTrustResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        post_mismatch_ratio=post_ratio.mean(),
        post_mismatch_ratio_per_sample=post_ratio,
        correction_ratio=correction_ratio.mean(),
        correction_ratio_per_sample=correction_ratio,
        source_response_energy=source_normalization.energy,
        source_response_energy_per_sample=source_energy,
        post_mismatch_energy_per_sample=post_energy,
        correction_energy_per_sample=correction_energy,
        source_response=source_normalization.source_response,
        correction_response=correction_response,
        trust_weight=float(resolved.trust_weight),
    )


def _validate_source_normalization(
    *,
    context: SymmetricTangentGreenResponseContext,
    reference: torch.Tensor,
    source_normalization: TangentSourceResponseNormalization,
) -> None:
    expected_response_shape = (reference.shape[0], 2, context.num_points)
    if source_normalization.source_response.shape != expected_response_shape:
        raise ValueError(
            "source_response must have shape "
            f"{expected_response_shape}, got "
            f"{tuple(source_normalization.source_response.shape)}."
        )
    if source_normalization.energy_per_sample.shape != (reference.shape[0],):
        raise ValueError("source-response energy must have shape (B,).")
    tensors = (
        source_normalization.source_response,
        source_normalization.energy,
        source_normalization.energy_per_sample,
    )
    if any(value.dtype != reference.dtype for value in tensors):
        raise ValueError("source normalization must share dtype with the batch.")
    if any(value.device != reference.device for value in tensors):
        raise ValueError("source normalization must share device with the batch.")
    if any(not torch.all(torch.isfinite(value)) for value in tensors):
        raise ValueError("source normalization contains non-finite values.")
    if torch.any(source_normalization.energy_per_sample < 0.0):
        raise ValueError("source-response energy must be non-negative.")


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
