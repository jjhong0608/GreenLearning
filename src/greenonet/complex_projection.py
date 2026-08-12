from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
)
from greenonet.complex_reconstruction import (
    ComplexReconstructionResult,
    reconstruct_from_projected_response,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    NormalizedPostLineSearchStationarityResult,
    SymmetricTangentGreenResponseContext,
    TangentResponseTrustResult,
    TangentSourceResponseNormalization,
    normalized_post_line_search_stationarity_loss,
    tangent_response_trust_loss,
    tangent_source_response_normalization,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexPostLineSearchStationarityConfig,
    ComplexResponseTrustConfig,
)


@dataclass(frozen=True)
class SymmetricTangentProjectionDiagnostics:
    """Audit tensors from a balance-preserving tangent subspace step."""

    symmetric_physical: torch.Tensor
    symmetric_solution: torch.Tensor
    mismatch_pre: torch.Tensor
    gradient: torch.Tensor
    preconditioner_base: torch.Tensor
    denominator: torch.Tensor
    delta: torch.Tensor
    projected_solution: torch.Tensor
    mismatch_post: torch.Tensor
    subspace_dimension: int
    eta_strategy: str
    eta_applied: torch.Tensor | None
    eta_cap: float | None
    eta_star: torch.Tensor | None
    eta_capped: torch.Tensor | None
    line_search_numerator: torch.Tensor | None
    line_search_denominator: torch.Tensor | None
    response_direction: torch.Tensor | None
    direction_0: torch.Tensor | None
    direction_1: torch.Tensor | None
    response_direction_0: torch.Tensor | None
    response_direction_1: torch.Tensor | None
    coefficient_0: torch.Tensor | None
    coefficient_1: torch.Tensor | None
    second_direction_active: torch.Tensor | None
    mismatch_k1: torch.Tensor | None
    cost_k1: torch.Tensor | None
    cost_k2: torch.Tensor | None
    residual_gradient_post: torch.Tensor | None
    subspace_result: KrylovSubspaceStepResult | None


@dataclass(frozen=True)
class ComplexProjectionResult:
    """Physical split with explicit correction and reference pull-back diagnostics."""

    mode: str
    raw_response: torch.Tensor
    raw_physical: torch.Tensor
    projected_response: torch.Tensor
    projected_physical: torch.Tensor
    raw_response_constraint_residual: torch.Tensor
    response_constraint_residual: torch.Tensor
    physical_balance_residual: torch.Tensor
    raw_difference: torch.Tensor
    projected_difference: torch.Tensor
    correction_phi: torch.Tensor
    correction_psi: torch.Tensor
    correction_weight_phi: torch.Tensor
    correction_weight_psi: torch.Tensor
    difference_update: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor
    column_diagonal_context: ColumnDiagonalGreenResponseContext | None
    symmetric_tangent_diagnostics: SymmetricTangentProjectionDiagnostics | None


@dataclass(frozen=True)
class TangentAuxiliaryLossResult:
    """Joint tangent diagnostics sharing one source-response normalization."""

    stationarity: NormalizedPostLineSearchStationarityResult | None
    response_trust: TangentResponseTrustResult | None
    source_normalization: TangentSourceResponseNormalization | None


def apply_complex_balance_projection(
    raw_response: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    config: BalanceProjectionConfig | str | dict[str, Any],
    column_diagonal_context: ColumnDiagonalGreenResponseContext | None = None,
    symmetric_tangent_context: SymmetricTangentGreenResponseContext | None = None,
    symmetric_tangent_eta_cap: float | None = None,
) -> ComplexProjectionResult:
    """Project in physical source space and pull back to reference responses."""

    projection = BalanceProjectionConfig.from_raw(config)
    if not projection.enabled:
        raise ValueError("Complex balance projection must be enabled.")
    if projection.mode not in {
        "physical_symmetric",
        "column_diagonal_green_response",
        "symmetric_tangent_green_response",
    }:
        raise ValueError(
            "Complex output-contract version 6 requires "
            "balance_projection.mode='physical_symmetric' or "
            "'column_diagonal_green_response' or "
            "'symmetric_tangent_green_response'."
        )

    sigma_x, sigma_y = _validate_inputs(
        raw_response=raw_response,
        rhs_phys=rhs_phys,
        geometry=geometry,
    )
    raw_physical = torch.stack(
        (raw_response[:, 0] / sigma_x, raw_response[:, 1] / sigma_y),
        dim=1,
    )
    raw_physical_difference = raw_physical[:, 0] - raw_physical[:, 1]
    raw_balance_residual = rhs_phys - raw_physical[:, 0] - raw_physical[:, 1]
    symmetric_physical = torch.stack(
        (
            0.5 * (rhs_phys + raw_physical_difference),
            0.5 * (rhs_phys - raw_physical_difference),
        ),
        dim=1,
    )
    tangent_diagnostics: SymmetricTangentProjectionDiagnostics | None = None
    if projection.mode == "physical_symmetric":
        projected_difference = raw_physical_difference
        projected_physical = symmetric_physical
        correction_weight_phi = torch.full_like(rhs_phys, 0.5)
        correction_weight_psi = torch.full_like(rhs_phys, 0.5)
        difference_update = torch.zeros_like(rhs_phys)
    elif projection.mode == "column_diagonal_green_response":
        if column_diagonal_context is None:
            raise ValueError(
                "column_diagonal_green_response projection requires a frozen "
                "column-diagonal Green-response context."
            )
        column_diagonal_context.validate_for(rhs_phys)
        correction_weight_phi = column_diagonal_context.correction_weight_phi.unsqueeze(
            0
        ).expand_as(rhs_phys)
        correction_weight_psi = 1.0 - correction_weight_phi
        difference_update = (
            correction_weight_phi - correction_weight_psi
        ) * raw_balance_residual
        projected_difference = raw_physical_difference + difference_update
        phi = 0.5 * (rhs_phys + projected_difference)
        projected_physical = torch.stack((phi, rhs_phys - phi), dim=1)
    else:
        if symmetric_tangent_context is None:
            raise ValueError(
                "symmetric_tangent_green_response projection requires a frozen "
                "tangent Green-response context."
            )
        symmetric_tangent_context.validate_for(rhs_phys)
        symmetric_solution = symmetric_tangent_context.response_operator.forward_pair(
            symmetric_physical
        )
        mismatch_pre = symmetric_solution[:, 0] - symmetric_solution[:, 1]
        gradient = symmetric_tangent_context.tangent_gradient(mismatch_pre)
        tangent_step = symmetric_tangent_context.tangent_step(
            mismatch=mismatch_pre,
            gradient=gradient,
            eta_cap=symmetric_tangent_eta_cap,
        )
        delta = tangent_step.delta
        if tangent_step.subspace_dimension >= 2:
            subspace = tangent_step.subspace_result
            if subspace is None:
                raise RuntimeError("K>=2 tangent subspace diagnostics are incomplete.")
            phi = symmetric_physical[:, 0] + delta
            projected_physical = torch.stack((phi, rhs_phys - phi), dim=1)
            projected_phi = symmetric_solution[:, 0]
            projected_psi = symmetric_solution[:, 1]
            for direction_index in range(tangent_step.subspace_dimension):
                coefficient = subspace.coefficients[direction_index].unsqueeze(1)
                response_pair = subspace.directional_responses[direction_index]
                projected_phi = projected_phi - coefficient * response_pair[:, 0]
                projected_psi = projected_psi + coefficient * response_pair[:, 1]
            projected_solution = torch.stack((projected_phi, projected_psi), dim=1)
        elif (
            symmetric_tangent_context.eta_strategy == "fixed"
            or tangent_step.directional_response is None
        ):
            if symmetric_tangent_context.eta == 0.0:
                projected_physical = symmetric_physical
            else:
                phi = symmetric_physical[:, 0] + delta
                projected_physical = torch.stack((phi, rhs_phys - phi), dim=1)
            projected_solution = (
                symmetric_tangent_context.response_operator.forward_pair(
                    projected_physical
                )
            )
        elif tangent_step.eta_cap == 0.0:
            projected_physical = symmetric_physical
            projected_solution = symmetric_solution
        else:
            if tangent_step.eta_applied is None:
                raise RuntimeError("K=1 adaptive eta diagnostics are incomplete.")
            phi = symmetric_physical[:, 0] + delta
            projected_physical = torch.stack((phi, rhs_phys - phi), dim=1)
            applied = tangent_step.eta_applied.unsqueeze(1)
            projected_solution = torch.stack(
                (
                    symmetric_solution[:, 0]
                    - applied * tangent_step.directional_response[:, 0],
                    symmetric_solution[:, 1]
                    + applied * tangent_step.directional_response[:, 1],
                ),
                dim=1,
            )
        mismatch_post = projected_solution[:, 0] - projected_solution[:, 1]
        projected_difference = projected_physical[:, 0] - projected_physical[:, 1]
        correction_weight_phi = torch.full_like(rhs_phys, 0.5)
        correction_weight_psi = torch.full_like(rhs_phys, 0.5)
        difference_update = projected_difference - raw_physical_difference
        tangent_diagnostics = SymmetricTangentProjectionDiagnostics(
            symmetric_physical=symmetric_physical,
            symmetric_solution=symmetric_solution,
            mismatch_pre=mismatch_pre,
            gradient=gradient,
            preconditioner_base=(symmetric_tangent_context.preconditioner_base),
            denominator=symmetric_tangent_context.denominator,
            delta=delta,
            projected_solution=projected_solution,
            mismatch_post=mismatch_post,
            subspace_dimension=tangent_step.subspace_dimension,
            eta_strategy=symmetric_tangent_context.eta_strategy,
            eta_applied=tangent_step.eta_applied,
            eta_cap=tangent_step.eta_cap,
            eta_star=tangent_step.eta_star,
            eta_capped=tangent_step.eta_capped,
            line_search_numerator=tangent_step.line_search_numerator,
            line_search_denominator=tangent_step.line_search_denominator,
            response_direction=tangent_step.response_direction,
            direction_0=tangent_step.direction_0,
            direction_1=tangent_step.direction_1,
            response_direction_0=tangent_step.response_direction_0,
            response_direction_1=tangent_step.response_direction_1,
            coefficient_0=tangent_step.coefficient_0,
            coefficient_1=tangent_step.coefficient_1,
            second_direction_active=tangent_step.second_direction_active,
            mismatch_k1=tangent_step.mismatch_k1,
            cost_k1=tangent_step.cost_k1,
            cost_k2=tangent_step.cost_k2,
            residual_gradient_post=tangent_step.residual_gradient_post,
            subspace_result=tangent_step.subspace_result,
        )
    correction_phi = projected_physical[:, 0] - raw_physical[:, 0]
    correction_psi = projected_physical[:, 1] - raw_physical[:, 1]
    projected_response = torch.stack(
        (
            sigma_x * projected_physical[:, 0],
            sigma_y * projected_physical[:, 1],
        ),
        dim=1,
    )
    raw_constraint_residual = (
        rhs_phys - raw_response[:, 0] / sigma_x - raw_response[:, 1] / sigma_y
    )
    response_constraint_residual = (
        rhs_phys
        - projected_response[:, 0] / sigma_x
        - projected_response[:, 1] / sigma_y
    )
    physical_balance_residual = (
        rhs_phys - projected_physical[:, 0] - projected_physical[:, 1]
    )
    return ComplexProjectionResult(
        mode=projection.mode,
        raw_response=raw_response,
        raw_physical=raw_physical,
        projected_response=projected_response,
        projected_physical=projected_physical,
        raw_response_constraint_residual=raw_constraint_residual,
        response_constraint_residual=response_constraint_residual,
        physical_balance_residual=physical_balance_residual,
        raw_difference=raw_physical_difference,
        projected_difference=(projected_physical[:, 0] - projected_physical[:, 1]),
        correction_phi=correction_phi,
        correction_psi=correction_psi,
        correction_weight_phi=correction_weight_phi,
        correction_weight_psi=correction_weight_psi,
        difference_update=difference_update,
        sigma_x=sigma_x.expand_as(rhs_phys),
        sigma_y=sigma_y.expand_as(rhs_phys),
        column_diagonal_context=column_diagonal_context,
        symmetric_tangent_diagnostics=tangent_diagnostics,
    )


def reconstruct_complex_projection(
    *,
    projection: ComplexProjectionResult,
    green_model: torch.nn.Module,
    geometry: ComplexGeometryMetadata,
    x_green_branch: torch.Tensor,
    y_green_branch: torch.Tensor,
) -> ComplexReconstructionResult:
    """Reuse tangent response blocks or run the existing Green reconstruction."""

    tangent = projection.symmetric_tangent_diagnostics
    if tangent is not None:
        return ComplexReconstructionResult(
            u_phi_valid=tangent.projected_solution[:, 0],
            u_psi_valid=tangent.projected_solution[:, 1],
            projected_response=projection.projected_response,
        )
    return reconstruct_from_projected_response(
        green_model=green_model,
        geometry=geometry,
        projected_response=projection.projected_response,
        x_green_branch=x_green_branch,
        y_green_branch=y_green_branch,
    )


def symmetric_tangent_metric_tensors(
    projection: ComplexProjectionResult,
) -> dict[str, torch.Tensor]:
    """Return detached-ready scalar diagnostics without changing the objective."""

    tangent = projection.symmetric_tangent_diagnostics
    if tangent is None:
        return {}
    eps = torch.finfo(tangent.mismatch_pre.dtype).eps
    mismatch_pre_rms = tangent.mismatch_pre.square().mean().sqrt()
    mismatch_post_rms = tangent.mismatch_post.square().mean().sqrt()
    correction_pair_norm = torch.linalg.vector_norm(
        torch.stack((tangent.delta, -tangent.delta), dim=1)
    )
    symmetric_pair_norm = torch.linalg.vector_norm(tangent.symmetric_physical)
    metrics = {
        "tangent_response_mismatch_pre": mismatch_pre_rms,
        "tangent_response_mismatch_post": mismatch_post_rms,
        "tangent_response_mismatch_ratio": (
            mismatch_post_rms / mismatch_pre_rms.clamp_min(eps)
        ),
        "tangent_gradient_rms": tangent.gradient.square().mean().sqrt(),
        "tangent_delta_rms": tangent.delta.square().mean().sqrt(),
        "tangent_delta_max_abs": tangent.delta.abs().max(),
        "tangent_correction_rel_symmetric_pair": (
            correction_pair_norm / symmetric_pair_norm.clamp_min(eps)
        ),
    }
    if tangent.eta_star is not None:
        if (
            tangent.eta_applied is None
            or tangent.eta_cap is None
            or tangent.eta_capped is None
            or tangent.line_search_numerator is None
            or tangent.line_search_denominator is None
        ):
            raise RuntimeError("Adaptive tangent diagnostics are incomplete.")
        metrics.update(
            {
                "tangent_eta_cap": tangent.mismatch_pre.new_tensor(tangent.eta_cap),
                "tangent_eta_star_mean": tangent.eta_star.mean(),
                "tangent_eta_applied_mean": tangent.eta_applied.mean(),
                "tangent_eta_cap_fraction": tangent.eta_capped.to(
                    tangent.mismatch_pre.dtype
                ).mean(),
                "tangent_line_search_numerator_mean": (
                    tangent.line_search_numerator.mean()
                ),
                "tangent_line_search_denominator_mean": (
                    tangent.line_search_denominator.mean()
                ),
            }
        )
    if tangent.subspace_dimension >= 2:
        subspace = tangent.subspace_result
        if subspace is None:
            raise RuntimeError("K>=2 tangent metric diagnostics are incomplete.")
        metrics["tangent_subspace_dimension"] = tangent.mismatch_pre.new_tensor(
            tangent.subspace_dimension
        )
        for direction_index in range(tangent.subspace_dimension):
            metrics[f"tangent_coefficient_{direction_index}_mean"] = (
                subspace.coefficients[direction_index].mean()
            )
            metrics[f"tangent_direction_{direction_index}_active_fraction"] = (
                subspace.direction_active[direction_index]
                .to(tangent.mismatch_pre.dtype)
                .mean()
            )
            metrics[f"tangent_response_cost_k{direction_index + 1}_mean"] = (
                subspace.costs[direction_index].mean()
            )
            if direction_index > 0:
                previous = subspace.costs[direction_index - 1].mean()
                metrics[
                    f"tangent_response_cost_k{direction_index + 1}_over_k"
                    f"{direction_index}"
                ] = subspace.costs[direction_index].mean() / previous.clamp_min(eps)
        metrics["tangent_second_direction_active_fraction"] = (
            subspace.direction_active[1].to(tangent.mismatch_pre.dtype).mean()
        )
        metrics["tangent_response_orthogonality_max"] = (
            subspace.response_orthogonality_max[-1].max()
        )
    return metrics


def post_line_search_stationarity_from_projection(
    *,
    projection: ComplexProjectionResult,
    context: SymmetricTangentGreenResponseContext | None,
    config: ComplexPostLineSearchStationarityConfig | dict[str, Any],
    source_normalization: TangentSourceResponseNormalization | None = None,
) -> NormalizedPostLineSearchStationarityResult | None:
    """Compute optional stationarity from the final tangent subspace residual."""

    resolved = ComplexPostLineSearchStationarityConfig.from_raw(config)
    if not resolved.enabled:
        return None
    if projection.mode != "symmetric_tangent_green_response" or context is None:
        raise ValueError(
            "Post-line-search stationarity requires symmetric tangent projection "
            "and its cached Green-response context."
        )
    tangent = projection.symmetric_tangent_diagnostics
    if tangent is None:
        raise RuntimeError(
            "Closed-loop tangent diagnostics are incomplete for stationarity loss."
        )
    normalization = source_normalization
    if normalization is None:
        normalization = tangent_source_response_normalization(
            context=context,
            rhs_phys=projection.projected_physical.sum(dim=1),
        )
    if tangent.subspace_dimension >= 2:
        if tangent.residual_gradient_post is None:
            raise RuntimeError(
                "K>=2 post-subspace residual gradient is missing for stationarity."
            )
        return normalized_post_line_search_stationarity_loss(
            context=context,
            gradient=tangent.gradient,
            stationarity_residual=tangent.residual_gradient_post,
            source_normalization=normalization,
            config=resolved,
        )
    if tangent.eta_star is None or tangent.response_direction is None:
        raise RuntimeError(
            "K=1 closed-loop diagnostics are incomplete for stationarity loss."
        )
    return normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=tangent.gradient,
        response_direction=tangent.response_direction,
        eta_star=tangent.eta_star,
        source_normalization=normalization,
        config=resolved,
    )


def post_line_search_stationarity_diagnostic_from_projection(
    *,
    projection: ComplexProjectionResult,
    context: SymmetricTangentGreenResponseContext | None,
    eps: float,
    source_normalization: TangentSourceResponseNormalization | None = None,
) -> NormalizedPostLineSearchStationarityResult:
    """Compute source-normalized stationarity and the legacy ratio diagnostically."""

    result = post_line_search_stationarity_from_projection(
        projection=projection,
        context=context,
        source_normalization=source_normalization,
        config=ComplexPostLineSearchStationarityConfig(
            enabled=True,
            weight=0.0,
            eps=eps,
        ),
    )
    if result is None:
        raise RuntimeError("Enabled stationarity diagnostic returned no result.")
    return result


def response_trust_from_projection(
    *,
    projection: ComplexProjectionResult,
    context: SymmetricTangentGreenResponseContext | None,
    rhs_phys: torch.Tensor,
    config: ComplexResponseTrustConfig | dict[str, Any],
    source_normalization: TangentSourceResponseNormalization | None = None,
) -> TangentResponseTrustResult | None:
    """Compute response trust from the actual applied tangent projection."""

    resolved = ComplexResponseTrustConfig.from_raw(config)
    if not resolved.enabled:
        return None
    if projection.mode != "symmetric_tangent_green_response" or context is None:
        raise ValueError(
            "Response-trust requires symmetric tangent projection and its cached "
            "Green-response context."
        )
    tangent = projection.symmetric_tangent_diagnostics
    if tangent is None or (
        tangent.subspace_dimension == 1 and tangent.eta_star is None
    ):
        raise RuntimeError(
            "Closed-loop tangent diagnostics are incomplete for response-trust."
        )
    normalization = source_normalization
    if normalization is None:
        normalization = tangent_source_response_normalization(
            context=context,
            rhs_phys=rhs_phys,
        )
    return tangent_response_trust_loss(
        context=context,
        mismatch_pre=tangent.mismatch_pre,
        mismatch_post=tangent.mismatch_post,
        source_normalization=normalization,
        config=resolved,
    )


def tangent_auxiliary_losses_from_projection(
    *,
    projection: ComplexProjectionResult,
    context: SymmetricTangentGreenResponseContext | None,
    rhs_phys: torch.Tensor,
    stationarity_config: ComplexPostLineSearchStationarityConfig | dict[str, Any],
    response_trust_config: ComplexResponseTrustConfig | dict[str, Any],
) -> TangentAuxiliaryLossResult:
    """Compute optional tangent losses with one shared Hx(f/2), Hy(f/2) pass."""

    stationarity_resolved = ComplexPostLineSearchStationarityConfig.from_raw(
        stationarity_config
    )
    response_trust_resolved = ComplexResponseTrustConfig.from_raw(response_trust_config)
    if not stationarity_resolved.enabled and not response_trust_resolved.enabled:
        return TangentAuxiliaryLossResult(
            stationarity=None,
            response_trust=None,
            source_normalization=None,
        )
    if projection.mode != "symmetric_tangent_green_response" or context is None:
        raise ValueError(
            "Tangent auxiliary losses require symmetric tangent projection and its "
            "cached Green-response context."
        )
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=rhs_phys,
    )
    response_trust = response_trust_from_projection(
        projection=projection,
        context=context,
        rhs_phys=rhs_phys,
        config=response_trust_resolved,
        source_normalization=source_normalization,
    )
    if stationarity_resolved.enabled:
        stationarity = post_line_search_stationarity_from_projection(
            projection=projection,
            context=context,
            config=stationarity_resolved,
            source_normalization=source_normalization,
        )
    else:
        stationarity = post_line_search_stationarity_diagnostic_from_projection(
            projection=projection,
            context=context,
            eps=response_trust_resolved.eps,
            source_normalization=source_normalization,
        )
    return TangentAuxiliaryLossResult(
        stationarity=stationarity,
        response_trust=response_trust,
        source_normalization=source_normalization,
    )


def _validate_inputs(
    *,
    raw_response: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_response.dim() != 3 or raw_response.shape[1] != 2:
        raise ValueError("raw_response must have shape (B, 2, P).")
    if rhs_phys.shape != raw_response[:, 0].shape:
        raise ValueError("rhs_phys must have shape (B, P).")

    sigma_x = (
        geometry.x_lengths_for_valid_points()
        .to(device=raw_response.device, dtype=raw_response.dtype)
        .square()
        .unsqueeze(0)
    )
    sigma_y = (
        geometry.y_lengths_for_valid_points()
        .to(device=raw_response.device, dtype=raw_response.dtype)
        .square()
        .unsqueeze(0)
    )
    if raw_response.shape[-1] != sigma_x.shape[-1]:
        raise ValueError("raw_response point count does not match geometry.")
    if torch.any(sigma_x <= 0.0) or torch.any(sigma_y <= 0.0):
        raise ValueError("Complex geometry segment lengths must be positive.")
    return sigma_x, sigma_y
