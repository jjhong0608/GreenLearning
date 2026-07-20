from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import BalanceProjectionConfig


@dataclass(frozen=True)
class ComplexProjectionResult:
    """Response-space split and physical-source diagnostics."""

    raw_response: torch.Tensor
    projected_response: torch.Tensor
    projected_physical: torch.Tensor
    raw_response_constraint_residual: torch.Tensor
    response_constraint_residual: torch.Tensor
    physical_balance_residual: torch.Tensor
    raw_difference: torch.Tensor
    projected_difference: torch.Tensor
    normal_x: torch.Tensor
    normal_y: torch.Tensor


def apply_complex_balance_projection(
    raw_response: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    config: BalanceProjectionConfig | str | dict[str, Any],
) -> ComplexProjectionResult:
    """Orthogonally project unit responses onto the physical balance constraint.

    The projected responses satisfy ``Phi / Lx^2 + Psi / Ly^2 = rhs``.
    No source pull-back is performed before or after this projection.
    """

    projection = BalanceProjectionConfig.from_raw(config)
    if not projection.enabled:
        raise ValueError("Complex balance projection must be enabled.")
    if projection.mode != "response_space":
        raise ValueError(
            "Complex output-contract version 5 requires "
            "balance_projection.mode='response_space'."
        )

    sigma_x, sigma_y = _validate_inputs(
        raw_response=raw_response,
        rhs_phys=rhs_phys,
        geometry=geometry,
    )
    scale = torch.maximum(sigma_x, sigma_y)
    normal_x = sigma_y / scale
    normal_y = sigma_x / scale
    constraint = sigma_x * sigma_y / scale * rhs_phys
    normal_norm_squared = normal_x.square() + normal_y.square()
    projection_residual = (
        normal_x * raw_response[:, 0] + normal_y * raw_response[:, 1] - constraint
    )
    projected_response = torch.stack(
        (
            raw_response[:, 0] - normal_x * projection_residual / normal_norm_squared,
            raw_response[:, 1] - normal_y * projection_residual / normal_norm_squared,
        ),
        dim=1,
    )
    projected_physical = torch.stack(
        (
            projected_response[:, 0] / sigma_x,
            projected_response[:, 1] / sigma_y,
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
        raw_response=raw_response,
        projected_response=projected_response,
        projected_physical=projected_physical,
        raw_response_constraint_residual=raw_constraint_residual,
        response_constraint_residual=response_constraint_residual,
        physical_balance_residual=physical_balance_residual,
        raw_difference=raw_response[:, 0] - raw_response[:, 1],
        projected_difference=projected_response[:, 0] - projected_response[:, 1],
        normal_x=normal_x.expand_as(rhs_phys),
        normal_y=normal_y.expand_as(rhs_phys),
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
