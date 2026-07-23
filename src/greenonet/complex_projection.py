from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import BalanceProjectionConfig


@dataclass(frozen=True)
class ComplexProjectionResult:
    """Physical symmetric split with explicit reference pull-back diagnostics."""

    raw_response: torch.Tensor
    raw_physical: torch.Tensor
    projected_response: torch.Tensor
    projected_physical: torch.Tensor
    raw_response_constraint_residual: torch.Tensor
    response_constraint_residual: torch.Tensor
    physical_balance_residual: torch.Tensor
    raw_difference: torch.Tensor
    projected_difference: torch.Tensor
    sigma_x: torch.Tensor
    sigma_y: torch.Tensor


def apply_complex_balance_projection(
    raw_response: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    config: BalanceProjectionConfig | str | dict[str, Any],
) -> ComplexProjectionResult:
    """Project in physical source space and pull back to reference responses."""

    projection = BalanceProjectionConfig.from_raw(config)
    if not projection.enabled:
        raise ValueError("Complex balance projection must be enabled.")
    if projection.mode != "physical_symmetric":
        raise ValueError(
            "Complex output-contract version 6 requires "
            "balance_projection.mode='physical_symmetric'."
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
    projected_physical = torch.stack(
        (
            0.5 * (rhs_phys + raw_physical_difference),
            0.5 * (rhs_phys - raw_physical_difference),
        ),
        dim=1,
    )
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
        raw_response=raw_response,
        raw_physical=raw_physical,
        projected_response=projected_response,
        projected_physical=projected_physical,
        raw_response_constraint_residual=raw_constraint_residual,
        response_constraint_residual=response_constraint_residual,
        physical_balance_residual=physical_balance_residual,
        raw_difference=raw_physical_difference,
        projected_difference=(projected_physical[:, 0] - projected_physical[:, 1]),
        sigma_x=sigma_x.expand_as(rhs_phys),
        sigma_y=sigma_y.expand_as(rhs_phys),
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
