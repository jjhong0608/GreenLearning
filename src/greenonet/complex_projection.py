from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata


@dataclass(frozen=True)
class ComplexProjectionResult:
    raw_unit: torch.Tensor
    raw_physical: torch.Tensor
    projected_physical: torch.Tensor
    projected_unit: torch.Tensor
    balance_residual: torch.Tensor


def apply_hard_symmetric_projection(
    raw_unit: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> ComplexProjectionResult:
    """Apply hard balance projection in physical variables."""

    raw_physical, x_scale, y_scale = _raw_unit_to_physical(
        raw_unit=raw_unit,
        rhs_phys=rhs_phys,
        geometry=geometry,
    )

    residual = rhs_phys - raw_physical[:, 0] - raw_physical[:, 1]
    projected_physical = raw_physical.clone()
    projected_physical[:, 0] = raw_physical[:, 0] + 0.5 * residual
    projected_physical[:, 1] = raw_physical[:, 1] + 0.5 * residual

    projected_unit = _physical_to_unit(
        projected_physical=projected_physical,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    return ComplexProjectionResult(
        raw_unit=raw_unit,
        raw_physical=raw_physical,
        projected_physical=projected_physical,
        projected_unit=projected_unit,
        balance_residual=residual,
    )


def apply_geometry_weighted_projection(
    raw_unit: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    *,
    eps: float = 1.0e-12,
) -> ComplexProjectionResult:
    """Apply direct length-squared geometry-weighted balance projection."""

    raw_physical, x_scale, y_scale = _raw_unit_to_physical(
        raw_unit=raw_unit,
        rhs_phys=rhs_phys,
        geometry=geometry,
    )
    denominator = (x_scale + y_scale).clamp_min(eps)
    w_phi = x_scale / denominator
    w_psi = y_scale / denominator
    beta = 2.0 * w_phi * w_psi
    raw_difference = raw_physical[:, 0] - raw_physical[:, 1]

    projected_physical = torch.empty_like(raw_unit)
    projected_physical[:, 0] = (
        w_phi.unsqueeze(0) * rhs_phys + beta.unsqueeze(0) * raw_difference
    )
    projected_physical[:, 1] = (
        w_psi.unsqueeze(0) * rhs_phys - beta.unsqueeze(0) * raw_difference
    )

    projected_unit = _physical_to_unit(
        projected_physical=projected_physical,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    return ComplexProjectionResult(
        raw_unit=raw_unit,
        raw_physical=raw_physical,
        projected_physical=projected_physical,
        projected_unit=projected_unit,
        balance_residual=rhs_phys - raw_physical[:, 0] - raw_physical[:, 1],
    )


def _raw_unit_to_physical(
    *,
    raw_unit: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if raw_unit.dim() != 3 or raw_unit.shape[1] != 2:
        raise ValueError("raw_unit must have shape (B, 2, P).")
    if rhs_phys.shape != raw_unit[:, 0].shape:
        raise ValueError("rhs_phys must have shape (B, P).")
    x_scale = geometry.x_lengths_for_valid_points().to(raw_unit.device).pow(2)
    y_scale = geometry.y_lengths_for_valid_points().to(raw_unit.device).pow(2)
    if raw_unit.shape[-1] != x_scale.numel():
        raise ValueError("raw_unit point count does not match geometry.")
    raw_physical = torch.empty_like(raw_unit)
    raw_physical[:, 0] = raw_unit[:, 0] / x_scale.unsqueeze(0)
    raw_physical[:, 1] = raw_unit[:, 1] / y_scale.unsqueeze(0)
    return raw_physical, x_scale, y_scale


def _physical_to_unit(
    *,
    projected_physical: torch.Tensor,
    x_scale: torch.Tensor,
    y_scale: torch.Tensor,
) -> torch.Tensor:
    projected_unit = torch.empty_like(projected_physical)
    projected_unit[:, 0] = projected_physical[:, 0] * x_scale.unsqueeze(0)
    projected_unit[:, 1] = projected_physical[:, 1] * y_scale.unsqueeze(0)
    return projected_unit
