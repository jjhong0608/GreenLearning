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

    residual = rhs_phys - raw_physical[:, 0] - raw_physical[:, 1]
    projected_physical = raw_physical.clone()
    projected_physical[:, 0] = raw_physical[:, 0] + 0.5 * residual
    projected_physical[:, 1] = raw_physical[:, 1] + 0.5 * residual

    projected_unit = torch.empty_like(raw_unit)
    projected_unit[:, 0] = projected_physical[:, 0] * x_scale.unsqueeze(0)
    projected_unit[:, 1] = projected_physical[:, 1] * y_scale.unsqueeze(0)
    return ComplexProjectionResult(
        raw_unit=raw_unit,
        raw_physical=raw_physical,
        projected_physical=projected_physical,
        projected_unit=projected_unit,
        balance_residual=residual,
    )
