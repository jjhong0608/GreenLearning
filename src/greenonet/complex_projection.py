from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata


@dataclass(frozen=True)
class ComplexProjectionResult:
    raw_physical: torch.Tensor
    projected_physical: torch.Tensor
    projected_unit: torch.Tensor
    balance_residual: torch.Tensor


def apply_hard_symmetric_projection(
    raw_physical: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> ComplexProjectionResult:
    """Project physical directional proposals onto ``phi + psi = rhs``."""

    x_scale, y_scale = _validate_and_length_scales(
        raw_physical=raw_physical,
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
        raw_physical=raw_physical,
        projected_physical=projected_physical,
        projected_unit=projected_unit,
        balance_residual=residual,
    )


def _validate_and_length_scales(
    *,
    raw_physical: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_physical.dim() != 3 or raw_physical.shape[1] != 2:
        raise ValueError("raw_physical must have shape (B, 2, P).")
    if rhs_phys.shape != raw_physical[:, 0].shape:
        raise ValueError("rhs_phys must have shape (B, P).")
    x_scale = (
        geometry.x_lengths_for_valid_points()
        .to(
            device=raw_physical.device,
            dtype=raw_physical.dtype,
        )
        .pow(2)
    )
    y_scale = (
        geometry.y_lengths_for_valid_points()
        .to(
            device=raw_physical.device,
            dtype=raw_physical.dtype,
        )
        .pow(2)
    )
    if raw_physical.shape[-1] != x_scale.numel():
        raise ValueError("raw_physical point count does not match geometry.")
    return x_scale, y_scale


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
