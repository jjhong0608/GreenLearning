from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import BalanceProjectionConfig


@dataclass(frozen=True)
class ComplexProjectionResult:
    """Physical source split and diagnostics before Green reconstruction."""

    raw_physical: torch.Tensor
    projected_physical: torch.Tensor
    balance_residual: torch.Tensor
    raw_difference: torch.Tensor
    projected_difference: torch.Tensor
    response_baseline_difference: torch.Tensor
    response_gain: torch.Tensor


def apply_complex_balance_projection(
    raw_physical: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    config: BalanceProjectionConfig | str | dict[str, Any],
) -> ComplexProjectionResult:
    """Project physical directional proposals onto ``phi + psi = rhs``.

    Projection stays entirely in physical source space. Axis-length pull-back is
    owned by complex reconstruction and is intentionally absent here.
    """

    projection = BalanceProjectionConfig.from_raw(config)
    if not projection.enabled:
        raise ValueError("Complex balance projection must be enabled.")
    if projection.mode not in {"symmetric", "response_preconditioned"}:
        raise ValueError(
            "Complex balance projection supports only 'symmetric' or "
            "'response_preconditioned'."
        )

    x_length_squared, y_length_squared = _validate_inputs(
        raw_physical=raw_physical,
        rhs_phys=rhs_phys,
        geometry=geometry,
    )
    raw_difference = raw_physical[:, 0] - raw_physical[:, 1]
    balance_residual = rhs_phys - raw_physical[:, 0] - raw_physical[:, 1]

    if projection.mode == "symmetric":
        response_baseline = torch.zeros_like(raw_difference)
        response_gain = torch.ones_like(raw_difference)
        projected_difference = raw_difference
    else:
        denominator = x_length_squared + y_length_squared
        response_baseline = (
            (y_length_squared - x_length_squared) / denominator * rhs_phys
        )
        response_gain = (
            4.0 * x_length_squared * y_length_squared / denominator.square()
        ).expand_as(raw_difference)
        projected_difference = response_baseline + response_gain * raw_difference

    projected_physical = torch.stack(
        (
            0.5 * (rhs_phys + projected_difference),
            0.5 * (rhs_phys - projected_difference),
        ),
        dim=1,
    )
    return ComplexProjectionResult(
        raw_physical=raw_physical,
        projected_physical=projected_physical,
        balance_residual=balance_residual,
        raw_difference=raw_difference,
        projected_difference=projected_difference,
        response_baseline_difference=response_baseline,
        response_gain=response_gain,
    )


def apply_hard_symmetric_projection(
    raw_physical: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> ComplexProjectionResult:
    """Apply the default physical symmetric projection."""

    return apply_complex_balance_projection(
        raw_physical=raw_physical,
        rhs_phys=rhs_phys,
        geometry=geometry,
        config=BalanceProjectionConfig(enabled=True, mode="symmetric"),
    )


def _validate_inputs(
    *,
    raw_physical: torch.Tensor,
    rhs_phys: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw_physical.dim() != 3 or raw_physical.shape[1] != 2:
        raise ValueError("raw_physical must have shape (B, 2, P).")
    if rhs_phys.shape != raw_physical[:, 0].shape:
        raise ValueError("rhs_phys must have shape (B, P).")

    x_length_squared = (
        geometry.x_lengths_for_valid_points()
        .to(device=raw_physical.device, dtype=raw_physical.dtype)
        .square()
        .unsqueeze(0)
    )
    y_length_squared = (
        geometry.y_lengths_for_valid_points()
        .to(device=raw_physical.device, dtype=raw_physical.dtype)
        .square()
        .unsqueeze(0)
    )
    if raw_physical.shape[-1] != x_length_squared.shape[-1]:
        raise ValueError("raw_physical point count does not match geometry.")
    if torch.any(x_length_squared <= 0.0) or torch.any(y_length_squared <= 0.0):
        raise ValueError("Complex geometry segment lengths must be positive.")
    return x_length_squared, y_length_squared
