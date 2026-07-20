from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import (
    ComplexLengthJumpBalanceConfig,
    ComplexRelativeSplitConsistencyConfig,
)


@dataclass(frozen=True)
class ComplexLengthJumpPartition:
    """Batch-shared transition classification for valid geometry edges."""

    x_score: torch.Tensor
    y_score: torch.Tensor
    x_transition_mask: torch.Tensor
    y_transition_mask: torch.Tensor

    @property
    def total_edges(self) -> int:
        return int(self.x_score.numel() + self.y_score.numel())

    @property
    def transition_edges(self) -> int:
        return int(
            self.x_transition_mask.sum().item() + self.y_transition_mask.sum().item()
        )

    def to(self, device: torch.device | str) -> ComplexLengthJumpPartition:
        return type(self)(
            x_score=self.x_score.to(device),
            y_score=self.y_score.to(device),
            x_transition_mask=self.x_transition_mask.to(device),
            y_transition_mask=self.y_transition_mask.to(device),
        )


@dataclass(frozen=True)
class ComplexEnergyLossResult:
    """Unweighted audit energy and transition-balanced optimization energy."""

    unweighted: torch.Tensor
    balanced: torch.Tensor
    regular_mean: torch.Tensor
    transition_mean: torch.Tensor
    transition_edge_fraction: torch.Tensor
    unweighted_per_sample: torch.Tensor
    balanced_per_sample: torch.Tensor
    regular_per_sample: torch.Tensor
    transition_per_sample: torch.Tensor


@dataclass(frozen=True)
class ComplexRelativeSplitLossResult:
    """Source-normalized split energy and solution-value consistency."""

    loss: torch.Tensor
    energy_relative: torch.Tensor
    mass_relative: torch.Tensor
    loss_per_sample: torch.Tensor
    energy_relative_per_sample: torch.Tensor
    mass_relative_per_sample: torch.Tensor
    mass_unscaled_per_sample: torch.Tensor
    rhs_l2_squared_per_sample: torch.Tensor
    domain_length_scale: torch.Tensor


def build_length_jump_partition(
    geometry: ComplexGeometryMetadata,
    config: ComplexLengthJumpBalanceConfig,
) -> ComplexLengthJumpPartition:
    """Classify edges by jumps in the two directional response scales."""

    sigma_x = geometry.x_lengths_for_valid_points().square().clamp_min(config.eps)
    sigma_y = geometry.y_lengths_for_valid_points().square().clamp_min(config.eps)
    log_sigma_x = torch.log(sigma_x)
    log_sigma_y = torch.log(sigma_y)
    x_score = _edge_jump_score(log_sigma_x, log_sigma_y, geometry.x_edges)
    y_score = _edge_jump_score(log_sigma_x, log_sigma_y, geometry.y_edges)
    threshold = float(config.log_sigma_jump_threshold)
    return ComplexLengthJumpPartition(
        x_score=x_score,
        y_score=y_score,
        x_transition_mask=x_score > threshold,
        y_transition_mask=y_score > threshold,
    )


def physical_edge_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> torch.Tensor:
    residual, a_valid = _validate_energy_inputs(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
    )
    x_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.x_edges.to(residual.device),
        spacing=geometry.hx.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    y_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.y_edges.to(residual.device),
        spacing=geometry.hy.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    return _sum_edge_energy(x_values, y_values, residual)


def length_jump_balanced_edge_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    config: ComplexLengthJumpBalanceConfig,
    partition: ComplexLengthJumpPartition | None = None,
) -> ComplexEnergyLossResult:
    """Return the canonical energy and a geometry-group-balanced objective."""

    residual, a_valid = _validate_energy_inputs(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
    )
    x_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.x_edges.to(residual.device),
        spacing=geometry.hx.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    y_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.y_edges.to(residual.device),
        spacing=geometry.hy.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    unweighted_per_sample = _sum_edge_energy_per_sample(
        x_values,
        y_values,
        residual,
    )
    unweighted = unweighted_per_sample.mean()
    if partition is None:
        partition = build_length_jump_partition(geometry, config)
    partition = partition.to(residual.device)
    transition_mask = torch.cat(
        (partition.x_transition_mask, partition.y_transition_mask), dim=0
    )
    edge_values = torch.cat((x_values, y_values), dim=-1)
    total_edges = edge_values.shape[-1]
    if total_edges != partition.total_edges:
        raise ValueError("Length-jump partition does not match geometry edge count.")
    if total_edges == 0:
        zero = residual.new_zeros(())
        zero_per_sample = residual.new_zeros((residual.shape[0],))
        return ComplexEnergyLossResult(
            unweighted=unweighted,
            balanced=unweighted,
            regular_mean=zero,
            transition_mean=zero,
            transition_edge_fraction=zero,
            unweighted_per_sample=unweighted_per_sample,
            balanced_per_sample=unweighted_per_sample,
            regular_per_sample=zero_per_sample,
            transition_per_sample=zero_per_sample,
        )

    regular_mask = ~transition_mask
    regular_count = int(regular_mask.sum().item())
    transition_count = int(transition_mask.sum().item())
    regular_per_sample = (
        edge_values[:, regular_mask].mean(dim=-1)
        if regular_count > 0
        else residual.new_zeros((residual.shape[0],))
    )
    transition_per_sample = (
        edge_values[:, transition_mask].mean(dim=-1)
        if transition_count > 0
        else residual.new_zeros((residual.shape[0],))
    )
    regular_mean = regular_per_sample.mean()
    transition_mean = transition_per_sample.mean()
    transition_fraction = residual.new_tensor(transition_count / total_edges)
    if not config.enabled or regular_count == 0 or transition_count == 0:
        balanced_per_sample = unweighted_per_sample
    else:
        alpha = float(config.transition_fraction)
        regular_sum = edge_values[:, regular_mask].sum(dim=-1)
        transition_sum = edge_values[:, transition_mask].sum(dim=-1)
        balanced_per_sample = (
            (1.0 - alpha) * total_edges / regular_count * regular_sum
            + alpha * total_edges / transition_count * transition_sum
        )
    balanced = balanced_per_sample.mean()
    return ComplexEnergyLossResult(
        unweighted=unweighted,
        balanced=balanced,
        regular_mean=regular_mean,
        transition_mean=transition_mean,
        transition_edge_fraction=transition_fraction,
        unweighted_per_sample=unweighted_per_sample,
        balanced_per_sample=balanced_per_sample,
        regular_per_sample=regular_per_sample,
        transition_per_sample=transition_per_sample,
    )


def relative_split_consistency_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    rhs_valid: torch.Tensor,
    energy: ComplexEnergyLossResult,
    geometry: ComplexGeometryMetadata,
    config: ComplexRelativeSplitConsistencyConfig,
) -> ComplexRelativeSplitLossResult:
    """Normalize split energy and value mismatch by each sample source scale."""

    if u_phi_valid.shape != u_psi_valid.shape:
        raise ValueError("u_phi_valid and u_psi_valid must have matching shapes.")
    if u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if rhs_valid.shape != u_phi_valid.shape:
        raise ValueError("rhs_valid must match represented solution shape.")
    if energy.balanced_per_sample.shape != (u_phi_valid.shape[0],):
        raise ValueError("energy per-sample values do not match batch size.")

    area = geometry.hx.to(u_phi_valid.device) * geometry.hy.to(u_phi_valid.device)
    split_residual = u_phi_valid - u_psi_valid
    mass_unscaled_per_sample = split_residual.square().sum(dim=-1) * area
    rhs_l2_squared_per_sample = rhs_valid.square().sum(dim=-1) * area
    domain_length_scale = _domain_length_scale(
        geometry,
        reference=u_phi_valid,
    )
    denominator = rhs_l2_squared_per_sample + float(config.eps)
    energy_relative_per_sample = energy.balanced_per_sample / denominator
    mass_relative_per_sample = (
        mass_unscaled_per_sample / domain_length_scale.square()
    ) / denominator
    loss_per_sample = float(config.weight) * (
        energy_relative_per_sample
        + float(config.mass_weight) * mass_relative_per_sample
    )
    return ComplexRelativeSplitLossResult(
        loss=loss_per_sample.mean(),
        energy_relative=energy_relative_per_sample.mean(),
        mass_relative=mass_relative_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        energy_relative_per_sample=energy_relative_per_sample,
        mass_relative_per_sample=mass_relative_per_sample,
        mass_unscaled_per_sample=mass_unscaled_per_sample,
        rhs_l2_squared_per_sample=rhs_l2_squared_per_sample,
        domain_length_scale=domain_length_scale,
    )


def _validate_energy_inputs(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if u_phi_valid.shape != u_psi_valid.shape:
        raise ValueError("u_phi_valid and u_psi_valid must have matching shapes.")
    if u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if a_valid.dim() == 1:
        a_valid = a_valid.unsqueeze(0).expand_as(u_phi_valid)
    if a_valid.shape != u_phi_valid.shape:
        raise ValueError("a_valid must have shape (P,) or (B, P).")
    return u_phi_valid - u_psi_valid, a_valid


def _edge_energy_values(
    *,
    residual: torch.Tensor,
    a_valid: torch.Tensor,
    edges: torch.Tensor,
    spacing: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    if edges.numel() == 0:
        return residual.new_zeros((residual.shape[0], 0))
    left = edges[:, 0]
    right = edges[:, 1]
    derivative = (residual[:, right] - residual[:, left]) / spacing
    a_face = 0.5 * (a_valid[:, left] + a_valid[:, right])
    return a_face * derivative.pow(2) * area


def _sum_edge_energy(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if x_values.shape[-1] + y_values.shape[-1] == 0:
        return reference.new_zeros(())
    return (x_values.sum(dim=-1) + y_values.sum(dim=-1)).mean()


def _sum_edge_energy_per_sample(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if x_values.shape[-1] + y_values.shape[-1] == 0:
        return reference.new_zeros((reference.shape[0],))
    return x_values.sum(dim=-1) + y_values.sum(dim=-1)


def _domain_length_scale(
    geometry: ComplexGeometryMetadata,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    x_span = geometry.y_transverse_max.to(
        reference.device
    ) - geometry.y_transverse_min.to(reference.device)
    y_span = geometry.x_transverse_max.to(
        reference.device
    ) - geometry.x_transverse_min.to(reference.device)
    scale = torch.maximum(x_span, y_span).to(dtype=reference.dtype)
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
        raise ValueError("Complex geometry domain length scale must be positive.")
    return scale


def _edge_jump_score(
    log_sigma_x: torch.Tensor,
    log_sigma_y: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    if edges.numel() == 0:
        return log_sigma_x.new_empty((0,))
    left = edges[:, 0]
    right = edges[:, 1]
    x_jump = (log_sigma_x[right] - log_sigma_x[left]).abs()
    y_jump = (log_sigma_y[right] - log_sigma_y[left]).abs()
    return torch.maximum(x_jump, y_jump)


def relative_l2_valid(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes.")
    numerator = torch.linalg.vector_norm(pred - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
    return cast(torch.Tensor, (numerator / denominator).mean())
