from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    ComplexEnergyLossResult,
    ComplexRelativeSplitLossResult,
    canonical_complex_energy_loss,
    relative_split_consistency_loss,
)
from greenonet.complex_weak_closure import (
    ComplexDirectionalWeakContext,
    ComplexWeakClosureResult,
    directional_weak_operator_closure_loss,
)
from greenonet.config import (
    ComplexCanonicalEnergyConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
)


@dataclass(frozen=True)
class ComplexCouplingObjectiveResult:
    """Shared reference-free objective for complex training and evaluation."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    energy: ComplexEnergyLossResult
    energy_optimized: torch.Tensor
    energy_optimized_per_sample: torch.Tensor
    relative_split: ComplexRelativeSplitLossResult | None
    weak_closure: ComplexWeakClosureResult | None
    relative_split_weight: float
    relative_split_mass_weight: float
    weak_closure_weight: float
    boundary_weight: float

    def metric_tensors(self) -> dict[str, torch.Tensor]:
        metrics = {
            "loss": self.loss,
            "boundary_weight": self.loss.new_tensor(self.boundary_weight),
            "loss_energy_optimized": self.energy_optimized,
            "loss_energy_consistency": self.energy.total,
            "loss_energy_bulk": self.energy.bulk,
            "loss_energy_boundary": self.energy.boundary,
            "loss_energy_boundary_x": self.energy.boundary_x,
            "loss_energy_boundary_y": self.energy.boundary_y,
        }
        if self.relative_split is not None:
            metrics.update(
                {
                    "loss_split_relative": self.relative_split.loss,
                    "loss_split_energy_relative": (
                        self.relative_split_weight * self.relative_split.energy_relative
                    ),
                    "loss_split_mass_relative": (
                        self.relative_split_weight
                        * self.relative_split_mass_weight
                        * self.relative_split.mass_relative
                    ),
                }
            )
        if self.weak_closure is not None:
            metrics.update(
                {
                    "loss_weak_operator_closure": (
                        self.weak_closure_weight * self.weak_closure.loss
                    ),
                    "loss_weak_operator_x": (
                        self.weak_closure_weight * self.weak_closure.x_loss
                    ),
                    "loss_weak_operator_y": (
                        self.weak_closure_weight * self.weak_closure.y_loss
                    ),
                }
            )
        return metrics

    def sample_metric_tensors(self, sample_offset: int) -> dict[str, torch.Tensor]:
        metrics = {
            "loss": self.loss_per_sample[sample_offset],
            "boundary_weight": self.loss_per_sample.new_tensor(self.boundary_weight),
            "loss_energy_optimized": self.energy_optimized_per_sample[sample_offset],
            "loss_energy_consistency": self.energy.total_per_sample[sample_offset],
            "loss_energy_bulk": self.energy.bulk_per_sample[sample_offset],
            "loss_energy_boundary": self.energy.boundary_per_sample[sample_offset],
            "loss_energy_boundary_x": self.energy.boundary_x_per_sample[sample_offset],
            "loss_energy_boundary_y": self.energy.boundary_y_per_sample[sample_offset],
        }
        if self.relative_split is not None:
            metrics.update(
                {
                    "loss_split_relative": self.relative_split.loss_per_sample[
                        sample_offset
                    ],
                    "loss_split_energy_relative": (
                        self.relative_split_weight
                        * self.relative_split.energy_relative_per_sample[sample_offset]
                    ),
                    "loss_split_mass_relative": (
                        self.relative_split_weight
                        * self.relative_split_mass_weight
                        * self.relative_split.mass_relative_per_sample[sample_offset]
                    ),
                }
            )
        if self.weak_closure is not None:
            metrics.update(
                {
                    "loss_weak_operator_closure": (
                        self.weak_closure_weight
                        * self.weak_closure.loss_per_sample[sample_offset]
                    ),
                    "loss_weak_operator_x": (
                        self.weak_closure_weight
                        * self.weak_closure.x_loss_per_sample[sample_offset]
                    ),
                    "loss_weak_operator_y": (
                        self.weak_closure_weight
                        * self.weak_closure.y_loss_per_sample[sample_offset]
                    ),
                }
            )
        return metrics


def optimized_complex_energy_per_sample(
    energy: ComplexEnergyLossResult,
    config: ComplexCanonicalEnergyConfig,
) -> torch.Tensor:
    """Apply the fixed boundary weight without changing canonical diagnostics."""

    boundary_weight = float(config.boundary_weight)
    if boundary_weight == 0.0:
        return energy.bulk_per_sample
    if boundary_weight == 1.0:
        return energy.total_per_sample
    return energy.bulk_per_sample + boundary_weight * energy.boundary_per_sample


def compute_complex_coupling_objective(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    rhs_valid: torch.Tensor,
    projected_physical: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    weak_context: ComplexDirectionalWeakContext,
    canonical_energy_config: ComplexCanonicalEnergyConfig,
    relative_split_config: ComplexRelativeSplitConsistencyConfig,
    weak_closure_config: ComplexWeakOperatorClosureConfig,
    boundary_context: ComplexBoundaryEnergyContext,
) -> ComplexCouplingObjectiveResult:
    """Compute the configured complex objective without reference targets."""

    energy = canonical_complex_energy_loss(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
        geometry=geometry,
        boundary_context=boundary_context,
    )
    boundary_weight = float(canonical_energy_config.boundary_weight)
    energy_optimized_per_sample = optimized_complex_energy_per_sample(
        energy,
        canonical_energy_config,
    )
    energy_optimized = energy_optimized_per_sample.mean()
    relative_split = None
    if relative_split_config.enabled:
        relative_split = relative_split_consistency_loss(
            u_phi_valid=u_phi_valid,
            u_psi_valid=u_psi_valid,
            rhs_valid=rhs_valid,
            optimized_energy_per_sample=energy_optimized_per_sample,
            geometry=geometry,
            config=relative_split_config,
        )
        split_loss_per_sample = relative_split.loss_per_sample
    else:
        split_loss_per_sample = energy_optimized_per_sample

    weak_closure = None
    if weak_closure_config.enabled:
        weak_closure = directional_weak_operator_closure_loss(
            u_pred_valid=0.5 * (u_phi_valid + u_psi_valid),
            projected_physical=projected_physical,
            rhs_valid=rhs_valid,
            context=weak_context,
            eps=weak_closure_config.eps,
        )
        loss_per_sample = (
            split_loss_per_sample
            + float(weak_closure_config.weight) * weak_closure.loss_per_sample
        )
    else:
        loss_per_sample = split_loss_per_sample

    return ComplexCouplingObjectiveResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        energy=energy,
        energy_optimized=energy_optimized,
        energy_optimized_per_sample=energy_optimized_per_sample,
        relative_split=relative_split,
        weak_closure=weak_closure,
        relative_split_weight=float(relative_split_config.weight),
        relative_split_mass_weight=float(relative_split_config.mass_weight),
        weak_closure_weight=float(weak_closure_config.weight),
        boundary_weight=boundary_weight,
    )
