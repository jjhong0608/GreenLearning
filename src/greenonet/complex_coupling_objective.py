from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    ComplexEnergyLossResult,
    ComplexLengthJumpPartition,
    ComplexRelativeSplitLossResult,
    length_jump_balanced_edge_energy_loss,
    relative_split_consistency_loss,
)
from greenonet.complex_weak_closure import (
    ComplexDirectionalWeakContext,
    ComplexWeakClosureResult,
    directional_weak_operator_closure_loss,
)
from greenonet.config import (
    ComplexLengthJumpBalanceConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
)


@dataclass(frozen=True)
class ComplexCouplingObjectiveResult:
    """Shared reference-free objective for complex training and evaluation."""

    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    energy: ComplexEnergyLossResult
    relative_split: ComplexRelativeSplitLossResult | None
    weak_closure: ComplexWeakClosureResult | None
    relative_split_weight: float
    relative_split_mass_weight: float
    weak_closure_weight: float

    def metric_tensors(self) -> dict[str, torch.Tensor]:
        metrics = {
            "loss": self.loss,
            "loss_energy_consistency": self.energy.unweighted,
            "loss_energy_length_balanced": self.energy.balanced,
            "loss_energy_bulk": self.energy.bulk_unweighted,
            "loss_energy_bulk_length_balanced": self.energy.bulk_balanced,
            "loss_energy_boundary": self.energy.boundary,
            "loss_energy_boundary_x": self.energy.boundary_x,
            "loss_energy_boundary_y": self.energy.boundary_y,
            "loss_energy_regular": self.energy.regular_mean,
            "loss_energy_transition": self.energy.transition_mean,
            "transition_edge_fraction": self.energy.transition_edge_fraction,
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
            "loss_energy_consistency": self.energy.unweighted_per_sample[sample_offset],
            "loss_energy_length_balanced": self.energy.balanced_per_sample[
                sample_offset
            ],
            "loss_energy_bulk": self.energy.bulk_unweighted_per_sample[sample_offset],
            "loss_energy_bulk_length_balanced": (
                self.energy.bulk_balanced_per_sample[sample_offset]
            ),
            "loss_energy_boundary": self.energy.boundary_per_sample[sample_offset],
            "loss_energy_boundary_x": self.energy.boundary_x_per_sample[sample_offset],
            "loss_energy_boundary_y": self.energy.boundary_y_per_sample[sample_offset],
            "loss_energy_regular": self.energy.regular_per_sample[sample_offset],
            "loss_energy_transition": self.energy.transition_per_sample[sample_offset],
            "transition_edge_fraction": self.energy.transition_edge_fraction,
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


def compute_complex_coupling_objective(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    rhs_valid: torch.Tensor,
    projected_physical: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    weak_context: ComplexDirectionalWeakContext,
    length_jump_config: ComplexLengthJumpBalanceConfig,
    relative_split_config: ComplexRelativeSplitConsistencyConfig,
    weak_closure_config: ComplexWeakOperatorClosureConfig,
    boundary_context: ComplexBoundaryEnergyContext,
    partition: ComplexLengthJumpPartition | None = None,
) -> ComplexCouplingObjectiveResult:
    """Compute the configured complex objective without reference targets."""

    energy = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
        geometry=geometry,
        config=length_jump_config,
        partition=partition,
        boundary_context=boundary_context,
    )
    relative_split = None
    if relative_split_config.enabled:
        relative_split = relative_split_consistency_loss(
            u_phi_valid=u_phi_valid,
            u_psi_valid=u_psi_valid,
            rhs_valid=rhs_valid,
            energy=energy,
            geometry=geometry,
            config=relative_split_config,
        )
        split_loss_per_sample = relative_split.loss_per_sample
    else:
        split_loss_per_sample = energy.balanced_per_sample

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
        relative_split=relative_split,
        weak_closure=weak_closure,
        relative_split_weight=float(relative_split_config.weight),
        relative_split_mass_weight=float(relative_split_config.mass_weight),
        weak_closure_weight=float(weak_closure_config.weight),
    )
