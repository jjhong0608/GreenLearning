from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_gluing import (
    ComplexGluingContext,
    ComplexGluingLossResult,
    complex_admissibility_gluing_loss,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_losses import (
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
    ComplexAdmissibilityGluingConfig,
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
    gluing: ComplexGluingLossResult | None
    relative_split_weight: float
    relative_split_mass_weight: float
    weak_closure_weight: float

    def metric_tensors(self) -> dict[str, torch.Tensor]:
        metrics = {
            "loss": self.loss,
            "loss_energy_consistency": self.energy.unweighted,
            "loss_energy_length_balanced": self.energy.balanced,
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
        if self.gluing is not None:
            metrics.update(
                {
                    "loss_trace_gluing": self.gluing.loss,
                    "loss_trace_self": self.gluing.self_loss,
                    "loss_trace_self_regular": self.gluing.self_regular,
                    "loss_trace_self_transition": self.gluing.self_transition,
                    "loss_trace_carrier_transition": (self.gluing.carrier_transition),
                    "trace_self_x_rms": self.gluing.x_self_rms,
                    "trace_self_y_rms": self.gluing.y_self_rms,
                    "trace_carrier_x_rms": self.gluing.x_carrier_rms,
                    "trace_carrier_y_rms": self.gluing.y_carrier_rms,
                    "transition_trace_fraction": (
                        self.gluing.transition_trace_fraction
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
        if self.gluing is not None:
            metrics.update(
                {
                    "loss_trace_gluing": self.gluing.loss_per_sample[sample_offset],
                    "loss_trace_self": self.gluing.self_loss_per_sample[sample_offset],
                    "loss_trace_self_regular": (
                        self.gluing.self_regular_per_sample[sample_offset]
                    ),
                    "loss_trace_self_transition": (
                        self.gluing.self_transition_per_sample[sample_offset]
                    ),
                    "loss_trace_carrier_transition": (
                        self.gluing.carrier_transition_per_sample[sample_offset]
                    ),
                    "transition_trace_fraction": (
                        self.gluing.transition_trace_fraction
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
    gluing_config: ComplexAdmissibilityGluingConfig,
    gluing_context: ComplexGluingContext | None = None,
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

    gluing = None
    if gluing_config.enabled:
        if gluing_context is None:
            raise ValueError(
                "gluing_context is required when admissibility_gluing is enabled."
            )
        gluing = complex_admissibility_gluing_loss(
            u_phi_valid=u_phi_valid,
            u_psi_valid=u_psi_valid,
            a_valid=a_valid,
            context=gluing_context,
            config=gluing_config,
        )
        loss_per_sample = loss_per_sample + gluing.loss_per_sample

    return ComplexCouplingObjectiveResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        energy=energy,
        relative_split=relative_split,
        weak_closure=weak_closure,
        gluing=gluing,
        relative_split_weight=float(relative_split_config.weight),
        relative_split_mass_weight=float(relative_split_config.mass_weight),
        weak_closure_weight=float(weak_closure_config.weight),
    )
