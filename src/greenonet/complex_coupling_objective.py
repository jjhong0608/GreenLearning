from __future__ import annotations

from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_tangent_projection import (
    NormalizedPostLineSearchStationarityResult,
    TangentResponseTrustResult,
)
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
    ComplexPostLineSearchStationarityConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexResponseTrustConfig,
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
    post_line_search_stationarity: NormalizedPostLineSearchStationarityResult | None
    response_trust: TangentResponseTrustResult | None
    relative_split_weight: float
    relative_split_mass_weight: float
    weak_closure_weight: float
    post_line_search_stationarity_weight: float
    post_line_search_stationarity_optimized: bool
    response_trust_weight: float
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
        if self.post_line_search_stationarity is not None:
            metrics["tangent_post_line_search_stationarity_ratio"] = (
                self.post_line_search_stationarity.loss
            )
            if self.post_line_search_stationarity_optimized:
                metrics["loss_tangent_post_line_search_stationarity"] = (
                    self.post_line_search_stationarity_weight
                    * self.post_line_search_stationarity.loss
                )
        if self.response_trust is not None:
            metrics.update(
                {
                    "loss_tangent_response_trust": (
                        self.response_trust_weight * self.response_trust.loss
                    ),
                    "tangent_response_trust_ratio": self.response_trust.loss,
                    "tangent_response_post_mismatch_ratio": (
                        self.response_trust.post_mismatch_ratio
                    ),
                    "tangent_response_correction_ratio": (
                        self.response_trust.correction_ratio
                    ),
                    "tangent_source_response_energy": (
                        self.response_trust.source_response_energy
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
        if self.post_line_search_stationarity is not None:
            ratio = self.post_line_search_stationarity.loss_per_sample[sample_offset]
            metrics["tangent_post_line_search_stationarity_ratio"] = ratio
            if self.post_line_search_stationarity_optimized:
                metrics["loss_tangent_post_line_search_stationarity"] = (
                    self.post_line_search_stationarity_weight * ratio
                )
        if self.response_trust is not None:
            response = self.response_trust
            ratio = response.loss_per_sample[sample_offset]
            metrics.update(
                {
                    "loss_tangent_response_trust": self.response_trust_weight * ratio,
                    "tangent_response_trust_ratio": ratio,
                    "tangent_response_post_mismatch_ratio": (
                        response.post_mismatch_ratio_per_sample[sample_offset]
                    ),
                    "tangent_response_correction_ratio": (
                        response.correction_ratio_per_sample[sample_offset]
                    ),
                    "tangent_source_response_energy": (
                        response.source_response_energy_per_sample[sample_offset]
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
    post_line_search_stationarity_config: (
        ComplexPostLineSearchStationarityConfig | None
    ) = None,
    post_line_search_stationarity: (
        NormalizedPostLineSearchStationarityResult | None
    ) = None,
    response_trust_config: ComplexResponseTrustConfig | None = None,
    response_trust: TangentResponseTrustResult | None = None,
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

    stationarity_config = (
        ComplexPostLineSearchStationarityConfig()
        if post_line_search_stationarity_config is None
        else ComplexPostLineSearchStationarityConfig.from_raw(
            post_line_search_stationarity_config
        )
    )
    resolved_response_trust = (
        ComplexResponseTrustConfig()
        if response_trust_config is None
        else ComplexResponseTrustConfig.from_raw(response_trust_config)
    )
    stationarity_diagnostic_expected = (
        stationarity_config.enabled or resolved_response_trust.enabled
    )
    if stationarity_diagnostic_expected != (post_line_search_stationarity is not None):
        raise ValueError(
            "Enabled stationarity or response-trust config and computed "
            "stationarity diagnostic must be provided together."
        )
    if post_line_search_stationarity is not None:
        if post_line_search_stationarity.loss_per_sample.shape != loss_per_sample.shape:
            raise ValueError(
                "Post-line-search stationarity batch shape does not match objective."
            )
        if stationarity_config.enabled:
            loss_per_sample = (
                loss_per_sample
                + float(stationarity_config.weight)
                * post_line_search_stationarity.loss_per_sample
            )
    if resolved_response_trust.enabled != (response_trust is not None):
        raise ValueError(
            "Enabled response-trust config and computed result must be provided "
            "together."
        )
    if response_trust is not None:
        if response_trust.loss_per_sample.shape != loss_per_sample.shape:
            raise ValueError("Response-trust batch shape does not match objective.")
        loss_per_sample = (
            loss_per_sample
            + float(resolved_response_trust.weight) * response_trust.loss_per_sample
        )

    return ComplexCouplingObjectiveResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        energy=energy,
        energy_optimized=energy_optimized,
        energy_optimized_per_sample=energy_optimized_per_sample,
        relative_split=relative_split,
        weak_closure=weak_closure,
        post_line_search_stationarity=post_line_search_stationarity,
        response_trust=response_trust,
        relative_split_weight=float(relative_split_config.weight),
        relative_split_mass_weight=float(relative_split_config.mass_weight),
        weak_closure_weight=float(weak_closure_config.weight),
        post_line_search_stationarity_weight=float(stationarity_config.weight),
        post_line_search_stationarity_optimized=stationarity_config.enabled,
        response_trust_weight=float(resolved_response_trust.weight),
        boundary_weight=boundary_weight,
    )
