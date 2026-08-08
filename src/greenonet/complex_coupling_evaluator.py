from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
    ComplexCrossAxisReconstructor,
)
from greenonet.complex_pre_projection_fusion import (
    ComplexPreProjectionFusionResult,
)
from greenonet.complex_coupling_objective import (
    ComplexCouplingObjectiveResult,
    compute_complex_coupling_objective,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
    ColumnDiagonalGreenResponseContextCache,
)
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    ComplexEnergyLossResult,
    build_boundary_energy_context,
    relative_l2_valid,
)
from greenonet.complex_projection import (
    ComplexProjectionResult,
    apply_complex_balance_projection,
    reconstruct_complex_projection,
    symmetric_tangent_metric_tensors,
    tangent_auxiliary_losses_from_projection,
)
from greenonet.complex_reconstruction import (
    ComplexReconstructionResult,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
    SymmetricTangentGreenResponseContextCache,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingTrainingConfig,
    SymmetricTangentGreenResponseProjectionConfig,
    validate_complex_post_line_search_stationarity_config,
    validate_complex_response_trust_config,
)
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexPredictionBatch:
    batch: ComplexCouplingBatch
    raw_response: torch.Tensor
    pre_projection_fusion: ComplexPreProjectionFusionResult | None
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult
    cross_axis_reconstruction: ComplexCrossAxisReconstructionResult
    energy: ComplexEnergyLossResult
    objective: ComplexCouplingObjectiveResult
    metrics: dict[str, torch.Tensor]


class ComplexCouplingEvaluator(LoggingMixin):
    """Evaluate complex CouplingNet with the configured final reconstruction."""

    def __init__(
        self,
        *,
        model: ComplexCouplingNet,
        green_model: torch.nn.Module,
        config: CouplingTrainingConfig,
        device: torch.device,
        work_dir: Path | str,
        terminal_width: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.balance_projection = BalanceProjectionConfig.from_raw(
            model.config.balance_projection
        )
        self.cross_axis_reconstruction_config = (
            ComplexCrossAxisReconstructionConfig.from_raw(
                model.config.cross_axis_reconstruction
            )
        )
        self.cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            self.cross_axis_reconstruction_config
        )
        self.canonical_energy_config = ComplexCanonicalEnergyConfig.from_raw(
            config.canonical_energy
        )
        self.relative_split_config = ComplexRelativeSplitConsistencyConfig.from_raw(
            config.relative_split_consistency
        )
        self.weak_closure_config = ComplexWeakOperatorClosureConfig.from_raw(
            config.weak_operator_closure
        )
        self.post_line_search_stationarity_config = (
            validate_complex_post_line_search_stationarity_config(
                training=config,
                balance_projection=self.balance_projection,
            )
        )
        self.response_trust_config = validate_complex_response_trust_config(
            training=config,
            balance_projection=self.balance_projection,
        )
        self.green_model = green_model.to(device)
        self.green_model.eval()
        for parameter in self.green_model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexCouplingEvaluator",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )
        self._boundary_context: ComplexBoundaryEnergyContext | None = None
        self._green_response_context_cache = ColumnDiagonalGreenResponseContextCache(
            self.balance_projection.column_diagonal_green_response
        )
        self._tangent_context_cache = SymmetricTangentGreenResponseContextCache(
            self.balance_projection.symmetric_tangent_green_response
        )
        tangent_config = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            self.balance_projection.symmetric_tangent_green_response
        )
        stationarity_residual_source = (
            "post_k2_residual_gradient"
            if tangent_config.subspace_dimension == 2
            else "uncapped_eta_star"
        )
        tangent_forward_source = (
            "unconstrained_k2_coefficients"
            if tangent_config.subspace_dimension == 2
            else "capped_eta_applied"
        )
        self.logger.info(
            "canonical energy boundary_weight=%.6e "
            "boundary_in_optimization=%s boundary_diagnostic_always=true "
            "optimized_formula=bulk+boundary_weight*boundary "
            "canonical_formula=bulk+boundary",
            self.canonical_energy_config.boundary_weight,
            self.canonical_energy_config.boundary_weight > 0.0,
        )
        self.logger.info(
            "final reconstruction enabled=%s mode=%s gamma=%.6f "
            "smoothing_steps=%d smoothing_relaxation=%.6f relative_floor=%.6f "
            "affects_training_objective=false",
            self.cross_axis_reconstruction_config.enabled,
            self.cross_axis_reconstruction_config.mode,
            self.cross_axis_reconstruction_config.gamma,
            self.cross_axis_reconstruction_config.smoothing_steps,
            self.cross_axis_reconstruction_config.smoothing_relaxation,
            self.cross_axis_reconstruction_config.relative_floor,
        )
        self.logger.info(
            "post-line-search stationarity enabled=%s weight=%.6e eps=%.6e "
            "eta_source=%s forward_eta_source=%s "
            "subspace_dimension=%d residual_source=%s forward_source=%s "
            "optimization_normalization=source_response "
            "legacy_initial_gradient_ratio=diagnostic_only "
            "matrix_free=true extra_adjoint_when_computed=%s "
            "shared_source_response_with_response_trust=%s "
            "uses_reference_targets=false",
            self.post_line_search_stationarity_config.enabled,
            self.post_line_search_stationarity_config.weight,
            self.post_line_search_stationarity_config.eps,
            (
                "not_applicable"
                if tangent_config.subspace_dimension == 2
                else stationarity_residual_source
            ),
            (
                "not_applicable"
                if tangent_config.subspace_dimension == 2
                else tangent_forward_source
            ),
            tangent_config.subspace_dimension,
            stationarity_residual_source,
            tangent_forward_source,
            tangent_config.subspace_dimension == 1
            and (
                self.post_line_search_stationarity_config.enabled
                or self.response_trust_config.enabled
            ),
            self.post_line_search_stationarity_config.enabled
            and self.response_trust_config.enabled,
        )
        self.logger.info(
            "response-trust enabled=%s weight=%.6e trust_weight=%.6e eps=%.6e "
            "eta_source=%s "
            "subspace_dimension=%d correction_source=%s "
            "source_normalization=Hx(f/2)^2+Hy(f/2)^2 "
            "matrix_free=true extra_forward_when_enabled=%s "
            "stationarity_diagnostic_when_enabled=%s "
            "extra_adjoint_when_enabled=%s joint_stationarity_enabled=%s "
            "source_response_shared_with_stationarity=%s "
            "uses_reference_targets=false",
            self.response_trust_config.enabled,
            self.response_trust_config.weight,
            self.response_trust_config.trust_weight,
            self.response_trust_config.eps,
            (
                "not_applicable"
                if tangent_config.subspace_dimension == 2
                else tangent_forward_source
            ),
            tangent_config.subspace_dimension,
            tangent_forward_source,
            self.response_trust_config.enabled,
            self.response_trust_config.enabled,
            tangent_config.subspace_dimension == 1
            and self.response_trust_config.enabled,
            self.response_trust_config.enabled
            and self.post_line_search_stationarity_config.enabled,
            self.response_trust_config.enabled
            and self.post_line_search_stationarity_config.enabled,
        )

    def evaluate(
        self,
        dataset: ComplexCouplingDataset,
        *,
        dataset_name: str,
        batch_size: int,
    ) -> dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        rows: list[dict[str, float | int | str]] = []
        totals: dict[str, float] = {}
        sample_count = 0
        with torch.no_grad():
            for batch in loader:
                prediction = self.predict_batch(batch.to(self.device))
                batch_size = int(prediction.batch.rhs_valid.shape[0])
                for sample_offset, sample_index in enumerate(
                    prediction.batch.sample_indices.cpu().tolist()
                ):
                    row = self._sample_metric_row(prediction, sample_offset)
                    row["sample_id"] = int(sample_index)
                    row["file_stem"] = prediction.batch.file_stems[sample_offset]
                    rows.append(row)
                for key, value in prediction.metrics.items():
                    totals[key] = totals.get(key, 0.0) + batch_size * float(
                        value.item()
                    )
                sample_count += batch_size
        summary = {key: value / max(sample_count, 1) for key, value in totals.items()}
        self._write_outputs(dataset_name, rows, summary)
        return summary

    def predict_batch(self, batch: ComplexCouplingBatch) -> ComplexPredictionBatch:
        projection_context = self._projection_context(batch)
        tangent_context = self._tangent_projection_context(batch)
        raw_response, fusion = self.model.forward_with_fusion_diagnostics(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
            rhs_phys=batch.rhs_valid,
        )
        projection = apply_complex_balance_projection(
            raw_response=raw_response,
            rhs_phys=batch.rhs_valid,
            geometry=batch.geometry,
            config=self.balance_projection,
            column_diagonal_context=projection_context,
            symmetric_tangent_context=tangent_context,
        )
        tangent_auxiliary = tangent_auxiliary_losses_from_projection(
            projection=projection,
            context=tangent_context,
            rhs_phys=batch.rhs_valid,
            stationarity_config=self.post_line_search_stationarity_config,
            response_trust_config=self.response_trust_config,
        )
        reconstruction = reconstruct_complex_projection(
            projection=projection,
            green_model=self.green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        cross_axis_reconstruction = self.cross_axis_reconstructor.reconstruct(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            projected_physical=projection.projected_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        objective = compute_complex_coupling_objective(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            rhs_valid=batch.rhs_valid,
            projected_physical=projection.projected_physical,
            a_valid=batch.a_valid,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
            canonical_energy_config=self.canonical_energy_config,
            relative_split_config=self.relative_split_config,
            weak_closure_config=self.weak_closure_config,
            boundary_context=self.boundary_energy_context(batch.geometry),
            post_line_search_stationarity_config=(
                self.post_line_search_stationarity_config
            ),
            post_line_search_stationarity=tangent_auxiliary.stationarity,
            response_trust_config=self.response_trust_config,
            response_trust=tangent_auxiliary.response_trust,
        )
        metrics = {
            key: value.detach() for key, value in objective.metric_tensors().items()
        }
        metrics.update(
            {
                key: value.detach()
                for key, value in symmetric_tangent_metric_tensors(projection).items()
            }
        )
        if torch.any(batch.has_solution):
            selected_solution = batch.has_solution
            metrics["rel_sol"] = relative_l2_valid(
                cross_axis_reconstruction.u_pred_valid[selected_solution],
                batch.sol_valid[selected_solution],
            ).detach()
            if self.cross_axis_reconstruction_config.enabled:
                metrics["rel_sol_equal_mean"] = relative_l2_valid(
                    cross_axis_reconstruction.u_equal_mean_valid[selected_solution],
                    batch.sol_valid[selected_solution],
                ).detach()
        if torch.any(batch.has_flux):
            selected = batch.has_flux
            metrics["rel_flux"] = relative_l2_valid(
                projection.projected_physical[selected],
                batch.flux_valid[selected],
            ).detach()
        return ComplexPredictionBatch(
            batch=batch,
            raw_response=raw_response,
            pre_projection_fusion=fusion,
            projection=projection,
            reconstruction=reconstruction,
            cross_axis_reconstruction=cross_axis_reconstruction,
            energy=objective.energy,
            objective=objective,
            metrics=metrics,
        )

    def _projection_context(
        self,
        batch: ComplexCouplingBatch,
    ) -> ColumnDiagonalGreenResponseContext | None:
        if self.balance_projection.mode != "column_diagonal_green_response":
            return None
        build_count_before = self._green_response_context_cache.build_count
        context = self._green_response_context_cache.get_or_build(
            green_model=self.green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        if self._green_response_context_cache.build_count != build_count_before:
            stats = context.statistics()
            self.logger.info(
                "column-diagonal Green-response context build_seconds=%.6f "
                "gain_exponent=%.6f "
                "gain_x_squared=[%.6e, %.6e] gain_y_squared=[%.6e, %.6e] "
                "weight_phi=[%.6e, %.6e] x_floored=%d y_floored=%d "
                "row_norm_used=false full_gram_solve=false",
                self._green_response_context_cache.build_seconds,
                context.gain_exponent,
                stats["gamma_x_squared_min"],
                stats["gamma_x_squared_max"],
                stats["gamma_y_squared_min"],
                stats["gamma_y_squared_max"],
                stats["weight_phi_min"],
                stats["weight_phi_max"],
                stats["x_floored_point_count"],
                stats["y_floored_point_count"],
            )
        return context

    def _tangent_projection_context(
        self,
        batch: ComplexCouplingBatch,
    ) -> SymmetricTangentGreenResponseContext | None:
        if self.balance_projection.mode != "symmetric_tangent_green_response":
            return None
        build_count_before = self._tangent_context_cache.build_count
        context = self._tangent_context_cache.get_or_build(
            green_model=self.green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        if self._tangent_context_cache.build_count != build_count_before:
            stats = context.statistics()
            self.logger.info(
                "symmetric-tangent Green-response context build_seconds=%.6f "
                "subspace_dimension=%d eta=%.6e eta_strategy=%s "
                "eta_applicability=%s line_search_relative_eps=%.6e "
                "relative_lambda=%.6e denominator_relative_eps=%.6e "
                "gain_scale=%.6e denominator=[%.6e, %.6e] "
                "x_blocks=%d y_blocks=%d row_norm_used=false "
                "global_matrix_materialized=false full_gram_solve=false",
                self._tangent_context_cache.build_seconds,
                context.subspace_dimension,
                context.eta,
                context.eta_strategy,
                stats["eta_applicability"],
                context.line_search_relative_eps,
                context.relative_lambda,
                context.denominator_relative_eps,
                stats["gain_scale"],
                stats["denominator_min"],
                stats["denominator_max"],
                stats["x_segment_block_count"],
                stats["y_segment_block_count"],
            )
        return context

    @property
    def column_diagonal_green_response_context(
        self,
    ) -> ColumnDiagonalGreenResponseContext | None:
        return self._green_response_context_cache.context

    @property
    def column_diagonal_green_response_context_build_count(self) -> int:
        return self._green_response_context_cache.build_count

    @property
    def column_diagonal_green_response_context_build_seconds(self) -> float:
        return self._green_response_context_cache.build_seconds

    @property
    def symmetric_tangent_green_response_context(
        self,
    ) -> SymmetricTangentGreenResponseContext | None:
        return self._tangent_context_cache.context

    @property
    def symmetric_tangent_green_response_context_build_count(self) -> int:
        return self._tangent_context_cache.build_count

    @property
    def symmetric_tangent_green_response_context_build_seconds(self) -> float:
        return self._tangent_context_cache.build_seconds

    def _sample_metric_row(
        self,
        prediction: ComplexPredictionBatch,
        sample_offset: int,
    ) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            key: float(value.item())
            for key, value in prediction.objective.sample_metric_tensors(
                sample_offset
            ).items()
        }
        tangent = prediction.projection.symmetric_tangent_diagnostics
        if tangent is not None:
            mismatch_pre = tangent.mismatch_pre[sample_offset]
            mismatch_post = tangent.mismatch_post[sample_offset]
            gradient = tangent.gradient[sample_offset]
            delta = tangent.delta[sample_offset]
            eps = torch.finfo(mismatch_pre.dtype).eps
            pre_rms = mismatch_pre.square().mean().sqrt()
            post_rms = mismatch_post.square().mean().sqrt()
            correction_pair_norm = torch.linalg.vector_norm(
                torch.stack((delta, -delta), dim=0)
            )
            symmetric_pair_norm = torch.linalg.vector_norm(
                tangent.symmetric_physical[sample_offset]
            )
            row.update(
                {
                    "tangent_response_mismatch_pre": float(pre_rms.item()),
                    "tangent_response_mismatch_post": float(post_rms.item()),
                    "tangent_response_mismatch_ratio": float(
                        (post_rms / pre_rms.clamp_min(eps)).item()
                    ),
                    "tangent_gradient_rms": float(
                        gradient.square().mean().sqrt().item()
                    ),
                    "tangent_delta_rms": float(delta.square().mean().sqrt().item()),
                    "tangent_delta_max_abs": float(delta.abs().max().item()),
                    "tangent_correction_rel_symmetric_pair": float(
                        (
                            correction_pair_norm / symmetric_pair_norm.clamp_min(eps)
                        ).item()
                    ),
                }
            )
            if tangent.eta_star is not None:
                if (
                    tangent.eta_applied is None
                    or tangent.eta_cap is None
                    or tangent.eta_capped is None
                    or tangent.line_search_numerator is None
                    or tangent.line_search_denominator is None
                ):
                    raise RuntimeError("Adaptive tangent diagnostics are incomplete.")
                row.update(
                    {
                        "tangent_eta_cap": tangent.eta_cap,
                        "tangent_eta_star": float(
                            tangent.eta_star[sample_offset].item()
                        ),
                        "tangent_eta_applied": float(
                            tangent.eta_applied[sample_offset].item()
                        ),
                        "tangent_eta_capped": int(
                            tangent.eta_capped[sample_offset].item()
                        ),
                        "tangent_line_search_numerator": float(
                            tangent.line_search_numerator[sample_offset].item()
                        ),
                        "tangent_line_search_denominator": float(
                            tangent.line_search_denominator[sample_offset].item()
                        ),
                    }
                )
            if tangent.subspace_dimension == 2:
                if (
                    tangent.coefficient_0 is None
                    or tangent.coefficient_1 is None
                    or tangent.second_direction_active is None
                    or tangent.cost_k1 is None
                    or tangent.cost_k2 is None
                ):
                    raise RuntimeError("K=2 tangent diagnostics are incomplete.")
                cost_k1 = tangent.cost_k1[sample_offset]
                cost_k2 = tangent.cost_k2[sample_offset]
                row.update(
                    {
                        "tangent_subspace_dimension": 2,
                        "tangent_coefficient_0": float(
                            tangent.coefficient_0[sample_offset].item()
                        ),
                        "tangent_coefficient_1": float(
                            tangent.coefficient_1[sample_offset].item()
                        ),
                        "tangent_second_direction_active": int(
                            tangent.second_direction_active[sample_offset].item()
                        ),
                        "tangent_response_cost_k1": float(cost_k1.item()),
                        "tangent_response_cost_k2": float(cost_k2.item()),
                        "tangent_response_cost_k2_over_k1": float(
                            (cost_k2 / cost_k1.clamp_min(eps)).item()
                        ),
                    }
                )
        if bool(prediction.batch.has_solution[sample_offset].item()):
            row["rel_sol"] = float(
                relative_l2_valid(
                    prediction.cross_axis_reconstruction.u_pred_valid[
                        sample_offset : sample_offset + 1
                    ],
                    prediction.batch.sol_valid[sample_offset : sample_offset + 1],
                ).item()
            )
            if self.cross_axis_reconstruction_config.enabled:
                row["rel_sol_equal_mean"] = float(
                    relative_l2_valid(
                        prediction.cross_axis_reconstruction.u_equal_mean_valid[
                            sample_offset : sample_offset + 1
                        ],
                        prediction.batch.sol_valid[sample_offset : sample_offset + 1],
                    ).item()
                )
        reliability = prediction.cross_axis_reconstruction.reliability
        if reliability is not None:
            weight = reliability.w_phi[sample_offset]
            row["weak_weight_phi_mean"] = float(weight.mean().item())
            row["weak_weight_phi_min"] = float(weight.min().item())
            row["weak_weight_phi_max"] = float(weight.max().item())
            row["weak_support_fraction"] = float(
                reliability.support_mask[sample_offset]
                .to(dtype=torch.float64)
                .mean()
                .item()
            )
        if bool(prediction.batch.has_flux[sample_offset].item()):
            row["rel_flux"] = float(
                relative_l2_valid(
                    prediction.projection.projected_physical[
                        sample_offset : sample_offset + 1
                    ],
                    prediction.batch.flux_valid[sample_offset : sample_offset + 1],
                ).item()
            )
        return row

    def boundary_energy_context(
        self,
        geometry: ComplexGeometryMetadata,
    ) -> ComplexBoundaryEnergyContext:
        if self._boundary_context is None:
            self._boundary_context = build_boundary_energy_context(geometry)
        return self._boundary_context

    def _write_outputs(
        self,
        dataset_name: str,
        rows: list[dict[str, float | int | str]],
        summary: dict[str, float],
    ) -> None:
        metrics_dir = self.work_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / f"{dataset_name}_metrics.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        csv_path = metrics_dir / f"{dataset_name}_per_sample_metrics.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
