from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_axial_response_operator import (
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    build_boundary_energy_context,
    canonical_complex_energy_loss,
)
from greenonet.complex_projection_response_audit import (
    ComplexProjectionResponseAudit,
    ProjectionTransitionEdges,
)
from greenonet.complex_reconstruction import reconstruct_from_projected_response
from greenonet.complex_symmetric_tangent_audit import (
    ClosedLoopTangentBatchDiagnostics,
    SymmetricTangentMetricMixin,
    SymmetricTangentPlotMixin,
    TangentBatchEvaluation,
    TangentMethod,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    SymmetricTangentGreenResponseContext,
    SymmetricTangentGreenResponseContextCache,
    matrix_free_krylov_k2_step as matrix_free_krylov_k2_step,
    matrix_free_krylov_subspace_step,
)
from greenonet.complex_tangent_context_io import resolve_tangent_context_path
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    SymmetricTangentGreenResponseProjectionConfig,
    validate_complex_tangent_context_checkpoint_config,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class TangentSubspaceAuditRequest:
    """Inputs for the frozen matrix-free tangent subspace comparison."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    batch_size: int = 10
    selected_samples: tuple[int, ...] | None = None
    transition_log_threshold: float = math.log(2.0)
    subspace_relative_eps: float = 1.0e-12
    metric_eps: float = 1.0e-30
    operator_equivalence_tol: float = 1.0e-10
    monotonicity_relative_tol: float = 1.0e-10
    max_subspace_dimension: int = 2
    save_generated_data: bool = True
    tangent_context: Path | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if (
            isinstance(self.max_subspace_dimension, bool)
            or not isinstance(self.max_subspace_dimension, int)
            or self.max_subspace_dimension < 2
        ):
            raise ValueError(
                "max_subspace_dimension must be an integer greater than or equal to 2."
            )
        for name, value, allow_zero in (
            ("transition_log_threshold", self.transition_log_threshold, True),
            ("subspace_relative_eps", self.subspace_relative_eps, False),
            ("metric_eps", self.metric_eps, False),
            ("operator_equivalence_tol", self.operator_equivalence_tol, False),
            ("monotonicity_relative_tol", self.monotonicity_relative_tol, False),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or (not allow_zero and float(value) == 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        if self.selected_samples is not None:
            if len(set(self.selected_samples)) != len(self.selected_samples):
                raise ValueError("selected_samples must not contain duplicates.")
            if any(sample_id < 0 for sample_id in self.selected_samples):
                raise ValueError("selected_samples must be non-negative.")
        if self.tangent_context is not None and not isinstance(
            self.tangent_context,
            Path,
        ):
            raise TypeError("tangent_context must be a pathlib.Path or None.")


@dataclass(frozen=True)
class PreparedTangentBatch:
    """Variant-independent network output and symmetric response for one batch."""

    raw_physical: torch.Tensor
    symmetric_physical: torch.Tensor
    mismatch: torch.Tensor
    gradient: torch.Tensor


KrylovSubspaceAuditResult = KrylovSubspaceStepResult


def matrix_free_krylov_subspace_audit(
    *,
    context: SymmetricTangentGreenResponseContext,
    mismatch: torch.Tensor,
    gradient: torch.Tensor,
    max_dimension: int,
    relative_eps: float,
    monotonicity_relative_tol: float,
) -> KrylovSubspaceStepResult:
    """Reuse the production K=2 through K=4 matrix-free subspace helper."""

    return matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=max_dimension,
        relative_eps=relative_eps,
        monotonicity_relative_tol=monotonicity_relative_tol,
    )


class ComplexTangentSubspaceAudit(
    SymmetricTangentMetricMixin,
    SymmetricTangentPlotMixin,
):
    """Compare frozen K=1 candidates with nested matrix-free K=2 through K=4."""

    def __init__(
        self,
        request: TangentSubspaceAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.methods = (
            TangentMethod("symmetric", "symmetric", "symmetric"),
            TangentMethod("k1_production", "K=1 configured cap", "k1_capped"),
            TangentMethod("k1_uncapped", "K=1 uncapped", "k1_uncapped"),
            *tuple(
                TangentMethod(
                    f"k{dimension}_unconstrained",
                    f"K={dimension} unconstrained",
                    f"k{dimension}_unconstrained",
                )
                for dimension in range(2, request.max_subspace_dimension + 1)
            ),
        )
        self.geometry: ComplexGeometryMetadata
        self.response_operator: FrozenBidirectionalResponseOperator
        self.tangent_context: SymmetricTangentGreenResponseContext
        self.boundary_context: ComplexBoundaryEnergyContext
        self._configs: CouplingArtifactConfigs
        self._coupling_model: ComplexCouplingNet
        self._green_model: torch.nn.Module
        self._cross_axis_reconstructor: ComplexCrossAxisReconstructor
        self._device: torch.device
        self._context_build_count = 0
        self._operator_equivalence_max_abs = math.nan
        self._tangent_context_cache: (
            SymmetricTangentGreenResponseContextCache | None
        ) = None

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(self.request.config)
        if self._configs.dataset.geometry_mode != "complex":
            raise ValueError("Tangent subspace audit requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        if projection.mode != "symmetric_tangent_green_response":
            raise ValueError(
                "Tangent subspace audit requires the frozen checkpoint config to use "
                "balance_projection.mode='symmetric_tangent_green_response'."
            )
        tangent_config = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        if tangent_config.eta_strategy != "closed_loop_exact_line_search":
            raise ValueError(
                "Tangent subspace audit requires closed_loop_exact_line_search."
            )
        geometry_path = self.request.geometry or self._configs.dataset.geometry_path
        test_path = self.request.test_path or self._configs.dataset.test_path
        coefficient_path = (
            self.request.coefficients
            or self._configs.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")
        for checkpoint in (
            self.request.coupling_checkpoint,
            self.request.green_checkpoint,
        ):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

        self._device = torch.device(
            self.request.device or self._configs.coupling_training.device
        )
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=self._configs.dataset.dtype,
        )
        coefficients = load_coefficient_functions(coefficient_path)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            coefficients,
            branch_input_dim=self._configs.coupling_model.branch_input_dim,
            dtype=self._configs.dataset.dtype,
            coefficient_terms=self._configs.coupling_model.coefficient_terms,
            integration_rule=self._configs.coupling_training.integration_rule,
        )
        if len(dataset) == 0:
            raise ValueError("The test dataset is empty.")
        self._load_models()
        self._cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            self._configs.coupling_model.cross_axis_reconstruction
        )
        self.boundary_context = build_boundary_energy_context(self.geometry)
        edges = ComplexProjectionResponseAudit.build_transition_edges(
            self.geometry,
            threshold=self.request.transition_log_threshold,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        rows: list[dict[str, float | int | str]] = []
        dataset_offset_by_sample: dict[int, int] = {}
        offset = 0
        for batch in loader:
            batch = batch.to(self._device)
            self._initialize_context(batch)
            evaluation, krylov = self._evaluate_batch(batch)
            rows.extend(self._metric_rows(batch, evaluation, krylov, edges))
            for sample_id in batch.sample_indices.tolist():
                dataset_offset_by_sample[int(sample_id)] = offset
                offset += 1

        aggregate = self._aggregate_rows(rows)
        paired = self._paired_comparisons(rows)
        selected, roles = self._select_samples(rows, dataset_offset_by_sample)
        suffix = self._subspace_suffix
        metric_path = self.request.outdir / "metrics" / f"per_sample_{suffix}.csv"
        self._write_csv(metric_path, rows)
        figure_paths = [
            self._write_aggregate_figure(aggregate),
            self._write_paired_figure(rows),
        ]

        selected_batch = complex_coupling_collate_fn(
            [dataset[dataset_offset_by_sample[sample_id]] for sample_id in selected]
        ).to(self._device)
        selected_evaluation, selected_krylov = self._evaluate_batch(selected_batch)
        all_method_indices = tuple(range(len(self.methods)))
        for sample_offset in range(len(selected)):
            figure_paths.append(
                self.write_selected_figure(
                    geometry=self.geometry,
                    batch=selected_batch,
                    evaluation=selected_evaluation,
                    sample_offset=sample_offset,
                    method_indices=all_method_indices,
                    request=self.request,  # type: ignore[arg-type]
                    logger=self.logger,
                )
            )
        if self.request.save_generated_data:
            self._write_selected_arrays(
                selected_batch,
                selected_evaluation,
                selected_krylov,
            )

        summary = self._build_summary(
            dataset=dataset,
            geometry_path=geometry_path,
            test_path=test_path,
            coefficient_path=coefficient_path,
            aggregate=aggregate,
            paired=paired,
            selected=selected,
            roles=roles,
            edges=edges,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "K=1 through K=%d tangent subspace audit complete: samples=%d",
                self.request.max_subspace_dimension,
                len(dataset),
            )
        return summary

    @property
    def _subspace_suffix(self) -> str:
        return f"k1_k{self.request.max_subspace_dimension}"

    def _load_models(self) -> None:
        loader_request = CouplingArtifactRequest(
            config=self.request.config,
            coupling_checkpoint=self.request.coupling_checkpoint,
            green_checkpoint=self.request.green_checkpoint,
            outdir=self.request.outdir,
            coefficients=self.request.coefficients,
            device=str(self._device),
            theme=self.request.theme,
        )
        loader = ComplexCouplingArtifactExporter(loader_request, logger=self.logger)
        self._coupling_model = loader._load_complex_model(self._configs, self._device)
        self._green_model = loader._load_green_model(self._configs, self._device)
        for model in (self._coupling_model, self._green_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _initialize_context(self, batch: ComplexCouplingBatch) -> None:
        if hasattr(self, "response_operator"):
            return
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        diagnostic_tangent = replace(tangent, subspace_dimension=1)
        checkpoint = validate_complex_tangent_context_checkpoint_config(
            training=self._configs.coupling_training,
            balance_projection=projection,
        )
        checkpoint_path = resolve_tangent_context_path(
            checkpoint=checkpoint,
            cli_override=self.request.tangent_context,
            default_path=(
                self.request.coupling_checkpoint.parent
                / "tangent_response_context.safetensors"
            ),
        )
        self._tangent_context_cache = SymmetricTangentGreenResponseContextCache(
            diagnostic_tangent,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )
        self.tangent_context = self._tangent_context_cache.get_or_build(
            green_model=self._green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        self.response_operator = self.tangent_context.response_operator
        self._context_build_count = self._tangent_context_cache.build_count
        self._verify_operator_equivalence(batch)

    def _verify_operator_equivalence(self, batch: ComplexCouplingBatch) -> None:
        physical = torch.stack((0.5 * batch.rhs_valid, 0.5 * batch.rhs_valid), dim=1)
        sigma_x = (
            batch.geometry.x_lengths_for_valid_points()
            .to(device=self._device, dtype=physical.dtype)
            .square()
        )
        sigma_y = (
            batch.geometry.y_lengths_for_valid_points()
            .to(device=self._device, dtype=physical.dtype)
            .square()
        )
        response = torch.stack(
            (
                sigma_x.unsqueeze(0) * physical[:, 0],
                sigma_y.unsqueeze(0) * physical[:, 1],
            ),
            dim=1,
        )
        production = reconstruct_from_projected_response(
            green_model=self._green_model,
            geometry=batch.geometry,
            projected_response=response,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        production_pair = torch.stack(
            (production.u_phi_valid, production.u_psi_valid),
            dim=1,
        )
        operator_pair = self.response_operator.forward_pair(physical)
        maximum = float((production_pair - operator_pair).abs().max().item())
        self._operator_equivalence_max_abs = maximum
        if maximum > self.request.operator_equivalence_tol:
            raise RuntimeError(
                "Segment response operator does not match production reconstruction: "
                f"max_abs={maximum:.6e}."
            )

    @torch.no_grad()
    def _prepare_batch(
        self,
        batch: ComplexCouplingBatch,
    ) -> PreparedTangentBatch:
        raw_response, _fusion = self._coupling_model.forward_with_fusion_diagnostics(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
            rhs_phys=batch.rhs_valid,
        )
        sigma_x = (
            batch.geometry.x_lengths_for_valid_points()
            .to(device=self._device, dtype=raw_response.dtype)
            .square()
        )
        sigma_y = (
            batch.geometry.y_lengths_for_valid_points()
            .to(device=self._device, dtype=raw_response.dtype)
            .square()
        )
        raw_physical = torch.stack(
            (raw_response[:, 0] / sigma_x, raw_response[:, 1] / sigma_y),
            dim=1,
        )
        raw_difference = raw_physical[:, 0] - raw_physical[:, 1]
        symmetric = torch.stack(
            (
                0.5 * (batch.rhs_valid + raw_difference),
                0.5 * (batch.rhs_valid - raw_difference),
            ),
            dim=1,
        )
        symmetric_solution = self.response_operator.forward_pair(symmetric)
        mismatch = symmetric_solution[:, 0] - symmetric_solution[:, 1]
        gradient = self.tangent_context.tangent_gradient(mismatch)
        return PreparedTangentBatch(
            raw_physical=raw_physical,
            symmetric_physical=symmetric,
            mismatch=mismatch,
            gradient=gradient,
        )

    @torch.no_grad()
    def _evaluate_prepared_batch(
        self,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
        *,
        context: SymmetricTangentGreenResponseContext,
    ) -> tuple[TangentBatchEvaluation, KrylovSubspaceAuditResult]:
        production_step = context.tangent_step(
            mismatch=prepared.mismatch,
            gradient=prepared.gradient,
            eta_cap=context.eta,
        )
        krylov = matrix_free_krylov_subspace_audit(
            context=context,
            mismatch=prepared.mismatch,
            gradient=prepared.gradient,
            max_dimension=self.request.max_subspace_dimension,
            relative_eps=self.request.subspace_relative_eps,
            monotonicity_relative_tol=self.request.monotonicity_relative_tol,
        )
        if (
            production_step.eta_star is None
            or production_step.eta_applied is None
            or production_step.eta_capped is None
            or production_step.line_search_numerator is None
            or production_step.line_search_denominator is None
        ):
            raise RuntimeError(
                "The configured production tangent step did not return "
                "closed-loop line-search diagnostics."
            )
        tangent_delta = torch.stack(
            (
                torch.zeros_like(prepared.gradient),
                production_step.delta,
                *tuple(
                    krylov.deltas[dimension]
                    for dimension in range(krylov.deltas.shape[0])
                ),
            ),
            dim=0,
        )
        symmetric = prepared.symmetric_physical
        candidate_physical = torch.stack(
            (
                symmetric.unsqueeze(0)[:, :, 0] + tangent_delta,
                symmetric.unsqueeze(0)[:, :, 1] - tangent_delta,
            ),
            dim=2,
        )
        method_count, batch_count, _axis, point_count = candidate_physical.shape
        flat_physical = candidate_physical.reshape(
            method_count * batch_count,
            2,
            point_count,
        )
        flat_solution = self.response_operator.forward_pair(flat_physical)
        candidate_solution = flat_solution.reshape(
            method_count,
            batch_count,
            2,
            point_count,
        )
        flat_a = batch.a_valid.repeat(method_count, 1)
        energy = canonical_complex_energy_loss(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            a_valid=flat_a,
            geometry=batch.geometry,
            boundary_context=self.boundary_context,
        )
        cross_axis = self._cross_axis_reconstructor.reconstruct(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            projected_physical=flat_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        configured = torch.stack(
            (
                symmetric[:, 0] + production_step.delta,
                symmetric[:, 1] - production_step.delta,
            ),
            dim=1,
        )
        return (
            TangentBatchEvaluation(
                methods=self.methods,
                raw_physical=prepared.raw_physical,
                symmetric_physical=symmetric,
                configured_physical=configured,
                tangent_gradient=prepared.gradient,
                tangent_preconditioner_base=context.preconditioner_base,
                tangent_delta=tangent_delta,
                candidate_physical=candidate_physical,
                candidate_solution=candidate_solution,
                candidate_equal_prediction=cross_axis.u_equal_mean_valid.reshape(
                    method_count,
                    batch_count,
                    point_count,
                ),
                candidate_prediction=cross_axis.u_pred_valid.reshape(
                    method_count,
                    batch_count,
                    point_count,
                ),
                canonical_energy=energy.total_per_sample.reshape(
                    method_count,
                    batch_count,
                ),
                canonical_bulk_energy=energy.bulk_per_sample.reshape(
                    method_count,
                    batch_count,
                ),
                canonical_boundary_energy=energy.boundary_per_sample.reshape(
                    method_count,
                    batch_count,
                ),
                closed_loop=(
                    ClosedLoopTangentBatchDiagnostics(
                        method_id="k1_production",
                        eta_cap=context.eta,
                        eta_star=production_step.eta_star,
                        eta_applied=production_step.eta_applied,
                        eta_capped=production_step.eta_capped,
                        line_search_numerator=(production_step.line_search_numerator),
                        line_search_denominator=(
                            production_step.line_search_denominator
                        ),
                    ),
                    ClosedLoopTangentBatchDiagnostics(
                        method_id="k1_uncapped",
                        eta_cap=None,
                        eta_star=krylov.coefficients[0],
                        eta_applied=krylov.coefficients[0],
                        eta_capped=torch.zeros_like(
                            krylov.coefficients[0],
                            dtype=torch.bool,
                        ),
                        line_search_numerator=krylov.line_search_numerator_0,
                        line_search_denominator=krylov.line_search_denominator_0,
                    ),
                ),
            ),
            krylov,
        )

    @torch.no_grad()
    def _evaluate_batch(
        self,
        batch: ComplexCouplingBatch,
    ) -> tuple[TangentBatchEvaluation, KrylovSubspaceAuditResult]:
        prepared = self._prepare_batch(batch)
        return self._evaluate_prepared_batch(
            batch,
            prepared,
            context=self.tangent_context,
        )

    def _metric_rows(
        self,
        batch: ComplexCouplingBatch,
        evaluation: TangentBatchEvaluation,
        krylov: KrylovSubspaceAuditResult,
        edges: ProjectionTransitionEdges,
        *,
        context: SymmetricTangentGreenResponseContext | None = None,
    ) -> list[dict[str, float | int | str]]:
        active_context = self.tangent_context if context is None else context
        rows = self.build_metric_rows(
            batch=batch,
            evaluation=evaluation,
            edges=edges,
            point_mass=active_context.point_mass,
            eps=self.request.metric_eps,
        )
        batch_count = batch.rhs_valid.shape[0]
        canonical = ComplexCanonicalEnergyConfig.from_raw(
            self._configs.coupling_training.canonical_energy
        )
        optimized = (
            evaluation.canonical_bulk_energy
            + float(canonical.boundary_weight) * evaluation.canonical_boundary_energy
        )
        optimized_baseline = optimized[0].clamp_min(self.request.metric_eps)
        response_mismatch = (
            evaluation.candidate_solution[:, :, 0]
            - evaluation.candidate_solution[:, :, 1]
        )
        for row_index, row in enumerate(rows):
            method_index = row_index // batch_count
            sample_offset = row_index % batch_count
            row["loss_energy_optimized"] = float(
                optimized[method_index, sample_offset].item()
            )
            row["loss_energy_optimized_ratio_vs_symmetric"] = float(
                (
                    optimized[method_index, sample_offset]
                    / optimized_baseline[sample_offset]
                ).item()
            )
            row["response_mismatch_rms"] = float(
                response_mismatch[method_index, sample_offset]
                .square()
                .mean()
                .sqrt()
                .item()
            )
            if bool(batch.has_solution[sample_offset].item()):
                target = batch.sol_valid[sample_offset].unsqueeze(0)
                row["rel_u_phi"] = float(
                    self._relative_l2(
                        evaluation.candidate_solution[
                            method_index,
                            sample_offset,
                            0,
                        ].unsqueeze(0),
                        target,
                        eps=self.request.metric_eps,
                    ).item()
                )
                row["rel_u_psi"] = float(
                    self._relative_l2(
                        evaluation.candidate_solution[
                            method_index,
                            sample_offset,
                            1,
                        ].unsqueeze(0),
                        target,
                        eps=self.request.metric_eps,
                    ).item()
                )
            row["k1_eta_star"] = float(krylov.coefficients[0, sample_offset].item())
            for direction_index in range(krylov.coefficients.shape[0]):
                dimension = direction_index + 1
                row[f"k{dimension}_coefficient_{direction_index}"] = float(
                    krylov.coefficients[direction_index, sample_offset].item()
                )
                row[f"k{dimension}_direction_active"] = int(
                    krylov.direction_active[direction_index, sample_offset].item()
                )
                row[f"k{dimension}_response_cost"] = float(
                    krylov.costs[direction_index, sample_offset].item()
                )
                row[f"k{dimension}_response_orthogonality_max"] = float(
                    krylov.response_orthogonality_max[
                        direction_index,
                        sample_offset,
                    ].item()
                )
                if direction_index > 0:
                    previous_cost = krylov.costs[
                        direction_index - 1,
                        sample_offset,
                    ].clamp_min(torch.finfo(krylov.costs.dtype).tiny)
                    row[f"k{dimension}_response_cost_ratio_vs_k{dimension - 1}"] = (
                        float(
                            (
                                krylov.costs[direction_index, sample_offset]
                                / previous_cost
                            ).item()
                        )
                    )
            method_id = str(row["method_id"])
            if method_id.startswith("k") and method_id.endswith("_unconstrained"):
                dimension = int(method_id[1 : method_id.index("_")])
                if dimension >= 2:
                    current_cost = krylov.costs[dimension - 1, sample_offset]
                    previous_cost = krylov.costs[dimension - 2, sample_offset]
                    row["subspace_dimension"] = dimension
                    row["response_cost_nonincrease_vs_previous"] = int(
                        bool(
                            current_cost
                            <= previous_cost
                            * (1.0 + self.request.monotonicity_relative_tol)
                            + torch.finfo(current_cost.dtype).tiny
                        )
                    )
        return rows

    def _aggregate_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, dict[str, float | int | str]]:
        aggregate = self.aggregate_rows(rows, self.methods)
        tail_metrics = (
            "response_mismatch_cost",
            "canonical_energy",
            "loss_energy_optimized",
            "rel_sol",
            "rel_sol_equal_mean",
            "rel_u_phi",
            "rel_u_psi",
            "rel_flux",
            "tangent_correction_rel_symmetric_pair",
        )
        for method in self.methods:
            selected = [row for row in rows if row["method_id"] == method.method_id]
            payload = aggregate[method.method_id]
            for metric in tail_metrics:
                values = np.asarray(
                    [float(row[metric]) for row in selected if metric in row],
                    dtype=np.float64,
                )
                values = values[np.isfinite(values)]
                if values.size:
                    payload[f"{metric}_std"] = float(values.std())
                    payload[f"{metric}_p90"] = float(np.quantile(values, 0.90))
                    payload[f"{metric}_p95"] = float(np.quantile(values, 0.95))
                    payload[f"{metric}_max"] = float(values.max())
        return aggregate

    @staticmethod
    def _paired_comparisons(
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, dict[str, dict[str, float | int]]]:
        metrics = (
            "response_mismatch_cost",
            "canonical_energy",
            "loss_energy_optimized",
            "rel_sol",
            "rel_sol_equal_mean",
            "rel_u_phi",
            "rel_u_psi",
            "rel_flux",
            "tangent_correction_rel_symmetric_pair",
        )
        method_ids = {str(row["method_id"]) for row in rows}
        subspace_method_ids = sorted(
            (
                method_id
                for method_id in method_ids
                if method_id.startswith("k")
                and method_id.endswith("_unconstrained")
                and method_id != "k1_uncapped"
            ),
            key=lambda method_id: int(method_id[1 : method_id.index("_")]),
        )
        by_method = {
            method_id: {
                int(row["sample_id"]): row
                for row in rows
                if row["method_id"] == method_id
            }
            for method_id in method_ids
        }
        output: dict[str, dict[str, dict[str, float | int]]] = {}
        for candidate_id in subspace_method_ids:
            dimension = int(candidate_id[1 : candidate_id.index("_")])
            candidate_label = f"k{dimension}"
            previous_id = (
                "k1_uncapped" if dimension == 2 else f"k{dimension - 1}_unconstrained"
            )
            for baseline in dict.fromkeys(("k1_production", previous_id)):
                comparison: dict[str, dict[str, float | int]] = {}
                common = sorted(set(by_method[baseline]) & set(by_method[candidate_id]))
                for metric in metrics:
                    pairs = [
                        (
                            float(by_method[baseline][sample_id][metric]),
                            float(by_method[candidate_id][sample_id][metric]),
                        )
                        for sample_id in common
                        if metric in by_method[baseline][sample_id]
                        and metric in by_method[candidate_id][sample_id]
                    ]
                    if not pairs:
                        continue
                    baseline_values = np.asarray([pair[0] for pair in pairs])
                    candidate_values = np.asarray([pair[1] for pair in pairs])
                    delta = candidate_values - baseline_values
                    comparison[metric] = {
                        "sample_count": len(pairs),
                        "baseline_mean": float(baseline_values.mean()),
                        "candidate_mean": float(candidate_values.mean()),
                        f"{candidate_label}_mean": float(candidate_values.mean()),
                        "mean_delta": float(delta.mean()),
                        "relative_mean_change": ComplexTangentSubspaceAudit._relative_change(
                            baseline=float(baseline_values.mean()),
                            candidate=float(candidate_values.mean()),
                        ),
                        "improved_sample_count": int(np.count_nonzero(delta < 0.0)),
                        "worsened_sample_count": int(np.count_nonzero(delta > 0.0)),
                        "unchanged_sample_count": int(np.count_nonzero(delta == 0.0)),
                        "max_improvement": float(max(0.0, -float(delta.min()))),
                        "max_worsening": float(max(0.0, float(delta.max()))),
                    }
                output[f"{candidate_label}_vs_{baseline}"] = comparison
        return output

    @staticmethod
    def _relative_change(*, baseline: float, candidate: float) -> float:
        if baseline == 0.0:
            if candidate == 0.0:
                return 0.0
            return math.copysign(math.inf, candidate)
        return candidate / baseline - 1.0

    def _select_samples(
        self,
        rows: Sequence[dict[str, float | int | str]],
        dataset_offset_by_sample: dict[int, int],
    ) -> tuple[tuple[int, ...], dict[str, str]]:
        available = set(dataset_offset_by_sample)
        if self.request.selected_samples is not None:
            missing = sorted(set(self.request.selected_samples) - available)
            if missing:
                raise ValueError(f"Selected sample IDs are unavailable: {missing}.")
            return self.request.selected_samples, {
                str(sample_id): "explicit"
                for sample_id in self.request.selected_samples
            }
        k1 = {
            int(row["sample_id"]): row
            for row in rows
            if row["method_id"] == "k1_production" and "rel_sol" in row
        }
        subspace_dimensions = sorted(
            {
                int(str(row["method_id"])[1 : str(row["method_id"]).index("_")])
                for row in rows
                if str(row["method_id"]).startswith("k")
                and str(row["method_id"]).endswith("_unconstrained")
                and str(row["method_id"]) != "k1_uncapped"
            }
        )
        final_dimension = subspace_dimensions[-1]
        final_method_id = f"k{final_dimension}_unconstrained"
        final = {
            int(row["sample_id"]): row
            for row in rows
            if row["method_id"] == final_method_id and "rel_sol" in row
        }
        ordered = sorted(k1, key=lambda sample_id: float(k1[sample_id]["rel_sol"]))
        if not ordered:
            fallback = tuple(sorted(available)[:1])
            return fallback, {
                str(sample_id): "first_available" for sample_id in fallback
            }
        candidates = [
            (ordered[len(ordered) // 2], "k1_rel_sol_q50"),
            (ordered[-1], "k1_rel_sol_worst"),
        ]
        common = sorted(set(k1) & set(final))
        deltas = {
            sample_id: float(final[sample_id]["rel_sol"])
            - float(k1[sample_id]["rel_sol"])
            for sample_id in common
        }
        largest_delta_sample = max(
            deltas,
            key=lambda sample_id: deltas[sample_id],
        )
        largest_delta_role = (
            f"largest_k{final_dimension}_rel_sol_worsening"
            if deltas[largest_delta_sample] > 0.0
            else f"smallest_k{final_dimension}_rel_sol_improvement"
        )
        candidates.extend(
            (
                (
                    min(deltas, key=lambda sample_id: deltas[sample_id]),
                    f"largest_k{final_dimension}_rel_sol_improvement",
                ),
                (
                    largest_delta_sample,
                    largest_delta_role,
                ),
            )
        )
        selected: list[int] = []
        roles: dict[str, str] = {}
        for sample_id, role in candidates:
            if sample_id not in selected:
                selected.append(sample_id)
                roles[str(sample_id)] = role
        return tuple(selected), roles

    @staticmethod
    def _write_csv(
        path: Path,
        rows: Sequence[dict[str, float | int | str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_aggregate_figure(
        self,
        aggregate: dict[str, dict[str, float | int | str]],
    ) -> Path:
        metrics = (
            ("response_mismatch_cost_mean", "Response mismatch"),
            ("loss_energy_optimized_mean", "Optimized energy"),
            ("rel_sol_mean", "rel_sol"),
            ("rel_u_phi_mean", "rel_u_phi"),
            ("rel_u_psi_mean", "rel_u_psi"),
            ("rel_flux_mean", "rel_flux"),
        )
        baseline = aggregate["k1_production"]
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=[label for _, label in metrics]
        )
        labels = [method.label for method in self.methods]
        colors = ["#64748b", "#2563eb", "#0f766e", "#dc2626", "#9333ea", "#ea580c"]
        for index, (metric, _label) in enumerate(metrics):
            base_value = float(baseline[metric])
            ratios = [
                float(aggregate[method.method_id][metric]) / base_value
                for method in self.methods
            ]
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=ratios,
                    marker_color=colors[: len(self.methods)],
                    hovertemplate="%{x}<br>ratio=%{y:.6f}<extra></extra>",
                ),
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="#334155",
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
        fig.update_layout(
            template=self.request.theme,
            title=(
                "Matrix-free K=1 through "
                f"K={self.request.max_subspace_dimension} tangent correction"
            ),
            width=1500,
            height=850,
            showlegend=False,
            margin={"l": 70, "r": 40, "t": 100, "b": 120},
        )
        fig.update_xaxes(tickangle=-20)
        fig.update_yaxes(title_text="ratio to K=1 production")
        path = (
            self.request.outdir
            / "figures"
            / "aggregate"
            / f"{self._subspace_suffix}_metric_ratios"
        )
        save_plotly_figure(fig, path, self.logger)
        return path.with_suffix(".json")

    def _write_paired_figure(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> Path:
        metrics = (
            ("response_mismatch_cost", "Response mismatch"),
            ("loss_energy_optimized", "Optimized energy"),
            ("rel_sol", "rel_sol"),
            ("rel_u_phi", "rel_u_phi"),
            ("rel_u_psi", "rel_u_psi"),
            ("rel_flux", "rel_flux"),
        )
        k1 = {
            int(row["sample_id"]): row
            for row in rows
            if row["method_id"] == "k1_production"
        }
        final_method_id = f"k{self.request.max_subspace_dimension}_unconstrained"
        final = {
            int(row["sample_id"]): row
            for row in rows
            if row["method_id"] == final_method_id
        }
        common = sorted(set(k1) & set(final))
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=[label for _, label in metrics]
        )
        for index, (metric, _label) in enumerate(metrics):
            x = np.asarray([float(k1[sample_id][metric]) for sample_id in common])
            y = np.asarray([float(final[sample_id][metric]) for sample_id in common])
            lower = float(min(x.min(), y.min()))
            upper = float(max(x.max(), y.max()))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    customdata=np.asarray(common),
                    mode="markers",
                    marker={"size": 7, "color": "#0f766e", "opacity": 0.75},
                    hovertemplate=(
                        "sample=%{customdata}<br>K1=%{x:.6e}<br>"
                        f"K{self.request.max_subspace_dimension}=%{{y:.6e}}"
                        "<extra></extra>"
                    ),
                ),
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[lower, upper],
                    y=[lower, upper],
                    mode="lines",
                    line={"dash": "dot", "color": "#64748b"},
                    hoverinfo="skip",
                ),
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
        fig.update_layout(
            template=self.request.theme,
            title=(
                f"Per-sample K={self.request.max_subspace_dimension} "
                "versus configured-cap K=1"
            ),
            width=1450,
            height=850,
            showlegend=False,
        )
        fig.update_xaxes(title_text="K=1 production")
        fig.update_yaxes(
            title_text=f"K={self.request.max_subspace_dimension} unconstrained"
        )
        path = (
            self.request.outdir
            / "figures"
            / "aggregate"
            / f"{self._subspace_suffix}_paired"
        )
        save_plotly_figure(fig, path, self.logger)
        return path.with_suffix(".json")

    def _write_selected_arrays(
        self,
        batch: ComplexCouplingBatch,
        evaluation: TangentBatchEvaluation,
        krylov: KrylovSubspaceAuditResult,
    ) -> None:
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "coords_valid": self.geometry.coords_valid.detach().cpu().numpy(),
            "selected_sample_ids": batch.sample_indices.detach().cpu().numpy(),
            "selected_file_stems": np.asarray(batch.file_stems),
            "method_ids": np.asarray([method.method_id for method in self.methods]),
            "subspace_dimensions": np.arange(
                1,
                self.request.max_subspace_dimension + 1,
                dtype=np.int64,
            ),
            "rhs": batch.rhs_valid.detach().cpu().numpy(),
            "sol": batch.sol_valid.detach().cpu().numpy(),
            "has_solution": batch.has_solution.detach().cpu().numpy(),
            "flux_target": batch.flux_valid.detach().cpu().numpy(),
            "has_flux": batch.has_flux.detach().cpu().numpy(),
            "raw_physical": evaluation.raw_physical.detach().cpu().numpy(),
            "symmetric_physical": evaluation.symmetric_physical.detach().cpu().numpy(),
            "tangent_gradient": evaluation.tangent_gradient.detach().cpu().numpy(),
            "tangent_denominator": self.tangent_context.denominator.detach()
            .cpu()
            .numpy(),
            "directions": krylov.directions.detach().cpu().numpy(),
            "directional_responses": krylov.directional_responses.detach()
            .cpu()
            .numpy(),
            "response_directions": krylov.response_directions.detach().cpu().numpy(),
            "coefficients": krylov.coefficients.detach().cpu().numpy(),
            "direction_active": krylov.direction_active.detach().cpu().numpy(),
            "subspace_deltas": krylov.deltas.detach().cpu().numpy(),
            "subspace_mismatches": krylov.mismatches.detach().cpu().numpy(),
            "subspace_costs": krylov.costs.detach().cpu().numpy(),
            "response_gram": krylov.response_gram.detach().cpu().numpy(),
            "response_orthogonality_max": (
                krylov.response_orthogonality_max.detach().cpu().numpy()
            ),
            "residual_gradient_post": (
                krylov.residual_gradient_post.detach().cpu().numpy()
            ),
            "line_search_numerator_0": (
                krylov.line_search_numerator_0.detach().cpu().numpy()
            ),
            "line_search_denominator_0": (
                krylov.line_search_denominator_0.detach().cpu().numpy()
            ),
            "tangent_delta": evaluation.tangent_delta.detach().cpu().numpy(),
            "candidate_physical": evaluation.candidate_physical.detach().cpu().numpy(),
            "candidate_solution": evaluation.candidate_solution.detach().cpu().numpy(),
            "candidate_prediction": evaluation.candidate_prediction.detach()
            .cpu()
            .numpy(),
            # Keep the original K=2 field names for existing audit consumers.
            "direction_0": krylov.directions[0].detach().cpu().numpy(),
            "direction_1": krylov.directions[1].detach().cpu().numpy(),
            "response_direction_0": (
                krylov.response_directions[0].detach().cpu().numpy()
            ),
            "response_direction_1": (
                krylov.response_directions[1].detach().cpu().numpy()
            ),
            "coefficient_0": krylov.coefficients[0].detach().cpu().numpy(),
            "coefficient_1": krylov.coefficients[1].detach().cpu().numpy(),
            "second_direction_active": (
                krylov.direction_active[1].detach().cpu().numpy()
            ),
        }
        np.savez_compressed(
            data_dir / f"selected_{self._subspace_suffix}_tangent_subspace.npz",
            **payload,
        )

    def _build_summary(
        self,
        *,
        dataset: ComplexCouplingDataset,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path,
        aggregate: dict[str, dict[str, float | int | str]],
        paired: dict[str, dict[str, dict[str, float | int]]],
        selected: tuple[int, ...],
        roles: dict[str, str],
        edges: ProjectionTransitionEdges,
        figure_paths: Sequence[Path],
    ) -> dict[str, Any]:
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        canonical = ComplexCanonicalEnergyConfig.from_raw(
            self._configs.coupling_training.canonical_energy
        )
        adjacent_monotonicity: dict[str, dict[str, int]] = {}
        for dimension in range(2, self.request.max_subspace_dimension + 1):
            baseline = (
                "k1_uncapped" if dimension == 2 else f"k{dimension - 1}_unconstrained"
            )
            comparison = paired[f"k{dimension}_vs_{baseline}"]["response_mismatch_cost"]
            adjacent_monotonicity[f"k{dimension}_vs_k{dimension - 1}"] = {
                "improved_sample_count": int(comparison["improved_sample_count"]),
                "worsened_sample_count": int(comparison["worsened_sample_count"]),
                "unchanged_sample_count": int(comparison["unchanged_sample_count"]),
            }
        return {
            "diagnostic": (
                f"matrix_free_{self._subspace_suffix}_tangent_subspace_posthoc_audit"
            ),
            "status": "frozen_checkpoint_posthoc",
            "production_code_changed": False,
            "training_or_checkpoint_updated": False,
            "config": str(self.request.config),
            "tangent_subspace_dimension_provenance": self._configs.raw.get(
                "tangent_subspace_dimension_provenance"
            ),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": str(coefficient_path),
            "sample_count": len(dataset),
            "configured_eta_cap": tangent.eta,
            "configured_relative_lambda": tangent.relative_lambda,
            "configured_preconditioner_variant": tangent.preconditioner_variant,
            "frozen_training_subspace_dimension": tangent.subspace_dimension,
            "maximum_audited_subspace_dimension": (self.request.max_subspace_dimension),
            "subspace_relative_eps": self.request.subspace_relative_eps,
            "canonical_boundary_weight": canonical.boundary_weight,
            "methods": [
                {
                    "method_id": method.method_id,
                    "label": method.label,
                    "kind": method.kind,
                }
                for method in self.methods
            ],
            "formula": {
                "objective": "J(delta)=||m0+(H_x+H_y)delta||_M^2",
                "k1": "z0=D^-1*g; delta1=-c0*z0",
                "residual": "r_k=S^T*M*(m0+S*delta_k); A*v=S^T*M*S*v",
                "next_direction": (
                    "z_k=response_modified_gram_schmidt(D^-1*r_k, z_0,...,z_{k-1})"
                ),
                "coefficient": "c_k=<m_k,S*z_k>_M/(<S*z_k,S*z_k>_M+eps_k)",
                "update": "delta_{k+1}=delta_k-c_k*z_k",
                "balance": "phi=p_tilde+delta; psi=q_tilde-delta",
            },
            "matrix_policy": {
                "global_matrix_materialized": False,
                "global_matrix_solve": False,
                "subspace_dimension": self.request.max_subspace_dimension,
                "sample_local_dense_solve_dimension": 0,
                "sample_local_scalar_line_search_coefficients": (
                    self.request.max_subspace_dimension
                ),
                "orthogonalization": "two_pass_modified_gram_schmidt_in_response_space",
                "segment_local_forward_adjoint_actions": True,
                "response_context_build_count": self._context_build_count,
                "operator_production_equivalence_max_abs": (
                    self._operator_equivalence_max_abs
                ),
            },
            "tangent_context": (
                {}
                if self._tangent_context_cache is None
                else self._tangent_context_cache.telemetry()
            ),
            "reference_policy": {
                "sol_and_flux_used_for_correction": False,
                "sol_and_flux_used_for_evaluation_only": True,
            },
            "response_objective_monotonicity": adjacent_monotonicity,
            "transition_definition": {
                "log_threshold": self.request.transition_log_threshold,
                "phi_transition_edge_count": int(edges.phi_transition.shape[0]),
                "psi_transition_edge_count": int(edges.psi_transition.shape[0]),
            },
            "aggregate_metrics": aggregate,
            "paired_comparisons": paired,
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "metric_csv": f"metrics/per_sample_{self._subspace_suffix}.csv",
            "raw_archive": (
                f"data/selected_{self._subspace_suffix}_tangent_subspace.npz"
                if self.request.save_generated_data
                else None
            ),
            "figure_json": [
                str(path.relative_to(self.request.outdir)) for path in figure_paths
            ],
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate_metrics"]
        paired = summary["paired_comparisons"]
        maximum_dimension = self.request.max_subspace_dimension
        lines = [
            f"# Matrix-Free K=1 through K={maximum_dimension} Tangent Subspace Audit",
            "",
            "The CouplingNet and GreenNet checkpoints are frozen. Reference solution",
            "and directional targets are used only for evaluation metrics.",
            "",
            "## Aggregate Results",
            "",
            "| method | mismatch / sym | optimized energy | rel_sol | rel_u_phi | rel_u_psi | rel_flux | correction / sym |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in self.methods:
            payload = aggregate[method.method_id]
            lines.append(
                "| "
                f"{method.label} | "
                f"{float(payload['response_mismatch_ratio_vs_symmetric_mean']):.6f} | "
                f"{float(payload['loss_energy_optimized_mean']):.6e} | "
                f"{100.0 * float(payload['rel_sol_mean']):.4f}% | "
                f"{100.0 * float(payload['rel_u_phi_mean']):.4f}% | "
                f"{100.0 * float(payload['rel_u_psi_mean']):.4f}% | "
                f"{100.0 * float(payload['rel_flux_mean']):.4f}% | "
                f"{float(payload['tangent_correction_rel_symmetric_pair_mean']):.6f} |"
            )
        production = aggregate["k1_production"]
        final_method_id = f"k{maximum_dimension}_unconstrained"
        final = aggregate[final_method_id]
        lines.extend(
            [
                "",
                "## Upper-Tail Results",
                "",
                (
                    "| metric | K=1 capped p95 | "
                    f"K={maximum_dimension} p95 | K=1 capped max | "
                    f"K={maximum_dimension} max |"
                ),
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in (
            "response_mismatch_cost",
            "loss_energy_optimized",
            "rel_sol",
            "rel_u_phi",
            "rel_u_psi",
            "rel_flux",
            "tangent_correction_rel_symmetric_pair",
        ):
            lines.append(
                f"| {metric} | "
                f"{float(production[f'{metric}_p95']):.6e} | "
                f"{float(final[f'{metric}_p95']):.6e} | "
                f"{float(production[f'{metric}_max']):.6e} | "
                f"{float(final[f'{metric}_max']):.6e} |"
            )
        lines.extend(
            [
                "",
                "## Paired Findings",
                "",
            ]
        )
        for dimension in range(2, maximum_dimension + 1):
            baseline = (
                "k1_uncapped" if dimension == 2 else f"k{dimension - 1}_unconstrained"
            )
            adjacent = paired[f"k{dimension}_vs_{baseline}"]
            versus_capped = paired[f"k{dimension}_vs_k1_production"]
            lines.extend(
                [
                    f"- K={dimension} versus K={dimension - 1} response mismatch "
                    "relative mean change: "
                    f"`{100.0 * float(adjacent['response_mismatch_cost']['relative_mean_change']):.3f}%`.",
                    f"- K={dimension} versus K={dimension - 1} response mismatch "
                    "improved samples: "
                    f"`{adjacent['response_mismatch_cost']['improved_sample_count']}/"
                    f"{adjacent['response_mismatch_cost']['sample_count']}`.",
                    f"- K={dimension} versus configured-cap K=1 rel_sol relative mean "
                    "change: "
                    f"`{100.0 * float(versus_capped['rel_sol']['relative_mean_change']):.3f}%`.",
                ]
            )
        lines.extend(
            [
                "",
                "## Interpretation Boundary",
                "",
                f"K=2 through K={maximum_dimension} are unconstrained frozen-checkpoint ",
                "diagnostics. They show whether additional matrix-free response directions",
                "help the configured surrogate on identical raw output. They do not establish",
                "the effect of training with the larger subspace; that requires paired retraining.",
            ]
        )
        (self.request.outdir / "diagnosis_report.md").write_text(
            "\n".join(lines) + "\n"
        )


def run_tangent_subspace_audit(
    request: TangentSubspaceAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the frozen K=1 versus K=2 matrix-free tangent audit."""

    return ComplexTangentSubspaceAudit(request, logger=logger).run()
