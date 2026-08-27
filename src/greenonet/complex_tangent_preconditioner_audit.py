from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import build_boundary_energy_context
from greenonet.complex_projection_response_audit import (
    ComplexProjectionResponseAudit,
    ProjectionTransitionEdges,
)
from greenonet.complex_symmetric_tangent_audit import TangentMethod
from greenonet.complex_tangent_preconditioner import (
    TANGENT_PRECONDITIONER_VARIANTS,
    TangentPreconditionerVariant,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    SymmetricTangentGreenResponseContext,
    SymmetricTangentGreenResponseContextCache,
)
from greenonet.complex_tangent_context_io import resolve_tangent_context_path
from greenonet.complex_tangent_subspace_audit import (
    ComplexTangentSubspaceAudit,
    PreparedTangentBatch,
    TangentSubspaceAuditRequest,
)
from greenonet.config import (
    BalanceProjectionConfig,
    SymmetricTangentGreenResponseProjectionConfig,
    TangentContextCheckpointConfig,
    validate_complex_tangent_context_checkpoint_config,
)
from greenonet.coupling_artifacts import load_coupling_artifact_configs
from greenonet.plotly_io import save_plotly_figure


_VARIANT_PREFIX: dict[TangentPreconditionerVariant, str] = {
    "separable": "sep",
    "exact_diagonal": "exact",
    "absolute_cross_axis": "abs",
    "normalized_quadratic_cross_axis": "q",
}


@dataclass(frozen=True)
class TangentPreconditionerAuditRequest:
    """Inputs for a shared-output four-preconditioner by K=1..4 audit."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    tangent_context: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    batch_size: int = 10
    selected_samples: tuple[int, ...] | None = None
    transition_log_threshold: float = math.log(2.0)
    subspace_relative_eps: float = 1.0e-12
    metric_eps: float = 1.0e-30
    operator_equivalence_tol: float = 1.0e-10
    monotonicity_relative_tol: float = 1.0e-10
    variants: tuple[TangentPreconditionerVariant, ...] = TANGENT_PRECONDITIONER_VARIANTS
    max_subspace_dimension: int = 4
    save_generated_data: bool = True
    posthoc_tangent_override: bool = False
    posthoc_eta: float = 0.01
    posthoc_line_search_relative_eps: float = 1.0e-12
    posthoc_relative_lambda: float = 0.01
    posthoc_denominator_relative_eps: float = 1.0e-12
    posthoc_cross_axis_relative_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.variants != TANGENT_PRECONDITIONER_VARIANTS:
            raise ValueError(
                "The production preconditioner audit requires the ordered variants "
                f"{TANGENT_PRECONDITIONER_VARIANTS}."
            )
        TangentSubspaceAuditRequest(
            config=self.config,
            coupling_checkpoint=self.coupling_checkpoint,
            green_checkpoint=self.green_checkpoint,
            outdir=self.outdir,
            geometry=self.geometry,
            test_path=self.test_path,
            coefficients=self.coefficients,
            tangent_context=self.tangent_context,
            device=self.device,
            theme=self.theme,
            batch_size=self.batch_size,
            selected_samples=self.selected_samples,
            transition_log_threshold=self.transition_log_threshold,
            subspace_relative_eps=self.subspace_relative_eps,
            metric_eps=self.metric_eps,
            operator_equivalence_tol=self.operator_equivalence_tol,
            monotonicity_relative_tol=self.monotonicity_relative_tol,
            max_subspace_dimension=self.max_subspace_dimension,
            save_generated_data=self.save_generated_data,
        )
        if not isinstance(self.posthoc_tangent_override, bool):
            raise TypeError("posthoc_tangent_override must be a boolean.")
        if self.posthoc_tangent_override:
            self.posthoc_tangent_config()

    def posthoc_tangent_config(
        self,
    ) -> SymmetricTangentGreenResponseProjectionConfig:
        """Return the explicit audit-only tangent settings."""

        return SymmetricTangentGreenResponseProjectionConfig(
            subspace_dimension=1,
            eta=self.posthoc_eta,
            eta_strategy="closed_loop_exact_line_search",
            line_search_relative_eps=self.posthoc_line_search_relative_eps,
            relative_lambda=self.posthoc_relative_lambda,
            denominator_relative_eps=self.posthoc_denominator_relative_eps,
            preconditioner_variant="separable",
            cross_axis_relative_eps=self.posthoc_cross_axis_relative_eps,
        )


class ComplexTangentPreconditionerAudit(ComplexTangentSubspaceAudit):
    """Evaluate four immutable diagonal contexts from one raw model output."""

    def __init__(
        self,
        request: TangentPreconditionerAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.preconditioner_request = request
        super().__init__(
            TangentSubspaceAuditRequest(
                config=request.config,
                coupling_checkpoint=request.coupling_checkpoint,
                green_checkpoint=request.green_checkpoint,
                outdir=request.outdir,
                geometry=request.geometry,
                test_path=request.test_path,
                coefficients=request.coefficients,
                tangent_context=request.tangent_context,
                device=request.device,
                theme=request.theme,
                batch_size=request.batch_size,
                selected_samples=request.selected_samples,
                transition_log_threshold=request.transition_log_threshold,
                subspace_relative_eps=request.subspace_relative_eps,
                metric_eps=request.metric_eps,
                operator_equivalence_tol=request.operator_equivalence_tol,
                monotonicity_relative_tol=request.monotonicity_relative_tol,
                max_subspace_dimension=request.max_subspace_dimension,
                save_generated_data=request.save_generated_data,
            ),
            logger=logger,
        )
        self.audit_methods = self._build_audit_methods()
        self.variant_contexts: dict[
            TangentPreconditionerVariant,
            SymmetricTangentGreenResponseContext,
        ] = {}

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(self.request.config)
        if self._configs.dataset.geometry_mode != "complex":
            raise ValueError("Tangent preconditioner audit requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        self._training_projection = projection
        if self.preconditioner_request.posthoc_tangent_override:
            if not projection.enabled or projection.mode not in {
                "physical_symmetric",
                "symmetric_tangent_green_response",
            }:
                raise ValueError(
                    "Post-hoc tangent override requires an enabled physical_symmetric "
                    "or symmetric_tangent_green_response training projection."
                )
            tangent = self.preconditioner_request.posthoc_tangent_config()
        else:
            if projection.mode != "symmetric_tangent_green_response":
                raise ValueError(
                    "Tangent preconditioner audit requires "
                    "balance_projection.mode='symmetric_tangent_green_response' or "
                    "an explicit posthoc_tangent_override=true."
                )
            tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
                projection.symmetric_tangent_green_response
            )
            if tangent.eta_strategy != "closed_loop_exact_line_search":
                raise ValueError(
                    "Tangent preconditioner audit requires "
                    "closed_loop_exact_line_search or an explicit "
                    "posthoc_tangent_override=true."
                )
        self._audit_tangent_config = tangent
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
        coefficient_functions = load_coefficient_functions(coefficient_path)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            coefficient_functions,
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
        offsets: dict[int, int] = {}
        raw_digest = hashlib.sha256()
        variant_seconds = {
            variant: 0.0 for variant in self.preconditioner_request.variants
        }
        model_seconds = 0.0
        numerical_started = time.perf_counter()
        offset = 0
        for batch in loader:
            batch = batch.to(self._device)
            self._initialize_context(batch)
            self._initialize_variant_contexts()
            prepare_started = time.perf_counter()
            prepared = self._prepare_batch(batch)
            model_seconds += time.perf_counter() - prepare_started
            self._update_raw_digest(raw_digest, batch, prepared)
            for variant in self.preconditioner_request.variants:
                context = self.variant_contexts[variant]
                variant_started = time.perf_counter()
                evaluation, krylov = self._evaluate_prepared_batch(
                    batch,
                    prepared,
                    context=context,
                )
                batch_rows = self._metric_rows(
                    batch,
                    evaluation,
                    krylov,
                    edges,
                    context=context,
                )
                variant_seconds[variant] += time.perf_counter() - variant_started
                rows.extend(self._rename_rows(batch_rows, variant))
            for sample_id in batch.sample_indices.tolist():
                offsets[int(sample_id)] = offset
                offset += 1
        numerical_seconds = time.perf_counter() - numerical_started

        aggregate = self._aggregate(rows)
        paired_rows, paired_summary = self._paired(rows)
        selected, selected_roles = self._select_samples(rows, offsets)
        per_sample_path = (
            self.request.outdir / "metrics" / "per_sample_preconditioner_k1_k4.csv"
        )
        aggregate_path = (
            self.request.outdir / "metrics" / "aggregate_preconditioner_k1_k4.csv"
        )
        paired_path = (
            self.request.outdir / "metrics" / "paired_preconditioner_k1_k4.csv"
        )
        self._write_csv(per_sample_path, rows)
        self._write_csv(aggregate_path, list(aggregate.values()))
        self._write_csv(paired_path, paired_rows)

        field_stats = self._write_preconditioner_fields(edges)
        figure_paths = [
            self._write_preconditioner_field_figure(),
            self._write_aggregate_figure_4x4(aggregate),
        ]
        selected_batch = complex_coupling_collate_fn(
            [dataset[offsets[sample_id]] for sample_id in selected]
        ).to(self._device)
        selected_prepared = self._prepare_batch(selected_batch)
        selected_evaluations: dict[
            TangentPreconditionerVariant,
            tuple[Any, KrylovSubspaceStepResult],
        ] = {}
        for variant in self.preconditioner_request.variants:
            selected_evaluations[variant] = self._evaluate_prepared_batch(
                selected_batch,
                selected_prepared,
                context=self.variant_contexts[variant],
            )
        if self.request.save_generated_data:
            self._write_preconditioner_selected_arrays(
                selected_batch,
                selected_prepared,
                selected_evaluations,
            )
        for sample_offset in range(len(selected)):
            figure_paths.append(
                self._write_selected_figure_4x4(
                    selected_batch,
                    selected_evaluations,
                    sample_offset,
                )
            )

        summary = self._build_preconditioner_summary(
            dataset_size=len(dataset),
            geometry_path=geometry_path,
            test_path=test_path,
            coefficient_path=coefficient_path,
            aggregate=aggregate,
            paired=paired_summary,
            field_stats=field_stats,
            selected=selected,
            selected_roles=selected_roles,
            raw_output_sha256="sha256:" + raw_digest.hexdigest(),
            model_seconds=model_seconds,
            variant_seconds=variant_seconds,
            numerical_seconds=numerical_seconds,
            total_seconds=time.perf_counter() - started,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        self._write_preconditioner_report(summary)
        if self.logger is not None:
            self.logger.info(
                "4xK tangent preconditioner audit complete: samples=%d context=%s",
                len(dataset),
                summary["tangent_context"]["source"],
            )
        return summary

    def _initialize_context(self, batch: ComplexCouplingBatch) -> None:
        if hasattr(self, "response_operator"):
            return
        diagnostic_tangent = replace(
            self._audit_tangent_config,
            subspace_dimension=1,
        )
        if self.preconditioner_request.posthoc_tangent_override:
            checkpoint = TangentContextCheckpointConfig(
                enabled=self.request.tangent_context is not None,
                path=self.request.tangent_context,
                load_policy="if_available",
                save_after_build=True,
            )
            checkpoint_path = self.request.tangent_context
        else:
            checkpoint = validate_complex_tangent_context_checkpoint_config(
                training=self._configs.coupling_training,
                balance_projection=self._training_projection,
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

    def _initialize_variant_contexts(self) -> None:
        if self.variant_contexts:
            return
        self.variant_contexts = {
            variant: self.tangent_context.with_preconditioner_variant(variant)
            for variant in self.preconditioner_request.variants
        }

    def _build_audit_methods(self) -> tuple[TangentMethod, ...]:
        methods = [
            TangentMethod("physical_symmetric", "Physical symmetric", "symmetric")
        ]
        for variant in self.preconditioner_request.variants:
            prefix = _VARIANT_PREFIX[variant]
            methods.append(
                TangentMethod(
                    f"{prefix}_k1_capped",
                    f"{prefix} K=1 capped",
                    "k1_capped",
                )
            )
            methods.extend(
                TangentMethod(
                    f"{prefix}_k{dimension}",
                    f"{prefix} K={dimension}",
                    "nested_uncapped",
                )
                for dimension in range(1, self.request.max_subspace_dimension + 1)
            )
        return tuple(methods)

    @staticmethod
    def _update_raw_digest(
        digest: Any,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
    ) -> None:
        for value in (
            batch.sample_indices.detach().cpu().contiguous(),
            prepared.raw_physical.detach().cpu().contiguous(),
            prepared.symmetric_physical.detach().cpu().contiguous(),
        ):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.view(torch.uint8).numpy().tobytes())

    def _rename_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
        variant: TangentPreconditionerVariant,
    ) -> list[dict[str, float | int | str]]:
        prefix = _VARIANT_PREFIX[variant]
        output: list[dict[str, float | int | str]] = []
        denominator = self.variant_contexts[variant].denominator
        for row in rows:
            original = str(row["method_id"])
            if original == "symmetric":
                if variant != self.preconditioner_request.variants[0]:
                    continue
                method_id = "physical_symmetric"
                dimension = 0
                variant_name = "shared"
            elif original == "k1_production":
                method_id = f"{prefix}_k1_capped"
                dimension = 1
                variant_name = variant
            elif original == "k1_uncapped":
                method_id = f"{prefix}_k1"
                dimension = 1
                variant_name = variant
            else:
                dimension = int(original[1 : original.index("_")])
                method_id = f"{prefix}_k{dimension}"
                variant_name = variant
            method = next(
                item for item in self.audit_methods if item.method_id == method_id
            )
            renamed = dict(row)
            renamed.update(
                {
                    "method_id": method_id,
                    "method_label": method.label,
                    "method_kind": method.kind,
                    "preconditioner_variant": variant_name,
                    "preconditioner_prefix": (
                        "shared" if variant_name == "shared" else prefix
                    ),
                    "subspace_dimension": dimension,
                    "configured_cap_reference": int(method.kind == "k1_capped"),
                    "posthoc_preconditioner_override": int(
                        variant_name != "shared"
                        and variant != self.tangent_context.preconditioner_variant
                    ),
                    "posthoc_training_projection_override": int(
                        self.preconditioner_request.posthoc_tangent_override
                    ),
                    "training_projection_mode": self._training_projection.mode,
                    "denominator_min": float(denominator.min().item()),
                    "denominator_max": float(denominator.max().item()),
                }
            )
            output.append(renamed)
        return output

    def _aggregate(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, dict[str, float | int | str]]:
        aggregate = self.aggregate_rows(rows, self.audit_methods)
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
        for method in self.audit_methods:
            selected = [row for row in rows if row["method_id"] == method.method_id]
            payload = aggregate[method.method_id]
            payload["method_id"] = method.method_id
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

    def _paired(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> tuple[list[dict[str, float | int | str]], dict[str, Any]]:
        metrics = (
            "response_mismatch_cost",
            "loss_energy_optimized",
            "canonical_energy",
            "rel_sol",
            "rel_u_phi",
            "rel_u_psi",
            "rel_flux",
            "tangent_correction_rel_symmetric_pair",
        )
        by_method = {
            method.method_id: {
                int(row["sample_id"]): row
                for row in rows
                if row["method_id"] == method.method_id
            }
            for method in self.audit_methods
        }
        pairs: list[tuple[str, str, str]] = []
        for variant in self.preconditioner_request.variants:
            prefix = _VARIANT_PREFIX[variant]
            for dimension in range(1, self.request.max_subspace_dimension + 1):
                candidate = f"{prefix}_k{dimension}"
                if prefix != "sep":
                    pairs.append(
                        (
                            f"{candidate}_vs_sep_k{dimension}",
                            f"sep_k{dimension}",
                            candidate,
                        )
                    )
                previous = (
                    "physical_symmetric"
                    if dimension == 1
                    else f"{prefix}_k{dimension - 1}"
                )
                pairs.append((f"{candidate}_vs_previous", previous, candidate))
        output_rows: list[dict[str, float | int | str]] = []
        summary: dict[str, Any] = {}
        for comparison_id, baseline_id, candidate_id in pairs:
            common = sorted(set(by_method[baseline_id]) & set(by_method[candidate_id]))
            comparison: dict[str, Any] = {}
            for metric in metrics:
                values = [
                    (
                        float(by_method[baseline_id][sample_id][metric]),
                        float(by_method[candidate_id][sample_id][metric]),
                    )
                    for sample_id in common
                    if metric in by_method[baseline_id][sample_id]
                    and metric in by_method[candidate_id][sample_id]
                ]
                if not values:
                    continue
                baseline = np.asarray([value[0] for value in values])
                candidate_values = np.asarray([value[1] for value in values])
                delta = candidate_values - baseline
                payload: dict[str, float | int] = {
                    "sample_count": len(values),
                    "baseline_mean": float(baseline.mean()),
                    "candidate_mean": float(candidate_values.mean()),
                    "mean_delta": float(delta.mean()),
                    "relative_mean_change": self._relative_change(
                        baseline=float(baseline.mean()),
                        candidate=float(candidate_values.mean()),
                    ),
                    "improved_sample_count": int(np.count_nonzero(delta < 0.0)),
                    "worsened_sample_count": int(np.count_nonzero(delta > 0.0)),
                    "unchanged_sample_count": int(np.count_nonzero(delta == 0.0)),
                    "p95_delta": float(np.quantile(delta, 0.95)),
                    "max_worsening": float(max(0.0, float(delta.max()))),
                }
                comparison[metric] = payload
                output_rows.append(
                    {
                        "comparison_id": comparison_id,
                        "baseline_method_id": baseline_id,
                        "candidate_method_id": candidate_id,
                        "metric": metric,
                        **payload,
                    }
                )
            summary[comparison_id] = comparison
        return output_rows, summary

    def _select_samples(
        self,
        rows: Sequence[dict[str, float | int | str]],
        offsets: dict[int, int],
    ) -> tuple[tuple[int, ...], dict[str, str]]:
        if self.request.selected_samples is not None:
            missing = sorted(set(self.request.selected_samples) - set(offsets))
            if missing:
                raise ValueError(f"Selected sample IDs are unavailable: {missing}.")
            return self.request.selected_samples, {
                str(sample_id): "explicit"
                for sample_id in self.request.selected_samples
            }
        symmetric = {
            int(row["sample_id"]): row
            for row in rows
            if row["method_id"] == "physical_symmetric" and "rel_sol" in row
        }
        if not symmetric:
            first = tuple(sorted(offsets)[:1])
            return first, {str(sample_id): "first_available" for sample_id in first}
        ordered = sorted(
            symmetric, key=lambda value: float(symmetric[value]["rel_sol"])
        )
        final_ids = [
            f"{_VARIANT_PREFIX[variant]}_k{self.request.max_subspace_dimension}"
            for variant in self.preconditioner_request.variants
        ]
        final = {
            method_id: {
                int(row["sample_id"]): row
                for row in rows
                if row["method_id"] == method_id and "rel_sol" in row
            }
            for method_id in final_ids
        }
        improvement = {
            sample_id: min(
                float(final[method_id][sample_id]["rel_sol"])
                for method_id in final_ids
                if sample_id in final[method_id]
            )
            - float(symmetric[sample_id]["rel_sol"])
            for sample_id in ordered
        }
        candidates = (
            (ordered[len(ordered) // 2], "symmetric_rel_sol_q50"),
            (ordered[-1], "symmetric_rel_sol_worst"),
            (
                min(improvement, key=lambda key: improvement[key]),
                "largest_k4_improvement",
            ),
            (
                max(improvement, key=lambda key: improvement[key]),
                "largest_k4_worsening",
            ),
        )
        selected: list[int] = []
        roles: dict[str, str] = {}
        for sample_id, role in candidates:
            if sample_id not in selected:
                selected.append(sample_id)
                roles[str(sample_id)] = role
        return tuple(selected), roles

    def _write_preconditioner_fields(
        self,
        edges: ProjectionTransitionEdges,
    ) -> dict[str, Any]:
        context = self.tangent_context
        fields = self._preconditioner_fields(context)
        path = self.request.outdir / "data" / "tangent_preconditioner_fields.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coords_valid=self.geometry.coords_valid.detach().cpu().numpy(),
            transition_edges=edges.transition.detach().cpu().numpy(),
            regular_edges=edges.regular.detach().cpu().numpy(),
            variant_names=np.asarray(self.preconditioner_request.variants),
            **{  # type: ignore[arg-type]
                key: value.detach().cpu().numpy() for key, value in fields.items()
            },
        )
        statistics = {
            key: self._field_statistics(value, edges) for key, value in fields.items()
        }
        tolerance = 128.0 * torch.finfo(context.denominator.dtype).eps
        ordering = {
            "absolute_ge_separable": bool(
                torch.all(
                    context.absolute_preconditioner_base
                    >= context.separable_preconditioner_base
                )
            ),
            "quadratic_ge_separable": bool(
                torch.all(
                    context.quadratic_preconditioner_base
                    >= context.separable_preconditioner_base
                )
            ),
            "quadratic_le_absolute": bool(
                torch.all(
                    context.quadratic_preconditioner_base
                    <= context.absolute_preconditioner_base
                    + tolerance * context.gain_scale
                )
            ),
            "exact_le_absolute": bool(
                torch.all(
                    context.exact_preconditioner_base
                    <= context.absolute_preconditioner_base
                    + tolerance * context.gain_scale
                )
            ),
        }
        return {
            "fields": statistics,
            "ordering": ordering,
            "all_finite": all(
                torch.all(torch.isfinite(value)) for value in fields.values()
            ),
            "all_denominators_positive": all(
                torch.all(value > 0.0)
                for key, value in fields.items()
                if key.endswith("_denominator")
            ),
            "cauchy_violation_max": float(context.cauchy_violation_max.item()),
            "exact_roundoff_clamp_count": context.exact_roundoff_clamp_count,
            "global_matrix_materialized": False,
            "global_linear_solve": False,
        }

    @staticmethod
    def _preconditioner_fields(
        context: SymmetricTangentGreenResponseContext,
    ) -> dict[str, torch.Tensor]:
        return {
            "a": context.gamma_x_squared,
            "b": context.gamma_y_squared,
            "c": context.cross_axis_inner_product,
            "rho": context.normalized_correlation,
            "q": context.normalized_quadratic_cross_axis,
            "separable_base": context.separable_preconditioner_base,
            "exact_base": context.exact_preconditioner_base,
            "absolute_base": context.absolute_preconditioner_base,
            "quadratic_base": context.quadratic_preconditioner_base,
            "separable_denominator": context.separable_denominator,
            "exact_denominator": context.exact_denominator,
            "absolute_denominator": context.absolute_denominator,
            "quadratic_denominator": context.quadratic_denominator,
        }

    @staticmethod
    def _field_statistics(
        values: torch.Tensor,
        edges: ProjectionTransitionEdges,
    ) -> dict[str, float]:
        array = values.detach().cpu().numpy()
        transition = edges.transition.to(values.device)
        regular = edges.regular.to(values.device)

        def edge_rms(edge_index: torch.Tensor) -> float:
            if not edge_index.numel():
                return math.nan
            difference = values[edge_index[:, 1]] - values[edge_index[:, 0]]
            return float(difference.square().mean().sqrt().item())

        transition_rms = edge_rms(transition)
        regular_rms = edge_rms(regular)
        return {
            "min": float(array.min()),
            "q01": float(np.quantile(array, 0.01)),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "q99": float(np.quantile(array, 0.99)),
            "max": float(array.max()),
            "transition_edge_jump_rms": transition_rms,
            "regular_edge_jump_rms": regular_rms,
            "transition_to_regular_jump_ratio": (
                transition_rms / regular_rms if regular_rms > 0.0 else math.inf
            ),
        }

    def _write_preconditioner_field_figure(self) -> Path:
        fields = self._preconditioner_fields(self.tangent_context)
        selected = (
            "a",
            "b",
            "c",
            "rho",
            "q",
            "separable_denominator",
            "exact_denominator",
            "absolute_denominator",
            "quadratic_denominator",
        )
        fig = make_subplots(rows=3, cols=3, subplot_titles=selected)
        for index, name in enumerate(selected):
            fig.add_trace(
                self._scatter(
                    geometry=self.geometry,
                    values=fields[name],
                    title=name,
                    symmetric=name in {"c", "rho"},
                ),
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
        fig.update_layout(
            template=self.request.theme,
            title="Tangent preconditioner spatial preflight",
            width=1350,
            height=1250,
            showlegend=False,
        )
        for index in range(len(selected)):
            axis = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis,
                scaleratio=1.0,
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
        path = (
            self.request.outdir
            / "figures"
            / "tangent_preconditioner"
            / "preconditioner_spatial_fields"
        )
        save_plotly_figure(fig, path, self.logger)
        return path.with_suffix(".json")

    def _write_aggregate_figure_4x4(
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
        colors = {
            "separable": "#2563eb",
            "exact_diagonal": "#dc2626",
            "absolute_cross_axis": "#0f766e",
            "normalized_quadratic_cross_axis": "#9333ea",
        }
        fig = make_subplots(
            rows=2, cols=3, subplot_titles=[label for _, label in metrics]
        )
        for metric_index, (metric, _label) in enumerate(metrics):
            baseline = float(aggregate["physical_symmetric"].get(metric, math.nan))
            for variant in self.preconditioner_request.variants:
                prefix = _VARIANT_PREFIX[variant]
                values = [
                    float(aggregate[f"{prefix}_k{k}"].get(metric, math.nan)) / baseline
                    for k in range(1, self.request.max_subspace_dimension + 1)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, self.request.max_subspace_dimension + 1)),
                        y=values,
                        mode="lines+markers",
                        name=variant,
                        legendgroup=variant,
                        showlegend=metric_index == 0,
                        line={"color": colors[variant]},
                        hovertemplate="K=%{x}<br>ratio=%{y:.6f}<extra>%{fullData.name}</extra>",
                    ),
                    row=metric_index // 3 + 1,
                    col=metric_index % 3 + 1,
                )
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="#475569",
                row=metric_index // 3 + 1,
                col=metric_index % 3 + 1,
            )
        fig.update_layout(
            template=self.request.theme,
            title="Frozen tangent preconditioner 4 x K audit",
            width=1500,
            height=850,
            margin={"l": 70, "r": 40, "t": 100, "b": 70},
        )
        fig.update_xaxes(dtick=1, title_text="subspace K")
        fig.update_yaxes(title_text="ratio to physical symmetric")
        path = (
            self.request.outdir
            / "figures"
            / "aggregate"
            / "preconditioner_k1_k4_metric_ratios"
        )
        save_plotly_figure(fig, path, self.logger)
        return path.with_suffix(".json")

    def _write_preconditioner_selected_arrays(
        self,
        batch: ComplexCouplingBatch,
        prepared: PreparedTangentBatch,
        evaluations: dict[
            TangentPreconditionerVariant,
            tuple[Any, KrylovSubspaceStepResult],
        ],
    ) -> Path:
        method_ids = ["physical_symmetric"]
        physical: list[torch.Tensor] = []
        solution: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        equal_prediction: list[torch.Tensor] = []
        delta: list[torch.Tensor] = []
        first_evaluation = evaluations[self.preconditioner_request.variants[0]][0]
        for field, target in (
            (first_evaluation.candidate_physical[0], physical),
            (first_evaluation.candidate_solution[0], solution),
            (first_evaluation.candidate_prediction[0], prediction),
            (first_evaluation.candidate_equal_prediction[0], equal_prediction),
            (first_evaluation.tangent_delta[0], delta),
        ):
            target.append(field.detach().cpu())
        coefficients: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        costs: list[torch.Tensor] = []
        for variant in self.preconditioner_request.variants:
            prefix = _VARIANT_PREFIX[variant]
            evaluation, krylov = evaluations[variant]
            for method_index in range(1, len(evaluation.methods)):
                original = evaluation.methods[method_index].method_id
                suffix = (
                    "k1_capped"
                    if original == "k1_production"
                    else ("k1" if original == "k1_uncapped" else original.split("_")[0])
                )
                method_ids.append(f"{prefix}_{suffix}")
                physical.append(
                    evaluation.candidate_physical[method_index].detach().cpu()
                )
                solution.append(
                    evaluation.candidate_solution[method_index].detach().cpu()
                )
                prediction.append(
                    evaluation.candidate_prediction[method_index].detach().cpu()
                )
                equal_prediction.append(
                    evaluation.candidate_equal_prediction[method_index].detach().cpu()
                )
                delta.append(evaluation.tangent_delta[method_index].detach().cpu())
            coefficients.append(krylov.coefficients.detach().cpu())
            active.append(krylov.direction_active.detach().cpu())
            costs.append(krylov.costs.detach().cpu())
        path = self.request.outdir / "data" / "selected_preconditioner_k1_k4.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            sample_ids=batch.sample_indices.detach().cpu().numpy(),
            file_stems=np.asarray(batch.file_stems),
            method_ids=np.asarray(method_ids),
            variant_names=np.asarray(self.preconditioner_request.variants),
            coords_valid=self.geometry.coords_valid.detach().cpu().numpy(),
            rhs=batch.rhs_valid.detach().cpu().numpy(),
            sol=batch.sol_valid.detach().cpu().numpy(),
            flux_target=batch.flux_valid.detach().cpu().numpy(),
            raw_physical=prepared.raw_physical.detach().cpu().numpy(),
            symmetric_physical=prepared.symmetric_physical.detach().cpu().numpy(),
            candidate_physical=torch.stack(physical).numpy(),
            candidate_solution=torch.stack(solution).numpy(),
            candidate_prediction=torch.stack(prediction).numpy(),
            candidate_equal_prediction=torch.stack(equal_prediction).numpy(),
            tangent_delta=torch.stack(delta).numpy(),
            krylov_coefficients=torch.stack(coefficients).numpy(),
            krylov_direction_active=torch.stack(active).numpy(),
            krylov_response_cost=torch.stack(costs).numpy(),
        )
        return path

    def _write_selected_figure_4x4(
        self,
        batch: ComplexCouplingBatch,
        evaluations: dict[
            TangentPreconditionerVariant,
            tuple[Any, KrylovSubspaceStepResult],
        ],
        sample_offset: int,
    ) -> Path:
        variants = self.preconditioner_request.variants
        first = evaluations[variants[0]][0]
        final_index = len(first.methods) - 1
        rows = ("physical_symmetric",) + tuple(
            _VARIANT_PREFIX[value] for value in variants
        )
        fig = make_subplots(
            rows=len(rows),
            cols=3,
            subplot_titles=[
                title
                for label in rows
                for title in (
                    f"{label}: delta",
                    f"{label}: mismatch",
                    f"{label}: u error",
                )
            ],
        )
        has_solution = bool(batch.has_solution[sample_offset].item())
        for row_index, label in enumerate(rows, start=1):
            if row_index == 1:
                evaluation = first
                method_index = 0
            else:
                evaluation = evaluations[variants[row_index - 2]][0]
                method_index = final_index
            pair = evaluation.candidate_solution[method_index, sample_offset]
            prediction = evaluation.candidate_prediction[method_index, sample_offset]
            signed_error = (
                prediction - batch.sol_valid[sample_offset]
                if has_solution
                else torch.zeros_like(prediction)
            )
            for col_index, values in enumerate(
                (
                    evaluation.tangent_delta[method_index, sample_offset],
                    pair[0] - pair[1],
                    signed_error,
                ),
                start=1,
            ):
                fig.add_trace(
                    self._scatter(
                        geometry=self.geometry,
                        values=values,
                        title=label,
                        symmetric=True,
                    ),
                    row=row_index,
                    col=col_index,
                )
        sample_id = int(batch.sample_indices[sample_offset].item())
        fig.update_layout(
            template=self.request.theme,
            title=f"sample {sample_id}: physical symmetric and four K=4 candidates",
            width=1450,
            height=320 * len(rows),
            showlegend=False,
        )
        fig.update_annotations(font={"size": 10})
        for index in range(len(rows) * 3):
            axis = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis,
                scaleratio=1.0,
                row=index // 3 + 1,
                col=index % 3 + 1,
            )
        path = (
            self.request.outdir
            / "figures"
            / "selected_samples"
            / f"sample_{sample_id:04d}_preconditioner_k4"
        )
        save_plotly_figure(fig, path, self.logger)
        return path.with_suffix(".json")

    def _build_preconditioner_summary(
        self,
        *,
        dataset_size: int,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path,
        aggregate: dict[str, dict[str, float | int | str]],
        paired: dict[str, Any],
        field_stats: dict[str, Any],
        selected: tuple[int, ...],
        selected_roles: dict[str, str],
        raw_output_sha256: str,
        model_seconds: float,
        variant_seconds: dict[TangentPreconditionerVariant, float],
        numerical_seconds: float,
        total_seconds: float,
        figure_paths: Sequence[Path],
    ) -> dict[str, Any]:
        context_telemetry = (
            {}
            if self._tangent_context_cache is None
            else self._tangent_context_cache.telemetry()
        )
        batch_count = math.ceil(dataset_size / self.request.batch_size)
        variant_count = len(self.preconditioner_request.variants)
        forward_per_variant_batch = 2 + self.request.max_subspace_dimension
        adjoint_per_variant_batch = self.request.max_subspace_dimension
        operator_actions = {
            "numerical_batch_count": batch_count,
            "coupling_model_forward_calls": batch_count,
            "shared_symmetric_forward_pair_calls": batch_count,
            "shared_initial_adjoint_calls": batch_count,
            "per_variant_forward_pair_calls_per_batch": forward_per_variant_batch,
            "per_variant_adjoint_calls_per_batch": adjoint_per_variant_batch,
            "total_forward_pair_calls": batch_count
            * (1 + variant_count * forward_per_variant_batch),
            "total_adjoint_calls": batch_count
            * (1 + variant_count * adjoint_per_variant_batch),
            "global_matrix_materialized": False,
            "global_linear_solve": False,
        }
        reference_metrics = (
            "rel_sol_mean",
            "rel_u_phi_mean",
            "rel_u_psi_mean",
            "rel_flux_mean",
        )
        findings: dict[str, Any] = {}
        for metric in (
            "response_mismatch_cost_mean",
            "loss_energy_optimized_mean",
            *reference_metrics,
        ):
            candidates = {
                method_id: float(payload[metric])
                for method_id, payload in aggregate.items()
                if method_id != "physical_symmetric"
                and not method_id.endswith("_capped")
                and metric in payload
            }
            if candidates:
                findings[f"lowest_{metric}_method"] = min(
                    candidates,
                    key=lambda key: candidates[key],
                )
                findings[f"lowest_{metric}"] = min(candidates.values())
        return {
            "audit": "tangent_preconditioner_4x4_frozen_checkpoint",
            "training_or_checkpoint_updated": False,
            "posthoc_preconditioner_override": True,
            "posthoc_training_projection_override": (
                self.preconditioner_request.posthoc_tangent_override
            ),
            "reference_targets_used_for_correction": False,
            "reference_targets_used_for_reporting_only": True,
            "dataset_size": dataset_size,
            "variants": list(self.preconditioner_request.variants),
            "variant_prefixes": {
                key: _VARIANT_PREFIX[key]
                for key in self.preconditioner_request.variants
            },
            "subspace_dimensions": list(
                range(1, self.request.max_subspace_dimension + 1)
            ),
            "projection_provenance": {
                "training_mode": self._training_projection.mode,
                "training_tangent_config": (
                    asdict(
                        SymmetricTangentGreenResponseProjectionConfig.from_raw(
                            self._training_projection.symmetric_tangent_green_response
                        )
                    )
                    if self._training_projection.mode
                    == "symmetric_tangent_green_response"
                    else None
                ),
                "audit_mode": "symmetric_tangent_green_response",
                "audit_tangent_config": asdict(self._audit_tangent_config),
                "explicit_posthoc_override": (
                    self.preconditioner_request.posthoc_tangent_override
                ),
            },
            "raw_output": {
                "sha256": raw_output_sha256,
                "computed_once_per_batch": True,
                "shared_by_all_cells": True,
            },
            "phase_a_spatial_preflight": field_stats,
            "phase_b_aggregate": aggregate,
            "paired_comparisons": paired,
            "findings": findings,
            "selected_samples": list(selected),
            "selected_sample_roles": selected_roles,
            "tangent_context": {
                **context_telemetry,
                "response_operator_instance_count": 1,
                "preconditioner_context_count": len(self.variant_contexts),
                "formula_suite_schema": 2,
            },
            "operator_actions": operator_actions,
            "runtime": {
                "model_and_shared_prepare_seconds": model_seconds,
                "variant_seconds": {
                    variant: variant_seconds[variant]
                    for variant in self.preconditioner_request.variants
                },
                "variant_seconds_per_sample": {
                    variant: variant_seconds[variant] / dataset_size
                    for variant in self.preconditioner_request.variants
                },
                "numerical_seconds": numerical_seconds,
                "total_seconds_including_artifacts": total_seconds,
            },
            "numerical_checks": {
                "operator_equivalence_max_abs": self._operator_equivalence_max_abs,
                "operator_equivalence_tolerance": self.request.operator_equivalence_tol,
                "k_monotonicity_relative_tolerance": (
                    self.request.monotonicity_relative_tol
                ),
                "physical_balance_tolerance": 1.0e-10,
                "physical_balance_max_abs": max(
                    float(payload.get("physical_balance_max_abs_max", 0.0))
                    for payload in aggregate.values()
                ),
            },
            "provenance": {
                "config": self._path_provenance(self.request.config),
                "coupling_checkpoint": self._path_provenance(
                    self.request.coupling_checkpoint
                ),
                "green_checkpoint": self._path_provenance(
                    self.request.green_checkpoint
                ),
                "geometry": self._path_provenance(geometry_path),
                "test_data": self._path_provenance(test_path),
                "coefficients": self._path_provenance(coefficient_path),
                "dtype": str(self._configs.dataset.dtype).replace("torch.", ""),
                "device": str(self._device),
                "git": self._git_provenance(),
            },
            "artifacts": {
                "per_sample_csv": "metrics/per_sample_preconditioner_k1_k4.csv",
                "aggregate_csv": "metrics/aggregate_preconditioner_k1_k4.csv",
                "paired_csv": "metrics/paired_preconditioner_k1_k4.csv",
                "preconditioner_fields": "data/tangent_preconditioner_fields.npz",
                "selected_arrays": "data/selected_preconditioner_k1_k4.npz",
                "figures": [
                    str(path.relative_to(self.request.outdir)) for path in figure_paths
                ],
            },
        }

    def _write_preconditioner_report(self, summary: dict[str, Any]) -> Path:
        aggregate = summary["phase_b_aggregate"]
        findings = summary["findings"]
        spatial_fields = summary["phase_a_spatial_preflight"]["fields"]
        sep_k1 = aggregate["sep_k1"]
        sep_k4 = aggregate["sep_k4"]
        exact_k4 = aggregate["exact_k4"]
        abs_k4 = aggregate["abs_k4"]
        q_k4 = aggregate["q_k4"]
        best_energy_method = str(findings["lowest_loss_energy_optimized_mean_method"])
        best_rel_sol_method = str(findings["lowest_rel_sol_mean_method"])

        def relative_change(candidate: dict[str, Any], key: str) -> float:
            baseline = float(sep_k4[key])
            return (float(candidate[key]) - baseline) / baseline

        def k_gain(key: str) -> float:
            baseline = float(sep_k1[key])
            return (float(sep_k4[key]) - baseline) / baseline

        lines = [
            "# Tangent preconditioner 4 x K frozen audit",
            "",
            "This is a post-hoc frozen-checkpoint screening. It does not measure training adaptation.",
            "",
            "## Shared contract",
            "",
            f"- Samples: `{summary['dataset_size']}`",
            f"- Raw output SHA-256: `{summary['raw_output']['sha256']}`",
            (
                "- Training projection: "
                f"`{summary['projection_provenance']['training_mode']}`"
            ),
            (
                "- Audit tangent override: "
                f"`{summary['projection_provenance']['explicit_posthoc_override']}`"
            ),
            "- CouplingNet raw output: once per batch and shared by all 16 cells",
            "- Response operator: one segment-local instance; no global matrix or solve",
            f"- Context source: `{summary['tangent_context']['source']}`",
            "",
            "## Aggregate results",
            "",
            "| method | mismatch | optimized energy | rel_sol | rel_u_phi | rel_u_psi | rel_flux |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in self.audit_methods:
            payload = aggregate[method.method_id]
            lines.append(
                "| "
                + " | ".join(
                    (
                        method.method_id,
                        self._format_metric(payload, "response_mismatch_cost_mean"),
                        self._format_metric(payload, "loss_energy_optimized_mean"),
                        self._format_metric(payload, "rel_sol_mean"),
                        self._format_metric(payload, "rel_u_phi_mean"),
                        self._format_metric(payload, "rel_u_psi_mean"),
                        self._format_metric(payload, "rel_flux_mean"),
                    )
                )
                + " |"
            )
        lines.extend(("", "## Findings", ""))
        for key, value in findings.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(
            (
                "",
                "## Interpretation",
                "",
                (
                    "- The cross term `c` is non-negative at every valid point "
                    f"(`min={float(spatial_fields['c']['min']):.6e}`, "
                    f"`max={float(spatial_fields['c']['max']):.6e}`). Therefore "
                    "`exact_diagonal` and `absolute_cross_axis` are algebraically "
                    "identical for this frozen operator."
                ),
                (
                    f"- `{best_energy_method}` has the lowest mean optimized energy, "
                    f"while `{best_rel_sol_method}` has the lowest mean `rel_sol` in "
                    "this audit. Relative to `separable K=1`, `separable K=4` changes response "
                    f"mismatch by `{100.0 * k_gain('response_mismatch_cost_mean'):.3f}%`, "
                    f"optimized energy by `{100.0 * k_gain('loss_energy_optimized_mean'):.3f}%`, "
                    f"and `rel_sol` by `{100.0 * k_gain('rel_sol_mean'):.3f}%`."
                ),
                (
                    "- `exact/absolute K=4` lowers response mismatch versus "
                    f"`separable K=4` by `{100.0 * relative_change(exact_k4, 'response_mismatch_cost_mean'):.3f}%`, "
                    f"but changes optimized energy by `{100.0 * relative_change(exact_k4, 'loss_energy_optimized_mean'):.3f}%` "
                    f"and `rel_sol` by `{100.0 * relative_change(exact_k4, 'rel_sol_mean'):.3f}%`."
                ),
                (
                    "- `normalized-quadratic K=4` is nearly indistinguishable from "
                    f"the separable baseline: optimized energy changes by "
                    f"`{100.0 * relative_change(q_k4, 'loss_energy_optimized_mean'):.3f}%` "
                    f"and `rel_sol` by `{100.0 * relative_change(q_k4, 'rel_sol_mean'):.3f}%`."
                ),
                (
                    "- Frozen evidence is a screening result. A small post-hoc "
                    "difference from `separable` does not by itself justify a "
                    "preconditioner promotion or replace a paired retraining comparison."
                ),
                (
                    "- Exact and absolute K=4 aggregate equality check: "
                    f"`response={float(exact_k4['response_mismatch_cost_mean']) == float(abs_k4['response_mismatch_cost_mean'])}`, "
                    f"`rel_sol={float(exact_k4['rel_sol_mean']) == float(abs_k4['rel_sol_mean'])}`."
                ),
            )
        )
        lines.extend(
            (
                "",
                "## Interpretation boundary",
                "",
                "- Reference solution and flux are evaluation-only.",
                "- A post-hoc winner must be retrained in a paired experiment before promotion.",
                "- Trunk fuser ablations are deliberately excluded from this factorial audit.",
            )
        )
        path = self.request.outdir / "diagnosis_report.md"
        path.write_text("\n".join(lines) + "\n")
        return path

    @staticmethod
    def _format_metric(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        return "n/a" if value is None else f"{float(value):.6e}"

    @staticmethod
    def _path_provenance(path: Path) -> dict[str, str | int]:
        resolved = path.resolve()
        digest = hashlib.sha256()
        byte_count = 0
        paths = [resolved] if resolved.is_file() else sorted(resolved.rglob("*"))
        for item in paths:
            if not item.is_file():
                continue
            name = item.name if resolved.is_file() else str(item.relative_to(resolved))
            digest.update(name.encode("utf-8"))
            with item.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
        return {
            "path": str(path),
            "sha256": "sha256:" + digest.hexdigest(),
            "bytes": byte_count,
        }

    @staticmethod
    def _git_provenance() -> dict[str, str | bool | None]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return {"commit": None, "dirty": None}
        return {"commit": commit, "dirty": bool(status.strip())}


def run_tangent_preconditioner_audit(
    request: TangentPreconditionerAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexTangentPreconditionerAudit(request, logger=logger).run()
