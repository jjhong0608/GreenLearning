from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_tangent_preconditioner import (
    TANGENT_PRECONDITIONER_VARIANTS,
    TangentPreconditionerVariant,
)
from greenonet.complex_tangent_preconditioner_audit import (
    ComplexTangentPreconditionerAudit,
    TangentPreconditionerAuditRequest,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from greenonet.config import (
    BalanceProjectionConfig,
    SymmetricTangentGreenResponseProjectionConfig,
)
from greenonet.coupling_artifacts import load_coupling_artifact_configs
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class TangentCrossAxisAuditRequest:
    """Inputs for a matrix-free cross-axis Gram audit."""

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
    probe_count: int = 256
    probe_batch_size: int = 16
    probe_seed: int = 20260826
    confidence_z: float = 1.96
    metric_eps: float = 1.0e-30
    operator_equivalence_tol: float = 1.0e-10
    preconditioner_variant: TangentPreconditionerVariant = "separable"
    posthoc_tangent_override: bool = False
    posthoc_eta: float = 0.01
    posthoc_line_search_relative_eps: float = 1.0e-12
    posthoc_relative_lambda: float = 0.01
    posthoc_denominator_relative_eps: float = 1.0e-12
    posthoc_cross_axis_relative_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("probe_count", self.probe_count),
            ("probe_batch_size", self.probe_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.probe_seed, bool)
            or not isinstance(self.probe_seed, int)
            or self.probe_seed < 0
        ):
            raise ValueError("probe_seed must be a non-negative integer.")
        for name, numeric_value in (
            ("confidence_z", self.confidence_z),
            ("metric_eps", self.metric_eps),
            ("operator_equivalence_tol", self.operator_equivalence_tol),
        ):
            if (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, (int, float))
                or not math.isfinite(float(numeric_value))
                or float(numeric_value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if self.preconditioner_variant not in TANGENT_PRECONDITIONER_VARIANTS:
            raise ValueError(
                "preconditioner_variant must be one of "
                f"{TANGENT_PRECONDITIONER_VARIANTS}."
            )
        self._preconditioner_request()

    def _preconditioner_request(self) -> TangentPreconditionerAuditRequest:
        return TangentPreconditionerAuditRequest(
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
            metric_eps=self.metric_eps,
            operator_equivalence_tol=self.operator_equivalence_tol,
            save_generated_data=False,
            posthoc_tangent_override=self.posthoc_tangent_override,
            posthoc_eta=self.posthoc_eta,
            posthoc_line_search_relative_eps=(self.posthoc_line_search_relative_eps),
            posthoc_relative_lambda=self.posthoc_relative_lambda,
            posthoc_denominator_relative_eps=(self.posthoc_denominator_relative_eps),
            posthoc_cross_axis_relative_eps=self.posthoc_cross_axis_relative_eps,
        )


@dataclass(frozen=True)
class MatrixFreeCrossAxisActions:
    """Matrix-free actions of cross, self, and full tangent Gram operators."""

    cross: torch.Tensor
    cross_transpose: torch.Tensor
    symmetric_cross: torch.Tensor
    self_hessian: torch.Tensor
    tangent_hessian: torch.Tensor


class MatrixFreeCrossAxisCouplingAnalyzer:
    """Measure implicit cross-axis coupling without assembling global matrices."""

    _OPERATOR_NAMES = (
        "cross",
        "symmetric_cross",
        "self_hessian",
        "tangent_hessian",
    )

    def __init__(
        self,
        context: SymmetricTangentGreenResponseContext,
        *,
        metric_eps: float,
        confidence_z: float,
    ) -> None:
        self.context = context
        self.metric_eps = float(metric_eps)
        self.confidence_z = float(confidence_z)

    @torch.no_grad()
    def actions(self, values: torch.Tensor) -> MatrixFreeCrossAxisActions:
        """Apply C, C^T, A_x+A_y, and A without materializing them."""

        self.context.validate_for(values)
        operator = self.context.response_operator
        mass = self.context.point_mass
        x_response = operator.x.forward(values)
        y_response = operator.y.forward(values)
        cross = operator.x.adjoint(mass * y_response)
        cross_transpose = operator.y.adjoint(mass * x_response)
        self_hessian = operator.x.adjoint(mass * x_response) + operator.y.adjoint(
            mass * y_response
        )
        symmetric_cross = cross + cross_transpose
        return MatrixFreeCrossAxisActions(
            cross=cross,
            cross_transpose=cross_transpose,
            symmetric_cross=symmetric_cross,
            self_hessian=self_hessian,
            tangent_hessian=self_hessian + symmetric_cross,
        )

    @torch.no_grad()
    def estimate_frobenius(
        self,
        *,
        probe_count: int,
        probe_batch_size: int,
        probe_seed: int,
    ) -> tuple[dict[str, Any], list[dict[str, float | int | str]]]:
        """Estimate implicit Frobenius norms with deterministic Rademacher probes."""

        generator = torch.Generator(device="cpu")
        generator.manual_seed(probe_seed)
        rows: list[dict[str, float | int | str]] = []
        point_count = self.context.num_points
        diagonal = self._operator_diagonals()
        probe_offset = 0
        while probe_offset < probe_count:
            count = min(probe_batch_size, probe_count - probe_offset)
            random_bits = torch.randint(
                0,
                2,
                (count, point_count),
                dtype=torch.int8,
                generator=generator,
                device="cpu",
            )
            probes = (2 * random_bits.to(torch.int16) - 1).to(
                device=self.context.denominator.device,
                dtype=self.context.denominator.dtype,
            )
            actions = self.actions(probes)
            tensors = {
                "cross": actions.cross,
                "symmetric_cross": actions.symmetric_cross,
                "self_hessian": actions.self_hessian,
                "tangent_hessian": actions.tangent_hessian,
            }
            for local_index in range(count):
                row: dict[str, float | int | str] = {
                    "probe_index": probe_offset + local_index,
                }
                for name, result in tensors.items():
                    full = result[local_index]
                    diagonal_action = diagonal[name] * probes[local_index]
                    off_diagonal = full - diagonal_action
                    row[f"{name}_action_norm_squared"] = float(
                        full.square().sum().item()
                    )
                    row[f"{name}_off_diagonal_action_norm_squared"] = float(
                        off_diagonal.square().sum().item()
                    )
                rows.append(row)
            probe_offset += count

        estimates: dict[str, Any] = {
            name: self._operator_estimate(name=name, rows=rows, diagonal=diagonal[name])
            for name in self._OPERATOR_NAMES
        }
        cross_frob = float(estimates["cross"]["frobenius_norm_estimate"])
        symmetric_cross_frob = float(
            estimates["symmetric_cross"]["frobenius_norm_estimate"]
        )
        tangent_frob = float(estimates["tangent_hessian"]["frobenius_norm_estimate"])
        estimates["ratios"] = {
            "cross_to_tangent_frobenius": cross_frob
            / max(tangent_frob, self.metric_eps),
            "symmetric_cross_to_tangent_frobenius": symmetric_cross_frob
            / max(tangent_frob, self.metric_eps),
        }
        estimates["probe_contract"] = {
            "distribution": "independent_rademacher",
            "count": probe_count,
            "batch_size": probe_batch_size,
            "seed": probe_seed,
            "confidence_z": self.confidence_z,
        }
        return estimates, rows

    @torch.no_grad()
    def sample_direction_rows(
        self,
        *,
        gradients: torch.Tensor,
        denominators: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> list[dict[str, float | int | str]]:
        """Measure cross-axis action on actual z_b=D^{-1}g_b directions."""

        self.context.validate_for(gradients)
        if denominators.shape != (self.context.num_points,):
            raise ValueError("denominators must have shape (P,).")
        if torch.any(denominators <= 0.0) or not torch.all(
            torch.isfinite(denominators)
        ):
            raise ValueError("denominators must be finite and positive.")
        if sample_indices.shape != (gradients.shape[0],):
            raise ValueError("sample_indices must match the gradient batch size.")

        directions = gradients / denominators.unsqueeze(0)
        actions = self.actions(directions)
        cross_diagonal = (
            2.0 * self.context.cross_axis_inner_product.unsqueeze(0) * directions
        )
        cross_off_diagonal = actions.symmetric_cross - cross_diagonal
        rows: list[dict[str, float | int | str]] = []
        for index, sample_id in enumerate(sample_indices.tolist()):
            tangent_norm = self._norm(actions.tangent_hessian[index])
            cross_norm = self._norm(actions.symmetric_cross[index])
            cross_off_diagonal_norm = self._norm(cross_off_diagonal[index])
            cross_diagonal_norm = self._norm(cross_diagonal[index])
            self_norm = self._norm(actions.self_hessian[index])
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "gradient_norm": self._norm(gradients[index]),
                    "preconditioned_direction_norm": self._norm(directions[index]),
                    "tangent_hessian_action_norm": tangent_norm,
                    "symmetric_cross_action_norm": cross_norm,
                    "self_hessian_action_norm": self_norm,
                    "cross_diagonal_action_norm": cross_diagonal_norm,
                    "cross_off_diagonal_action_norm": cross_off_diagonal_norm,
                    "cross_to_tangent_action_ratio": self._ratio(
                        cross_norm, tangent_norm
                    ),
                    "self_to_tangent_action_ratio": self._ratio(
                        self_norm, tangent_norm
                    ),
                    "cross_diagonal_to_cross_action_ratio": self._ratio(
                        cross_diagonal_norm, cross_norm
                    ),
                    "cross_off_diagonal_to_cross_action_ratio": self._ratio(
                        cross_off_diagonal_norm, cross_norm
                    ),
                    "cross_off_diagonal_to_tangent_action_ratio": self._ratio(
                        cross_off_diagonal_norm, tangent_norm
                    ),
                    "cross_tangent_cosine": self._cosine(
                        actions.symmetric_cross[index],
                        actions.tangent_hessian[index],
                    ),
                    "cross_diagonal_full_cosine": self._cosine(
                        cross_diagonal[index],
                        actions.symmetric_cross[index],
                    ),
                }
            )
        return rows

    def _operator_diagonals(self) -> dict[str, torch.Tensor]:
        a = self.context.gamma_x_squared
        b = self.context.gamma_y_squared
        c = self.context.cross_axis_inner_product
        return {
            "cross": c,
            "symmetric_cross": 2.0 * c,
            "self_hessian": a + b,
            "tangent_hessian": a + b + 2.0 * c,
        }

    def _operator_estimate(
        self,
        *,
        name: str,
        rows: Sequence[dict[str, float | int | str]],
        diagonal: torch.Tensor,
    ) -> dict[str, float | int]:
        full_values = np.asarray(
            [float(row[f"{name}_action_norm_squared"]) for row in rows],
            dtype=np.float64,
        )
        off_values = np.asarray(
            [float(row[f"{name}_off_diagonal_action_norm_squared"]) for row in rows],
            dtype=np.float64,
        )
        full = self._mean_confidence_interval(full_values)
        off = self._mean_confidence_interval(off_values)
        diagonal_energy = float(diagonal.square().sum().item())
        full_mean = float(full["mean"])
        off_mean = float(off["mean"])
        formula_fraction_raw = 1.0 - diagonal_energy / max(full_mean, self.metric_eps)
        direct_fraction = off_mean / max(full_mean, self.metric_eps)
        expected_off = max(0.0, full_mean - diagonal_energy)
        return {
            "probe_count": len(rows),
            "frobenius_norm_squared_estimate": full_mean,
            "frobenius_norm_estimate": math.sqrt(max(0.0, full_mean)),
            "frobenius_norm_squared_standard_error": float(full["standard_error"]),
            "frobenius_norm_squared_ci_low": float(full["ci_low"]),
            "frobenius_norm_squared_ci_high": float(full["ci_high"]),
            "diagonal_frobenius_norm_squared_exact": diagonal_energy,
            "off_diagonal_frobenius_norm_squared_estimate": off_mean,
            "off_diagonal_frobenius_norm_squared_standard_error": float(
                off["standard_error"]
            ),
            "off_diagonal_fraction": min(1.0, max(0.0, formula_fraction_raw)),
            "off_diagonal_fraction_raw": formula_fraction_raw,
            "off_diagonal_fraction_direct_probe": direct_fraction,
            "probe_decomposition_relative_difference": abs(off_mean - expected_off)
            / max(full_mean, self.metric_eps),
        }

    def _mean_confidence_interval(self, values: np.ndarray) -> dict[str, float]:
        mean = float(values.mean())
        if values.size == 1:
            standard_error = 0.0
        else:
            standard_error = float(values.std(ddof=1) / math.sqrt(values.size))
        radius = self.confidence_z * standard_error
        return {
            "mean": mean,
            "standard_error": standard_error,
            "ci_low": max(0.0, mean - radius),
            "ci_high": mean + radius,
        }

    @staticmethod
    def _norm(values: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(values).item())

    def _ratio(self, numerator: float, denominator: float) -> float:
        return numerator / max(denominator, self.metric_eps)

    def _cosine(self, left: torch.Tensor, right: torch.Tensor) -> float:
        numerator = float(torch.dot(left, right).item())
        denominator = self._norm(left) * self._norm(right)
        return numerator / max(denominator, self.metric_eps)


class ComplexTangentCrossAxisAudit(ComplexTangentPreconditionerAudit):
    """Run the operator-global and network-direction cross-axis audit."""

    _SAMPLE_METRICS = (
        "gradient_norm",
        "preconditioned_direction_norm",
        "tangent_hessian_action_norm",
        "symmetric_cross_action_norm",
        "self_hessian_action_norm",
        "cross_diagonal_action_norm",
        "cross_off_diagonal_action_norm",
        "cross_to_tangent_action_ratio",
        "self_to_tangent_action_ratio",
        "cross_diagonal_to_cross_action_ratio",
        "cross_off_diagonal_to_cross_action_ratio",
        "cross_off_diagonal_to_tangent_action_ratio",
        "cross_tangent_cosine",
        "cross_diagonal_full_cosine",
    )

    def __init__(
        self,
        request: TangentCrossAxisAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cross_axis_request = request
        super().__init__(request._preconditioner_request(), logger=logger)

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        request = self.cross_axis_request
        request.outdir.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(request.config)
        if self._configs.dataset.geometry_mode != "complex":
            raise ValueError("Cross-axis audit requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        self._training_projection = projection
        if request.posthoc_tangent_override:
            if not projection.enabled or projection.mode not in {
                "physical_symmetric",
                "symmetric_tangent_green_response",
            }:
                raise ValueError(
                    "Post-hoc tangent override requires physical_symmetric or "
                    "symmetric_tangent_green_response training projection."
                )
            tangent = self.preconditioner_request.posthoc_tangent_config()
        else:
            if projection.mode != "symmetric_tangent_green_response":
                raise ValueError(
                    "Cross-axis audit requires symmetric_tangent_green_response or "
                    "an explicit posthoc_tangent_override=true."
                )
            tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
                projection.symmetric_tangent_green_response
            )
        self._audit_tangent_config = tangent

        geometry_path = request.geometry or self._configs.dataset.geometry_path
        test_path = request.test_path or self._configs.dataset.test_path
        coefficient_path = (
            request.coefficients or self._configs.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")
        for checkpoint in (request.coupling_checkpoint, request.green_checkpoint):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

        self._device = torch.device(
            request.device or self._configs.coupling_training.device
        )
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=self._configs.dataset.dtype,
        )
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            load_coefficient_functions(coefficient_path),
            branch_input_dim=self._configs.coupling_model.branch_input_dim,
            dtype=self._configs.dataset.dtype,
            coefficient_terms=self._configs.coupling_model.coefficient_terms,
            integration_rule=self._configs.coupling_training.integration_rule,
        )
        if len(dataset) == 0:
            raise ValueError("The test dataset is empty.")
        self._load_models()
        loader = DataLoader(
            dataset,
            batch_size=min(request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )

        analyzer: MatrixFreeCrossAxisCouplingAnalyzer | None = None
        selected_context: SymmetricTangentGreenResponseContext | None = None
        probe_summary: dict[str, Any] | None = None
        probe_rows: list[dict[str, float | int | str]] = []
        sample_rows: list[dict[str, float | int | str]] = []
        raw_digest = hashlib.sha256()
        probe_seconds = 0.0
        sample_seconds = 0.0
        for batch in loader:
            batch = batch.to(self._device)
            self._initialize_context(batch)
            if analyzer is None:
                selected_context = self.tangent_context.with_preconditioner_variant(
                    request.preconditioner_variant
                )
                analyzer = MatrixFreeCrossAxisCouplingAnalyzer(
                    selected_context,
                    metric_eps=request.metric_eps,
                    confidence_z=request.confidence_z,
                )
                probe_started = time.perf_counter()
                probe_summary, probe_rows = analyzer.estimate_frobenius(
                    probe_count=request.probe_count,
                    probe_batch_size=request.probe_batch_size,
                    probe_seed=request.probe_seed,
                )
                probe_seconds = time.perf_counter() - probe_started
            prepared = self._prepare_batch(batch)
            self._update_raw_digest(raw_digest, batch, prepared)
            sample_started = time.perf_counter()
            if selected_context is None:
                raise RuntimeError("Selected preconditioner context is unavailable.")
            rows = analyzer.sample_direction_rows(
                gradients=prepared.gradient,
                denominators=selected_context.denominator,
                sample_indices=batch.sample_indices,
            )
            sample_seconds += time.perf_counter() - sample_started
            for row in rows:
                row["preconditioner_variant"] = request.preconditioner_variant
                row["training_projection_mode"] = projection.mode
                row["posthoc_tangent_override"] = int(request.posthoc_tangent_override)
            sample_rows.extend(rows)

        if analyzer is None or selected_context is None or probe_summary is None:
            raise RuntimeError(
                "Cross-axis audit did not initialize its response context."
            )

        aggregate = self._aggregate_sample_rows(sample_rows)
        metrics_dir = request.outdir / "metrics"
        self._write_csv(metrics_dir / "probe_cross_axis_coupling.csv", probe_rows)
        self._write_csv(metrics_dir / "per_sample_cross_axis_coupling.csv", sample_rows)
        self._write_csv(metrics_dir / "aggregate_cross_axis_coupling.csv", [aggregate])
        self._write_fields(selected_context)
        figure_path = self._write_figure(probe_summary, sample_rows)

        context_telemetry = (
            {}
            if self._tangent_context_cache is None
            else self._tangent_context_cache.telemetry()
        )
        summary: dict[str, Any] = {
            "diagnostic": "matrix_free_cross_axis_gram_and_direction_audit",
            "tangent_subspace_dimension_provenance": self._configs.raw.get(
                "tangent_subspace_dimension_provenance"
            ),
            "sample_count": len(sample_rows),
            "preconditioner_variant_for_z": request.preconditioner_variant,
            "formulas": {
                "cross_gram": "C=H_x^T M_Omega H_y",
                "symmetric_cross": "B=C+C^T",
                "tangent_hessian": "A=(H_x+H_y)^T M_Omega (H_x+H_y)",
                "probe_identity": "E[||Cz||_2^2]=||C||_F^2",
                "off_diagonal_fraction": ("1-||diag(C)||_2^2/(||C||_F^2+eps)"),
                "sample_direction": "z_b=D^{-1}g_b",
                "sample_cross_ratio": "||(C+C^T)z_b||_2/(||Az_b||_2+eps)",
            },
            "operator_global": probe_summary,
            "sample_direction": aggregate,
            "projection_provenance": {
                "training_mode": projection.mode,
                "audit_mode": "symmetric_tangent_green_response",
                "explicit_posthoc_override": request.posthoc_tangent_override,
                "audit_tangent_config": asdict(tangent),
            },
            "raw_output": {
                "sha256": "sha256:" + raw_digest.hexdigest(),
                "computed_once_per_batch": True,
            },
            "tangent_context": {
                **context_telemetry,
                "response_operator_instance_count": 1,
            },
            "matrix_policy": {
                "global_matrix_materialized": False,
                "global_gram_materialized": False,
                "global_linear_solve": False,
                "operator_access": "segment_local_forward_and_adjoint_only",
            },
            "reference_policy": {
                "sol_used": False,
                "phi_used": False,
                "psi_used": False,
                "rhs_used_to_form_symmetric_proposal": True,
            },
            "numerical_checks": {
                "operator_equivalence_max_abs": self._operator_equivalence_max_abs,
                "operator_equivalence_tolerance": request.operator_equivalence_tol,
            },
            "runtime": {
                "probe_seconds": probe_seconds,
                "sample_direction_seconds": sample_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            "provenance": {
                "config": self._path_provenance(request.config),
                "coupling_checkpoint": self._path_provenance(
                    request.coupling_checkpoint
                ),
                "green_checkpoint": self._path_provenance(request.green_checkpoint),
                "geometry": self._path_provenance(geometry_path),
                "test_data": self._path_provenance(test_path),
                "coefficients": self._path_provenance(coefficient_path),
                "dtype": str(self._configs.dataset.dtype).replace("torch.", ""),
                "device": str(self._device),
                "git": self._git_provenance(),
            },
            "artifacts": {
                "probe_csv": "metrics/probe_cross_axis_coupling.csv",
                "per_sample_csv": "metrics/per_sample_cross_axis_coupling.csv",
                "aggregate_csv": "metrics/aggregate_cross_axis_coupling.csv",
                "fields_npz": "data/cross_axis_operator_fields.npz",
                "figure": str(figure_path.relative_to(request.outdir)),
                "report": "diagnosis_report.md",
            },
        }
        (request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        self._write_cross_axis_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Cross-axis audit complete: samples=%d probes=%d R_off,C=%.6f",
                len(sample_rows),
                request.probe_count,
                float(probe_summary["cross"]["off_diagonal_fraction"]),
            )
        return summary

    def _aggregate_sample_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, float | int | str]:
        aggregate: dict[str, float | int | str] = {
            "sample_count": len(rows),
            "preconditioner_variant": self.cross_axis_request.preconditioner_variant,
        }
        for metric in self._SAMPLE_METRICS:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            aggregate[f"{metric}_mean"] = float(values.mean())
            aggregate[f"{metric}_std"] = float(values.std(ddof=0))
            aggregate[f"{metric}_median"] = float(np.median(values))
            aggregate[f"{metric}_p05"] = float(np.quantile(values, 0.05))
            aggregate[f"{metric}_p95"] = float(np.quantile(values, 0.95))
            aggregate[f"{metric}_min"] = float(values.min())
            aggregate[f"{metric}_max"] = float(values.max())
        return aggregate

    def _write_fields(
        self,
        context: SymmetricTangentGreenResponseContext,
    ) -> Path:
        path = (
            self.cross_axis_request.outdir / "data" / "cross_axis_operator_fields.npz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        a = context.gamma_x_squared.detach().cpu().numpy()
        b = context.gamma_y_squared.detach().cpu().numpy()
        c = context.cross_axis_inner_product.detach().cpu().numpy()
        np.savez(
            path,
            gamma_x_squared=a,
            gamma_y_squared=b,
            cross_axis_diagonal=c,
            symmetric_cross_diagonal=2.0 * c,
            self_hessian_diagonal=a + b,
            tangent_hessian_diagonal=a + b + 2.0 * c,
            selected_denominator=context.denominator.detach().cpu().numpy(),
            valid_x=self.geometry.coords_valid[:, 0].detach().cpu().numpy(),
            valid_y=self.geometry.coords_valid[:, 1].detach().cpu().numpy(),
        )
        return path

    def _write_figure(
        self,
        probe_summary: dict[str, Any],
        sample_rows: Sequence[dict[str, float | int | str]],
    ) -> Path:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Operator off-diagonal Frobenius fraction",
                "Operator Frobenius norm relative to A",
                "Actual z: symmetric cross / full tangent action",
                "Actual z: cross off-diagonal / cross action",
            ),
        )
        names = ("cross", "symmetric_cross", "self_hessian", "tangent_hessian")
        labels = ("C", "C+C^T", "A_x+A_y", "A")
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[
                    float(probe_summary[name]["off_diagonal_fraction"])
                    for name in names
                ],
                marker_color="#0f766e",
                name="off-diagonal fraction",
            ),
            row=1,
            col=1,
        )
        tangent_norm = float(
            probe_summary["tangent_hessian"]["frobenius_norm_estimate"]
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[
                    float(probe_summary[name]["frobenius_norm_estimate"])
                    / max(tangent_norm, self.cross_axis_request.metric_eps)
                    for name in names
                ],
                marker_color="#2563eb",
                name="Frobenius / ||A||_F",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Histogram(
                x=[float(row["cross_to_tangent_action_ratio"]) for row in sample_rows],
                nbinsx=20,
                marker_color="#dc2626",
                name="cross / tangent",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=[
                    float(row["cross_off_diagonal_to_cross_action_ratio"])
                    for row in sample_rows
                ],
                nbinsx=20,
                marker_color="#9333ea",
                name="off-diagonal / cross",
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="fraction", row=1, col=1)
        fig.update_yaxes(title_text="relative norm", row=1, col=2)
        fig.update_xaxes(title_text="ratio", row=2, col=1)
        fig.update_xaxes(title_text="ratio", row=2, col=2)
        fig.update_layout(
            title="Matrix-free cross-axis coupling audit",
            template=self.cross_axis_request.theme,
            width=1200,
            height=820,
            showlegend=False,
        )
        path = (
            self.cross_axis_request.outdir
            / "figures"
            / "cross_axis_coupling_audit.html"
        )
        save_plotly_figure(fig, path.with_suffix(""), logger=self.logger)
        return path

    def _write_cross_axis_report(self, summary: dict[str, Any]) -> Path:
        operator = summary["operator_global"]
        sample = summary["sample_direction"]
        cross_off = float(operator["cross"]["off_diagonal_fraction"])
        symmetric_cross_off = float(
            operator["symmetric_cross"]["off_diagonal_fraction"]
        )
        tangent_off = float(operator["tangent_hessian"]["off_diagonal_fraction"])
        sample_cross = float(sample["cross_to_tangent_action_ratio_mean"])
        sample_cross_off = float(
            sample["cross_off_diagonal_to_cross_action_ratio_mean"]
        )
        sample_diag = float(sample["cross_diagonal_to_cross_action_ratio_mean"])
        lines = [
            "# Matrix-free cross-axis coupling audit",
            "",
            "## Question",
            "",
            (
                "Does the small difference between separable and exact-diagonal "
                "preconditioners mean that cross-axis coupling is globally small, "
                "or only that its diagonal is small?"
            ),
            "",
            "## Matrix-free contract",
            "",
            "- `C = H_x^T M_Omega H_y` is never assembled.",
            "- Rademacher probes estimate `||C||_F^2` through `E||Cz||^2`.",
            "- No global Gram matrix or linear solve is used.",
            "- Reference `sol/phi/psi` fields are not used.",
            "",
            "## Operator-global result",
            "",
            f"- `R_off,C = {cross_off:.6f}`.",
            f"- `R_off,C+C^T = {symmetric_cross_off:.6f}`.",
            f"- `R_off,A = {tangent_off:.6f}`.",
            (
                "- `||C+C^T||_F / ||A||_F = "
                f"{float(operator['ratios']['symmetric_cross_to_tangent_frobenius']):.6f}`."
            ),
            "",
            "## Network-direction result",
            "",
            (f"- Mean `||(C+C^T)z_b||/||Az_b|| = {sample_cross:.6f}`."),
            (
                "- Mean `||(C+C^T-2diag(c))z_b||/||(C+C^T)z_b|| = "
                f"{sample_cross_off:.6f}`."
            ),
            (f"- Mean `||2diag(c)z_b||/||(C+C^T)z_b|| = {sample_diag:.6f}`."),
            "",
            "## Interpretation",
            "",
            (
                "- `exact_diagonal` changes only `2 diag(C)`. A small separable-versus-"
                "exact difference therefore establishes only that this diagonal term "
                "is weak relative to the damped Jacobi denominator."
            ),
            (
                "- A large `R_off,C` means nonlocal cross-axis coupling exists even "
                "when `rho_i` and the exact-diagonal adjustment are small."
            ),
            (
                "- The sample-direction ratios determine whether that nonlocal part is "
                "actually encountered by frozen CouplingNet tangent directions."
            ),
            (
                "- K>=2 matrix-free Krylov updates can use the full `A` action and thus "
                "respond to off-diagonal structure without placing it in the Jacobi "
                "preconditioner."
            ),
        ]
        path = self.cross_axis_request.outdir / "diagnosis_report.md"
        path.write_text("\n".join(lines) + "\n")
        return path


def run_tangent_cross_axis_audit(
    request: TangentCrossAxisAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexTangentCrossAxisAudit(request, logger=logger).run()
