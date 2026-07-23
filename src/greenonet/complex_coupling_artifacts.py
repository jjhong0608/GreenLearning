from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader

from greenonet.coefficients import CoefficientFunctions, load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexLengthJumpBalanceConfig,
    ComplexPreProjectionFusionConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingBestEnergyCheckpointConfig,
    CouplingBestPhysicsCheckpointConfig,
    CouplingCoefficientTermsConfig,
)
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class ComplexSelectedSample:
    sample_id: int
    file_stem: str
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ComplexCoefficientFields:
    """Physical coefficient fields and deterministic quiver metadata."""

    coords_valid: np.ndarray
    a: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    b_magnitude: np.ndarray
    c: np.ndarray
    quiver_indices: np.ndarray
    quiver_stride: int
    quiver_scale: float

    def npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "coords_valid": self.coords_valid,
            "a": self.a,
            "bx": self.bx,
            "by": self.by,
            "b_magnitude": self.b_magnitude,
            "c": self.c,
            "quiver_indices": self.quiver_indices,
        }


class ComplexCoefficientArtifactMixin:
    """Create run-level physical coefficient artifacts for complex geometry."""

    COEFFICIENT_ZERO_TOLERANCE: ClassVar[float] = 1e-12
    QUIVER_ARROW_GRID_FRACTION: ClassVar[float] = 0.75

    request: CouplingArtifactRequest
    logger: logging.Logger | None

    def _evaluate_coefficient_fields(
        self,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
    ) -> ComplexCoefficientFields:
        coords = geometry.coords_valid
        x = coords[:, 0]
        y = coords[:, 1]
        with torch.no_grad():
            a = self._evaluate_coefficient_function(coeffs.a_fun, x, y, "a")
            bx = self._evaluate_coefficient_function(coeffs.bx_fun, x, y, "bx")
            by = self._evaluate_coefficient_function(coeffs.by_fun, x, y, "by")
            c = self._evaluate_coefficient_function(coeffs.c_fun, x, y, "c")

        coords_numpy = coords.detach().cpu().numpy()
        a_numpy = a.detach().cpu().numpy()
        bx_numpy = bx.detach().cpu().numpy()
        by_numpy = by.detach().cpu().numpy()
        c_numpy = c.detach().cpu().numpy()
        magnitude = np.sqrt(np.square(bx_numpy) + np.square(by_numpy))
        quiver_indices, stride = self._select_quiver_indices(
            geometry,
            self.request.coefficient_vector_max_points,
        )
        if magnitude.size:
            max_index = int(np.argmax(magnitude))
            if max_index not in quiver_indices:
                if quiver_indices.size < self.request.coefficient_vector_max_points:
                    quiver_indices = np.append(quiver_indices, max_index)
                else:
                    quiver_indices = quiver_indices.copy()
                    quiver_indices[-1] = max_index
                quiver_indices = np.unique(quiver_indices).astype(np.int64)

        max_magnitude = float(np.max(magnitude)) if magnitude.size else 0.0
        grid_spacing = stride * min(float(geometry.hx), float(geometry.hy))
        quiver_scale = (
            self.QUIVER_ARROW_GRID_FRACTION * grid_spacing / max_magnitude
            if max_magnitude > self.COEFFICIENT_ZERO_TOLERANCE
            else 0.0
        )
        return ComplexCoefficientFields(
            coords_valid=coords_numpy,
            a=a_numpy,
            bx=bx_numpy,
            by=by_numpy,
            b_magnitude=magnitude,
            c=c_numpy,
            quiver_indices=quiver_indices,
            quiver_stride=stride,
            quiver_scale=quiver_scale,
        )

    @staticmethod
    def _evaluate_coefficient_function(
        function: Any,
        x: torch.Tensor,
        y: torch.Tensor,
        field_name: str,
    ) -> torch.Tensor:
        raw = function(x, y)
        values = torch.as_tensor(raw, dtype=x.dtype, device=x.device)
        try:
            values = torch.broadcast_to(values, x.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Physical coefficient '{field_name}' returned shape "
                f"{tuple(values.shape)}, which cannot broadcast to {tuple(x.shape)}."
            ) from exc
        if not torch.all(torch.isfinite(values)):
            raise ValueError(
                f"Physical coefficient '{field_name}' contains non-finite values."
            )
        return values

    @staticmethod
    def _select_quiver_indices(
        geometry: ComplexGeometryMetadata,
        max_points: int,
    ) -> tuple[np.ndarray, int]:
        x_indices = geometry.valid_grid_x_index.detach().cpu().numpy()
        y_indices = geometry.valid_grid_y_index.detach().cpu().numpy()
        point_count = int(x_indices.size)
        if point_count == 0:
            raise ValueError("Complex coefficient visualization requires valid points.")
        if point_count <= max_points:
            return np.arange(point_count, dtype=np.int64), 1

        x_origin = int(np.min(x_indices))
        y_origin = int(np.min(y_indices))
        max_span = max(
            int(np.max(x_indices) - x_origin),
            int(np.max(y_indices) - y_origin),
        )
        for stride in range(2, max_span + 2):
            mask = ((x_indices - x_origin) % stride == 0) & (
                (y_indices - y_origin) % stride == 0
            )
            selected = np.flatnonzero(mask).astype(np.int64)
            if 0 < selected.size <= max_points:
                return selected, stride

        fallback_step = max(1, int(np.ceil(point_count / max_points)))
        selected = np.arange(0, point_count, fallback_step, dtype=np.int64)[:max_points]
        return selected, max(1, int(np.ceil(np.sqrt(point_count / max_points))))

    def _coefficient_figure_fields(
        self,
        fields: ComplexCoefficientFields,
        terms: CouplingCoefficientTermsConfig,
    ) -> tuple[str, ...]:
        names = ["diffusion_a"]
        if self._is_physical_nonzero(fields.c) or terms.reaction:
            names.append("reaction_c")
        if self._is_physical_nonzero(fields.b_magnitude) or terms.convection:
            names.extend(
                (
                    "convection_bx",
                    "convection_by",
                    "convection_magnitude",
                    "convection_vector",
                )
            )
        return tuple(names)

    def _coefficient_field_statistics(
        self,
        fields: ComplexCoefficientFields,
        terms: CouplingCoefficientTermsConfig,
        figure_fields: tuple[str, ...],
    ) -> dict[str, dict[str, float | bool]]:
        figures = set(figure_fields)
        specifications = {
            "a": (fields.a, terms.diffusion, "diffusion_a"),
            "bx": (fields.bx, terms.convection, "convection_bx"),
            "by": (fields.by, terms.convection, "convection_by"),
            "b_magnitude": (
                fields.b_magnitude,
                terms.convection,
                "convection_magnitude",
            ),
            "c": (fields.c, terms.reaction, "reaction_c"),
        }
        statistics: dict[str, dict[str, float | bool]] = {}
        for name, (values, branch_enabled, figure_name) in specifications.items():
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            max_abs = float(np.max(np.abs(values)))
            constant_tolerance = self.COEFFICIENT_ZERO_TOLERANCE * max(1.0, max_abs)
            statistics[name] = {
                "min": minimum,
                "max": maximum,
                "mean": float(np.mean(values)),
                "physical_nonzero": max_abs > self.COEFFICIENT_ZERO_TOLERANCE,
                "constant": maximum - minimum <= constant_tolerance,
                "branch_enabled": bool(branch_enabled),
                "figure_exported": figure_name in figures,
            }
        return statistics

    def _is_physical_nonzero(self, values: np.ndarray) -> bool:
        return bool(
            values.size
            and float(np.max(np.abs(values))) > self.COEFFICIENT_ZERO_TOLERANCE
        )

    def _write_coefficient_npz(self, fields: ComplexCoefficientFields) -> None:
        if not self.request.save_generated_data:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / "coefficient_fields.npz"
        np.savez(output_path, **fields.npz_payload())  # type: ignore[arg-type]

    def _write_coefficient_figures(
        self,
        fields: ComplexCoefficientFields,
        terms: CouplingCoefficientTermsConfig,
        theme: str,
    ) -> tuple[list[str], tuple[str, ...]]:
        figure_fields = self._coefficient_figure_fields(fields, terms)
        paths: list[str] = []
        scalar_specs = {
            "diffusion_a": (fields.a, "Diffusion coefficient a(x, y)", "a", False),
            "reaction_c": (
                fields.c,
                "Reaction coefficient c(x, y)",
                "c",
                bool(np.min(fields.c) < 0.0 < np.max(fields.c)),
            ),
            "convection_bx": (
                fields.bx,
                "Convection coefficient b_x(x, y)",
                "b_x",
                True,
            ),
            "convection_by": (
                fields.by,
                "Convection coefficient b_y(x, y)",
                "b_y",
                True,
            ),
            "convection_magnitude": (
                fields.b_magnitude,
                "Convection magnitude |b(x, y)|",
                "|b|",
                False,
            ),
        }
        for name in figure_fields:
            if name == "convection_vector":
                figure = self._convection_vector_figure(fields, theme)
            else:
                values, title, label, signed = scalar_specs[name]
                figure = self._coefficient_scalar_figure(
                    title=title,
                    label=label,
                    coords=fields.coords_valid,
                    values=values,
                    theme=theme,
                    signed=signed,
                )
            base_path = self.request.outdir / "figures" / "coefficients" / name
            save_plotly_figure(figure, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths, figure_fields

    def _coefficient_scalar_figure(
        self,
        *,
        title: str,
        label: str,
        coords: np.ndarray,
        values: np.ndarray,
        theme: str,
        signed: bool,
    ) -> go.Figure:
        max_abs = float(np.max(np.abs(values))) if values.size else 0.0
        marker_range: dict[str, float] = {}
        if signed and max_abs > 0.0:
            marker_range = {"cmin": -max_abs, "cmax": max_abs}
        figure = go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                customdata=values,
                hovertemplate=(
                    "x=%{x:.6g}<br>y=%{y:.6g}<br>"
                    f"{label}=%{{customdata:.6g}}<extra></extra>"
                ),
                marker={
                    "color": values,
                    "colorscale": "RdBu" if signed else "Viridis",
                    "showscale": True,
                    "size": 6,
                    "colorbar": {
                        "title": label,
                        "exponentformat": "power",
                        "showexponent": "all",
                    },
                    **marker_range,
                },
            ),
            layout=self._coefficient_layout(title, theme),
        )
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        tolerance = self.COEFFICIENT_ZERO_TOLERANCE * max(1.0, max_abs)
        if maximum - minimum <= tolerance:
            figure.add_annotation(
                x=0.5,
                y=1.04,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"Constant field: {label}={minimum:.6g}",
            )
        return figure

    def _convection_vector_figure(
        self,
        fields: ComplexCoefficientFields,
        theme: str,
    ) -> go.Figure:
        customdata = np.column_stack((fields.bx, fields.by, fields.b_magnitude))
        figure = go.Figure(
            data=go.Scattergl(
                x=fields.coords_valid[:, 0],
                y=fields.coords_valid[:, 1],
                mode="markers",
                customdata=customdata,
                hovertemplate=(
                    "x=%{x:.6g}<br>y=%{y:.6g}<br>"
                    "b_x=%{customdata[0]:.6g}<br>"
                    "b_y=%{customdata[1]:.6g}<br>"
                    "|b|=%{customdata[2]:.6g}<extra></extra>"
                ),
                marker={
                    "color": fields.b_magnitude,
                    "colorscale": "Viridis",
                    "showscale": True,
                    "size": 5,
                    "opacity": 0.72,
                    "colorbar": {
                        "title": "|b|",
                        "exponentformat": "power",
                        "showexponent": "all",
                    },
                },
                name="|b|",
                showlegend=False,
            ),
            layout=self._coefficient_layout("Convection vector field b(x, y)", theme),
        )
        if fields.quiver_scale > 0.0:
            indices = fields.quiver_indices
            quiver = ff.create_quiver(
                fields.coords_valid[indices, 0],
                fields.coords_valid[indices, 1],
                fields.quiver_scale * fields.bx[indices],
                fields.quiver_scale * fields.by[indices],
                scale=1.0,
                arrow_scale=0.3,
                line={"color": "rgba(20, 20, 20, 1.0)", "width": 1.3},
                name="b direction",
            )
            for trace in quiver.data:
                figure.add_trace(
                    go.Scatter(
                        x=trace.x,
                        y=trace.y,
                        mode="lines",
                        line={"color": "rgba(255, 255, 255, 0.9)", "width": 3.2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                trace.update(hoverinfo="skip", showlegend=False)
                figure.add_trace(trace)
        else:
            figure.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                text="Zero convection field on valid points",
            )
        return figure

    @staticmethod
    def _coefficient_layout(title: str, theme: str) -> go.Layout:
        return go.Layout(
            template=theme,
            width=900,
            height=800,
            title=title,
            xaxis_title="x",
            yaxis_title="y",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
            margin={"t": 100},
        )


class ComplexCouplingArtifactExporter(ComplexCoefficientArtifactMixin):
    """Export complex-geometry CouplingNet metrics, raw archives, and scatter plots."""

    COLOR_RANGE_POLICY: ClassVar[str] = "shared_reference_prediction_groups"
    COLOR_RANGE_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "solution": ("sol", "u_pred", "u_phi", "u_psi"),
        "phi": ("target_phi", "phi"),
        "psi": ("target_psi", "psi"),
    }
    FIGURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "rhs",
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "u_split_mismatch",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
        "weak_residual_x",
        "weak_residual_y",
        "split_mass_relative_contribution",
        "base_physical_difference",
        "fused_physical_difference",
        "linear_difference_correction",
        "nonlinear_difference_correction",
        "blended_difference_correction",
    )
    SIGNED_FIGURE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "u_pred_error",
            "u_phi_error",
            "u_psi_error",
            "u_split_mismatch",
            "phi_error",
            "psi_error",
            "weak_residual_x",
            "weak_residual_y",
            "base_physical_difference",
            "fused_physical_difference",
            "linear_difference_correction",
            "nonlinear_difference_correction",
            "blended_difference_correction",
        }
    )
    FIGURE_TITLES: ClassVar[dict[str, str]] = {
        "rhs": "Source rhs",
        "sol": "Exact solution sol",
        "u_pred": "Predicted solution u_pred",
        "u_phi": "Reconstructed solution u_phi",
        "u_psi": "Reconstructed solution u_psi",
        "u_pred_error": "Signed error u_pred - sol",
        "u_phi_error": "Signed error u_phi - sol",
        "u_psi_error": "Signed error u_psi - sol",
        "u_split_mismatch": "Mismatch u_phi - u_psi",
        "phi": "Projected phi",
        "psi": "Projected psi",
        "target_phi": "Target phi",
        "target_psi": "Target psi",
        "phi_error": "Signed error phi - target_phi",
        "psi_error": "Signed error psi - target_psi",
        "weak_residual_x": "Directional weak residual Bx(u_pred) - phi",
        "weak_residual_y": "Directional weak residual By(u_pred) - psi",
        "split_mass_relative_contribution": (
            "Normalized pointwise split value mismatch"
        ),
        "base_physical_difference": "Base physical difference p - q",
        "fused_physical_difference": "Fused physical difference",
        "linear_difference_correction": "Linear physical difference correction",
        "nonlinear_difference_correction": ("Nonlinear physical difference correction"),
        "blended_difference_correction": "Blended physical difference correction",
    }

    def __init__(
        self,
        request: CouplingArtifactRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.request.outdir.mkdir(parents=True, exist_ok=True)

    def export(self) -> dict[str, Any]:
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError(
                "Complex artifact exporter requires geometry_mode='complex'."
            )
        if configs.dataset.geometry_path is None:
            raise ValueError("dataset.geometry_path is required for complex artifacts.")
        if configs.dataset.test_path is None:
            raise ValueError("dataset.test_path is required for complex artifacts.")

        device = torch.device(self.request.device or configs.coupling_training.device)
        coeff_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        coeffs = load_coefficient_functions(coeff_path)
        geometry = load_complex_geometry(
            configs.dataset.geometry_path,
            dtype=configs.dataset.dtype,
        )
        coefficient_fields = self._evaluate_coefficient_fields(geometry, coeffs)
        dataset = ComplexCouplingDataset(
            configs.dataset.test_path,
            geometry,
            coeffs,
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        coupling_model = self._load_complex_model(configs, device)
        green_model = self._load_green_model(configs, device)
        evaluator = ComplexCouplingEvaluator(
            model=coupling_model,
            green_model=green_model,
            config=configs.coupling_training,
            device=device,
            work_dir=self.request.outdir,
        )
        metric_rows = self._evaluate_rows(dataset, evaluator, configs)
        selected, roles, policy = self._select_sample_indices(metric_rows)
        selected_samples = self._evaluate_selected(dataset, evaluator, selected, device)
        self._write_metric_csv(metric_rows)
        self._write_selected_npz(selected_samples)
        self._write_coefficient_npz(coefficient_fields)
        figure_fields = self._figure_fields(selected_samples)
        sample_figure_paths = self._write_figures(
            selected_samples,
            self.request.theme,
        )
        coefficient_figure_paths, coefficient_figure_fields = (
            self._write_coefficient_figures(
                coefficient_fields,
                configs.coupling_model.coefficient_terms,
                self.request.theme,
            )
        )
        coefficient_statistics = self._coefficient_field_statistics(
            coefficient_fields,
            configs.coupling_model.coefficient_terms,
            coefficient_figure_fields,
        )
        aggregate = self._aggregate_metrics(metric_rows)
        axis_1d_trunk = Axis1DTrunkConfig.from_raw(configs.coupling_model.axis_1d_trunk)
        balance_projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        pre_projection_fusion = ComplexPreProjectionFusionConfig.from_raw(
            configs.coupling_model.pre_projection_fusion
        )
        length_jump = ComplexLengthJumpBalanceConfig.from_raw(
            configs.coupling_training.length_jump_balance
        )
        relative_split = ComplexRelativeSplitConsistencyConfig.from_raw(
            configs.coupling_training.relative_split_consistency
        )
        weak_closure = ComplexWeakOperatorClosureConfig.from_raw(
            configs.coupling_training.weak_operator_closure
        )
        best_energy = CouplingBestEnergyCheckpointConfig.from_raw(
            configs.coupling_training.best_energy_checkpoint
        )
        best_physics = CouplingBestPhysicsCheckpointConfig.from_raw(
            configs.coupling_training.best_physics_checkpoint
        )
        projection_formula = (
            "p=P_raw/Lx^2; q=Q_raw/Ly^2; d=p-q; "
            "phi=(rhs+d)/2; psi=(rhs-d)/2; "
            "Phi=Lx^2*phi; Psi=Ly^2*psi"
        )
        boundary_context = evaluator.boundary_energy_context(geometry)
        summary = {
            "geometry_mode": "complex",
            "device": str(device),
            "coefficients": None if coeff_path is None else str(coeff_path),
            "geometry_path": str(configs.dataset.geometry_path),
            "test_path": str(configs.dataset.test_path),
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "selected_sample_policy": policy,
            "plot_workers": self.request.plot_workers,
            "save_generated_data": self.request.save_generated_data,
            "aggregate_metrics": aggregate,
            "figure_count": (len(sample_figure_paths) + len(coefficient_figure_paths)),
            "figure_fields": list(figure_fields),
            "coefficient_figure_count": len(coefficient_figure_paths),
            "coefficient_figure_fields": list(coefficient_figure_fields),
            "coefficient_field_space": "physical",
            "coefficient_evaluation": "direct_at_coords_valid",
            "coefficient_zero_tolerance": self.COEFFICIENT_ZERO_TOLERANCE,
            "coefficient_field_statistics": coefficient_statistics,
            "coefficient_vector": {
                "max_points": self.request.coefficient_vector_max_points,
                "selected_points": int(coefficient_fields.quiver_indices.size),
                "stride": coefficient_fields.quiver_stride,
                "arrow_scale_factor": coefficient_fields.quiver_scale,
                "max_arrow_length": (
                    coefficient_fields.quiver_scale
                    * float(np.max(coefficient_fields.b_magnitude))
                ),
                "arrow_grid_fraction": self.QUIVER_ARROW_GRID_FRACTION,
                "background_points": int(coefficient_fields.coords_valid.shape[0]),
            },
            "coefficient_raw_archive": (
                "data/coefficient_fields.npz"
                if self.request.save_generated_data
                else None
            ),
            "error_convention": "signed_difference",
            "solution_prediction": "u_pred=0.5*(u_phi+u_psi)",
            "raw_output_space": "reference_response",
            "output_contract_version": ComplexCouplingNet.OUTPUT_CONTRACT_VERSION,
            "balance_projection": {
                "enabled": balance_projection.enabled,
                "mode": balance_projection.mode,
                "space": "physical_source",
                "formula": projection_formula,
                "constraint": "phi + psi = rhs",
                "raw_physical_difference_preserved": True,
                "pre_projection_scaling": "p=P_raw/Lx^2; q=Q_raw/Ly^2",
                "post_projection_pull_back": "Phi=Lx^2*phi; Psi=Ly^2*psi",
                "uses_reference_targets": False,
            },
            "pre_projection_fusion": {
                "enabled": pre_projection_fusion.enabled,
                "space": "physical_directional_source",
                "correction_mode": "antisymmetric_difference",
                "linear_input": ["normalized_difference", "normalized_rhs"],
                "nonlinear_input": [
                    "normalized_difference",
                    "normalized_rhs",
                    "x_local_t",
                    "y_local_t",
                    "log_x_length_over_reference",
                    "log_y_length_over_reference",
                    "log_x_over_y_length",
                    "length_balance_kappa",
                ],
                "nonlinear_hidden_dim": pre_projection_fusion.nonlinear_hidden_dim,
                "nonlinear_depth": pre_projection_fusion.nonlinear_depth,
                "gate_initial_value": pre_projection_fusion.gate_initial_value,
                "common_mode_preserved": True,
                "gate_value": (
                    None
                    if coupling_model.pre_projection_fusion is None
                    else float(
                        torch.sigmoid(
                            coupling_model.pre_projection_fusion.gate_logit.detach()
                        )
                        .cpu()
                        .item()
                    )
                ),
                "source_scale": "sqrt((A_x^2+A_y^2)/2)",
                "linear_bias": False,
                "identity_initialized": True,
                "uses_reference_targets": False,
            },
            "reconstruction_response_input": {
                "phi": "projected Phi is used directly",
                "psi": "projected Psi is used directly",
                "additional_length_scaling": False,
            },
            "canonical_boundary_energy": {
                "enabled": True,
                "definition": "endpoint_p1_edge",
                "formula": "a_i * r_i^2 * h_perp / d_endpoint",
                "coefficient_evaluation": "one_sided_nearest_valid_point",
                "endpoint_value": 0.0,
                "anchor_count": boundary_context.total_anchors,
                "x_anchor_count": boundary_context.x_anchor_count,
                "y_anchor_count": boundary_context.y_anchor_count,
                "covers_all_connected_segment_endpoints": True,
                "uses_reference_targets": False,
            },
            "length_jump_balance": {
                "enabled": length_jump.enabled,
                "log_sigma_jump_threshold": length_jump.log_sigma_jump_threshold,
                "transition_fraction": length_jump.transition_fraction,
                "eps": length_jump.eps,
                "checkpoint_metric": "loss_energy_length_balanced",
            },
            "relative_split_consistency": {
                "enabled": relative_split.enabled,
                "weight": relative_split.weight,
                "mass_weight": relative_split.mass_weight,
                "eps": relative_split.eps,
                "source_normalization": "physical_rhs_l2_squared",
                "domain_length_scale": "max_global_extent",
                "uses_reference_targets": False,
            },
            "weak_operator_closure": {
                "enabled": weak_closure.enabled,
                "weight": weak_closure.weight,
                "eps": weak_closure.eps,
                "trial_solution": "u_pred=0.5*(u_phi+u_psi)",
                "test_space": "directional_segment_p1_nodal",
                "coefficient_evaluation": "direct_at_physical_element_midpoints",
                "reaction_split": "c/2_per_direction",
                "uses_reference_targets": False,
            },
            "checkpoint_selection": {
                "best_energy": best_energy.enabled,
                "best_physics": best_physics.enabled,
                "reference_metric_used": False,
            },
            "reference_targets_used_for_training": False,
            "non_error_color_range_policy": self.COLOR_RANGE_POLICY,
            "non_error_color_range_groups": {
                name: list(fields) for name, fields in self.COLOR_RANGE_GROUPS.items()
            },
            "optional_flux_targets_exported": self._has_flux_target_artifacts(
                selected_samples
            ),
            "source_branch": {
                "enabled": True,
                "space": "physical",
                "normalization": "unit_coordinate_l2_of_physical_source",
                "amplitude": "A=sqrt(integral_0^1 f_phys(s(t))^2 dt)",
                "model_output_scaling": "primary_length_squared_times_physical_amplitude",
            },
            "coefficient_terms": {
                "diffusion": configs.coupling_model.coefficient_terms.diffusion,
                "convection": configs.coupling_model.coefficient_terms.convection,
                "reaction": configs.coupling_model.coefficient_terms.reaction,
            },
            "coefficient_branch_channel_order": self._coefficient_branch_channel_order(
                configs
            ),
            "coefficient_branch_convection": (
                "primary_transverse"
                if configs.coupling_model.coefficient_terms.convection
                else "disabled"
            ),
            "coefficient_branch_transverse_convection_scaling": (
                "primary_segment_length"
            ),
            "transverse_encoding": {
                "coordinate": "global_normalized_transverse",
                "num_frequencies": axis_1d_trunk.num_frequencies,
                "max_frequency": axis_1d_trunk.max_frequency,
            },
            "transverse_trunk": {
                "enabled": axis_1d_trunk.transverse_trunk.enabled,
                "fusion": axis_1d_trunk.transverse_trunk.fusion,
                "length_context": axis_1d_trunk.transverse_trunk.length_context,
                "features": [
                    "t_perpendicular",
                    "log(L_perpendicular/L_ref)",
                    "log(L_parallel/L_perpendicular)",
                    "kappa",
                ],
            },
        }
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        return summary

    @staticmethod
    def _coefficient_branch_channel_order(
        configs: CouplingArtifactConfigs,
    ) -> list[str]:
        terms = configs.coupling_model.coefficient_terms
        order: list[str] = []
        if terms.diffusion:
            order.append("a")
        if terms.convection:
            order.extend(("b_primary", "b_transverse"))
        if terms.reaction:
            order.append("c")
        return order

    def _load_complex_model(
        self,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> ComplexCouplingNet:
        model = ComplexCouplingNet(configs.coupling_model)
        load_state_dict_auto(model, self.request.coupling_checkpoint)
        model.to(device)
        model.eval()
        return model

    def _load_green_model(
        self,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> torch.nn.Module:
        model: torch.nn.Module
        try:
            model, _loaded_config = load_model_with_config(
                self.request.green_checkpoint
            )
        except Exception:
            model = GreenONetModel(configs.green_model)
            load_state_dict_auto(model, self.request.green_checkpoint)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _evaluate_rows(
        dataset: ComplexCouplingDataset,
        evaluator: ComplexCouplingEvaluator,
        configs: CouplingArtifactConfigs,
    ) -> list[dict[str, float | int | str]]:
        loader = DataLoader(
            dataset,
            batch_size=configs.coupling_training.batch_size,
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        rows: list[dict[str, float | int | str]] = []
        with torch.no_grad():
            for batch in loader:
                prediction = evaluator.predict_batch(batch.to(evaluator.device))
                for offset, sample_index in enumerate(
                    prediction.batch.sample_indices.cpu().tolist()
                ):
                    row = evaluator._sample_metric_row(prediction, offset)
                    row["sample_id"] = int(sample_index)
                    row["file_stem"] = prediction.batch.file_stems[offset]
                    rows.append(row)
        return rows

    def _select_sample_indices(
        self,
        metric_rows: list[dict[str, float | int | str]],
    ) -> tuple[tuple[int, ...], dict[str, int], str]:
        if self.request.selected_samples is not None:
            seen: set[int] = set()
            explicit_selected: list[int] = []
            for value in self.request.selected_samples:
                if value not in seen:
                    explicit_selected.append(int(value))
                    seen.add(int(value))
            return tuple(explicit_selected), {}, "explicit"
        if not metric_rows:
            return (), {}, "empty"
        sorted_rows = sorted(metric_rows, key=lambda row: float(row["rel_sol"]))
        positions = {
            "min": 0,
            "q25": round(0.25 * (len(sorted_rows) - 1)),
            "q50": round(0.50 * (len(sorted_rows) - 1)),
            "q75": round(0.75 * (len(sorted_rows) - 1)),
            "max": len(sorted_rows) - 1,
        }
        roles = {
            role: int(sorted_rows[position]["sample_id"])
            for role, position in positions.items()
        }
        quantile_selected = tuple(dict.fromkeys(roles.values()))
        return quantile_selected, roles, "rel_sol_quantiles"

    def _evaluate_selected(
        self,
        dataset: ComplexCouplingDataset,
        evaluator: ComplexCouplingEvaluator,
        selected: tuple[int, ...],
        device: torch.device,
    ) -> list[ComplexSelectedSample]:
        samples: list[ComplexSelectedSample] = []
        coords = dataset.geometry.coords_valid.detach().cpu().numpy()
        with torch.no_grad():
            for sample_id in selected:
                batch = complex_coupling_collate_fn([dataset[sample_id]]).to(device)
                prediction = evaluator.predict_batch(batch)
                rhs = prediction.batch.rhs_valid[0].detach().cpu().numpy()
                sol = prediction.batch.sol_valid[0].detach().cpu().numpy()
                phi = (
                    prediction.projection.projected_physical[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                psi = (
                    prediction.projection.projected_physical[0, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
                u_phi = prediction.reconstruction.u_phi_valid[0].detach().cpu().numpy()
                u_psi = prediction.reconstruction.u_psi_valid[0].detach().cpu().numpy()
                u_pred = (
                    prediction.reconstruction.u_mean_valid[0].detach().cpu().numpy()
                )
                x_length_squared = (
                    prediction.batch.geometry.x_lengths_for_valid_points()
                    .square()
                    .detach()
                    .cpu()
                    .numpy()
                )
                y_length_squared = (
                    prediction.batch.geometry.y_lengths_for_valid_points()
                    .square()
                    .detach()
                    .cpu()
                    .numpy()
                )
                arrays = {
                    "coords_valid": coords,
                    "rhs": rhs,
                    "sol": sol,
                    "raw_response_phi": (
                        prediction.raw_response[0, 0].detach().cpu().numpy()
                    ),
                    "raw_response_psi": (
                        prediction.raw_response[0, 1].detach().cpu().numpy()
                    ),
                    "raw_physical_phi": (
                        prediction.projection.raw_physical[0, 0].detach().cpu().numpy()
                    ),
                    "raw_physical_psi": (
                        prediction.projection.raw_physical[0, 1].detach().cpu().numpy()
                    ),
                    "projected_response_phi": (
                        prediction.projection.projected_response[0, 0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "projected_response_psi": (
                        prediction.projection.projected_response[0, 1]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "x_length_squared": x_length_squared,
                    "y_length_squared": y_length_squared,
                    "raw_difference": (
                        prediction.projection.raw_difference[0].detach().cpu().numpy()
                    ),
                    "projected_difference": (
                        prediction.projection.projected_difference[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "raw_response_constraint_residual": (
                        prediction.projection.raw_response_constraint_residual[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "response_constraint_residual": (
                        prediction.projection.response_constraint_residual[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "physical_balance_residual": (
                        prediction.projection.physical_balance_residual[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "phi": phi,
                    "psi": psi,
                    "projected_physical_phi": phi,
                    "projected_physical_psi": psi,
                    "u_pred": u_pred,
                    "u_phi": u_phi,
                    "u_psi": u_psi,
                    "u_pred_error": u_pred - sol,
                    "u_phi_error": u_phi - sol,
                    "u_psi_error": u_psi - sol,
                    "u_split_mismatch": u_phi - u_psi,
                }
                if prediction.pre_projection_fusion is not None:
                    fusion = prediction.pre_projection_fusion
                    arrays.update(
                        {
                            "base_raw_response_phi": (
                                fusion.base_response[0, 0].detach().cpu().numpy()
                            ),
                            "base_raw_response_psi": (
                                fusion.base_response[0, 1].detach().cpu().numpy()
                            ),
                            "base_physical_p": (
                                fusion.base_physical[0, 0].detach().cpu().numpy()
                            ),
                            "base_physical_q": (
                                fusion.base_physical[0, 1].detach().cpu().numpy()
                            ),
                            "base_physical_difference": (
                                fusion.base_difference[0].detach().cpu().numpy()
                            ),
                            "fused_physical_p": (
                                fusion.fused_physical[0, 0].detach().cpu().numpy()
                            ),
                            "fused_physical_q": (
                                fusion.fused_physical[0, 1].detach().cpu().numpy()
                            ),
                            "fused_physical_difference": (
                                fusion.fused_difference[0].detach().cpu().numpy()
                            ),
                            "linear_difference_correction": (
                                fusion.linear_correction[0].detach().cpu().numpy()
                            ),
                            "nonlinear_difference_correction": (
                                fusion.nonlinear_correction[0].detach().cpu().numpy()
                            ),
                            "blended_difference_correction": (
                                fusion.blended_correction[0].detach().cpu().numpy()
                            ),
                            "fusion_source_scale": (
                                fusion.source_scale[0].detach().cpu().numpy()
                            ),
                            "fusion_gate": np.asarray(
                                float(fusion.gate.detach().cpu().item())
                            ),
                        }
                    )
                if prediction.objective.relative_split is not None:
                    relative = prediction.objective.relative_split
                    split_residual = u_phi - u_psi
                    point_area = float(
                        (
                            prediction.batch.geometry.hx * prediction.batch.geometry.hy
                        ).item()
                    )
                    denominator = float(
                        relative.rhs_l2_squared_per_sample[0].item()
                        + evaluator.relative_split_config.eps
                    )
                    domain_scale = float(relative.domain_length_scale.item())
                    arrays["split_mass_relative_contribution"] = (
                        evaluator.relative_split_config.weight
                        * evaluator.relative_split_config.mass_weight
                        * point_area
                        * np.square(split_residual)
                        / (domain_scale * domain_scale)
                        / denominator
                    )
                if prediction.objective.weak_closure is not None:
                    weak = prediction.objective.weak_closure
                    arrays["weak_residual_x"] = (
                        weak.x_residual[0].detach().cpu().numpy()
                    )
                    arrays["weak_residual_y"] = (
                        weak.y_residual[0].detach().cpu().numpy()
                    )
                    arrays["weak_nodal_mass_x"] = (
                        prediction.batch.weak_context.x.nodal_mass.detach()
                        .cpu()
                        .numpy()
                    )
                    arrays["weak_nodal_mass_y"] = (
                        prediction.batch.weak_context.y.nodal_mass.detach()
                        .cpu()
                        .numpy()
                    )
                boundary_context = evaluator.boundary_energy_context(
                    prediction.batch.geometry
                )
                boundary_indices = boundary_context.point_indices.detach().cpu()
                arrays["boundary_endpoint_coords"] = (
                    boundary_context.endpoint_coords.detach().cpu().numpy()
                )
                arrays["boundary_nearest_valid_index"] = boundary_indices.numpy()
                arrays["boundary_physical_distance"] = (
                    boundary_context.physical_distance.detach().cpu().numpy()
                )
                arrays["boundary_transverse_measure"] = (
                    boundary_context.transverse_measure.detach().cpu().numpy()
                )
                arrays["boundary_axis_id"] = (
                    boundary_context.axis_id.detach().cpu().numpy()
                )
                arrays["boundary_split_residual"] = (u_phi - u_psi)[
                    boundary_indices.numpy()
                ]
                partition = evaluator.energy_partition(prediction.batch.geometry)
                arrays["x_length_jump_score"] = partition.x_score.detach().cpu().numpy()
                arrays["y_length_jump_score"] = partition.y_score.detach().cpu().numpy()
                arrays["x_transition_edge_mask"] = (
                    partition.x_transition_mask.detach().cpu().numpy()
                )
                arrays["y_transition_edge_mask"] = (
                    partition.y_transition_mask.detach().cpu().numpy()
                )
                arrays["x_transverse_length_context"] = (
                    evaluator.model.transverse_length_context_features(
                        prediction.batch.geometry,
                        "x",
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                arrays["y_transverse_length_context"] = (
                    evaluator.model.transverse_length_context_features(
                        prediction.batch.geometry,
                        "y",
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                if bool(prediction.batch.has_flux[0].item()):
                    target_phi = (
                        prediction.batch.flux_valid[0, 0].detach().cpu().numpy()
                    )
                    target_psi = (
                        prediction.batch.flux_valid[0, 1].detach().cpu().numpy()
                    )
                    arrays["target_phi"] = target_phi
                    arrays["target_psi"] = target_psi
                    arrays["phi_error"] = phi - target_phi
                    arrays["psi_error"] = psi - target_psi
                samples.append(
                    ComplexSelectedSample(
                        sample_id=sample_id,
                        file_stem=prediction.batch.file_stems[0],
                        arrays=arrays,
                    )
                )
        return samples

    def _write_metric_csv(
        self,
        metric_rows: list[dict[str, float | int | str]],
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if not metric_rows:
            return
        fieldnames = list(metric_rows[0].keys())
        for row in metric_rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (metrics_dir / "per_sample_metrics.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    def _write_selected_npz(
        self,
        selected_samples: list[ComplexSelectedSample],
    ) -> None:
        if not self.request.save_generated_data:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for sample in selected_samples:
            prefix = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            for key, value in sample.arrays.items():
                payload[f"{prefix}_{key}"] = value
        np.savez(data_dir / "selected_raw_arrays.npz", **payload)  # type: ignore[arg-type]

    def _write_figures(
        self,
        selected_samples: list[ComplexSelectedSample],
        theme: str,
    ) -> list[str]:
        figure_paths: list[str] = []
        for sample in selected_samples:
            stem = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            color_ranges = self._color_ranges_for_sample(sample.arrays)
            for field in self._figure_fields_for_sample(sample.arrays):
                fig = self._scatter_figure(
                    title=f"{stem} {self.FIGURE_TITLES[field]}",
                    coords=sample.arrays["coords_valid"],
                    values=sample.arrays[field],
                    theme=theme,
                    signed=field in self.SIGNED_FIGURE_FIELDS,
                    color_range=color_ranges.get(field),
                )
                base_path = self.request.outdir / "figures" / field / f"{stem}_{field}"
                save_plotly_figure(fig, base_path, logger=self.logger)
                figure_paths.append(str(base_path.with_suffix(".json")))
        return figure_paths

    @classmethod
    def _color_ranges_for_sample(
        cls,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        ranges: dict[str, dict[str, float]] = {}
        for fields in cls.COLOR_RANGE_GROUPS.values():
            color_range = cls._shared_color_range(arrays, fields)
            if not color_range:
                continue
            for field in fields:
                if field in arrays:
                    ranges[field] = color_range
        return ranges

    @staticmethod
    def _shared_color_range(
        arrays: dict[str, np.ndarray],
        fields: tuple[str, ...],
    ) -> dict[str, float]:
        finite_values: list[np.ndarray] = []
        for field in fields:
            if field not in arrays:
                continue
            values = np.asarray(arrays[field])
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)
        if not finite_values:
            return {}
        joined = np.concatenate(finite_values)
        return {
            "cmin": float(np.min(joined)),
            "cmax": float(np.max(joined)),
        }

    @classmethod
    def _figure_fields_for_sample(
        cls, arrays: dict[str, np.ndarray]
    ) -> tuple[str, ...]:
        return tuple(field for field in cls.FIGURE_FIELDS if field in arrays)

    @classmethod
    def _figure_fields(
        cls,
        selected_samples: list[ComplexSelectedSample],
    ) -> tuple[str, ...]:
        fields: list[str] = []
        seen: set[str] = set()
        for sample in selected_samples:
            for field in cls._figure_fields_for_sample(sample.arrays):
                if field not in seen:
                    fields.append(field)
                    seen.add(field)
        return tuple(fields)

    @staticmethod
    def _has_flux_target_artifacts(
        selected_samples: list[ComplexSelectedSample],
    ) -> bool:
        return any(
            {"target_phi", "target_psi", "phi_error", "psi_error"}.issubset(
                sample.arrays
            )
            for sample in selected_samples
        )

    @staticmethod
    def _scatter_figure(
        *,
        title: str,
        coords: np.ndarray,
        values: np.ndarray,
        theme: str,
        signed: bool = False,
        color_range: dict[str, float] | None = None,
    ) -> go.Figure:
        finite_values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        marker_color_range: dict[str, float] = dict(color_range or {})
        if signed and max_abs > 0.0:
            marker_color_range = {"cmin": -max_abs, "cmax": max_abs}
        return go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": "RdBu" if signed else "Viridis",
                    "showscale": True,
                    "size": 6,
                    "colorbar": {"exponentformat": "power", "showexponent": "all"},
                    **marker_color_range,
                },
            ),
            layout=go.Layout(
                template=theme,
                width=900,
                height=800,
                title=title,
                xaxis_title="x",
                yaxis_title="y",
                yaxis={"scaleanchor": "x", "scaleratio": 1},
            ),
        )

    @staticmethod
    def _aggregate_metrics(
        metric_rows: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        for key in (
            "loss",
            "loss_energy_consistency",
            "loss_energy_length_balanced",
            "loss_energy_bulk",
            "loss_energy_bulk_length_balanced",
            "loss_energy_boundary",
            "loss_energy_boundary_x",
            "loss_energy_boundary_y",
            "loss_energy_regular",
            "loss_energy_transition",
            "transition_edge_fraction",
            "rel_sol",
            "rel_flux",
        ):
            values = [float(row[key]) for row in metric_rows if key in row]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_max"] = float(np.max(values))
        return aggregate


def export_complex_coupling_artifacts(
    request: CouplingArtifactRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexCouplingArtifactExporter(request, logger=logger).export()
