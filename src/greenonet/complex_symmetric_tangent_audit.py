from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_axial_response_operator import (
    FrozenAxialResponseOperatorBuilder,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructor,
)
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
    ColumnDiagonalGreenResponseContextBuilder,
)
from greenonet.complex_losses import (
    build_boundary_energy_context,
    canonical_complex_energy_loss,
)
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.complex_projection_response_audit import (
    ComplexProjectionResponseAudit,
    ProjectionTransitionEdges,
)
from greenonet.complex_reconstruction import reconstruct_from_projected_response
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from greenonet.config import BalanceProjectionConfig
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class SymmetricTangentAuditRequest:
    """Frozen-checkpoint inputs for the response-gradient tangent audit."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    etas: tuple[float, ...] = (
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.015,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    )
    relative_lambdas: tuple[float, ...] = (
        0.0,
        1.0e-4,
        1.0e-3,
        1.0e-2,
        1.0e-1,
        1.0,
    )
    transition_log_threshold: float = math.log(2.0)
    selected_samples: tuple[int, ...] | None = None
    batch_size: int = 10
    denominator_relative_eps: float = 1.0e-12
    closed_loop_enabled: bool = False
    closed_loop_eta_cap: float = 0.01
    closed_loop_eta_caps: tuple[float, ...] | None = None
    closed_loop_relative_lambda: float = 0.01
    line_search_relative_eps: float = 1.0e-12
    metric_eps: float = 1.0e-30
    operator_equivalence_tol: float = 1.0e-10
    save_generated_data: bool = True

    def __post_init__(self) -> None:
        self._validate_unique_numeric(self.etas, "etas", lower=0.0, strict=True)
        self._validate_unique_numeric(
            self.relative_lambdas,
            "relative_lambdas",
            lower=0.0,
            strict=False,
        )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(self.closed_loop_enabled, bool):
            raise TypeError("closed_loop_enabled must be a boolean.")
        for name, value, allow_zero in (
            ("transition_log_threshold", self.transition_log_threshold, True),
            ("denominator_relative_eps", self.denominator_relative_eps, False),
            ("closed_loop_eta_cap", self.closed_loop_eta_cap, True),
            (
                "closed_loop_relative_lambda",
                self.closed_loop_relative_lambda,
                True,
            ),
            ("line_search_relative_eps", self.line_search_relative_eps, False),
            ("metric_eps", self.metric_eps, False),
            ("operator_equivalence_tol", self.operator_equivalence_tol, False),
        ):
            if (
                not math.isfinite(value)
                or value < 0.0
                or (not allow_zero and value == 0.0)
            ):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        if self.selected_samples is not None and len(set(self.selected_samples)) != len(
            self.selected_samples
        ):
            raise ValueError("selected_samples must not contain duplicates.")
        if self.closed_loop_eta_caps is not None:
            self._validate_eta_caps(self.closed_loop_eta_caps)

    @property
    def resolved_closed_loop_eta_caps(self) -> tuple[float, ...]:
        if self.closed_loop_eta_caps is None:
            return (self.closed_loop_eta_cap,)
        return self.closed_loop_eta_caps

    @staticmethod
    def _validate_eta_caps(values: tuple[float, ...]) -> None:
        if not values:
            raise ValueError("closed_loop_eta_caps must not be empty.")
        normalized: list[float] = []
        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or math.isnan(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(
                    "closed_loop_eta_caps must contain non-negative values or +inf."
                )
            normalized.append(float(value))
        if len(set(normalized)) != len(normalized):
            raise ValueError("closed_loop_eta_caps must not contain duplicates.")

    @staticmethod
    def _validate_unique_numeric(
        values: tuple[float, ...],
        name: str,
        *,
        lower: float,
        strict: bool,
    ) -> None:
        if not values:
            raise ValueError(f"{name} must not be empty.")
        normalized: list[float] = []
        for value in values:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must contain finite numeric values.")
            numeric = float(value)
            if numeric < lower or (strict and numeric == lower):
                qualifier = "greater than" if strict else "at least"
                raise ValueError(f"{name} values must be {qualifier} {lower}.")
            normalized.append(numeric)
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} must not contain duplicate values.")


@dataclass(frozen=True)
class TangentMethod:
    method_id: str
    label: str
    kind: str
    eta: float | None = None
    relative_lambda: float | None = None


@dataclass(frozen=True)
class ClosedLoopTangentBatchDiagnostics:
    """Per-sample exact-line-search values for the optional closed-loop method."""

    method_id: str
    eta_cap: float | None
    eta_star: torch.Tensor
    eta_applied: torch.Tensor
    eta_capped: torch.Tensor
    line_search_numerator: torch.Tensor
    line_search_denominator: torch.Tensor


@dataclass(frozen=True)
class TangentBatchEvaluation:
    methods: tuple[TangentMethod, ...]
    raw_physical: torch.Tensor
    symmetric_physical: torch.Tensor
    configured_physical: torch.Tensor
    tangent_gradient: torch.Tensor
    tangent_preconditioner_base: torch.Tensor
    tangent_delta: torch.Tensor
    candidate_physical: torch.Tensor
    candidate_solution: torch.Tensor
    candidate_equal_prediction: torch.Tensor
    candidate_prediction: torch.Tensor
    canonical_energy: torch.Tensor
    canonical_bulk_energy: torch.Tensor
    canonical_boundary_energy: torch.Tensor
    closed_loop: tuple[ClosedLoopTangentBatchDiagnostics, ...] = ()


class SymmetricTangentMetricMixin:
    """Metric helpers for feasible tangent source corrections."""

    @staticmethod
    def _relative_l2(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(prediction - target, dim=-1)
        denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
        result: torch.Tensor = numerator / denominator
        return result

    @staticmethod
    def _relative_pair_l2(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(
            (prediction - target).flatten(start_dim=1),
            dim=-1,
        )
        denominator = torch.linalg.vector_norm(
            target.flatten(start_dim=1),
            dim=-1,
        ).clamp_min(eps)
        result: torch.Tensor = numerator / denominator
        return result

    @staticmethod
    def _edge_rms(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return values.new_full(values.shape[:-1], math.nan)
        difference = values[..., edges[:, 1]] - values[..., edges[:, 0]]
        return torch.sqrt(difference.square().mean(dim=-1))

    @staticmethod
    def _point_masks(
        edges: ProjectionTransitionEdges,
        *,
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = (
            torch.unique(edges.transition.flatten())
            if edges.transition.numel()
            else torch.empty(0, dtype=torch.long)
        )
        mask = torch.ones(point_count, dtype=torch.bool)
        mask[transition] = False
        regular = torch.nonzero(mask, as_tuple=False).flatten()
        return transition, regular

    def build_metric_rows(
        self,
        *,
        batch: ComplexCouplingBatch,
        evaluation: TangentBatchEvaluation,
        edges: ProjectionTransitionEdges,
        point_mass: torch.Tensor,
        eps: float,
    ) -> list[dict[str, float | int | str]]:
        candidate_solution = evaluation.candidate_solution
        mismatch = candidate_solution[:, :, 0] - candidate_solution[:, :, 1]
        response_cost = point_mass * mismatch.square().sum(dim=-1)
        symmetric_cost = response_cost[0].clamp_min(eps)
        canonical = evaluation.canonical_energy
        symmetric_canonical = canonical[0].clamp_min(eps)
        symmetric_pair = evaluation.symmetric_physical
        symmetric_pair_norm = torch.linalg.vector_norm(
            symmetric_pair.flatten(start_dim=1),
            dim=-1,
        ).clamp_min(eps)
        correction_norm = torch.sqrt(torch.tensor(2.0, dtype=mismatch.dtype)) * (
            torch.linalg.vector_norm(evaluation.tangent_delta, dim=-1)
        )
        transition_points, regular_points = self._point_masks(
            edges,
            point_count=batch.geometry.num_points,
        )
        transition_edges = edges.transition.to(mismatch.device)
        regular_edges = edges.regular.to(mismatch.device)
        closed_loop_by_method = {
            diagnostics.method_id: diagnostics for diagnostics in evaluation.closed_loop
        }
        rows: list[dict[str, float | int | str]] = []
        for method_index, method in enumerate(evaluation.methods):
            for sample_offset, sample_id in enumerate(batch.sample_indices.tolist()):
                delta = evaluation.tangent_delta[method_index, sample_offset]
                prediction = evaluation.candidate_prediction[
                    method_index, sample_offset
                ]
                equal_prediction = evaluation.candidate_equal_prediction[
                    method_index, sample_offset
                ]
                row: dict[str, float | int | str] = {
                    "sample_id": int(sample_id),
                    "file_stem": batch.file_stems[sample_offset],
                    "method_id": method.method_id,
                    "method_label": method.label,
                    "method_kind": method.kind,
                    "eta": math.nan if method.eta is None else method.eta,
                    "relative_lambda": (
                        math.nan
                        if method.relative_lambda is None
                        else method.relative_lambda
                    ),
                    "response_mismatch_cost": float(
                        response_cost[method_index, sample_offset].item()
                    ),
                    "response_mismatch_ratio_vs_symmetric": float(
                        (
                            response_cost[method_index, sample_offset]
                            / symmetric_cost[sample_offset]
                        ).item()
                    ),
                    "canonical_energy": float(
                        canonical[method_index, sample_offset].item()
                    ),
                    "canonical_energy_ratio_vs_symmetric": float(
                        (
                            canonical[method_index, sample_offset]
                            / symmetric_canonical[sample_offset]
                        ).item()
                    ),
                    "canonical_bulk_energy": float(
                        evaluation.canonical_bulk_energy[
                            method_index, sample_offset
                        ].item()
                    ),
                    "canonical_boundary_energy": float(
                        evaluation.canonical_boundary_energy[
                            method_index, sample_offset
                        ].item()
                    ),
                    "tangent_delta_rms": float(
                        torch.sqrt(delta.square().mean()).item()
                    ),
                    "tangent_delta_max_abs": float(delta.abs().max().item()),
                    "tangent_correction_rel_symmetric_pair": float(
                        (
                            correction_norm[method_index, sample_offset]
                            / symmetric_pair_norm[sample_offset]
                        ).item()
                    ),
                    "tangent_gradient_rms": float(
                        torch.sqrt(
                            evaluation.tangent_gradient[sample_offset].square().mean()
                        ).item()
                    ),
                    "physical_balance_max_abs": float(
                        (
                            batch.rhs_valid[sample_offset]
                            - evaluation.candidate_physical[
                                method_index, sample_offset
                            ].sum(dim=0)
                        )
                        .abs()
                        .max()
                        .item()
                    ),
                    "split_transition_jump_rms": float(
                        self._edge_rms(
                            mismatch[method_index, sample_offset].unsqueeze(0),
                            transition_edges,
                        )[0].item()
                    ),
                    "split_regular_jump_rms": float(
                        self._edge_rms(
                            mismatch[method_index, sample_offset].unsqueeze(0),
                            regular_edges,
                        )[0].item()
                    ),
                }
                closed_loop = closed_loop_by_method.get(method.method_id)
                if closed_loop is not None:
                    row.update(
                        {
                            "eta_strategy": "closed_loop_exact_line_search",
                            "eta_cap": (
                                math.nan
                                if closed_loop.eta_cap is None
                                else closed_loop.eta_cap
                            ),
                            "eta_cap_label": (
                                "uncapped"
                                if closed_loop.eta_cap is None
                                else f"{closed_loop.eta_cap:g}"
                            ),
                            "eta_star": float(
                                closed_loop.eta_star[sample_offset].item()
                            ),
                            "eta_applied": float(
                                closed_loop.eta_applied[sample_offset].item()
                            ),
                            "eta_capped": int(
                                closed_loop.eta_capped[sample_offset].item()
                            ),
                            "line_search_numerator": float(
                                closed_loop.line_search_numerator[sample_offset].item()
                            ),
                            "line_search_denominator": float(
                                closed_loop.line_search_denominator[
                                    sample_offset
                                ].item()
                            ),
                        }
                    )
                if bool(batch.has_solution[sample_offset].item()):
                    solution = batch.sol_valid[sample_offset]
                    row["rel_sol"] = float(
                        self._relative_l2(
                            prediction.unsqueeze(0),
                            solution.unsqueeze(0),
                            eps=eps,
                        ).item()
                    )
                    row["rel_sol_equal_mean"] = float(
                        self._relative_l2(
                            equal_prediction.unsqueeze(0),
                            solution.unsqueeze(0),
                            eps=eps,
                        ).item()
                    )
                    signed_error = prediction - solution
                    row["transition_solution_error_jump_rms"] = float(
                        self._edge_rms(
                            signed_error.unsqueeze(0),
                            transition_edges,
                        )[0].item()
                    )
                    row["regular_solution_error_jump_rms"] = float(
                        self._edge_rms(
                            signed_error.unsqueeze(0),
                            regular_edges,
                        )[0].item()
                    )
                if bool(batch.has_flux[sample_offset].item()):
                    physical = evaluation.candidate_physical[
                        method_index, sample_offset
                    ]
                    target = batch.flux_valid[sample_offset]
                    row["pair_rel_flux_target"] = float(
                        self._relative_pair_l2(
                            physical.unsqueeze(0),
                            target.unsqueeze(0),
                            eps=eps,
                        ).item()
                    )
                    row["rel_flux"] = float(
                        self._relative_l2(
                            physical,
                            target,
                            eps=eps,
                        )
                        .mean()
                        .item()
                    )
                    if transition_points.numel():
                        row["transition_pair_rel_flux_target"] = float(
                            self._relative_pair_l2(
                                physical[:, transition_points].unsqueeze(0),
                                target[:, transition_points].unsqueeze(0),
                                eps=eps,
                            ).item()
                        )
                    if regular_points.numel():
                        row["regular_pair_rel_flux_target"] = float(
                            self._relative_pair_l2(
                                physical[:, regular_points].unsqueeze(0),
                                target[:, regular_points].unsqueeze(0),
                                eps=eps,
                            ).item()
                        )
                rows.append(row)
        return rows

    @staticmethod
    def aggregate_rows(
        rows: Sequence[dict[str, float | int | str]],
        methods: tuple[TangentMethod, ...],
    ) -> dict[str, dict[str, float | int | str]]:
        aggregate: dict[str, dict[str, float | int | str]] = {}
        excluded = {
            "sample_id",
            "file_stem",
            "method_id",
            "method_label",
            "method_kind",
            "eta",
            "relative_lambda",
            "eta_strategy",
        }
        for method in methods:
            selected = [row for row in rows if row["method_id"] == method.method_id]
            payload: dict[str, float | int | str] = {
                "method_label": method.label,
                "method_kind": method.kind,
                "sample_count": len(selected),
                "eta": math.nan if method.eta is None else method.eta,
                "relative_lambda": (
                    math.nan
                    if method.relative_lambda is None
                    else method.relative_lambda
                ),
            }
            metric_keys = {
                key
                for row in selected
                for key, value in row.items()
                if key not in excluded and isinstance(value, (int, float))
            }
            for key in sorted(metric_keys):
                values = np.asarray(
                    [float(row[key]) for row in selected if key in row],
                    dtype=np.float64,
                )
                finite = values[np.isfinite(values)]
                if finite.size:
                    payload[f"{key}_mean"] = float(finite.mean())
                    payload[f"{key}_median"] = float(np.median(finite))
            aggregate[method.method_id] = payload
        return aggregate


class SymmetricTangentPlotMixin:
    """Plot aggregate eta/lambda sweeps and selected candidate fields."""

    @staticmethod
    def _scatter(
        *,
        geometry: ComplexGeometryMetadata,
        values: torch.Tensor,
        title: str,
        symmetric: bool,
    ) -> go.Scattergl:
        array = values.detach().cpu().numpy()
        marker: dict[str, Any] = {
            "size": 4,
            "color": array,
            "colorscale": "RdBu" if symmetric else "Viridis",
            "showscale": False,
        }
        if symmetric:
            limit = float(np.max(np.abs(array), initial=0.0)) or 1.0
            marker.update(cmin=-limit, cmax=limit, cmid=0.0)
        coords = geometry.coords_valid.detach().cpu().numpy()
        return go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker=marker,
            customdata=array,
            hovertemplate=(
                "x=%{x:.6g}<br>y=%{y:.6g}<br>value=%{customdata:.6e}<extra></extra>"
            ),
            name=title,
            showlegend=False,
        )

    def write_sweep_figure(
        self,
        *,
        aggregate: dict[str, dict[str, float | int | str]],
        methods: tuple[TangentMethod, ...],
        request: SymmetricTangentAuditRequest,
        logger: logging.Logger | None,
    ) -> Path:
        metrics = (
            (
                "response_mismatch_ratio_vs_symmetric_mean",
                "Response mismatch / sym",
                "log10 ratio",
            ),
            (
                "canonical_energy_ratio_vs_symmetric_mean",
                "Canonical energy / sym",
                "log10 ratio",
            ),
            ("rel_sol_mean", "Production-blend rel_sol", "log10 rel_sol"),
            ("rel_flux_mean", "Directional rel_flux", "log10 rel_flux"),
        )
        tangent = [method for method in methods if method.kind == "tangent_gradient"]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[label for _, label, _ in metrics],
            horizontal_spacing=0.14,
            vertical_spacing=0.16,
        )
        eta_labels = [f"{value:g}" for value in request.etas]
        lambda_labels = [f"{value:g}" for value in request.relative_lambdas]
        colorbar_positions = (
            {"x": 0.44, "y": 0.79},
            {"x": 1.02, "y": 0.79},
            {"x": 0.44, "y": 0.21},
            {"x": 1.02, "y": 0.21},
        )
        for index, (metric, _label, colorbar_title) in enumerate(metrics):
            values = np.full(
                (len(request.relative_lambdas), len(request.etas)),
                np.nan,
                dtype=np.float64,
            )
            for method in tangent:
                if method.relative_lambda is None or method.eta is None:
                    raise RuntimeError("Tangent sweep methods require eta and lambda.")
                lambda_index = request.relative_lambdas.index(method.relative_lambda)
                eta_index = request.etas.index(method.eta)
                raw_value = aggregate[method.method_id].get(metric, math.nan)
                values[lambda_index, eta_index] = float(raw_value)
            log_values = np.log10(np.maximum(values, np.finfo(np.float64).tiny))
            colorbar = {
                **colorbar_positions[index],
                "len": 0.34,
                "thickness": 12,
                "title": {"text": colorbar_title, "side": "right"},
            }
            fig.add_trace(
                go.Heatmap(
                    x=eta_labels,
                    y=lambda_labels,
                    z=log_values,
                    customdata=values,
                    colorscale="Viridis",
                    colorbar=colorbar,
                    hovertemplate=(
                        "eta=%{x}<br>lambda_rel=%{y}<br>"
                        "value=%{customdata:.6e}<extra></extra>"
                    ),
                ),
                row=index // 2 + 1,
                col=index % 2 + 1,
            )
        fig.update_layout(
            template=request.theme,
            title="Symmetric tangent response-gradient sweep",
            width=1450,
            height=900,
            margin={"l": 80, "r": 145, "t": 100, "b": 80},
        )
        fig.update_xaxes(title_text="eta", type="category")
        fig.update_yaxes(title_text="lambda_rel", type="category")
        path = request.outdir / "figures" / "aggregate" / "eta_lambda_sweep"
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")

    def write_closed_loop_cap_figure(
        self,
        *,
        aggregate: dict[str, dict[str, float | int | str]],
        methods: tuple[TangentMethod, ...],
        request: SymmetricTangentAuditRequest,
        logger: logging.Logger | None,
    ) -> Path:
        closed_loop = tuple(
            method
            for method in methods
            if method.kind == "closed_loop_exact_line_search"
        )
        if len(closed_loop) < 2:
            raise ValueError("A closed-loop cap figure requires at least two caps.")
        baseline = aggregate["symmetric"]
        cap_labels = [
            "uncapped" if method.eta is None else f"{method.eta:g}"
            for method in closed_loop
        ]
        panels = (
            (
                ("response_mismatch_cost_mean",),
                ("response mismatch",),
                "Response mismatch / symmetric",
            ),
            (
                (
                    "canonical_energy_mean",
                    "canonical_bulk_energy_mean",
                    "canonical_boundary_energy_mean",
                ),
                ("total", "bulk", "boundary"),
                "Canonical energy / symmetric",
            ),
            (
                ("rel_sol_mean", "rel_flux_mean"),
                ("rel_sol", "rel_flux"),
                "Evaluation error / symmetric",
            ),
            (
                ("split_regular_jump_rms_mean", "split_transition_jump_rms_mean"),
                ("split regular", "split transition"),
                "Split jump RMS / symmetric",
            ),
            (
                (
                    "regular_solution_error_jump_rms_mean",
                    "transition_solution_error_jump_rms_mean",
                ),
                ("error regular", "error transition"),
                "Solution-error jump RMS / symmetric",
            ),
        )
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[panel[2] for panel in panels]
            + ["Closed-loop cap activity"],
            horizontal_spacing=0.1,
            vertical_spacing=0.16,
        )
        for panel_index, (metrics, labels, _title) in enumerate(panels):
            row = panel_index // 3 + 1
            col = panel_index % 3 + 1
            for metric, label in zip(metrics, labels, strict=True):
                baseline_value = float(baseline[metric])
                ratios = [
                    float(aggregate[method.method_id][metric]) / baseline_value
                    for method in closed_loop
                ]
                fig.add_trace(
                    go.Scatter(
                        x=cap_labels,
                        y=ratios,
                        mode="lines+markers",
                        name=label,
                        legendgroup=f"panel-{panel_index}",
                        hovertemplate=(
                            "cap=%{x}<br>ratio=%{y:.6f}<extra>" + label + "</extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="#64748b",
                row=row,
                col=col,
            )
        capped_fraction = [
            float(aggregate[method.method_id]["eta_capped_mean"])
            for method in closed_loop
        ]
        applied_fraction = [
            float(aggregate[method.method_id]["eta_applied_mean"])
            / float(aggregate[method.method_id]["eta_star_mean"])
            for method in closed_loop
        ]
        for values, label in (
            (capped_fraction, "cap-hit fraction"),
            (applied_fraction, "mean applied / eta_star"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=cap_labels,
                    y=values,
                    mode="lines+markers",
                    name=label,
                    hovertemplate=(
                        "cap=%{x}<br>fraction=%{y:.6f}<extra>" + label + "</extra>"
                    ),
                ),
                row=2,
                col=3,
            )
        fig.update_layout(
            template=request.theme,
            title="Closed-loop exact-line-search cap sweep",
            width=1500,
            height=850,
            margin={"l": 80, "r": 40, "t": 100, "b": 80},
        )
        fig.update_xaxes(title_text="eta cap", type="category")
        fig.update_yaxes(title_text="ratio")
        fig.update_yaxes(range=[0.0, 1.05], row=2, col=3)
        path = request.outdir / "figures" / "aggregate" / "closed_loop_cap_sweep"
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")

    def write_selected_figure(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        batch: ComplexCouplingBatch,
        evaluation: TangentBatchEvaluation,
        sample_offset: int,
        method_indices: tuple[int, ...],
        request: SymmetricTangentAuditRequest,
        logger: logging.Logger | None,
    ) -> Path:
        has_solution = bool(batch.has_solution[sample_offset].item())
        has_flux = bool(batch.has_flux[sample_offset].item())
        columns = 4
        fig = make_subplots(
            rows=len(method_indices),
            cols=columns,
            subplot_titles=[
                title
                for method_index in method_indices
                for title in (
                    f"{evaluation.methods[method_index].label}: delta",
                    f"{evaluation.methods[method_index].label}: u_phi-u_psi",
                    f"{evaluation.methods[method_index].label}: u_pred-sol",
                    f"{evaluation.methods[method_index].label}: source pair error",
                )
            ],
        )
        for row_index, method_index in enumerate(method_indices, start=1):
            delta = evaluation.tangent_delta[method_index, sample_offset]
            solution_pair = evaluation.candidate_solution[method_index, sample_offset]
            mismatch = solution_pair[0] - solution_pair[1]
            prediction = evaluation.candidate_prediction[method_index, sample_offset]
            signed_error = (
                prediction - batch.sol_valid[sample_offset]
                if has_solution
                else torch.zeros_like(prediction)
            )
            if has_flux:
                source_difference = (
                    evaluation.candidate_physical[method_index, sample_offset]
                    - batch.flux_valid[sample_offset]
                )
                source_error = torch.sqrt(source_difference.square().sum(dim=0))
            else:
                source_error = torch.zeros_like(prediction)
            for col_index, (values, symmetric) in enumerate(
                (
                    (delta, True),
                    (mismatch, True),
                    (signed_error, True),
                    (source_error, False),
                ),
                start=1,
            ):
                fig.add_trace(
                    self._scatter(
                        geometry=geometry,
                        values=values,
                        title="field",
                        symmetric=symmetric,
                    ),
                    row=row_index,
                    col=col_index,
                )
        fig.update_layout(
            template=request.theme,
            title=(
                f"sample {int(batch.sample_indices[sample_offset].item())}: "
                "symmetric tangent candidates"
            ),
            width=1550,
            height=360 * len(method_indices),
            showlegend=False,
        )
        fig.update_annotations(font={"size": 10})
        for index in range(len(method_indices) * columns):
            axis_reference = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis_reference,
                scaleratio=1.0,
                row=index // columns + 1,
                col=index % columns + 1,
            )
        sample_id = int(batch.sample_indices[sample_offset].item())
        path = (
            request.outdir
            / "figures"
            / "selected"
            / f"sample_{sample_id:04d}_symmetric_tangent_audit"
        )
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")


class ComplexSymmetricTangentAudit(
    SymmetricTangentMetricMixin,
    SymmetricTangentPlotMixin,
):
    """Compare matrix-free tangent updates on one frozen CouplingNet checkpoint."""

    def __init__(
        self,
        request: SymmetricTangentAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.geometry: ComplexGeometryMetadata
        self.response_operator: FrozenBidirectionalResponseOperator
        self.response_context: ColumnDiagonalGreenResponseContext
        self.closed_loop_context: SymmetricTangentGreenResponseContext | None = None
        self.methods: tuple[TangentMethod, ...]
        self._coupling_model: ComplexCouplingNet
        self._green_model: torch.nn.Module
        self._cross_axis_reconstructor: ComplexCrossAxisReconstructor
        self._configs: CouplingArtifactConfigs
        self._device: torch.device
        self._operator_equivalence_max_abs = math.nan
        self._context_build_count = 0
        self._operator_build_count = 0
        self._configured_projection_mode = ""

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        self._configs = load_coupling_artifact_configs(self.request.config)
        if self._configs.dataset.geometry_mode != "complex":
            raise ValueError("Symmetric tangent audit requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        if projection.mode not in {
            "physical_symmetric",
            "column_diagonal_green_response",
        }:
            raise ValueError(
                "The configured baseline must use physical_symmetric or "
                "column_diagonal_green_response."
            )
        self._configured_projection_mode = projection.mode
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
        dataset_offset = 0
        for batch in loader:
            batch = batch.to(self._device)
            self._initialize_response_context(batch)
            evaluation = self._evaluate_batch(batch)
            rows.extend(
                self.build_metric_rows(
                    batch=batch,
                    evaluation=evaluation,
                    edges=edges,
                    point_mass=self.response_context.point_mass,
                    eps=self.request.metric_eps,
                )
            )
            for sample_id in batch.sample_indices.tolist():
                dataset_offset_by_sample[int(sample_id)] = dataset_offset
                dataset_offset += 1
        aggregate = self.aggregate_rows(rows, self.methods)
        findings = self._automated_findings(aggregate)
        selected, roles = self._select_samples(rows, dataset_offset_by_sample)
        metric_path = self.request.outdir / "metrics" / "per_sample_tangent_sweep.csv"
        self._write_csv(metric_path, rows)
        figure_paths = [
            self.write_sweep_figure(
                aggregate=aggregate,
                methods=self.methods,
                request=self.request,
                logger=self.logger,
            )
        ]
        if (
            sum(
                method.kind == "closed_loop_exact_line_search"
                for method in self.methods
            )
            > 1
        ):
            figure_paths.append(
                self.write_closed_loop_cap_figure(
                    aggregate=aggregate,
                    methods=self.methods,
                    request=self.request,
                    logger=self.logger,
                )
            )
        selected_batch = complex_coupling_collate_fn(
            [dataset[dataset_offset_by_sample[sample_id]] for sample_id in selected]
        ).to(self._device)
        selected_evaluation = self._evaluate_batch(selected_batch)
        method_ids = self._selected_method_ids(findings)
        method_indices = tuple(
            next(
                index
                for index, method in enumerate(self.methods)
                if method.method_id == method_id
            )
            for method_id in method_ids
        )
        for sample_offset in range(len(selected)):
            figure_paths.append(
                self.write_selected_figure(
                    geometry=self.geometry,
                    batch=selected_batch,
                    evaluation=selected_evaluation,
                    sample_offset=sample_offset,
                    method_indices=method_indices,
                    request=self.request,
                    logger=self.logger,
                )
            )
        if self.request.save_generated_data:
            self._write_selected_arrays(
                batch=selected_batch,
                evaluation=selected_evaluation,
                method_indices=method_indices,
                edges=edges,
            )
        summary = self._build_summary(
            dataset=dataset,
            geometry_path=geometry_path,
            test_path=test_path,
            coefficient_path=coefficient_path,
            aggregate=aggregate,
            findings=findings,
            selected=selected,
            roles=roles,
            method_ids=method_ids,
            edges=edges,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Symmetric tangent audit complete: samples=%d methods=%d",
                len(dataset),
                len(self.methods),
            )
        return summary

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
        model_loader = ComplexCouplingArtifactExporter(
            loader_request,
            logger=self.logger,
        )
        self._coupling_model = model_loader._load_complex_model(
            self._configs,
            self._device,
        )
        self._green_model = model_loader._load_green_model(
            self._configs,
            self._device,
        )
        for model in (self._coupling_model, self._green_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    def _initialize_response_context(self, batch: ComplexCouplingBatch) -> None:
        if hasattr(self, "response_operator"):
            return
        projection = BalanceProjectionConfig.from_raw(
            self._configs.coupling_model.balance_projection
        )
        builder = ColumnDiagonalGreenResponseContextBuilder(
            projection.column_diagonal_green_response
        )
        self.response_context = builder.build(
            green_model=self._green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        self._context_build_count += 1
        self.response_operator = FrozenAxialResponseOperatorBuilder.build(
            green_model=self._green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        self._operator_build_count += 1
        if self.request.closed_loop_enabled:
            caps = self.request.resolved_closed_loop_eta_caps
            finite_caps = tuple(cap for cap in caps if math.isfinite(cap))
            context_eta = max(finite_caps, default=0.0)
            if any(math.isinf(cap) for cap in caps):
                context_eta = max(context_eta, 1.0)
            self.closed_loop_context = (
                SymmetricTangentGreenResponseContext.from_response_operator(
                    response_operator=self.response_operator,
                    point_mass=self.response_context.point_mass,
                    config={
                        "eta": context_eta,
                        "eta_strategy": "closed_loop_exact_line_search",
                        "line_search_relative_eps": (
                            self.request.line_search_relative_eps
                        ),
                        "relative_lambda": (self.request.closed_loop_relative_lambda),
                        "denominator_relative_eps": (
                            self.request.denominator_relative_eps
                        ),
                    },
                )
            )
        self.methods = self._build_methods(
            configured_mode=projection.mode,
            configured_alpha=self.response_context.gain_exponent,
        )
        self._verify_operator_equivalence(batch)

    def _verify_operator_equivalence(self, batch: ComplexCouplingBatch) -> None:
        physical = torch.stack(
            (batch.rhs_valid * 0.5, batch.rhs_valid * 0.5),
            dim=1,
        )
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

    def _build_methods(
        self,
        *,
        configured_mode: str,
        configured_alpha: float,
    ) -> tuple[TangentMethod, ...]:
        methods = [TangentMethod("symmetric", "symmetric", "symmetric")]
        if configured_mode == "column_diagonal_green_response":
            methods.append(
                TangentMethod(
                    "configured_column",
                    f"configured column alpha={configured_alpha:g}",
                    "configured_column",
                )
            )
        for relative_lambda in self.request.relative_lambdas:
            for eta in self.request.etas:
                methods.append(
                    TangentMethod(
                        method_id=(
                            f"tangent_eta_{self._number_label(eta)}_lambda_"
                            f"{self._number_label(relative_lambda)}"
                        ),
                        label=(f"tangent eta={eta:g}, lambda_rel={relative_lambda:g}"),
                        kind="tangent_gradient",
                        eta=eta,
                        relative_lambda=relative_lambda,
                    )
                )
        if self.request.closed_loop_enabled:
            for cap in self.request.resolved_closed_loop_eta_caps:
                uncapped = math.isinf(cap)
                cap_id = "uncapped" if uncapped else self._number_label(cap)
                cap_label = "uncapped" if uncapped else f"{cap:g}"
                methods.append(
                    TangentMethod(
                        method_id=(
                            f"closed_loop_eta_cap_{cap_id}_lambda_"
                            f"{self._number_label(self.request.closed_loop_relative_lambda)}"
                        ),
                        label=(
                            "closed-loop exact line search "
                            f"cap={cap_label}, "
                            f"lambda_rel={self.request.closed_loop_relative_lambda:g}"
                        ),
                        kind="closed_loop_exact_line_search",
                        eta=None if uncapped else cap,
                        relative_lambda=self.request.closed_loop_relative_lambda,
                    )
                )
        return tuple(methods)

    @staticmethod
    def _number_label(value: float) -> str:
        return f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")

    @torch.no_grad()
    def _evaluate_batch(self, batch: ComplexCouplingBatch) -> TangentBatchEvaluation:
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
        configured = apply_complex_balance_projection(
            raw_response=raw_response,
            rhs_phys=batch.rhs_valid,
            geometry=batch.geometry,
            config=self._configs.coupling_model.balance_projection,
            column_diagonal_context=self.response_context,
        )
        raw_physical = configured.raw_physical
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
        gradient = self.response_operator.tangent_gradient(
            mismatch,
            point_mass=self.response_context.point_mass,
        )
        preconditioner_base = (
            self.response_context.gamma_x_squared
            + self.response_context.gamma_y_squared
        )
        gain_scale = preconditioner_base.mean()
        if not torch.isfinite(gain_scale) or bool((gain_scale <= 0.0).item()):
            raise RuntimeError(
                "The tangent preconditioner gain scale must be positive."
            )
        deltas: list[torch.Tensor] = []
        closed_loop_diagnostics: list[ClosedLoopTangentBatchDiagnostics] = []
        closed_loop_step = None
        closed_loop_direction = None
        for method in self.methods:
            if method.kind == "symmetric":
                deltas.append(torch.zeros_like(gradient))
            elif method.kind == "configured_column":
                deltas.append(configured.projected_physical[:, 0] - symmetric[:, 0])
            elif method.kind == "tangent_gradient":
                if method.relative_lambda is None or method.eta is None:
                    raise RuntimeError("Tangent sweep methods require eta and lambda.")
                denominator = (
                    preconditioner_base
                    + (method.relative_lambda + self.request.denominator_relative_eps)
                    * gain_scale
                )
                deltas.append(-method.eta * gradient / denominator.unsqueeze(0))
            elif method.kind == "closed_loop_exact_line_search":
                if self.closed_loop_context is None:
                    raise RuntimeError(
                        "Closed-loop audit method requires its frozen context."
                    )
                if closed_loop_step is None:
                    closed_loop_step = self.closed_loop_context.tangent_step(
                        mismatch=mismatch,
                        gradient=gradient,
                        eta_cap=self.closed_loop_context.eta,
                    )
                    closed_loop_direction = (
                        gradient / self.closed_loop_context.denominator.unsqueeze(0)
                    )
                step = closed_loop_step
                if (
                    step.eta_star is None
                    or step.line_search_numerator is None
                    or step.line_search_denominator is None
                    or closed_loop_direction is None
                ):
                    raise RuntimeError(
                        "Closed-loop exact-line-search diagnostics are incomplete."
                    )
                if method.eta is None:
                    eta_applied = step.eta_star
                    eta_capped = torch.zeros_like(step.eta_star, dtype=torch.bool)
                else:
                    cap = step.eta_star.new_full(step.eta_star.shape, method.eta)
                    eta_applied = torch.minimum(step.eta_star, cap)
                    eta_capped = step.eta_star > cap
                deltas.append(-eta_applied.unsqueeze(1) * closed_loop_direction)
                closed_loop_diagnostics.append(
                    ClosedLoopTangentBatchDiagnostics(
                        method_id=method.method_id,
                        eta_cap=method.eta,
                        eta_star=step.eta_star,
                        eta_applied=eta_applied,
                        eta_capped=eta_capped,
                        line_search_numerator=step.line_search_numerator,
                        line_search_denominator=step.line_search_denominator,
                    )
                )
            else:
                raise RuntimeError(f"Unsupported tangent method kind: {method.kind}.")
        tangent_delta = torch.stack(deltas, dim=0)
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
        boundary_context = build_boundary_energy_context(batch.geometry)
        energy = canonical_complex_energy_loss(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            a_valid=flat_a,
            geometry=batch.geometry,
            boundary_context=boundary_context,
        )
        cross_axis = self._cross_axis_reconstructor.reconstruct(
            u_phi_valid=flat_solution[:, 0],
            u_psi_valid=flat_solution[:, 1],
            projected_physical=flat_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        return TangentBatchEvaluation(
            methods=self.methods,
            raw_physical=raw_physical,
            symmetric_physical=symmetric,
            configured_physical=configured.projected_physical,
            tangent_gradient=gradient,
            tangent_preconditioner_base=preconditioner_base,
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
            closed_loop=tuple(closed_loop_diagnostics),
        )

    @staticmethod
    def _best_method(
        aggregate: dict[str, dict[str, float | int | str]],
        *,
        metric: str,
        kinds: frozenset[str] | None = None,
    ) -> str | None:
        candidates = [
            (method_id, float(payload[metric]))
            for method_id, payload in aggregate.items()
            if metric in payload and (kinds is None or payload["method_kind"] in kinds)
        ]
        return min(candidates, key=lambda item: item[1])[0] if candidates else None

    def _automated_findings(
        self,
        aggregate: dict[str, dict[str, float | int | str]],
    ) -> dict[str, str | None]:
        tangent_kinds = frozenset({"tangent_gradient", "closed_loop_exact_line_search"})
        closed_loop_method = self._best_method(
            aggregate,
            metric="response_mismatch_cost_mean",
            kinds=frozenset({"closed_loop_exact_line_search"}),
        )
        return {
            "configured_projection_method": (
                "configured_column" if "configured_column" in aggregate else "symmetric"
            ),
            "lowest_mean_response_mismatch_method": self._best_method(
                aggregate,
                metric="response_mismatch_cost_mean",
            ),
            "lowest_mean_response_mismatch_tangent_method": self._best_method(
                aggregate,
                metric="response_mismatch_cost_mean",
                kinds=tangent_kinds,
            ),
            "lowest_mean_normalized_response_ratio_tangent_method": (
                self._best_method(
                    aggregate,
                    metric="response_mismatch_ratio_vs_symmetric_mean",
                    kinds=tangent_kinds,
                )
            ),
            "lowest_mean_canonical_energy_method": self._best_method(
                aggregate,
                metric="canonical_energy_mean",
            ),
            "lowest_mean_canonical_energy_tangent_method": self._best_method(
                aggregate,
                metric="canonical_energy_mean",
                kinds=tangent_kinds,
            ),
            "lowest_mean_rel_sol_method_evaluation_only": self._best_method(
                aggregate,
                metric="rel_sol_mean",
            ),
            "lowest_mean_rel_flux_method_evaluation_only": self._best_method(
                aggregate,
                metric="rel_flux_mean",
            ),
            "closed_loop_method": closed_loop_method,
        }

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
        baseline_method = (
            "configured_column"
            if any(row["method_id"] == "configured_column" for row in rows)
            else "symmetric"
        )
        configured = [row for row in rows if row["method_id"] == baseline_method]
        metric = (
            "rel_sol"
            if configured and "rel_sol" in configured[0]
            else ("response_mismatch_cost")
        )
        ordered = sorted(configured, key=lambda row: float(row[metric]))
        selected: list[int] = []
        roles: dict[str, str] = {}
        for quantile in (0.0, 0.25, 0.5, 0.75, 1.0):
            index = round(quantile * max(len(ordered) - 1, 0))
            sample_id = int(ordered[index]["sample_id"])
            if sample_id not in selected:
                selected.append(sample_id)
                roles[str(sample_id)] = (
                    f"{baseline_method}_{metric}_q{int(100 * quantile):02d}"
                )
        return tuple(selected), roles

    def _selected_method_ids(
        self,
        findings: dict[str, str | None],
    ) -> tuple[str, ...]:
        ordered = [
            "symmetric",
            findings["configured_projection_method"],
            *(
                method.method_id
                for method in self.methods
                if method.kind == "closed_loop_exact_line_search"
            ),
            findings["lowest_mean_response_mismatch_tangent_method"],
            findings["lowest_mean_normalized_response_ratio_tangent_method"],
            findings["lowest_mean_canonical_energy_tangent_method"],
        ]
        selected: list[str] = []
        for method_id in ordered:
            if method_id is not None and method_id not in selected:
                selected.append(method_id)
        return tuple(selected)

    @staticmethod
    def _write_csv(
        path: Path,
        rows: Sequence[dict[str, float | int | str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise ValueError("Cannot write an empty tangent metric table.")
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_selected_arrays(
        self,
        *,
        batch: ComplexCouplingBatch,
        evaluation: TangentBatchEvaluation,
        method_indices: tuple[int, ...],
        edges: ProjectionTransitionEdges,
    ) -> None:
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "coords_valid": self.geometry.coords_valid.detach().cpu().numpy(),
            "selected_sample_ids": batch.sample_indices.detach().cpu().numpy(),
            "selected_file_stems": np.asarray(batch.file_stems),
            "selected_method_ids": np.asarray(
                [evaluation.methods[index].method_id for index in method_indices]
            ),
            "selected_method_eta": np.asarray(
                [
                    math.nan
                    if evaluation.methods[index].eta is None
                    else evaluation.methods[index].eta
                    for index in method_indices
                ],
                dtype=np.float64,
            ),
            "selected_method_relative_lambda": np.asarray(
                [
                    math.nan
                    if evaluation.methods[index].relative_lambda is None
                    else evaluation.methods[index].relative_lambda
                    for index in method_indices
                ],
                dtype=np.float64,
            ),
            "rhs": batch.rhs_valid.detach().cpu().numpy(),
            "sol": batch.sol_valid.detach().cpu().numpy(),
            "has_solution": batch.has_solution.detach().cpu().numpy(),
            "flux_target": batch.flux_valid.detach().cpu().numpy(),
            "has_flux": batch.has_flux.detach().cpu().numpy(),
            "raw_physical": evaluation.raw_physical.detach().cpu().numpy(),
            "symmetric_physical": evaluation.symmetric_physical.detach().cpu().numpy(),
            "configured_physical": evaluation.configured_physical.detach()
            .cpu()
            .numpy(),
            "tangent_gradient": evaluation.tangent_gradient.detach().cpu().numpy(),
            "tangent_preconditioner_base": (
                evaluation.tangent_preconditioner_base.detach().cpu().numpy()
            ),
            "tangent_delta": evaluation.tangent_delta[list(method_indices)]
            .detach()
            .cpu()
            .numpy(),
            "candidate_physical": evaluation.candidate_physical[list(method_indices)]
            .detach()
            .cpu()
            .numpy(),
            "candidate_solution": evaluation.candidate_solution[list(method_indices)]
            .detach()
            .cpu()
            .numpy(),
            "candidate_equal_prediction": evaluation.candidate_equal_prediction[
                list(method_indices)
            ]
            .detach()
            .cpu()
            .numpy(),
            "candidate_prediction": evaluation.candidate_prediction[
                list(method_indices)
            ]
            .detach()
            .cpu()
            .numpy(),
            "phi_transition_edges": edges.phi_transition.detach().cpu().numpy(),
            "psi_transition_edges": edges.psi_transition.detach().cpu().numpy(),
        }
        if evaluation.closed_loop:
            closed_loop = evaluation.closed_loop
            payload.update(
                {
                    "closed_loop_method_ids": np.asarray(
                        [diagnostics.method_id for diagnostics in closed_loop]
                    ),
                    "closed_loop_eta_caps": np.asarray(
                        [
                            math.nan
                            if diagnostics.eta_cap is None
                            else diagnostics.eta_cap
                            for diagnostics in closed_loop
                        ],
                        dtype=np.float64,
                    ),
                    "closed_loop_cap_is_unbounded": np.asarray(
                        [diagnostics.eta_cap is None for diagnostics in closed_loop],
                        dtype=np.bool_,
                    ),
                    "closed_loop_eta_star": torch.stack(
                        [diagnostics.eta_star for diagnostics in closed_loop]
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    "closed_loop_eta_applied": torch.stack(
                        [diagnostics.eta_applied for diagnostics in closed_loop]
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    "closed_loop_eta_capped": torch.stack(
                        [diagnostics.eta_capped for diagnostics in closed_loop]
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    "closed_loop_line_search_numerator": (
                        torch.stack(
                            [
                                diagnostics.line_search_numerator
                                for diagnostics in closed_loop
                            ]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "closed_loop_line_search_denominator": (
                        torch.stack(
                            [
                                diagnostics.line_search_denominator
                                for diagnostics in closed_loop
                            ]
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                }
            )
            if len(closed_loop) == 1:
                diagnostics = closed_loop[0]
                payload.update(
                    {
                        "closed_loop_method_id": np.asarray(diagnostics.method_id),
                        "closed_loop_eta_cap": np.asarray(
                            diagnostics.eta_cap,
                            dtype=np.float64,
                        ),
                        "closed_loop_eta_star": diagnostics.eta_star.detach()
                        .cpu()
                        .numpy(),
                        "closed_loop_eta_applied": diagnostics.eta_applied.detach()
                        .cpu()
                        .numpy(),
                        "closed_loop_eta_capped": diagnostics.eta_capped.detach()
                        .cpu()
                        .numpy(),
                        "closed_loop_line_search_numerator": (
                            diagnostics.line_search_numerator.detach().cpu().numpy()
                        ),
                        "closed_loop_line_search_denominator": (
                            diagnostics.line_search_denominator.detach().cpu().numpy()
                        ),
                    }
                )
        np.savez_compressed(
            data_dir / "selected_symmetric_tangent_audit.npz",
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
        findings: dict[str, str | None],
        selected: tuple[int, ...],
        roles: dict[str, str],
        method_ids: tuple[str, ...],
        edges: ProjectionTransitionEdges,
        figure_paths: Sequence[Path],
    ) -> dict[str, Any]:
        closed_loop_summary: dict[str, Any] = {
            "enabled": self.request.closed_loop_enabled,
            "eta_cap": self.request.closed_loop_eta_cap,
            "relative_lambda": self.request.closed_loop_relative_lambda,
            "line_search_relative_eps": self.request.line_search_relative_eps,
            "cap_policy": "final_cap_posthoc_evaluation",
            "sample_adaptive": True,
            "batch_independent": True,
            "reference_targets_used": False,
        }
        if self.request.closed_loop_eta_caps is not None:
            caps = self.request.resolved_closed_loop_eta_caps
            closed_loop_summary.update(
                {
                    "eta_cap": None,
                    "eta_caps": [None if math.isinf(cap) else cap for cap in caps],
                    "uncapped_included": any(math.isinf(cap) for cap in caps),
                    "shared_eta_star_and_direction": True,
                }
            )
        return {
            "diagnostic": "symmetric_tangent_response_gradient_posthoc_audit",
            "status": "frozen_checkpoint_posthoc",
            "production_code_changed": False,
            "training_or_checkpoint_updated": False,
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": str(coefficient_path),
            "sample_count": len(dataset),
            "etas": list(self.request.etas),
            "relative_lambdas": list(self.request.relative_lambdas),
            "closed_loop_exact_line_search": closed_loop_summary,
            "configured_projection_mode": self._configured_projection_mode,
            "damping_formula": (
                "D=gamma_x_squared+gamma_y_squared+"
                "(lambda_relative+denominator_relative_eps)*mean(gamma_sum)"
            ),
            "tangent_formula": {
                "symmetric_pair": ("p_tilde=0.5*(f+p-q); q_tilde=0.5*(f-p+q)"),
                "mismatch": "m0=H_x*p_tilde-H_y*q_tilde",
                "gradient": "g=(H_x+H_y)^T*M_Omega*m0",
                "update": "delta=-eta*D^{-1}*g",
                "closed_loop_update": (
                    "z=D^{-1}g; v=(H_x+H_y)z; "
                    "eta_star=(g^T z)/(<v,v>_M+eps_sample); "
                    "eta_applied=min(eta_star,eta_cap); delta=-eta_applied*z"
                ),
                "balanced_pair": "phi=p_tilde+delta; psi=q_tilde-delta",
            },
            "matrix_policy": {
                "global_matrix_materialized": False,
                "global_matrix_solve": False,
                "segment_local_kernel_blocks": True,
                "transpose_action": "segment_local_matvec",
                "column_cross_terms_in_preconditioner": False,
                "response_context_build_count": self._context_build_count,
                "response_operator_build_count": self._operator_build_count,
                "operator_production_equivalence_max_abs": (
                    self._operator_equivalence_max_abs
                ),
            },
            "operator_statistics": {
                "x": self.response_operator.x.statistics(),
                "y": self.response_operator.y.statistics(),
            },
            "reference_policy": {
                "primary_method_selection_metric": "response_mismatch_cost",
                "secondary_normalized_metric": (
                    "mean_per_sample_response_mismatch_ratio_vs_symmetric"
                ),
                "sol_and_flux_used_for_update": False,
                "sol_and_flux_used_for_evaluation_only": True,
            },
            "cross_axis_reconstruction": {
                "enabled": self._cross_axis_reconstructor.config.enabled,
                "mode": self._cross_axis_reconstructor.config.mode,
                "rel_sol_uses_configured_production_blend": True,
                "rel_sol_equal_mean_reported_separately": True,
            },
            "transition_definition": {
                "log_threshold": self.request.transition_log_threshold,
                "phi_transition_edge_count": int(edges.phi_transition.shape[0]),
                "psi_transition_edge_count": int(edges.psi_transition.shape[0]),
            },
            "aggregate_metrics": aggregate,
            "automated_findings": findings,
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "selected_method_ids": list(method_ids),
            "metric_csv": "metrics/per_sample_tangent_sweep.csv",
            "raw_archive": (
                "data/selected_symmetric_tangent_audit.npz"
                if self.request.save_generated_data
                else None
            ),
            "figure_count": len(figure_paths),
            "figure_json": [
                str(path.relative_to(self.request.outdir)) for path in figure_paths
            ],
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate_metrics"]
        findings = summary["automated_findings"]
        method_ids = self._selected_method_ids(findings)
        lines = [
            "# Symmetric Tangent Response-Gradient Post-Hoc Audit",
            "",
            "## Scope",
            "",
            "The CouplingNet and GreenNet checkpoints are frozen. Every candidate",
            "starts from the exact-balanced symmetric pair. Tangent updates use only",
            "rhs, the frozen axial Green response, and reconstructed mismatch.",
            "Reference solution and directional targets are evaluation-only.",
            "",
            "## Update",
            "",
            "`m0=H_x*p_tilde-H_y*q_tilde`,",
            "`g=(H_x+H_y)^T*M_Omega*m0`, and",
            "fixed candidates use `delta=-eta*D^{-1}*g`. The optional closed-loop",
            "candidate uses `z=D^{-1}g`, `v=(H_x+H_y)z`,",
            "`eta_star=(g^T*z)/(<v,v>_M+eps)`, and",
            "`delta=-min(eta_star,eta_cap)*z`. Every candidate uses",
            "`phi=p_tilde+delta`, `psi=q_tilde-delta`. No global matrix or solve",
            "is used.",
            "",
            "## Selected Aggregate Results",
            "",
            "| method | mismatch / sym | canonical / sym | rel_sol | "
            "rel_sol equal | rel_flux | correction / sym |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method_id in method_ids:
            payload = aggregate[method_id]
            lines.append(
                "| "
                f"{payload['method_label']} | "
                f"{float(payload.get('response_mismatch_ratio_vs_symmetric_mean', math.nan)):.6f} | "
                f"{float(payload.get('canonical_energy_ratio_vs_symmetric_mean', math.nan)):.6f} | "
                f"{float(payload.get('rel_sol_mean', math.nan)):.6f} | "
                f"{float(payload.get('rel_sol_equal_mean_mean', math.nan)):.6f} | "
                f"{float(payload.get('rel_flux_mean', math.nan)):.6f} | "
                f"{float(payload.get('tangent_correction_rel_symmetric_pair_mean', math.nan)):.6f} |"
            )
        lines.extend(
            [
                "",
                "## Automated Findings",
                "",
                "- Lowest reference-free response mismatch: "
                f"`{findings['lowest_mean_response_mismatch_method']}`.",
                "- Lowest tangent-grid response mismatch: "
                f"`{findings['lowest_mean_response_mismatch_tangent_method']}`.",
                "- Lowest tangent-grid mean per-sample normalized response ratio: "
                f"`{findings['lowest_mean_normalized_response_ratio_tangent_method']}`.",
                "- Lowest reference-free canonical energy: "
                f"`{findings['lowest_mean_canonical_energy_method']}`.",
                "- Lowest tangent-grid canonical energy: "
                f"`{findings['lowest_mean_canonical_energy_tangent_method']}`.",
                "- Lowest evaluation-only rel_sol: "
                f"`{findings['lowest_mean_rel_sol_method_evaluation_only']}`.",
                "- Lowest evaluation-only rel_flux: "
                f"`{findings['lowest_mean_rel_flux_method_evaluation_only']}`.",
                "",
                "## Interpretation Boundary",
                "",
                "This audit tests fixed and optional sample-adaptive",
                "Jacobi-preconditioned tangent steps on one",
                "frozen checkpoint. It can reject unstable eta/lambda regions and show",
                "whether the reference-free surrogate improves the intended response",
                "metric. It does not replace paired retraining if the production",
                "projection is changed.",
            ]
        )
        (self.request.outdir / "diagnosis_report.md").write_text(
            "\n".join(lines) + "\n"
        )


def run_symmetric_tangent_response_audit(
    request: SymmetricTangentAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the frozen symmetric-tangent response-gradient audit."""

    return ComplexSymmetricTangentAudit(request, logger=logger).run()
