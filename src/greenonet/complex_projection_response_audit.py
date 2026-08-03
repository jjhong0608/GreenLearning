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
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
    ColumnDiagonalGreenResponseContextBuilder,
)
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.complex_reconstruction import reconstruct_from_projected_response
from greenonet.config import (
    BalanceProjectionConfig,
    ColumnDiagonalGreenResponseProjectionConfig,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class ProjectionResponseAuditRequest:
    """Frozen-checkpoint column-diagonal projection audit inputs."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    transition_log_threshold: float = math.log(2.0)
    selected_samples: tuple[int, ...] | None = None
    batch_size: int = 10
    metric_eps: float = 1.0e-30
    save_generated_data: bool = True

    def __post_init__(self) -> None:
        if not self.alphas:
            raise ValueError("alphas must contain at least alpha=0 and alpha=1.")
        if any(
            not isinstance(alpha, (int, float))
            or isinstance(alpha, bool)
            or not math.isfinite(float(alpha))
            or not 0.0 <= float(alpha) <= 1.0
            for alpha in self.alphas
        ):
            raise ValueError("alphas must contain finite numeric values in [0, 1].")
        normalized = tuple(float(alpha) for alpha in self.alphas)
        if len(set(normalized)) != len(normalized):
            raise ValueError("alphas must not contain duplicate values.")
        if 0.0 not in normalized or 1.0 not in normalized:
            raise ValueError("alphas must include 0.0 and 1.0.")
        if (
            not math.isfinite(self.transition_log_threshold)
            or self.transition_log_threshold < 0.0
        ):
            raise ValueError(
                "transition_log_threshold must be finite and non-negative."
            )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if not math.isfinite(self.metric_eps) or self.metric_eps <= 0.0:
            raise ValueError("metric_eps must be finite and positive.")
        if self.selected_samples is not None and len(set(self.selected_samples)) != len(
            self.selected_samples
        ):
            raise ValueError("selected_samples must not contain duplicates.")


@dataclass(frozen=True)
class ProjectionTransitionEdges:
    """Cross-axis edges separated by the line-response jump criterion."""

    phi_transition: torch.Tensor
    psi_transition: torch.Tensor
    phi_regular: torch.Tensor
    psi_regular: torch.Tensor

    @property
    def transition(self) -> torch.Tensor:
        return torch.cat((self.phi_transition, self.psi_transition), dim=0)

    @property
    def regular(self) -> torch.Tensor:
        return torch.cat((self.phi_regular, self.psi_regular), dim=0)


@dataclass(frozen=True)
class ProjectionResponseAuditEvaluation:
    """CPU tensors required for alpha-wise projection response comparisons."""

    sample_ids: torch.Tensor
    file_stems: tuple[str, ...]
    has_solution: torch.Tensor
    has_flux: torch.Tensor
    rhs: torch.Tensor
    sol: torch.Tensor
    flux_target: torch.Tensor
    raw_response: torch.Tensor
    raw_physical: torch.Tensor
    raw_balance_residual: torch.Tensor
    weights_phi: torch.Tensor
    projected_physical: torch.Tensor
    correction_physical: torch.Tensor
    correction_response: torch.Tensor
    correction_solution: torch.Tensor
    raw_solution: torch.Tensor

    @property
    def final_solution(self) -> torch.Tensor:
        return self.raw_solution.unsqueeze(0) + self.correction_solution

    @property
    def final_equal_prediction(self) -> torch.Tensor:
        return self.final_solution.mean(dim=2)

    @property
    def symmetric_balanced_physical(self) -> torch.Tensor:
        """Return the exact-balanced pair that preserves the raw difference."""

        raw_difference = self.raw_physical[:, 0] - self.raw_physical[:, 1]
        return torch.stack(
            (
                0.5 * (self.rhs + raw_difference),
                0.5 * (self.rhs - raw_difference),
            ),
            dim=1,
        )


class ProjectionResponseAuditMetricsMixin:
    """Compute learned Green-response costs and transition trace jumps."""

    @staticmethod
    def build_transition_edges(
        geometry: ComplexGeometryMetadata,
        *,
        threshold: float,
    ) -> ProjectionTransitionEdges:
        sigma_x = geometry.x_lengths_for_valid_points().square()
        sigma_y = geometry.y_lengths_for_valid_points().square()
        if torch.any(sigma_x <= 0.0) or torch.any(sigma_y <= 0.0):
            raise ValueError("All pointwise segment response scales must be positive.")
        phi_edges = geometry.y_edges.to(dtype=torch.long)
        psi_edges = geometry.x_edges.to(dtype=torch.long)
        phi_score = torch.abs(
            torch.log(sigma_x[phi_edges[:, 1]]) - torch.log(sigma_x[phi_edges[:, 0]])
        )
        psi_score = torch.abs(
            torch.log(sigma_y[psi_edges[:, 1]]) - torch.log(sigma_y[psi_edges[:, 0]])
        )
        phi_mask = phi_score > threshold
        psi_mask = psi_score > threshold
        return ProjectionTransitionEdges(
            phi_transition=phi_edges[phi_mask],
            psi_transition=psi_edges[psi_mask],
            phi_regular=phi_edges[~phi_mask],
            psi_regular=psi_edges[~psi_mask],
        )

    @staticmethod
    def _edge_squared_jump(values: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        if edges.numel() == 0:
            return values.new_empty((*values.shape[:-1], 0))
        return (values[..., edges[:, 1]] - values[..., edges[:, 0]]).square()

    @classmethod
    def _directional_edge_rms(
        cls,
        values: torch.Tensor,
        phi_edges: torch.Tensor,
        psi_edges: torch.Tensor,
    ) -> torch.Tensor:
        phi_squared = cls._edge_squared_jump(values[:, :, 0], phi_edges)
        psi_squared = cls._edge_squared_jump(values[:, :, 1], psi_edges)
        count = int(phi_squared.shape[-1] + psi_squared.shape[-1])
        if count == 0:
            return values.new_full(values.shape[:2], math.nan)
        return torch.sqrt(
            (phi_squared.sum(dim=-1) + psi_squared.sum(dim=-1)) / float(count)
        )

    @classmethod
    def _scalar_edge_rms(
        cls,
        values: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        squared = cls._edge_squared_jump(values, edges)
        if squared.shape[-1] == 0:
            return values.new_full(values.shape[:2], math.nan)
        return torch.sqrt(squared.mean(dim=-1))

    @staticmethod
    def _relative_l2_per_sample(
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
    def _finite_mean(values: Sequence[float]) -> float:
        array = np.asarray(values, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return math.nan if finite.size == 0 else float(finite.mean())

    @staticmethod
    def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        valid = denominator > 0.0
        return torch.where(
            valid,
            numerator / denominator.clamp_min(torch.finfo(denominator.dtype).tiny),
            torch.full_like(numerator, math.nan),
        )

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
    def _relative_correction_l2(
        proposal: torch.Tensor,
        projected: torch.Tensor,
        *,
        eps: float,
    ) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(
            (projected - proposal).flatten(start_dim=1),
            dim=-1,
        )
        denominator = torch.linalg.vector_norm(
            projected.flatten(start_dim=1),
            dim=-1,
        ).clamp_min(eps)
        result: torch.Tensor = numerator / denominator
        return result

    @staticmethod
    def _candidate_point_indices(
        edges: ProjectionTransitionEdges,
        *,
        point_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edges.transition.numel() == 0:
            transition = torch.empty(0, dtype=torch.long)
        else:
            transition = torch.unique(edges.transition.flatten())
        mask = torch.ones(point_count, dtype=torch.bool)
        mask[transition] = False
        regular = torch.nonzero(mask, as_tuple=False).flatten()
        return transition, regular

    def build_candidate_rows(
        self,
        *,
        evaluation: ProjectionResponseAuditEvaluation,
        edges: ProjectionTransitionEdges,
        configured_alpha: float,
        alphas: tuple[float, ...],
        eps: float,
    ) -> list[dict[str, float | int | str]]:
        """Compare raw, symmetric-balanced, and configured projected sources."""

        alpha_index = alphas.index(configured_alpha)
        raw = evaluation.raw_physical
        symmetric = evaluation.symmetric_balanced_physical
        projected = evaluation.projected_physical[alpha_index]
        transition, regular = self._candidate_point_indices(
            edges,
            point_count=int(evaluation.rhs.shape[-1]),
        )

        raw_balance_norm = torch.linalg.vector_norm(
            evaluation.raw_balance_residual,
            dim=-1,
        )
        rhs_norm = torch.linalg.vector_norm(evaluation.rhs, dim=-1).clamp_min(eps)
        symmetric_correction = self._relative_correction_l2(
            raw,
            symmetric,
            eps=eps,
        )
        projected_correction = self._relative_correction_l2(
            raw,
            projected,
            eps=eps,
        )
        tangent_transfer = self._relative_correction_l2(
            symmetric,
            projected,
            eps=eps,
        )

        rows: list[dict[str, float | int | str]] = []
        for sample_offset, sample_id in enumerate(evaluation.sample_ids.tolist()):
            row: dict[str, float | int | str] = {
                "sample_id": int(sample_id),
                "file_stem": evaluation.file_stems[sample_offset],
                "configured_gain_exponent": configured_alpha,
                "raw_balance_residual_rms": float(
                    torch.sqrt(
                        evaluation.raw_balance_residual[sample_offset].square().mean()
                    ).item()
                ),
                "raw_balance_residual_rel_rhs": float(
                    (raw_balance_norm[sample_offset] / rhs_norm[sample_offset]).item()
                ),
                "symmetric_correction_rel_balanced_pair": float(
                    symmetric_correction[sample_offset].item()
                ),
                "configured_correction_rel_projected_pair": float(
                    projected_correction[sample_offset].item()
                ),
                "configured_tangent_transfer_rel_projected_pair": float(
                    tangent_transfer[sample_offset].item()
                ),
                "configured_correction_phi_rel_projected": float(
                    self._relative_l2_per_sample(
                        raw[sample_offset, 0].unsqueeze(0),
                        projected[sample_offset, 0].unsqueeze(0),
                        eps=eps,
                    ).item()
                ),
                "configured_correction_psi_rel_projected": float(
                    self._relative_l2_per_sample(
                        raw[sample_offset, 1].unsqueeze(0),
                        projected[sample_offset, 1].unsqueeze(0),
                        eps=eps,
                    ).item()
                ),
                "symmetric_balance_max_abs": float(
                    torch.abs(
                        evaluation.rhs[sample_offset]
                        - symmetric[sample_offset].sum(dim=0)
                    )
                    .max()
                    .item()
                ),
                "configured_balance_max_abs": float(
                    torch.abs(
                        evaluation.rhs[sample_offset]
                        - projected[sample_offset].sum(dim=0)
                    )
                    .max()
                    .item()
                ),
            }
            if bool(evaluation.has_flux[sample_offset].item()):
                target = evaluation.flux_target[sample_offset]
                stage_fields = {
                    "raw": raw[sample_offset],
                    "symmetric": symmetric[sample_offset],
                    "configured": projected[sample_offset],
                }
                for stage, field in stage_fields.items():
                    row[f"{stage}_pair_rel_target"] = float(
                        self._relative_pair_l2(
                            field.unsqueeze(0),
                            target.unsqueeze(0),
                            eps=eps,
                        ).item()
                    )
                    for axis_index, axis in enumerate(("phi", "psi")):
                        row[f"{stage}_{axis}_rel_target"] = float(
                            self._relative_l2_per_sample(
                                field[axis_index].unsqueeze(0),
                                target[axis_index].unsqueeze(0),
                                eps=eps,
                            ).item()
                        )
                    if transition.numel() > 0:
                        row[f"{stage}_transition_pair_rel_target"] = float(
                            self._relative_pair_l2(
                                field[:, transition].unsqueeze(0),
                                target[:, transition].unsqueeze(0),
                                eps=eps,
                            ).item()
                        )
                    if regular.numel() > 0:
                        row[f"{stage}_regular_pair_rel_target"] = float(
                            self._relative_pair_l2(
                                field[:, regular].unsqueeze(0),
                                target[:, regular].unsqueeze(0),
                                eps=eps,
                            ).item()
                        )
                target_difference = target[0] - target[1]
                raw_difference = raw[sample_offset, 0] - raw[sample_offset, 1]
                projected_difference = (
                    projected[sample_offset, 0] - projected[sample_offset, 1]
                )
                row["raw_difference_rel_target"] = float(
                    self._relative_l2_per_sample(
                        raw_difference.unsqueeze(0),
                        target_difference.unsqueeze(0),
                        eps=eps,
                    ).item()
                )
                row["configured_difference_rel_target"] = float(
                    self._relative_l2_per_sample(
                        projected_difference.unsqueeze(0),
                        target_difference.unsqueeze(0),
                        eps=eps,
                    ).item()
                )
                row["symmetric_pair_error_ratio_vs_raw"] = float(
                    float(row["symmetric_pair_rel_target"])
                    / max(float(row["raw_pair_rel_target"]), eps)
                )
                row["configured_pair_error_ratio_vs_symmetric"] = float(
                    float(row["configured_pair_rel_target"])
                    / max(float(row["symmetric_pair_rel_target"]), eps)
                )
            rows.append(row)
        return rows

    def aggregate_candidate_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
    ) -> dict[str, dict[str, float | int]]:
        metric_keys: list[str] = []
        for row in rows:
            for key, value in row.items():
                if key in {"sample_id", "file_stem", "configured_gain_exponent"}:
                    continue
                if isinstance(value, (int, float)) and key not in metric_keys:
                    metric_keys.append(key)
        aggregate: dict[str, dict[str, float | int]] = {
            "sample_count": {"value": len(rows)}
        }
        for key in metric_keys:
            values = np.asarray(
                [float(row[key]) for row in rows if key in row],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            aggregate[key] = {
                "count": int(finite.size),
                "mean": float(finite.mean()),
                "median": float(np.median(finite)),
                "q25": float(np.quantile(finite, 0.25)),
                "q75": float(np.quantile(finite, 0.75)),
                "max": float(finite.max()),
            }
        return aggregate

    def build_metric_rows(
        self,
        *,
        evaluation: ProjectionResponseAuditEvaluation,
        contexts: tuple[ColumnDiagonalGreenResponseContext, ...],
        edges: ProjectionTransitionEdges,
        point_mass: torch.Tensor,
        alphas: tuple[float, ...],
        eps: float,
    ) -> list[dict[str, float | int | str]]:
        correction = evaluation.correction_physical
        correction_solution = evaluation.correction_solution
        actual_cost = point_mass * correction_solution.square().sum(dim=(-1, -2))
        equal_correction = correction_solution.mean(dim=2)
        equal_response_cost = point_mass * equal_correction.square().sum(dim=-1)
        residual_cost = point_mass * evaluation.raw_balance_residual.square().sum(
            dim=-1
        )
        response_gain = torch.sqrt(
            self._safe_ratio(actual_cost, residual_cost.unsqueeze(0))
        )
        diagonal_cost = torch.stack(
            [
                (
                    context.regularized_gamma_x_squared.cpu()
                    * correction[alpha_index, :, 0].square()
                    + context.regularized_gamma_y_squared.cpu()
                    * correction[alpha_index, :, 1].square()
                ).sum(dim=-1)
                for alpha_index, context in enumerate(contexts)
            ],
            dim=0,
        )
        directional_transition_rms = self._directional_edge_rms(
            correction_solution,
            edges.phi_transition,
            edges.psi_transition,
        )
        directional_regular_rms = self._directional_edge_rms(
            correction_solution,
            edges.phi_regular,
            edges.psi_regular,
        )
        source_transition_rms = self._directional_edge_rms(
            correction,
            edges.phi_transition,
            edges.psi_transition,
        )
        source_regular_rms = self._directional_edge_rms(
            correction,
            edges.phi_regular,
            edges.psi_regular,
        )
        equal_transition_rms = self._scalar_edge_rms(
            equal_correction,
            edges.transition,
        )
        equal_regular_rms = self._scalar_edge_rms(
            equal_correction,
            edges.regular,
        )
        balance_max = torch.abs(
            evaluation.rhs.unsqueeze(0) - evaluation.projected_physical.sum(dim=2)
        ).amax(dim=-1)

        symmetric_index = alphas.index(0.0)
        actual_ratio = self._safe_ratio(
            actual_cost,
            actual_cost[symmetric_index].unsqueeze(0),
        )
        diagonal_ratio = self._safe_ratio(
            diagonal_cost,
            diagonal_cost[symmetric_index].unsqueeze(0),
        )
        transition_ratio = self._safe_ratio(
            directional_transition_rms,
            directional_transition_rms[symmetric_index].unsqueeze(0),
        )
        source_transition_ratio = self._safe_ratio(
            source_transition_rms,
            source_transition_rms[symmetric_index].unsqueeze(0),
        )

        rows: list[dict[str, float | int | str]] = []
        for alpha_index, alpha in enumerate(alphas):
            for sample_offset, sample_id in enumerate(evaluation.sample_ids.tolist()):
                row: dict[str, float | int | str] = {
                    "sample_id": int(sample_id),
                    "file_stem": evaluation.file_stems[sample_offset],
                    "gain_exponent": alpha,
                    "raw_balance_residual_rms": float(
                        torch.sqrt(
                            evaluation.raw_balance_residual[sample_offset]
                            .square()
                            .mean()
                        ).item()
                    ),
                    "diagonal_surrogate_cost": float(
                        diagonal_cost[alpha_index, sample_offset].item()
                    ),
                    "diagonal_surrogate_cost_ratio_vs_symmetric": float(
                        diagonal_ratio[alpha_index, sample_offset].item()
                    ),
                    "actual_directional_response_cost": float(
                        actual_cost[alpha_index, sample_offset].item()
                    ),
                    "actual_directional_response_cost_ratio_vs_symmetric": float(
                        actual_ratio[alpha_index, sample_offset].item()
                    ),
                    "equal_correction_response_cost": float(
                        equal_response_cost[alpha_index, sample_offset].item()
                    ),
                    "response_gain": float(
                        response_gain[alpha_index, sample_offset].item()
                    ),
                    "directional_transition_source_correction_jump_rms": float(
                        source_transition_rms[alpha_index, sample_offset].item()
                    ),
                    "directional_transition_source_jump_ratio_vs_symmetric": float(
                        source_transition_ratio[alpha_index, sample_offset].item()
                    ),
                    "directional_regular_source_correction_jump_rms": float(
                        source_regular_rms[alpha_index, sample_offset].item()
                    ),
                    "directional_transition_correction_jump_rms": float(
                        directional_transition_rms[alpha_index, sample_offset].item()
                    ),
                    "directional_transition_jump_ratio_vs_symmetric": float(
                        transition_ratio[alpha_index, sample_offset].item()
                    ),
                    "directional_regular_correction_jump_rms": float(
                        directional_regular_rms[alpha_index, sample_offset].item()
                    ),
                    "equal_transition_correction_jump_rms": float(
                        equal_transition_rms[alpha_index, sample_offset].item()
                    ),
                    "equal_regular_correction_jump_rms": float(
                        equal_regular_rms[alpha_index, sample_offset].item()
                    ),
                    "projected_balance_max_abs": float(
                        balance_max[alpha_index, sample_offset].item()
                    ),
                }
                if bool(evaluation.has_solution[sample_offset].item()):
                    final_prediction = evaluation.final_equal_prediction[
                        alpha_index, sample_offset
                    ]
                    solution = evaluation.sol[sample_offset]
                    error = final_prediction - solution
                    row["equal_mean_rel_sol"] = float(
                        self._relative_l2_per_sample(
                            final_prediction.unsqueeze(0),
                            solution.unsqueeze(0),
                            eps=eps,
                        ).item()
                    )
                    row["equal_mean_error_rms"] = float(
                        torch.sqrt(error.square().mean()).item()
                    )
                    row["equal_mean_transition_error_jump_rms"] = float(
                        self._scalar_edge_rms(
                            error.unsqueeze(0).unsqueeze(0),
                            edges.transition,
                        )[0, 0].item()
                    )
                if bool(evaluation.has_flux[sample_offset].item()):
                    flux_prediction = evaluation.projected_physical[
                        alpha_index, sample_offset
                    ]
                    flux_target = evaluation.flux_target[sample_offset]
                    flux_axis_relative = torch.linalg.vector_norm(
                        flux_prediction - flux_target,
                        dim=-1,
                    ) / torch.linalg.vector_norm(flux_target, dim=-1).clamp_min(eps)
                    row["rel_flux"] = float(flux_axis_relative.mean().item())
                rows.append(row)
        return rows

    def aggregate_metric_rows(
        self,
        rows: Sequence[dict[str, float | int | str]],
        *,
        alphas: tuple[float, ...],
    ) -> dict[str, dict[str, float | int]]:
        metric_keys = (
            "diagonal_surrogate_cost",
            "diagonal_surrogate_cost_ratio_vs_symmetric",
            "actual_directional_response_cost",
            "actual_directional_response_cost_ratio_vs_symmetric",
            "equal_correction_response_cost",
            "response_gain",
            "directional_transition_source_correction_jump_rms",
            "directional_transition_source_jump_ratio_vs_symmetric",
            "directional_regular_source_correction_jump_rms",
            "directional_transition_correction_jump_rms",
            "directional_transition_jump_ratio_vs_symmetric",
            "directional_regular_correction_jump_rms",
            "equal_transition_correction_jump_rms",
            "equal_regular_correction_jump_rms",
            "projected_balance_max_abs",
            "equal_mean_rel_sol",
            "equal_mean_error_rms",
            "equal_mean_transition_error_jump_rms",
            "rel_flux",
        )
        by_alpha: dict[str, dict[str, float | int]] = {}
        symmetric_rows = [row for row in rows if float(row["gain_exponent"]) == 0.0]
        symmetric_by_sample = {int(row["sample_id"]): row for row in symmetric_rows}
        for alpha in alphas:
            alpha_rows = [
                row for row in rows if float(row["gain_exponent"]) == float(alpha)
            ]
            metrics: dict[str, float | int] = {"sample_count": len(alpha_rows)}
            for key in metric_keys:
                values = [float(row[key]) for row in alpha_rows if key in row]
                if values:
                    metrics[f"{key}_mean"] = self._finite_mean(values)
            for key in (
                "actual_directional_response_cost",
                "directional_transition_correction_jump_rms",
                "equal_mean_rel_sol",
                "rel_flux",
            ):
                wins = 0
                comparable = 0
                for row in alpha_rows:
                    baseline = symmetric_by_sample[int(row["sample_id"])]
                    if key in row and key in baseline:
                        comparable += 1
                        wins += float(row[key]) < float(baseline[key])
                if comparable:
                    metrics[f"{key}_wins_vs_symmetric"] = wins
            by_alpha[self._alpha_label(alpha)] = metrics
        return by_alpha

    @staticmethod
    def _alpha_label(alpha: float) -> str:
        return f"alpha_{alpha:g}".replace(".", "p")


class ProjectionResponseAuditPlotMixin:
    """Write compact Plotly evidence for the projection response audit."""

    @staticmethod
    def _scatter(
        *,
        geometry: ComplexGeometryMetadata,
        values: torch.Tensor,
        title: str,
        colorscale: str,
        symmetric: bool = False,
        show_scale: bool = True,
    ) -> go.Scattergl:
        array = values.detach().cpu().numpy()
        marker: dict[str, Any] = {
            "size": 4,
            "color": array,
            "colorscale": colorscale,
            "showscale": show_scale,
            "colorbar": {"title": title} if show_scale else None,
        }
        if symmetric:
            limit = float(np.max(np.abs(array), initial=0.0))
            if limit == 0.0:
                limit = 1.0
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

    def _write_weight_figure(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        weights: torch.Tensor,
        alphas: tuple[float, ...],
        theme: str,
        outdir: Path,
        logger: logging.Logger | None,
    ) -> Path:
        columns = 2
        rows = math.ceil(len(alphas) / columns)
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[f"alpha={alpha:g}" for alpha in alphas],
        )
        for index, alpha in enumerate(alphas):
            trace = self._scatter(
                geometry=geometry,
                values=weights[index],
                title=f"w_phi alpha={alpha:g}",
                colorscale="Viridis",
                show_scale=index == len(alphas) - 1,
            )
            trace.marker.cmin = 0.0
            trace.marker.cmax = 1.0
            fig.add_trace(
                trace,
                row=index // columns + 1,
                col=index % columns + 1,
            )
        fig.update_layout(
            template=theme,
            title="Column-diagonal correction weight by fixed gain exponent",
            height=450 * rows,
            width=1100,
            showlegend=False,
        )
        for index in range(len(alphas)):
            axis_reference = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis_reference,
                scaleratio=1.0,
                row=index // columns + 1,
                col=index % columns + 1,
            )
        path = outdir / "figures" / "geometry" / "correction_weight_phi"
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")

    def _write_aggregate_figure(
        self,
        *,
        rows: Sequence[dict[str, float | int | str]],
        alphas: tuple[float, ...],
        theme: str,
        outdir: Path,
        logger: logging.Logger | None,
    ) -> Path:
        keys = (
            (
                "actual_directional_response_cost_ratio_vs_symmetric",
                "Actual response cost / symmetric",
            ),
            (
                "diagonal_surrogate_cost_ratio_vs_symmetric",
                "Diagonal surrogate / symmetric",
            ),
            (
                "directional_transition_jump_ratio_vs_symmetric",
                "Transition jump / symmetric",
            ),
        )
        fig = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in keys])
        for col, (key, _label) in enumerate(keys, start=1):
            for alpha in alphas:
                values = [
                    float(row[key])
                    for row in rows
                    if float(row["gain_exponent"]) == alpha and key in row
                ]
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=f"alpha={alpha:g}",
                        boxmean=True,
                        showlegend=col == 1,
                    ),
                    row=1,
                    col=col,
                )
            fig.add_hline(y=1.0, line_dash="dash", line_color="#555", row=1, col=col)
        fig.update_layout(
            template=theme,
            title="Frozen raw-output projection response audit",
            height=520,
            width=1400,
        )
        path = outdir / "figures" / "aggregate" / "projection_response_ratios"
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")

    def _write_selected_figure(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        evaluation: ProjectionResponseAuditEvaluation,
        sample_offset: int,
        alpha_index: int,
        symmetric_index: int,
        alpha: float,
        theme: str,
        outdir: Path,
        logger: logging.Logger | None,
    ) -> Path:
        residual = evaluation.raw_balance_residual[sample_offset]
        correction = evaluation.correction_physical[alpha_index, sample_offset]
        delta_u = evaluation.correction_solution[alpha_index, sample_offset]
        fields: list[tuple[str, torch.Tensor, bool]] = [
            ("raw balance residual", residual, True),
            ("w_phi", evaluation.weights_phi[alpha_index], False),
            ("w_phi - 0.5", evaluation.weights_phi[alpha_index] - 0.5, True),
            ("delta phi", correction[0], True),
            ("delta psi", correction[1], True),
            ("delta phi - delta psi", correction[0] - correction[1], True),
            ("H_x delta phi", delta_u[0], True),
            ("H_y delta psi", delta_u[1], True),
            ("delta u_pred", delta_u.mean(dim=0), True),
        ]
        if bool(evaluation.has_solution[sample_offset].item()):
            solution = evaluation.sol[sample_offset]
            fields.extend(
                [
                    (
                        "symmetric final error",
                        evaluation.final_equal_prediction[
                            symmetric_index, sample_offset
                        ]
                        - solution,
                        True,
                    ),
                    (
                        f"alpha={alpha:g} final error",
                        evaluation.final_equal_prediction[alpha_index, sample_offset]
                        - solution,
                        True,
                    ),
                    (
                        "weighted - symmetric prediction",
                        evaluation.final_equal_prediction[alpha_index, sample_offset]
                        - evaluation.final_equal_prediction[
                            symmetric_index, sample_offset
                        ],
                        True,
                    ),
                ]
            )
        columns = 3
        rows = math.ceil(len(fields) / columns)
        fig = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=[name for name, _, _ in fields],
        )
        for index, (name, values, symmetric) in enumerate(fields):
            fig.add_trace(
                self._scatter(
                    geometry=geometry,
                    values=values,
                    title=name,
                    colorscale="RdBu" if symmetric else "Viridis",
                    symmetric=symmetric,
                    show_scale=False,
                ),
                row=index // columns + 1,
                col=index % columns + 1,
            )
        sample_id = int(evaluation.sample_ids[sample_offset].item())
        fig.update_layout(
            template=theme,
            title=(f"sample {sample_id}: correction response audit at alpha={alpha:g}"),
            height=380 * rows,
            width=1300,
            showlegend=False,
        )
        fig.update_annotations(font={"size": 12})
        for index in range(len(fields)):
            axis_reference = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis_reference,
                scaleratio=1.0,
                row=index // columns + 1,
                col=index % columns + 1,
            )
        path = (
            outdir
            / "figures"
            / "selected"
            / f"sample_{sample_id:04d}_projection_response_audit"
        )
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")

    def _write_candidate_figure(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        evaluation: ProjectionResponseAuditEvaluation,
        sample_offset: int,
        configured_index: int,
        configured_alpha: float,
        theme: str,
        outdir: Path,
        logger: logging.Logger | None,
    ) -> Path:
        if not bool(evaluation.has_flux[sample_offset].item()):
            raise ValueError("Directional candidate figures require flux targets.")
        raw = evaluation.raw_physical[sample_offset]
        symmetric = evaluation.symmetric_balanced_physical[sample_offset]
        projected = evaluation.projected_physical[configured_index, sample_offset]
        target = evaluation.flux_target[sample_offset]
        correction = projected - raw
        fields = (
            ("raw p - target phi", raw[0] - target[0]),
            ("symmetric p_tilde - target phi", symmetric[0] - target[0]),
            ("configured phi - target phi", projected[0] - target[0]),
            ("configured phi - raw p", correction[0]),
            ("raw q - target psi", raw[1] - target[1]),
            ("symmetric q_tilde - target psi", symmetric[1] - target[1]),
            ("configured psi - target psi", projected[1] - target[1]),
            ("configured psi - raw q", correction[1]),
        )
        phi_limit = max(
            float(torch.max(torch.abs(values)).item()) for _, values in fields[:3]
        )
        psi_limit = max(
            float(torch.max(torch.abs(values)).item()) for _, values in fields[4:7]
        )
        correction_limit = max(
            float(torch.max(torch.abs(fields[3][1])).item()),
            float(torch.max(torch.abs(fields[7][1])).item()),
        )
        limits = (
            phi_limit,
            phi_limit,
            phi_limit,
            correction_limit,
            psi_limit,
            psi_limit,
            psi_limit,
            correction_limit,
        )
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=[name for name, _ in fields],
        )
        for index, ((name, values), limit) in enumerate(
            zip(fields, limits, strict=True)
        ):
            trace = self._scatter(
                geometry=geometry,
                values=values,
                title=name,
                colorscale="RdBu",
                symmetric=True,
                show_scale=False,
            )
            stable_limit = limit if limit > 0.0 else 1.0
            trace.marker.cmin = -stable_limit
            trace.marker.cmax = stable_limit
            trace.marker.cmid = 0.0
            fig.add_trace(trace, row=index // 4 + 1, col=index % 4 + 1)
        sample_id = int(evaluation.sample_ids[sample_offset].item())
        fig.update_layout(
            template=theme,
            title=(
                f"sample {sample_id}: directional candidates at configured "
                f"alpha={configured_alpha:g}"
            ),
            height=760,
            width=1550,
            showlegend=False,
        )
        fig.update_annotations(font={"size": 11})
        for index in range(len(fields)):
            axis_reference = "x" if index == 0 else f"x{index + 1}"
            fig.update_yaxes(
                scaleanchor=axis_reference,
                scaleratio=1.0,
                row=index // 4 + 1,
                col=index % 4 + 1,
            )
        path = (
            outdir
            / "figures"
            / "selected"
            / f"sample_{sample_id:04d}_directional_candidate_audit"
        )
        save_plotly_figure(fig, path, logger)
        return path.with_suffix(".json")


class ComplexProjectionResponseAudit(
    ProjectionResponseAuditMetricsMixin,
    ProjectionResponseAuditPlotMixin,
):
    """Audit the immediate learned-response effect of fixed projection weights."""

    def __init__(
        self,
        request: ProjectionResponseAuditRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.geometry: ComplexGeometryMetadata
        self.contexts: tuple[ColumnDiagonalGreenResponseContext, ...]

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError("Projection response audit requires complex geometry.")
        geometry_path = self.request.geometry or configs.dataset.geometry_path
        test_path = self.request.test_path or configs.dataset.test_path
        coefficient_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")
        if not self.request.coupling_checkpoint.is_file():
            raise FileNotFoundError(self.request.coupling_checkpoint)
        if not self.request.green_checkpoint.is_file():
            raise FileNotFoundError(self.request.green_checkpoint)

        device = torch.device(self.request.device or configs.coupling_training.device)
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=configs.dataset.dtype,
        )
        coeffs = load_coefficient_functions(coefficient_path)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            coeffs,
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        if len(dataset) == 0:
            raise ValueError("The test dataset is empty.")

        resolved_projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        resolved_column_config = ColumnDiagonalGreenResponseProjectionConfig.from_raw(
            resolved_projection.column_diagonal_green_response
        )
        configured_alpha = float(resolved_column_config.gain_exponent)
        requested_alphas = tuple(float(alpha) for alpha in self.request.alphas)
        alphas = (
            requested_alphas
            if configured_alpha in requested_alphas
            else (*requested_alphas, configured_alpha)
        )

        evaluation, context_build_count = self._evaluate_dataset(
            dataset=dataset,
            configs=configs,
            device=device,
            alphas=alphas,
        )
        edges = self.build_transition_edges(
            self.geometry,
            threshold=self.request.transition_log_threshold,
        )
        point_mass = self.contexts[0].point_mass.cpu()
        rows = self.build_metric_rows(
            evaluation=evaluation,
            contexts=self.contexts,
            edges=edges,
            point_mass=point_mass,
            alphas=alphas,
            eps=self.request.metric_eps,
        )
        aggregate = self.aggregate_metric_rows(rows, alphas=alphas)
        candidate_rows = self.build_candidate_rows(
            evaluation=evaluation,
            edges=edges,
            configured_alpha=configured_alpha,
            alphas=alphas,
            eps=self.request.metric_eps,
        )
        candidate_aggregate = self.aggregate_candidate_rows(candidate_rows)
        selected, roles = self._select_samples(evaluation, rows, alphas)

        metrics_path = (
            self.request.outdir / "metrics" / "per_sample_projection_response_audit.csv"
        )
        self._write_csv(metrics_path, rows)
        candidate_metrics_path = (
            self.request.outdir
            / "metrics"
            / "per_sample_directional_candidate_audit.csv"
        )
        self._write_csv(candidate_metrics_path, candidate_rows)
        if self.request.save_generated_data:
            self._write_selected_arrays(
                evaluation,
                edges,
                selected,
                alphas,
                configured_alpha=configured_alpha,
            )

        figure_paths = [
            self._write_weight_figure(
                geometry=self.geometry,
                weights=evaluation.weights_phi,
                alphas=alphas,
                theme=self.request.theme,
                outdir=self.request.outdir,
                logger=self.logger,
            ),
            self._write_aggregate_figure(
                rows=rows,
                alphas=alphas,
                theme=self.request.theme,
                outdir=self.request.outdir,
                logger=self.logger,
            ),
        ]
        primary_alpha = configured_alpha
        primary_index = alphas.index(primary_alpha)
        symmetric_index = alphas.index(0.0)
        sample_offset_by_id = {
            int(sample_id): offset
            for offset, sample_id in enumerate(evaluation.sample_ids.tolist())
        }
        figure_paths.extend(
            self._write_selected_figure(
                geometry=self.geometry,
                evaluation=evaluation,
                sample_offset=sample_offset_by_id[sample_id],
                alpha_index=primary_index,
                symmetric_index=symmetric_index,
                alpha=primary_alpha,
                theme=self.request.theme,
                outdir=self.request.outdir,
                logger=self.logger,
            )
            for sample_id in selected
        )
        figure_paths.extend(
            self._write_candidate_figure(
                geometry=self.geometry,
                evaluation=evaluation,
                sample_offset=sample_offset_by_id[sample_id],
                configured_index=primary_index,
                configured_alpha=configured_alpha,
                theme=self.request.theme,
                outdir=self.request.outdir,
                logger=self.logger,
            )
            for sample_id in selected
            if bool(evaluation.has_flux[sample_offset_by_id[sample_id]].item())
        )

        summary = self._build_summary(
            configs=configs,
            geometry_path=Path(geometry_path),
            test_path=Path(test_path),
            coefficient_path=Path(coefficient_path),
            dataset=dataset,
            aggregate=aggregate,
            candidate_aggregate=candidate_aggregate,
            candidate_rows=candidate_rows,
            rows=rows,
            evaluation=evaluation,
            edges=edges,
            selected=selected,
            roles=roles,
            figure_paths=figure_paths,
            context_build_count=context_build_count,
            configured_alpha=configured_alpha,
            primary_alpha=primary_alpha,
            alphas=alphas,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Projection response audit complete: samples=%d alphas=%s",
                len(dataset),
                alphas,
            )
        return summary

    def _evaluate_dataset(
        self,
        *,
        dataset: ComplexCouplingDataset,
        configs: CouplingArtifactConfigs,
        device: torch.device,
        alphas: tuple[float, ...],
    ) -> tuple[ProjectionResponseAuditEvaluation, int]:
        loader_request = CouplingArtifactRequest(
            config=self.request.config,
            coupling_checkpoint=self.request.coupling_checkpoint,
            green_checkpoint=self.request.green_checkpoint,
            outdir=self.request.outdir,
            coefficients=self.request.coefficients,
            device=str(device),
            theme=self.request.theme,
        )
        model_loader = ComplexCouplingArtifactExporter(
            loader_request,
            logger=self.logger,
        )
        coupling_model = model_loader._load_complex_model(configs, device)
        green_model = model_loader._load_green_model(configs, device)
        for model in (coupling_model, green_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        loader = DataLoader(
            dataset,
            batch_size=min(self.request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        configured_projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        builder = ColumnDiagonalGreenResponseContextBuilder(
            configured_projection.column_diagonal_green_response
        )
        base_context: ColumnDiagonalGreenResponseContext | None = None
        contexts: tuple[ColumnDiagonalGreenResponseContext, ...] | None = None
        context_build_count = 0

        collected: dict[str, list[torch.Tensor]] = {
            name: []
            for name in (
                "sample_ids",
                "has_solution",
                "has_flux",
                "rhs",
                "sol",
                "flux_target",
                "raw_response",
                "raw_physical",
                "raw_balance_residual",
                "projected_physical",
                "correction_physical",
                "correction_response",
                "correction_solution",
                "raw_solution",
            )
        }
        file_stems: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                raw_response, _fusion = coupling_model.forward_with_fusion_diagnostics(
                    geometry=batch.geometry,
                    x_source_branch=batch.x_source_branch,
                    y_source_branch=batch.y_source_branch,
                    x_source_amplitude=batch.x_source_amplitude,
                    y_source_amplitude=batch.y_source_amplitude,
                    x_coefficient_branch=batch.x_coefficient_branch,
                    y_coefficient_branch=batch.y_coefficient_branch,
                    rhs_phys=batch.rhs_valid,
                )
                if base_context is None:
                    base_context = builder.build(
                        green_model=green_model,
                        geometry=batch.geometry,
                        x_green_branch=batch.x_green_branch,
                        y_green_branch=batch.y_green_branch,
                    )
                    context_build_count += 1
                    contexts = tuple(
                        ColumnDiagonalGreenResponseContext.from_gain_squared(
                            gamma_x_squared=base_context.gamma_x_squared,
                            gamma_y_squared=base_context.gamma_y_squared,
                            point_mass=base_context.point_mass,
                            gain_squared_eps=base_context.gain_squared_eps,
                            gain_exponent=alpha,
                        )
                        for alpha in alphas
                    )
                if contexts is None:
                    raise RuntimeError("Projection contexts were not initialized.")

                projections = []
                for alpha, context in zip(alphas, contexts, strict=True):
                    projection_config = BalanceProjectionConfig(
                        enabled=True,
                        mode="column_diagonal_green_response",
                        column_diagonal_green_response=(
                            ColumnDiagonalGreenResponseProjectionConfig(
                                gain_squared_eps=context.gain_squared_eps,
                                gain_exponent=alpha,
                            )
                        ),
                    )
                    projections.append(
                        apply_complex_balance_projection(
                            raw_response=raw_response,
                            rhs_phys=batch.rhs_valid,
                            geometry=batch.geometry,
                            config=projection_config,
                            column_diagonal_context=context,
                        )
                    )
                correction_response = torch.stack(
                    [
                        projection.projected_response - raw_response
                        for projection in projections
                    ],
                    dim=0,
                )
                alpha_count, batch_count, _axis, point_count = correction_response.shape
                correction_reconstruction = reconstruct_from_projected_response(
                    green_model=green_model,
                    geometry=batch.geometry,
                    projected_response=correction_response.reshape(
                        alpha_count * batch_count,
                        2,
                        point_count,
                    ),
                    x_green_branch=batch.x_green_branch,
                    y_green_branch=batch.y_green_branch,
                )
                correction_solution = torch.stack(
                    (
                        correction_reconstruction.u_phi_valid,
                        correction_reconstruction.u_psi_valid,
                    ),
                    dim=1,
                ).reshape(alpha_count, batch_count, 2, point_count)
                raw_reconstruction = reconstruct_from_projected_response(
                    green_model=green_model,
                    geometry=batch.geometry,
                    projected_response=raw_response,
                    x_green_branch=batch.x_green_branch,
                    y_green_branch=batch.y_green_branch,
                )
                raw_solution = torch.stack(
                    (
                        raw_reconstruction.u_phi_valid,
                        raw_reconstruction.u_psi_valid,
                    ),
                    dim=1,
                )
                projected_physical = torch.stack(
                    [projection.projected_physical for projection in projections],
                    dim=0,
                )
                correction_physical = torch.stack(
                    [
                        torch.stack(
                            (projection.correction_phi, projection.correction_psi),
                            dim=1,
                        )
                        for projection in projections
                    ],
                    dim=0,
                )
                collected["sample_ids"].append(batch.sample_indices.detach().cpu())
                collected["has_solution"].append(batch.has_solution.detach().cpu())
                collected["has_flux"].append(batch.has_flux.detach().cpu())
                collected["rhs"].append(batch.rhs_valid.detach().cpu())
                collected["sol"].append(batch.sol_valid.detach().cpu())
                collected["flux_target"].append(batch.flux_valid.detach().cpu())
                collected["raw_response"].append(raw_response.detach().cpu())
                collected["raw_physical"].append(
                    projections[0].raw_physical.detach().cpu()
                )
                collected["raw_balance_residual"].append(
                    (
                        batch.rhs_valid
                        - projections[0].raw_physical[:, 0]
                        - projections[0].raw_physical[:, 1]
                    )
                    .detach()
                    .cpu()
                )
                collected["projected_physical"].append(
                    projected_physical.detach().cpu()
                )
                collected["correction_physical"].append(
                    correction_physical.detach().cpu()
                )
                collected["correction_response"].append(
                    correction_response.detach().cpu()
                )
                collected["correction_solution"].append(
                    correction_solution.detach().cpu()
                )
                collected["raw_solution"].append(raw_solution.detach().cpu())
                file_stems.extend(batch.file_stems)
        if contexts is None:
            raise RuntimeError("No batch was evaluated.")
        self.contexts = tuple(
            ColumnDiagonalGreenResponseContext.from_gain_squared(
                gamma_x_squared=context.gamma_x_squared.cpu(),
                gamma_y_squared=context.gamma_y_squared.cpu(),
                point_mass=context.point_mass.cpu(),
                gain_squared_eps=context.gain_squared_eps,
                gain_exponent=context.gain_exponent,
            )
            for context in contexts
        )
        weights_phi = torch.stack(
            [context.correction_weight_phi for context in self.contexts], dim=0
        )
        return (
            ProjectionResponseAuditEvaluation(
                sample_ids=torch.cat(collected["sample_ids"], dim=0),
                file_stems=tuple(file_stems),
                has_solution=torch.cat(collected["has_solution"], dim=0),
                has_flux=torch.cat(collected["has_flux"], dim=0),
                rhs=torch.cat(collected["rhs"], dim=0),
                sol=torch.cat(collected["sol"], dim=0),
                flux_target=torch.cat(collected["flux_target"], dim=0),
                raw_response=torch.cat(collected["raw_response"], dim=0),
                raw_physical=torch.cat(collected["raw_physical"], dim=0),
                raw_balance_residual=torch.cat(
                    collected["raw_balance_residual"], dim=0
                ),
                weights_phi=weights_phi,
                projected_physical=torch.cat(collected["projected_physical"], dim=1),
                correction_physical=torch.cat(collected["correction_physical"], dim=1),
                correction_response=torch.cat(collected["correction_response"], dim=1),
                correction_solution=torch.cat(collected["correction_solution"], dim=1),
                raw_solution=torch.cat(collected["raw_solution"], dim=0),
            ),
            context_build_count,
        )

    def _select_samples(
        self,
        evaluation: ProjectionResponseAuditEvaluation,
        rows: Sequence[dict[str, float | int | str]],
        alphas: tuple[float, ...],
    ) -> tuple[tuple[int, ...], dict[str, str]]:
        available = {int(value) for value in evaluation.sample_ids.tolist()}
        if self.request.selected_samples is not None:
            missing = sorted(set(self.request.selected_samples) - available)
            if missing:
                raise ValueError(f"Selected sample IDs are unavailable: {missing}.")
            return self.request.selected_samples, {
                str(sample_id): "explicit"
                for sample_id in self.request.selected_samples
            }
        candidate_alpha = alphas[1] if len(alphas) > 2 else alphas[-1]
        alpha_rows = [
            row for row in rows if float(row["gain_exponent"]) == candidate_alpha
        ]
        metric = (
            "equal_mean_rel_sol"
            if alpha_rows and "equal_mean_rel_sol" in alpha_rows[0]
            else "actual_directional_response_cost_ratio_vs_symmetric"
        )
        ordered = sorted(alpha_rows, key=lambda row: float(row[metric]))
        positions = (0.0, 0.25, 0.5, 0.75, 1.0)
        selected: list[int] = []
        roles: dict[str, str] = {}
        for quantile in positions:
            index = round(quantile * max(len(ordered) - 1, 0))
            sample_id = int(ordered[index]["sample_id"])
            if sample_id not in selected:
                selected.append(sample_id)
                roles[str(sample_id)] = f"{metric}_q{int(100 * quantile):02d}"
        return tuple(selected), roles

    @staticmethod
    def _write_csv(
        path: Path,
        rows: Sequence[dict[str, float | int | str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            raise ValueError("Cannot write an empty metric table.")
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
        evaluation: ProjectionResponseAuditEvaluation,
        edges: ProjectionTransitionEdges,
        selected: tuple[int, ...],
        alphas: tuple[float, ...],
        *,
        configured_alpha: float,
    ) -> None:
        offset_by_id = {
            int(sample_id): offset
            for offset, sample_id in enumerate(evaluation.sample_ids.tolist())
        }
        offsets = torch.as_tensor(
            [offset_by_id[sample_id] for sample_id in selected], dtype=torch.long
        )
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        configured_index = alphas.index(configured_alpha)
        np.savez_compressed(
            data_dir / "selected_projection_response_audit.npz",
            coords_valid=self.geometry.coords_valid.detach().cpu().numpy(),
            alpha_values=np.asarray(alphas, dtype=np.float64),
            selected_sample_ids=np.asarray(selected, dtype=np.int64),
            selected_file_stems=np.asarray(
                [evaluation.file_stems[index] for index in offsets.tolist()]
            ),
            rhs=evaluation.rhs[offsets].numpy(),
            sol=evaluation.sol[offsets].numpy(),
            has_solution=evaluation.has_solution[offsets].numpy(),
            flux_target=evaluation.flux_target[offsets].numpy(),
            has_flux=evaluation.has_flux[offsets].numpy(),
            raw_response=evaluation.raw_response[offsets].numpy(),
            raw_physical=evaluation.raw_physical[offsets].numpy(),
            symmetric_balanced_physical=(
                evaluation.symmetric_balanced_physical[offsets].numpy()
            ),
            raw_balance_residual=evaluation.raw_balance_residual[offsets].numpy(),
            correction_weight_phi=evaluation.weights_phi.numpy(),
            projected_physical=evaluation.projected_physical[:, offsets].numpy(),
            configured_gain_exponent=np.asarray(configured_alpha, dtype=np.float64),
            configured_projected_physical=(
                evaluation.projected_physical[configured_index, offsets].numpy()
            ),
            configured_correction_physical=(
                evaluation.correction_physical[configured_index, offsets].numpy()
            ),
            correction_physical=evaluation.correction_physical[:, offsets].numpy(),
            correction_response=evaluation.correction_response[:, offsets].numpy(),
            correction_solution=evaluation.correction_solution[:, offsets].numpy(),
            raw_solution=evaluation.raw_solution[offsets].numpy(),
            final_equal_prediction=evaluation.final_equal_prediction[
                :, offsets
            ].numpy(),
            gamma_x_squared=self.contexts[0].gamma_x_squared.numpy(),
            gamma_y_squared=self.contexts[0].gamma_y_squared.numpy(),
            regularized_gamma_x_squared=(
                self.contexts[0].regularized_gamma_x_squared.numpy()
            ),
            regularized_gamma_y_squared=(
                self.contexts[0].regularized_gamma_y_squared.numpy()
            ),
            phi_transition_edges=edges.phi_transition.numpy(),
            psi_transition_edges=edges.psi_transition.numpy(),
            phi_regular_edges=edges.phi_regular.numpy(),
            psi_regular_edges=edges.psi_regular.numpy(),
        )

    def _build_summary(
        self,
        *,
        configs: CouplingArtifactConfigs,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path,
        dataset: ComplexCouplingDataset,
        aggregate: dict[str, dict[str, float | int]],
        candidate_aggregate: dict[str, dict[str, float | int]],
        candidate_rows: Sequence[dict[str, float | int | str]],
        rows: Sequence[dict[str, float | int | str]],
        evaluation: ProjectionResponseAuditEvaluation,
        edges: ProjectionTransitionEdges,
        selected: tuple[int, ...],
        roles: dict[str, str],
        figure_paths: Sequence[Path],
        context_build_count: int,
        configured_alpha: float,
        primary_alpha: float,
        alphas: tuple[float, ...],
    ) -> dict[str, Any]:
        alpha_one_rows = [row for row in rows if float(row["gain_exponent"]) == 1.0]
        diagonal_optimal_count = 0
        for alpha_one in alpha_one_rows:
            sample_id = int(alpha_one["sample_id"])
            sample_rows = [row for row in rows if int(row["sample_id"]) == sample_id]
            minimum = min(float(row["diagonal_surrogate_cost"]) for row in sample_rows)
            diagonal_optimal_count += math.isclose(
                float(alpha_one["diagonal_surrogate_cost"]),
                minimum,
                rel_tol=1.0e-12,
                abs_tol=1.0e-18,
            )
        actual_best = min(
            alphas,
            key=lambda alpha: float(
                aggregate[self._alpha_label(alpha)][
                    "actual_directional_response_cost_mean"
                ]
            ),
        )
        transition_best = min(
            alphas,
            key=lambda alpha: float(
                aggregate[self._alpha_label(alpha)][
                    "directional_transition_correction_jump_rms_mean"
                ]
            ),
        )
        rel_sol_best = (
            min(
                alphas,
                key=lambda alpha: float(
                    aggregate[self._alpha_label(alpha)]["equal_mean_rel_sol_mean"]
                ),
            )
            if all(
                "equal_mean_rel_sol_mean" in aggregate[self._alpha_label(alpha)]
                for alpha in alphas
            )
            else None
        )
        rel_flux_best = (
            min(
                alphas,
                key=lambda alpha: float(
                    aggregate[self._alpha_label(alpha)]["rel_flux_mean"]
                ),
            )
            if all(
                "rel_flux_mean" in aggregate[self._alpha_label(alpha)]
                for alpha in alphas
            )
            else None
        )
        context_index = (
            alphas.index(configured_alpha) if configured_alpha in alphas else 0
        )
        context = self.contexts[context_index]
        weight_geometry: dict[str, dict[str, float]] = {}
        for alpha_index, alpha in enumerate(alphas):
            weight = evaluation.weights_phi[alpha_index]
            transition_jump = self._edge_squared_jump(
                weight,
                edges.transition,
            )
            regular_jump = self._edge_squared_jump(weight, edges.regular)
            weight_geometry[self._alpha_label(alpha)] = {
                "weight_phi_min": float(weight.min().item()),
                "weight_phi_max": float(weight.max().item()),
                "transition_weight_jump_rms": (
                    math.nan
                    if transition_jump.numel() == 0
                    else float(torch.sqrt(transition_jump.mean()).item())
                ),
                "transition_weight_jump_max_abs": (
                    math.nan
                    if transition_jump.numel() == 0
                    else float(torch.sqrt(transition_jump).max().item())
                ),
                "regular_weight_jump_rms": (
                    math.nan
                    if regular_jump.numel() == 0
                    else float(torch.sqrt(regular_jump.mean()).item())
                ),
            }
        return {
            "diagnostic": "column_diagonal_green_response_posthoc_audit",
            "status": "frozen_checkpoint_posthoc",
            "production_code_changed": False,
            "coupling_model_rerun": True,
            "coupling_model_trained_or_updated": False,
            "green_model_trained_or_updated": False,
            "fair_comparison_scope": (
                "same frozen raw directional response and balance residual; only "
                "fixed projection gain exponent changes"
            ),
            "conclusion_scope": (
                "immediate projection correction response only; not a replacement "
                "for paired retraining"
            ),
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": str(coefficient_path),
            "sample_count": len(dataset),
            "gain_exponents": list(alphas),
            "configured_gain_exponent": configured_alpha,
            "selected_figure_gain_exponent": primary_alpha,
            "projection_formula": {
                "residual": "r=f-p-q",
                "weight_phi": (
                    "w_phi=(gamma_y_squared+eps)^alpha/"
                    "((gamma_x_squared+eps)^alpha+"
                    "(gamma_y_squared+eps)^alpha)"
                ),
                "correction": "delta_phi=w_phi*r; delta_psi=(1-w_phi)*r",
                "balance": "phi+psi=f",
            },
            "directional_candidate_audit": {
                "raw": "raw physical proposals (p,q)",
                "symmetric_balanced": ("p_tilde=0.5*(f+p-q); q_tilde=0.5*(f-p+q)"),
                "configured": (
                    "configured column-diagonal projected physical (phi,psi)"
                ),
                "target_policy": (
                    "sample phi/psi are evaluation-only directional targets"
                ),
                "transition_points": (
                    "unique endpoints of the configured cross-axis transition edges"
                ),
                "aggregate_metrics": candidate_aggregate,
                "flux_target_sample_count": sum(
                    "raw_pair_rel_target" in row for row in candidate_rows
                ),
            },
            "response_cost_formula": {
                "actual": (
                    "J_actual=point_mass*(||H_x delta_phi||_2^2+||H_y delta_psi||_2^2)"
                ),
                "diagonal_surrogate": (
                    "J_diag=sum((gamma_x_squared+eps)*delta_phi^2+"
                    "(gamma_y_squared+eps)*delta_psi^2)"
                ),
                "trace_jump": (
                    "RMS of correction-solution jumps on cross-axis edges with "
                    "abs(delta(log(L_axis^2)))>threshold"
                ),
            },
            "matrix_policy": {
                "column_diagonal_only": True,
                "row_norm_used": False,
                "full_gram_matrix_materialized": False,
                "global_matrix_solve": False,
                "green_response_context_build_count": context_build_count,
            },
            "reference_policy": {
                "sol_and_flux_used_for_training": False,
                "sol_and_flux_used_for_projection": False,
                "sol_and_flux_used_for_optional_evaluation_metrics": True,
                "primary_response_cost_requires_reference": False,
            },
            "transition_definition": {
                "log_threshold": self.request.transition_log_threshold,
                "phi_cross_axis": "geometry.y_edges with jump in log(L_x^2)",
                "psi_cross_axis": "geometry.x_edges with jump in log(L_y^2)",
                "phi_transition_edge_count": int(edges.phi_transition.shape[0]),
                "psi_transition_edge_count": int(edges.psi_transition.shape[0]),
                "total_transition_edge_count": int(edges.transition.shape[0]),
            },
            "green_response_context": context.statistics(),
            "projection_weight_geometry": weight_geometry,
            "aggregate_metrics": aggregate,
            "automated_findings": {
                "alpha_1_diagonal_surrogate_optimal_sample_count": (
                    diagonal_optimal_count
                ),
                "alpha_1_diagonal_surrogate_sample_count": len(alpha_one_rows),
                "lowest_mean_actual_response_cost_alpha": actual_best,
                "lowest_mean_transition_correction_jump_alpha": transition_best,
                "lowest_mean_equal_rel_sol_alpha": rel_sol_best,
                "lowest_mean_rel_flux_alpha": rel_flux_best,
                "response_cost_and_accuracy_select_same_alpha": (
                    actual_best == rel_sol_best if rel_sol_best is not None else None
                ),
            },
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "save_generated_data": self.request.save_generated_data,
            "raw_archive": (
                "data/selected_projection_response_audit.npz"
                if self.request.save_generated_data
                else None
            ),
            "metric_csv": ("metrics/per_sample_projection_response_audit.csv"),
            "candidate_metric_csv": (
                "metrics/per_sample_directional_candidate_audit.csv"
            ),
            "figure_count": len(figure_paths),
            "figure_json": [
                str(path.relative_to(self.request.outdir)) for path in figure_paths
            ],
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate_metrics"]
        candidate = summary["directional_candidate_audit"]["aggregate_metrics"]

        def candidate_mean(key: str) -> float:
            payload = candidate.get(key)
            if not isinstance(payload, dict):
                return math.nan
            return float(payload.get("mean", math.nan))

        lines = [
            "# Column-Diagonal Green-Response Post-Hoc Audit",
            "",
            "## Scope",
            "",
            "The same frozen CouplingNet raw response and balance residual are used",
            "for every fixed gain exponent. Only the projection correction weights",
            "change. This isolates the immediate projection effect, but it does not",
            "replace paired retraining from identical initial conditions.",
            "",
            "## Aggregate Results",
            "",
            "| alpha | actual response cost ratio | diagonal cost ratio | "
            "source jump ratio | response jump ratio | equal rel_sol | rel_flux |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for alpha in summary["gain_exponents"]:
            metrics = aggregate[self._alpha_label(float(alpha))]
            lines.append(
                "| "
                f"{alpha:g} | "
                f"{metrics.get('actual_directional_response_cost_ratio_vs_symmetric_mean', math.nan):.6f} | "
                f"{metrics.get('diagonal_surrogate_cost_ratio_vs_symmetric_mean', math.nan):.6f} | "
                f"{metrics.get('directional_transition_source_jump_ratio_vs_symmetric_mean', math.nan):.6f} | "
                f"{metrics.get('directional_transition_jump_ratio_vs_symmetric_mean', math.nan):.6f} | "
                f"{metrics.get('equal_mean_rel_sol_mean', math.nan):.6f} | "
                f"{metrics.get('rel_flux_mean', math.nan):.6f} |"
            )
        lines.extend(
            [
                "",
                "## Directional Candidate Quality",
                "",
                "The table compares the same raw network output before balance,",
                "after symmetric balancing with the raw difference preserved, and",
                "after the configured column-diagonal projection. Directional target",
                "fields are evaluation-only and never enter training or projection.",
                "",
                "| stage | global pair rel target | transition pair rel target | "
                "regular pair rel target | phi rel target | psi rel target |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for stage in ("raw", "symmetric", "configured"):
            lines.append(
                "| "
                f"{stage} | "
                f"{candidate_mean(f'{stage}_pair_rel_target'):.6f} | "
                f"{candidate_mean(f'{stage}_transition_pair_rel_target'):.6f} | "
                f"{candidate_mean(f'{stage}_regular_pair_rel_target'):.6f} | "
                f"{candidate_mean(f'{stage}_phi_rel_target'):.6f} | "
                f"{candidate_mean(f'{stage}_psi_rel_target'):.6f} |"
            )
        lines.extend(
            [
                "",
                "- Mean raw balance defect / rhs norm: "
                f"{candidate_mean('raw_balance_residual_rel_rhs'):.6f}.",
                "- Mean symmetric target-error ratio versus raw: "
                f"{candidate_mean('symmetric_pair_error_ratio_vs_raw'):.6f}.",
                "- Mean configured target-error ratio versus symmetric: "
                f"{candidate_mean('configured_pair_error_ratio_vs_symmetric'):.6f}.",
                "- Mean configured correction / projected pair norm: "
                f"{candidate_mean('configured_correction_rel_projected_pair'):.6f}.",
                "- Mean configured tangent transfer / projected pair norm: "
                f"{candidate_mean('configured_tangent_transfer_rel_projected_pair'):.6f}.",
            ]
        )
        findings = summary["automated_findings"]
        weight_geometry = summary["projection_weight_geometry"]
        lines.extend(
            [
                "",
                "## Automated Checks",
                "",
                "- alpha=1 minimizes the diagonal surrogate on "
                f"{findings['alpha_1_diagonal_surrogate_optimal_sample_count']}/"
                f"{findings['alpha_1_diagonal_surrogate_sample_count']} samples.",
                "- Lowest mean actual learned-Green response cost: "
                f"alpha={findings['lowest_mean_actual_response_cost_alpha']:g}.",
                "- Lowest mean transition correction jump: "
                f"alpha={findings['lowest_mean_transition_correction_jump_alpha']:g}.",
                "- Transition weight-jump RMS increases from "
                f"{weight_geometry['alpha_0']['transition_weight_jump_rms']:.6f} "
                "at alpha=0 to "
                f"{weight_geometry['alpha_1']['transition_weight_jump_rms']:.6f} "
                "at alpha=1; response-jump behavior must therefore be measured "
                "after Green reconstruction.",
                "- Lowest frozen-checkpoint equal-mean rel_sol: "
                f"alpha={findings['lowest_mean_equal_rel_sol_alpha']}; lowest "
                f"rel_flux: alpha={findings['lowest_mean_rel_flux_alpha']}. These "
                "evaluation optima reflect the exponent used during training and "
                "must not be used as a post-hoc model-selection result.",
                "",
                "## Interpretation Boundary",
                "",
                "A lower diagonal surrogate confirms the optimization target of the",
                "column-diagonal approximation. A lower actual response cost confirms",
                "that this approximation also reduces the learned Green response for",
                "the frozen correction. A lower transition jump is a separate property;",
                "it is not guaranteed by minimizing either response cost. Reference",
                "solution and flux fields appear only in optional evaluation metrics.",
            ]
        )
        (self.request.outdir / "diagnosis_report.md").write_text(
            "\n".join(lines) + "\n"
        )


def run_complex_projection_response_audit(
    request: ProjectionResponseAuditRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the frozen-checkpoint projection response audit."""

    return ComplexProjectionResponseAudit(request, logger=logger).run()
