from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence, cast

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
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    CouplingArtifactConfigs,
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class FixedSmoothBlendConfig:
    """Fixed geometry-only reconstruction blend parameters."""

    weight_construction: Literal["jump_smoothed", "compact_c2_ramp"] = "jump_smoothed"
    alpha: float = 1.0 / math.log(2.0)
    smoothing_steps: int = 2
    smoothing_relaxation: float = 0.5
    reliability_floor: float = 1.0e-6
    transition_log_threshold: float = math.log(2.0)
    transition_dilation_steps: int = 2
    ramp_gamma: float = 0.5
    ramp_width: float | None = None

    def __post_init__(self) -> None:
        if self.weight_construction not in {"jump_smoothed", "compact_c2_ramp"}:
            raise ValueError(
                "weight_construction must be 'jump_smoothed' or 'compact_c2_ramp'."
            )
        finite_positive = {
            "alpha": self.alpha,
            "reliability_floor": self.reliability_floor,
            "transition_log_threshold": self.transition_log_threshold,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            not math.isfinite(self.smoothing_relaxation)
            or not 0.0 < self.smoothing_relaxation <= 1.0
        ):
            raise ValueError("smoothing_relaxation must be in (0, 1].")
        if (
            isinstance(self.smoothing_steps, bool)
            or not isinstance(self.smoothing_steps, int)
            or self.smoothing_steps < 0
        ):
            raise ValueError("smoothing_steps must be a non-negative integer.")
        if (
            isinstance(self.transition_dilation_steps, bool)
            or not isinstance(self.transition_dilation_steps, int)
            or self.transition_dilation_steps < 0
        ):
            raise ValueError(
                "transition_dilation_steps must be a non-negative integer."
            )
        if (
            not math.isfinite(self.ramp_gamma)
            or self.ramp_gamma < 0.0
            or self.ramp_gamma > 1.0
        ):
            raise ValueError("ramp_gamma must be finite and in [0, 1].")
        if self.ramp_width is not None and (
            not math.isfinite(self.ramp_width) or self.ramp_width <= 0.0
        ):
            raise ValueError("ramp_width must be finite and positive when provided.")


@dataclass(frozen=True)
class FixedSmoothBlendDiagnosticRequest:
    """Checkpoint-backed request for a post-reconstruction blend diagnostic."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    selected_samples: tuple[int, ...] | None = None
    batch_size: int = 10
    save_generated_data: bool = True
    run_compact_sweep: bool = False
    sweep_gammas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    sweep_width_steps: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0)
    blend: FixedSmoothBlendConfig = FixedSmoothBlendConfig()

    def __post_init__(self) -> None:
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        if self.selected_samples is not None and any(
            index < 0 for index in self.selected_samples
        ):
            raise ValueError("selected_samples must contain non-negative indices.")
        if not self.sweep_gammas or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.sweep_gammas
        ):
            raise ValueError("sweep_gammas must contain finite values in [0, 1].")
        if not self.sweep_width_steps or any(
            not math.isfinite(value) or value <= 0.0 for value in self.sweep_width_steps
        ):
            raise ValueError("sweep_width_steps must contain finite positive values.")


@dataclass(frozen=True)
class FixedSmoothBlendGeometryFields:
    """Sample-independent geometry fields used by the fixed blend."""

    weight_construction: str
    j_phi_raw: torch.Tensor
    j_psi_raw: torch.Tensor
    j_phi: torch.Tensor
    j_psi: torch.Tensor
    rho_phi: torch.Tensor
    rho_psi: torch.Tensor
    distance_phi: torch.Tensor
    distance_psi: torch.Tensor
    influence_phi: torch.Tensor
    influence_psi: torch.Tensor
    theta: torch.Tensor
    w_phi: torch.Tensor
    w_psi: torch.Tensor
    ramp_support_mask: torch.Tensor
    phi_transition_coordinates: torch.Tensor
    psi_transition_coordinates: torch.Tensor
    resolved_ramp_width: float | None
    transition_point_mask: torch.Tensor
    phi_transition_edge_mask: torch.Tensor
    psi_transition_edge_mask: torch.Tensor


@dataclass(frozen=True)
class FixedSmoothBlendEvaluation:
    """Full-test reference and reconstruction arrays on CPU."""

    sample_ids: torch.Tensor
    file_stems: tuple[str, ...]
    sol: torch.Tensor
    u_phi: torch.Tensor
    u_psi: torch.Tensor
    baseline: torch.Tensor
    blend: torch.Tensor


class FixedSmoothBlendGeometryMixin:
    """Construct geometry-only reliability and partition-of-unity fields."""

    @staticmethod
    def _pointwise_edge_jump(
        values: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.dim() != 1:
            raise ValueError("values must be one-dimensional.")
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2).")
        edge_jump = (values[edges[:, 1]] - values[edges[:, 0]]).abs()
        point_jump = torch.zeros_like(values)
        if edges.numel() > 0:
            point_jump.scatter_reduce_(
                0,
                edges[:, 0],
                edge_jump,
                reduce="amax",
                include_self=True,
            )
            point_jump.scatter_reduce_(
                0,
                edges[:, 1],
                edge_jump,
                reduce="amax",
                include_self=True,
            )
        return point_jump, edge_jump

    @staticmethod
    def _smooth_on_edges(
        values: torch.Tensor,
        edges: torch.Tensor,
        *,
        steps: int,
        relaxation: float,
    ) -> torch.Tensor:
        current = values.clone()
        if steps == 0 or edges.numel() == 0:
            return current
        degree = torch.zeros_like(values)
        ones = torch.ones(
            edges.shape[0],
            dtype=values.dtype,
            device=values.device,
        )
        degree.index_add_(0, edges[:, 0], ones)
        degree.index_add_(0, edges[:, 1], ones)
        connected = degree > 0
        for _ in range(steps):
            neighbor_sum = torch.zeros_like(values)
            neighbor_sum.index_add_(0, edges[:, 0], current[edges[:, 1]])
            neighbor_sum.index_add_(0, edges[:, 1], current[edges[:, 0]])
            neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
            relaxed = (1.0 - relaxation) * current + relaxation * neighbor_mean
            current = torch.where(connected, relaxed, current)
        return current

    @staticmethod
    def _dilate_point_mask(
        mask: torch.Tensor,
        edges: torch.Tensor,
        *,
        steps: int,
    ) -> torch.Tensor:
        current = mask.clone()
        if steps == 0 or edges.numel() == 0:
            return current
        for _ in range(steps):
            expanded = current.clone()
            left_from_right = current[edges[:, 1]].to(torch.int8)
            right_from_left = current[edges[:, 0]].to(torch.int8)
            expanded_int = expanded.to(torch.int8)
            expanded_int.scatter_reduce_(
                0,
                edges[:, 0],
                left_from_right,
                reduce="amax",
                include_self=True,
            )
            expanded_int.scatter_reduce_(
                0,
                edges[:, 1],
                right_from_left,
                reduce="amax",
                include_self=True,
            )
            current = expanded_int.bool()
        return current

    @staticmethod
    def _infer_transition_coordinates(
        fixed_coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Locate split/merge interfaces from adjacent axial-line multiplicities."""

        if fixed_coordinates.dim() != 1 or fixed_coordinates.numel() == 0:
            raise ValueError("fixed_coordinates must be a non-empty 1D tensor.")
        unique_raw, counts_raw = torch.unique(
            fixed_coordinates,
            sorted=True,
            return_counts=True,
        )
        unique = cast(torch.Tensor, unique_raw)
        counts = cast(torch.Tensor, counts_raw)
        if unique.numel() < 2:
            return unique.new_empty((0,))
        multiplicity_change = counts[1:] != counts[:-1]
        return 0.5 * (
            unique[:-1][multiplicity_change] + unique[1:][multiplicity_change]
        )

    @staticmethod
    def _distance_to_interfaces(
        coordinates: torch.Tensor,
        interfaces: torch.Tensor,
    ) -> torch.Tensor:
        if coordinates.dim() != 1:
            raise ValueError("coordinates must be one-dimensional.")
        if interfaces.dim() != 1 or interfaces.numel() == 0:
            raise ValueError("At least one transition interface is required.")
        return (coordinates[:, None] - interfaces[None, :]).abs().amin(dim=1)

    @staticmethod
    def _compact_c2_bump(
        distance: torch.Tensor,
        *,
        width: float,
    ) -> torch.Tensor:
        """Return a compact quintic ramp with zero first/second endpoint slopes."""

        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("width must be finite and positive.")
        scaled = (distance / width).clamp(min=0.0, max=1.0)
        polynomial = (
            1.0 - 10.0 * scaled.pow(3) + 15.0 * scaled.pow(4) - 6.0 * scaled.pow(5)
        )
        return torch.where(
            distance < width,
            polynomial.clamp(min=0.0, max=1.0),
            torch.zeros_like(distance),
        )

    @staticmethod
    def _validate_partition_weights(
        w_phi: torch.Tensor,
        w_psi: torch.Tensor,
    ) -> None:
        if not all(torch.all(torch.isfinite(field)) for field in (w_phi, w_psi)):
            raise RuntimeError("Fixed smooth blend produced non-finite weights.")
        if (
            torch.any(w_phi < 0.0)
            or torch.any(w_phi > 1.0)
            or torch.any(w_psi < 0.0)
            or torch.any(w_psi > 1.0)
        ):
            raise RuntimeError("Fixed smooth blend weights must be in [0, 1].")
        if not torch.allclose(
            w_phi + w_psi,
            torch.ones_like(w_phi),
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise RuntimeError("Fixed smooth blend weights do not sum to one.")

    @classmethod
    def build_fixed_blend_fields(
        cls,
        geometry: ComplexGeometryMetadata,
        config: FixedSmoothBlendConfig,
    ) -> FixedSmoothBlendGeometryFields:
        x_length = geometry.x_lengths_for_valid_points()
        y_length = geometry.y_lengths_for_valid_points()
        if torch.any(x_length <= 0.0) or torch.any(y_length <= 0.0):
            raise ValueError("All pointwise segment lengths must be positive.")

        log_sigma_x = x_length.square().log()
        log_sigma_y = y_length.square().log()
        j_phi_raw, phi_edge_jump = cls._pointwise_edge_jump(
            log_sigma_x,
            geometry.y_edges,
        )
        j_psi_raw, psi_edge_jump = cls._pointwise_edge_jump(
            log_sigma_y,
            geometry.x_edges,
        )
        j_phi = cls._smooth_on_edges(
            j_phi_raw,
            geometry.y_edges,
            steps=config.smoothing_steps,
            relaxation=config.smoothing_relaxation,
        )
        j_psi = cls._smooth_on_edges(
            j_psi_raw,
            geometry.x_edges,
            steps=config.smoothing_steps,
            relaxation=config.smoothing_relaxation,
        )

        phi_transition_edge_mask = phi_edge_jump > config.transition_log_threshold
        psi_transition_edge_mask = psi_edge_jump > config.transition_log_threshold
        transition_point_mask = (j_phi_raw > config.transition_log_threshold) | (
            j_psi_raw > config.transition_log_threshold
        )
        all_edges = torch.cat((geometry.x_edges, geometry.y_edges), dim=0)
        transition_point_mask = cls._dilate_point_mask(
            transition_point_mask,
            all_edges,
            steps=config.transition_dilation_steps,
        )
        if not torch.any(transition_point_mask):
            raise ValueError(
                "No geometry transition was detected with the configured threshold."
            )
        if not (
            torch.any(phi_transition_edge_mask) or torch.any(psi_transition_edge_mask)
        ):
            raise ValueError(
                "No transition edge was detected with the configured threshold."
            )

        if config.weight_construction == "compact_c2_ramp":
            phi_transition_coordinates = cls._infer_transition_coordinates(
                geometry.x_segment_y
            )
            psi_transition_coordinates = cls._infer_transition_coordinates(
                geometry.y_segment_x
            )
            if (
                phi_transition_coordinates.numel() == 0
                or psi_transition_coordinates.numel() == 0
            ):
                raise ValueError(
                    "compact_c2_ramp requires split/merge transitions in both "
                    "axial-line families."
                )
            distance_phi = cls._distance_to_interfaces(
                geometry.coords_valid[:, 1],
                phi_transition_coordinates,
            )
            distance_psi = cls._distance_to_interfaces(
                geometry.coords_valid[:, 0],
                psi_transition_coordinates,
            )
            resolved_ramp_width = config.ramp_width or (
                4.0 * max(float(geometry.hx.item()), float(geometry.hy.item()))
            )
            influence_phi = cls._compact_c2_bump(
                distance_phi,
                width=resolved_ramp_width,
            )
            influence_psi = cls._compact_c2_bump(
                distance_psi,
                width=resolved_ramp_width,
            )
            theta = config.ramp_gamma * (influence_psi - influence_phi)
            w_phi = 0.5 * (1.0 + theta)
            w_psi = 0.5 * (1.0 - theta)
            rho_phi = w_phi
            rho_psi = w_psi
            ramp_support_mask = (influence_phi > 0.0) | (influence_psi > 0.0)
        else:
            rho_phi = config.reliability_floor + torch.exp(-config.alpha * j_phi)
            rho_psi = config.reliability_floor + torch.exp(-config.alpha * j_psi)
            denominator = rho_phi + rho_psi
            w_phi = rho_phi / denominator
            w_psi = rho_psi / denominator
            theta = 2.0 * w_phi - 1.0
            influence_phi = j_phi
            influence_psi = j_psi
            distance_phi = torch.full_like(j_phi, torch.inf)
            distance_psi = torch.full_like(j_psi, torch.inf)
            ramp_support_mask = theta.abs() > 0.0
            phi_transition_coordinates = j_phi.new_empty((0,))
            psi_transition_coordinates = j_psi.new_empty((0,))
            resolved_ramp_width = None

        if not all(
            torch.all(torch.isfinite(field))
            for field in (
                j_phi,
                j_psi,
                rho_phi,
                rho_psi,
                influence_phi,
                influence_psi,
                theta,
            )
        ):
            raise RuntimeError("Fixed smooth blend produced non-finite fields.")
        cls._validate_partition_weights(w_phi, w_psi)

        return FixedSmoothBlendGeometryFields(
            weight_construction=config.weight_construction,
            j_phi_raw=j_phi_raw,
            j_psi_raw=j_psi_raw,
            j_phi=j_phi,
            j_psi=j_psi,
            rho_phi=rho_phi,
            rho_psi=rho_psi,
            distance_phi=distance_phi,
            distance_psi=distance_psi,
            influence_phi=influence_phi,
            influence_psi=influence_psi,
            theta=theta,
            w_phi=w_phi,
            w_psi=w_psi,
            ramp_support_mask=ramp_support_mask,
            phi_transition_coordinates=phi_transition_coordinates,
            psi_transition_coordinates=psi_transition_coordinates,
            resolved_ramp_width=resolved_ramp_width,
            transition_point_mask=transition_point_mask,
            phi_transition_edge_mask=phi_transition_edge_mask,
            psi_transition_edge_mask=psi_transition_edge_mask,
        )


class FixedSmoothBlendMetricMixin:
    """Compute global, transition-zone, and one-sided trace diagnostics."""

    request: FixedSmoothBlendDiagnosticRequest
    geometry: ComplexGeometryMetadata
    blend_fields: FixedSmoothBlendGeometryFields

    @staticmethod
    def _rms(values: torch.Tensor) -> float:
        if values.numel() == 0:
            raise ValueError("Cannot compute RMS over an empty tensor.")
        return float(torch.sqrt(torch.mean(values.square())).item())

    @staticmethod
    def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(target).clamp_min(1.0e-12)
        return float(
            (torch.linalg.vector_norm(prediction - target) / denominator).item()
        )

    @classmethod
    def _edge_jump_rms(
        cls,
        values: torch.Tensor,
        edges: torch.Tensor,
    ) -> float:
        if edges.numel() == 0:
            raise ValueError("Cannot compute edge jump over an empty edge set.")
        return cls._rms(values[edges[:, 1]] - values[edges[:, 0]])

    def _transition_edges(
        self,
        blend_fields: FixedSmoothBlendGeometryFields | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fields = self.blend_fields if blend_fields is None else blend_fields
        transition = torch.cat(
            (
                self.geometry.y_edges[fields.phi_transition_edge_mask],
                self.geometry.x_edges[fields.psi_transition_edge_mask],
            ),
            dim=0,
        )
        regular = torch.cat(
            (
                self.geometry.y_edges[~fields.phi_transition_edge_mask],
                self.geometry.x_edges[~fields.psi_transition_edge_mask],
            ),
            dim=0,
        )
        return transition, regular

    def _sample_metric_rows(
        self,
        evaluation: FixedSmoothBlendEvaluation,
        blend_fields: FixedSmoothBlendGeometryFields | None = None,
        sample_support_masks: torch.Tensor | None = None,
    ) -> list[dict[str, float | int | str]]:
        fields = self.blend_fields if blend_fields is None else blend_fields
        transition_mask = fields.transition_point_mask
        regular_mask = ~transition_mask
        if sample_support_masks is not None and (
            sample_support_masks.shape != evaluation.baseline.shape
            or sample_support_masks.dtype != torch.bool
        ):
            raise ValueError("sample_support_masks must be boolean with shape (B, P).")
        transition_edges, regular_edges = self._transition_edges(fields)
        rows: list[dict[str, float | int | str]] = []
        for offset, sample_id in enumerate(evaluation.sample_ids.tolist()):
            outside_support_mask = (
                ~fields.ramp_support_mask
                if sample_support_masks is None
                else ~sample_support_masks[offset]
            )
            sol = evaluation.sol[offset]
            baseline = evaluation.baseline[offset]
            blend = evaluation.blend[offset]
            baseline_error = baseline - sol
            blend_error = blend - sol
            correction = blend - baseline
            baseline_rel_sol = self._relative_l2(baseline, sol)
            blend_rel_sol = self._relative_l2(blend, sol)
            baseline_transition_rms = self._rms(baseline_error[transition_mask])
            blend_transition_rms = self._rms(blend_error[transition_mask])
            baseline_trace_error_jump = self._edge_jump_rms(
                baseline_error,
                transition_edges,
            )
            blend_trace_error_jump = self._edge_jump_rms(
                blend_error,
                transition_edges,
            )
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "file_stem": evaluation.file_stems[offset],
                    "baseline_rel_sol": baseline_rel_sol,
                    "blend_rel_sol": blend_rel_sol,
                    "rel_sol_change": blend_rel_sol - baseline_rel_sol,
                    "rel_sol_relative_change": (
                        (blend_rel_sol - baseline_rel_sol)
                        / max(baseline_rel_sol, 1.0e-12)
                    ),
                    "u_phi_rel_sol": self._relative_l2(
                        evaluation.u_phi[offset],
                        sol,
                    ),
                    "u_psi_rel_sol": self._relative_l2(
                        evaluation.u_psi[offset],
                        sol,
                    ),
                    "baseline_error_rms": self._rms(baseline_error),
                    "blend_error_rms": self._rms(blend_error),
                    "blend_correction_rms": self._rms(correction),
                    "blend_correction_max_abs": float(correction.abs().max().item()),
                    "blend_transition_correction_rms": self._rms(
                        correction[transition_mask]
                    ),
                    "blend_regular_correction_rms": self._rms(correction[regular_mask]),
                    "blend_outside_support_correction_max_abs": (
                        0.0
                        if not torch.any(outside_support_mask)
                        else float(correction[outside_support_mask].abs().max().item())
                    ),
                    "mse_change_linear_term": float(
                        (2.0 * baseline_error * correction).mean().item()
                    ),
                    "mse_change_quadratic_term": float(
                        correction.square().mean().item()
                    ),
                    "transition_mse_change_linear_term": float(
                        (
                            2.0
                            * baseline_error[transition_mask]
                            * correction[transition_mask]
                        )
                        .mean()
                        .item()
                    ),
                    "transition_mse_change_quadratic_term": float(
                        correction[transition_mask].square().mean().item()
                    ),
                    "baseline_transition_error_rms": baseline_transition_rms,
                    "blend_transition_error_rms": blend_transition_rms,
                    "transition_error_rms_relative_change": (
                        (blend_transition_rms - baseline_transition_rms)
                        / max(baseline_transition_rms, 1.0e-12)
                    ),
                    "baseline_regular_error_rms": self._rms(
                        baseline_error[regular_mask]
                    ),
                    "blend_regular_error_rms": self._rms(blend_error[regular_mask]),
                    "baseline_transition_trace_prediction_jump_rms": (
                        self._edge_jump_rms(baseline, transition_edges)
                    ),
                    "blend_transition_trace_prediction_jump_rms": (
                        self._edge_jump_rms(blend, transition_edges)
                    ),
                    "baseline_transition_trace_error_jump_rms": (
                        baseline_trace_error_jump
                    ),
                    "blend_transition_trace_error_jump_rms": (blend_trace_error_jump),
                    "transition_trace_error_jump_relative_change": (
                        (blend_trace_error_jump - baseline_trace_error_jump)
                        / max(baseline_trace_error_jump, 1.0e-12)
                    ),
                    "baseline_regular_trace_error_jump_rms": (
                        self._edge_jump_rms(baseline_error, regular_edges)
                    ),
                    "blend_regular_trace_error_jump_rms": self._edge_jump_rms(
                        blend_error,
                        regular_edges,
                    ),
                }
            )
        return rows

    @staticmethod
    def _mean(rows: Sequence[dict[str, float | int | str]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    def _aggregate_metrics(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> dict[str, float | int | str]:
        baseline_rel = self._mean(rows, "baseline_rel_sol")
        blend_rel = self._mean(rows, "blend_rel_sol")
        baseline_transition = self._mean(
            rows,
            "baseline_transition_error_rms",
        )
        blend_transition = self._mean(rows, "blend_transition_error_rms")
        baseline_trace = self._mean(
            rows,
            "baseline_transition_trace_error_jump_rms",
        )
        blend_trace = self._mean(
            rows,
            "blend_transition_trace_error_jump_rms",
        )
        global_change = (blend_rel - baseline_rel) / max(baseline_rel, 1.0e-12)
        transition_change = (blend_transition - baseline_transition) / max(
            baseline_transition,
            1.0e-12,
        )
        trace_change = (blend_trace - baseline_trace) / max(
            baseline_trace,
            1.0e-12,
        )
        if global_change < 0.0 and transition_change < 0.0 and trace_change < 0.0:
            verdict = "improves_all_primary_diagnostics"
        elif global_change < 0.0 and (transition_change < 0.0 or trace_change < 0.0):
            verdict = "promising_but_mixed"
        elif transition_change < 0.0 or trace_change < 0.0:
            verdict = "local_improvement_with_global_tradeoff"
        else:
            verdict = "not_supported_by_fixed_preset"
        return {
            "sample_count": len(rows),
            "baseline_rel_sol_mean": baseline_rel,
            "blend_rel_sol_mean": blend_rel,
            "rel_sol_mean_relative_change": global_change,
            "rel_sol_relative_change_median": float(
                np.median([float(row["rel_sol_relative_change"]) for row in rows])
            ),
            "rel_sol_blend_win_count": sum(
                float(row["blend_rel_sol"]) < float(row["baseline_rel_sol"])
                for row in rows
            ),
            "blend_correction_rms_mean": self._mean(
                rows,
                "blend_correction_rms",
            ),
            "blend_transition_correction_rms_mean": self._mean(
                rows,
                "blend_transition_correction_rms",
            ),
            "blend_outside_support_correction_max_abs": max(
                float(row["blend_outside_support_correction_max_abs"]) for row in rows
            ),
            "mse_change_linear_term_mean": self._mean(
                rows,
                "mse_change_linear_term",
            ),
            "mse_change_quadratic_term_mean": self._mean(
                rows,
                "mse_change_quadratic_term",
            ),
            "transition_mse_change_linear_term_mean": self._mean(
                rows,
                "transition_mse_change_linear_term",
            ),
            "transition_mse_change_quadratic_term_mean": self._mean(
                rows,
                "transition_mse_change_quadratic_term",
            ),
            "baseline_transition_error_rms_mean": baseline_transition,
            "blend_transition_error_rms_mean": blend_transition,
            "transition_error_rms_mean_relative_change": transition_change,
            "baseline_transition_trace_error_jump_rms_mean": baseline_trace,
            "blend_transition_trace_error_jump_rms_mean": blend_trace,
            "transition_trace_error_jump_mean_relative_change": trace_change,
            "verdict": verdict,
        }

    @staticmethod
    def _paired_bootstrap_summary(
        rows: list[dict[str, float | int | str]],
        *,
        draws: int = 100_000,
        seed: int = 0,
    ) -> dict[str, Any]:
        if not rows:
            raise ValueError("Paired bootstrap requires at least one sample row.")
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(
            0,
            len(rows),
            size=(draws, len(rows)),
        )
        metric_fields = {
            "rel_sol": ("baseline_rel_sol", "blend_rel_sol"),
            "transition_error_rms": (
                "baseline_transition_error_rms",
                "blend_transition_error_rms",
            ),
            "transition_trace_error_jump_rms": (
                "baseline_transition_trace_error_jump_rms",
                "blend_transition_trace_error_jump_rms",
            ),
        }
        metrics: dict[str, Any] = {}
        for name, (baseline_key, blend_key) in metric_fields.items():
            baseline = np.asarray(
                [float(row[baseline_key]) for row in rows],
                dtype=np.float64,
            )
            blend = np.asarray(
                [float(row[blend_key]) for row in rows],
                dtype=np.float64,
            )
            bootstrap_baseline = baseline[sample_indices].mean(axis=1)
            bootstrap_blend = blend[sample_indices].mean(axis=1)
            relative_change = (bootstrap_blend - bootstrap_baseline) / np.maximum(
                bootstrap_baseline, 1.0e-12
            )
            observed = (blend.mean() - baseline.mean()) / max(
                baseline.mean(),
                1.0e-12,
            )
            metrics[name] = {
                "observed_relative_change": float(observed),
                "relative_change_ci95": [
                    float(value)
                    for value in np.quantile(relative_change, (0.025, 0.975))
                ],
                "bootstrap_probability_improvement": float(
                    np.mean(relative_change < 0.0)
                ),
            }
        return {
            "method": "paired sample bootstrap of aggregate relative change",
            "draws": draws,
            "seed": seed,
            "metrics": metrics,
        }


class FixedSmoothBlendPlotMixin:
    """Write Plotly figures for geometry weights and paired solution errors."""

    request: FixedSmoothBlendDiagnosticRequest
    geometry: ComplexGeometryMetadata
    blend_fields: FixedSmoothBlendGeometryFields
    logger: logging.Logger | None

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    @staticmethod
    def _signed_limit(*values: np.ndarray) -> float:
        limit = max(float(np.max(np.abs(value))) for value in values)
        return max(limit, 1.0e-15)

    def _add_scatter(
        self,
        figure: go.Figure,
        *,
        row: int,
        col: int,
        values: np.ndarray,
        title: str,
        colorscale: str,
        cmin: float | None = None,
        cmax: float | None = None,
        subplot_columns: int = 3,
        colorbar_column: int = 3,
        colorbar_y: float | None = None,
        colorbar_length: float = 0.36,
    ) -> None:
        coords = self._numpy(self.geometry.coords_valid)
        figure.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "size": 4,
                    "color": values,
                    "colorscale": colorscale,
                    "cmin": cmin,
                    "cmax": cmax,
                    "showscale": col == colorbar_column,
                    "colorbar": {
                        "len": colorbar_length,
                        "x": 1.02,
                        "y": (
                            colorbar_y
                            if colorbar_y is not None
                            else (0.79 if row == 1 else 0.21)
                        ),
                    },
                },
                customdata=values,
                hovertemplate=(
                    "x=%{x:.6f}<br>y=%{y:.6f}<br>value=%{customdata:.6e}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        figure.layout.annotations[(row - 1) * subplot_columns + col - 1].text = title

    def _write_geometry_figure(self) -> str:
        if self.blend_fields.weight_construction == "compact_c2_ramp":
            return self._write_compact_ramp_geometry_figure()
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("J_phi", "J_psi", "w_phi", "w_psi"),
            horizontal_spacing=0.12,
            vertical_spacing=0.12,
        )
        fields = (
            (self.blend_fields.j_phi, "J_phi", "Viridis", None, None),
            (self.blend_fields.j_psi, "J_psi", "Viridis", None, None),
            (self.blend_fields.w_phi, "w_phi", "Viridis", 0.0, 1.0),
            (self.blend_fields.w_psi, "w_psi", "Viridis", 0.0, 1.0),
        )
        coords = self._numpy(self.geometry.coords_valid)
        for index, (tensor, title, colorscale, cmin, cmax) in enumerate(fields):
            row = index // 2 + 1
            col = index % 2 + 1
            values = self._numpy(tensor)
            colorbar_y = 0.79 if row == 1 else 0.21
            figure.add_trace(
                go.Scattergl(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker={
                        "size": 4,
                        "color": values,
                        "colorscale": colorscale,
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": col == 2,
                        "colorbar": {
                            "len": 0.35,
                            "x": 1.02,
                            "y": colorbar_y,
                        },
                    },
                    customdata=values,
                    hovertemplate=(
                        "x=%{x:.6f}<br>y=%{y:.6f}<br>"
                        "value=%{customdata:.6e}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(scaleanchor="y", scaleratio=1)
        figure.update_layout(
            title="Fixed geometry-only cross-axis blend fields",
            template=self.request.theme,
            width=1180,
            height=940,
        )
        base = self.request.outdir / "figures" / "geometry" / "fixed_blend_fields"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _write_compact_ramp_geometry_figure(self) -> str:
        figure = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=(
                "distance to Gamma_phi",
                "distance to Gamma_psi",
                "B_phi",
                "B_psi",
                "theta",
                "w_phi",
                "w_psi",
                "compact support",
            ),
            horizontal_spacing=0.07,
            vertical_spacing=0.12,
        )
        fields = (
            (self.blend_fields.distance_phi, "Viridis", None, None),
            (self.blend_fields.distance_psi, "Viridis", None, None),
            (self.blend_fields.influence_phi, "Viridis", 0.0, 1.0),
            (self.blend_fields.influence_psi, "Viridis", 0.0, 1.0),
            (self.blend_fields.theta, "RdBu", -1.0, 1.0),
            (self.blend_fields.w_phi, "Viridis", 0.0, 1.0),
            (self.blend_fields.w_psi, "Viridis", 0.0, 1.0),
            (
                self.blend_fields.ramp_support_mask.to(torch.float64),
                "Viridis",
                0.0,
                1.0,
            ),
        )
        coords = self._numpy(self.geometry.coords_valid)
        for index, (tensor, colorscale, cmin, cmax) in enumerate(fields):
            row = index // 4 + 1
            col = index % 4 + 1
            values = self._numpy(tensor)
            figure.add_trace(
                go.Scattergl(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker={
                        "size": 3,
                        "color": values,
                        "colorscale": colorscale,
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": col == 4,
                        "colorbar": {
                            "len": 0.35,
                            "x": 1.01,
                            "y": 0.79 if row == 1 else 0.21,
                        },
                    },
                    customdata=values,
                    hovertemplate=(
                        "x=%{x:.6f}<br>y=%{y:.6f}<br>"
                        "value=%{customdata:.6e}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        figure.update_xaxes(scaleanchor="y", scaleratio=1)
        figure.update_layout(
            title="Compact C2 topology-distance cross-axis blend fields",
            template=self.request.theme,
            width=1720,
            height=900,
        )
        base = self.request.outdir / "figures" / "geometry" / "compact_c2_ramp_fields"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _write_paired_metric_figure(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> str:
        baseline = np.asarray([float(row["baseline_rel_sol"]) for row in rows])
        blend = np.asarray([float(row["blend_rel_sol"]) for row in rows])
        sample_ids = np.asarray([int(row["sample_id"]) for row in rows])
        limit = max(float(np.max(baseline)), float(np.max(blend))) * 1.03
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=100.0 * baseline,
                y=100.0 * blend,
                mode="markers",
                marker={"size": 8, "color": sample_ids, "colorscale": "Viridis"},
                customdata=sample_ids,
                hovertemplate=(
                    "sample=%{customdata}<br>baseline=%{x:.4f}%"
                    "<br>blend=%{y:.4f}%<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[0.0, 100.0 * limit],
                y=[0.0, 100.0 * limit],
                mode="lines",
                line={"color": "#333333", "dash": "dash"},
                name="equal",
            )
        )
        figure.update_layout(
            title="Per-sample solution error: equal mean vs fixed smooth blend",
            xaxis_title="Baseline rel_sol (%)",
            yaxis_title="Fixed blend rel_sol (%)",
            template=self.request.theme,
            width=820,
            height=760,
        )
        base = self.request.outdir / "figures" / "aggregate" / "paired_rel_sol"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _write_selected_figures(
        self,
        evaluation: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> list[str]:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(evaluation.sample_ids.tolist())
        }
        paths: list[str] = []
        transition = self._numpy(
            self.blend_fields.transition_point_mask.to(torch.float64)
        )
        for sample_id in selected:
            offset = sample_to_offset[sample_id]
            sol = self._numpy(evaluation.sol[offset])
            u_phi_error = self._numpy(evaluation.u_phi[offset]) - sol
            u_psi_error = self._numpy(evaluation.u_psi[offset]) - sol
            baseline_error = self._numpy(evaluation.baseline[offset]) - sol
            blend_error = self._numpy(evaluation.blend[offset]) - sol
            blend_delta = self._numpy(evaluation.blend[offset]) - self._numpy(
                evaluation.baseline[offset]
            )
            shared_limit = self._signed_limit(
                u_phi_error,
                u_psi_error,
                baseline_error,
                blend_error,
            )
            delta_limit = self._signed_limit(blend_delta)
            figure = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=(
                    "u_phi - sol",
                    "u_psi - sol",
                    "0.5(u_phi+u_psi) - sol",
                    "u_blend - sol",
                    "u_blend - u_baseline",
                    "transition zone",
                ),
                horizontal_spacing=0.1,
                vertical_spacing=0.12,
            )
            panels = (
                (u_phi_error, "u_phi - sol", "RdBu", -shared_limit, shared_limit),
                (u_psi_error, "u_psi - sol", "RdBu", -shared_limit, shared_limit),
                (
                    baseline_error,
                    "equal mean error",
                    "RdBu",
                    -shared_limit,
                    shared_limit,
                ),
                (
                    blend_error,
                    "fixed blend error",
                    "RdBu",
                    -shared_limit,
                    shared_limit,
                ),
                (
                    blend_delta,
                    "blend - baseline",
                    "RdBu",
                    -delta_limit,
                    delta_limit,
                ),
                (transition, "transition zone", "Viridis", 0.0, 1.0),
            )
            for panel_index, (values, title, scale, cmin, cmax) in enumerate(panels):
                self._add_scatter(
                    figure,
                    row=panel_index // 3 + 1,
                    col=panel_index % 3 + 1,
                    values=values,
                    title=title,
                    colorscale=scale,
                    cmin=cmin,
                    cmax=cmax,
                )
            figure.update_xaxes(scaleanchor="y", scaleratio=1)
            figure.update_layout(
                title=(f"Sample {sample_id}: fixed smooth cross-axis blend comparison"),
                template=self.request.theme,
                width=1500,
                height=900,
            )
            base = (
                self.request.outdir
                / "figures"
                / "selected"
                / f"sample_{sample_id:04d}_blend_comparison"
            )
            save_plotly_figure(figure, base, logger=self.logger)
            paths.append(
                str(base.with_suffix(".html").relative_to(self.request.outdir))
            )
        return paths


class FixedSmoothCrossAxisBlendDiagnostic(
    FixedSmoothBlendGeometryMixin,
    FixedSmoothBlendMetricMixin,
    FixedSmoothBlendPlotMixin,
):
    """Evaluate a fixed smooth blend without changing production inference."""

    def __init__(
        self,
        request: FixedSmoothBlendDiagnosticRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.geometry: ComplexGeometryMetadata
        self.blend_fields: FixedSmoothBlendGeometryFields

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError(
                "Fixed smooth blend diagnostic requires geometry_mode='complex'."
            )
        geometry_path = self.request.geometry or configs.dataset.geometry_path
        test_path = self.request.test_path or configs.dataset.test_path
        coefficient_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        if geometry_path is None:
            raise ValueError("A complex geometry path is required.")
        if test_path is None:
            raise ValueError("A full-reference test path is required.")

        device = torch.device(self.request.device or configs.coupling_training.device)
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=configs.dataset.dtype,
        )
        self.blend_fields = self.build_fixed_blend_fields(
            self.geometry,
            self.request.blend,
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
        evaluation = self._evaluate_dataset(dataset, configs, device)
        rows = self._sample_metric_rows(evaluation)
        aggregate = self._aggregate_metrics(rows)
        paired_bootstrap = self._paired_bootstrap_summary(rows)
        sweep_rows: list[dict[str, float | int | str]] = []
        sweep_summary: dict[str, Any] = {}
        if self.request.run_compact_sweep:
            sweep_rows = self._run_compact_sweep(evaluation)
            sweep_summary = self._compact_sweep_summary(sweep_rows)
            self._write_csv(
                self.request.outdir / "metrics" / "compact_c2_ramp_parameter_sweep.csv",
                sweep_rows,
            )
        selected, roles = self._select_samples(rows)

        metrics_path = (
            self.request.outdir / "metrics" / "per_sample_blend_comparison.csv"
        )
        self._write_csv(metrics_path, rows)
        if self.request.save_generated_data:
            self._write_selected_npz(evaluation, selected)

        figure_paths = [
            self._write_geometry_figure(),
            self._write_paired_metric_figure(rows),
        ]
        if sweep_rows:
            figure_paths.append(self._write_compact_sweep_figure(sweep_rows))
        figure_paths.extend(self._write_selected_figures(evaluation, selected))
        summary = self._build_summary(
            configs=configs,
            dataset=dataset,
            geometry_path=Path(geometry_path),
            test_path=Path(test_path),
            coefficient_path=(
                None if coefficient_path is None else Path(coefficient_path)
            ),
            device=device,
            rows=rows,
            aggregate=aggregate,
            paired_bootstrap=paired_bootstrap,
            sweep_summary=sweep_summary,
            selected=selected,
            roles=roles,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Fixed blend diagnostic complete: baseline_rel_sol=%.6f "
                "blend_rel_sol=%.6f verdict=%s",
                aggregate["baseline_rel_sol_mean"],
                aggregate["blend_rel_sol_mean"],
                aggregate["verdict"],
            )
        return summary

    def _evaluate_dataset(
        self,
        dataset: ComplexCouplingDataset,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> FixedSmoothBlendEvaluation:
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
        evaluator = ComplexCouplingEvaluator(
            model=coupling_model,
            green_model=green_model,
            config=configs.coupling_training,
            device=device,
            work_dir=self.request.outdir / "_evaluator",
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        sample_ids: list[torch.Tensor] = []
        file_stems: list[str] = []
        sol: list[torch.Tensor] = []
        u_phi: list[torch.Tensor] = []
        u_psi: list[torch.Tensor] = []
        w_phi = self.blend_fields.w_phi.to(device)
        w_psi = self.blend_fields.w_psi.to(device)
        with torch.no_grad():
            for batch in loader:
                prediction = evaluator.predict_batch(batch.to(device))
                if not bool(torch.all(prediction.batch.has_solution).item()):
                    raise ValueError(
                        "All diagnostic test samples must contain reference sol."
                    )
                sample_ids.append(prediction.batch.sample_indices.detach().cpu())
                file_stems.extend(prediction.batch.file_stems)
                sol.append(prediction.batch.sol_valid.detach().cpu())
                u_phi_batch = prediction.reconstruction.u_phi_valid
                u_psi_batch = prediction.reconstruction.u_psi_valid
                u_phi.append(u_phi_batch.detach().cpu())
                u_psi.append(u_psi_batch.detach().cpu())
        sample_ids_tensor = torch.cat(sample_ids, dim=0)
        sol_tensor = torch.cat(sol, dim=0)
        u_phi_tensor = torch.cat(u_phi, dim=0)
        u_psi_tensor = torch.cat(u_psi, dim=0)
        baseline = 0.5 * (u_phi_tensor + u_psi_tensor)
        blend = (
            w_phi.detach().cpu().unsqueeze(0) * u_phi_tensor
            + w_psi.detach().cpu().unsqueeze(0) * u_psi_tensor
        )
        return FixedSmoothBlendEvaluation(
            sample_ids=sample_ids_tensor,
            file_stems=tuple(file_stems),
            sol=sol_tensor,
            u_phi=u_phi_tensor,
            u_psi=u_psi_tensor,
            baseline=baseline,
            blend=blend,
        )

    @staticmethod
    def _evaluation_with_blend_fields(
        evaluation: FixedSmoothBlendEvaluation,
        blend_fields: FixedSmoothBlendGeometryFields,
    ) -> FixedSmoothBlendEvaluation:
        w_phi = blend_fields.w_phi.detach().cpu().unsqueeze(0)
        w_psi = blend_fields.w_psi.detach().cpu().unsqueeze(0)
        return replace(
            evaluation,
            blend=w_phi * evaluation.u_phi + w_psi * evaluation.u_psi,
        )

    def _compact_sweep_row(
        self,
        *,
        estimator: str,
        gamma: float | None,
        width_steps: float | None,
        width_physical: float | None,
        evaluation: FixedSmoothBlendEvaluation,
        blend_fields: FixedSmoothBlendGeometryFields,
    ) -> dict[str, float | int | str]:
        rows = self._sample_metric_rows(evaluation, blend_fields)
        aggregate = self._aggregate_metrics(rows)
        geometry = self._geometry_statistics(blend_fields)
        return {
            "estimator": estimator,
            "gamma": "" if gamma is None else gamma,
            "ramp_width_steps": "" if width_steps is None else width_steps,
            "ramp_width_physical": ("" if width_physical is None else width_physical),
            "blend_rel_sol_mean": float(aggregate["blend_rel_sol_mean"]),
            "rel_sol_mean_relative_change": float(
                aggregate["rel_sol_mean_relative_change"]
            ),
            "rel_sol_blend_win_count": int(aggregate["rel_sol_blend_win_count"]),
            "blend_transition_error_rms_mean": float(
                aggregate["blend_transition_error_rms_mean"]
            ),
            "transition_error_rms_mean_relative_change": float(
                aggregate["transition_error_rms_mean_relative_change"]
            ),
            "blend_transition_trace_error_jump_rms_mean": float(
                aggregate["blend_transition_trace_error_jump_rms_mean"]
            ),
            "transition_trace_error_jump_mean_relative_change": float(
                aggregate["transition_trace_error_jump_mean_relative_change"]
            ),
            "blend_correction_rms_mean": float(aggregate["blend_correction_rms_mean"]),
            "mse_change_linear_term_mean": float(
                aggregate["mse_change_linear_term_mean"]
            ),
            "mse_change_quadratic_term_mean": float(
                aggregate["mse_change_quadratic_term_mean"]
            ),
            "transition_mse_change_linear_term_mean": float(
                aggregate["transition_mse_change_linear_term_mean"]
            ),
            "transition_mse_change_quadratic_term_mean": float(
                aggregate["transition_mse_change_quadratic_term_mean"]
            ),
            "w_phi_min": float(geometry["w_phi_min"]),
            "w_phi_max": float(geometry["w_phi_max"]),
            "w_neighbor_jump_max": max(
                float(geometry["w_phi_neighbor_jump_max"]),
                float(geometry["w_psi_neighbor_jump_max"]),
            ),
            "w_neighbor_slope_max": max(
                float(geometry["w_phi_neighbor_slope_max"]),
                float(geometry["w_psi_neighbor_slope_max"]),
            ),
            "ramp_support_point_count": int(geometry["ramp_support_point_count"]),
            "outside_support_theta_max_abs": float(
                geometry["outside_support_theta_max_abs"]
            ),
            "outside_support_correction_max_abs": float(
                aggregate["blend_outside_support_correction_max_abs"]
            ),
        }

    def _run_compact_sweep(
        self,
        evaluation: FixedSmoothBlendEvaluation,
    ) -> list[dict[str, float | int | str]]:
        step = max(float(self.geometry.hx.item()), float(self.geometry.hy.item()))
        sweep_rows: list[dict[str, float | int | str]] = []

        legacy_config = replace(
            self.request.blend,
            weight_construction="jump_smoothed",
        )
        legacy_fields = self.build_fixed_blend_fields(self.geometry, legacy_config)
        legacy_evaluation = self._evaluation_with_blend_fields(
            evaluation,
            legacy_fields,
        )
        sweep_rows.append(
            self._compact_sweep_row(
                estimator="legacy_jump_smoothed",
                gamma=None,
                width_steps=None,
                width_physical=None,
                evaluation=legacy_evaluation,
                blend_fields=legacy_fields,
            )
        )

        for gamma in self.request.sweep_gammas:
            for width_steps in self.request.sweep_width_steps:
                width = width_steps * step
                compact_config = replace(
                    self.request.blend,
                    weight_construction="compact_c2_ramp",
                    ramp_gamma=gamma,
                    ramp_width=width,
                )
                compact_fields = self.build_fixed_blend_fields(
                    self.geometry,
                    compact_config,
                )
                compact_evaluation = self._evaluation_with_blend_fields(
                    evaluation,
                    compact_fields,
                )
                sweep_rows.append(
                    self._compact_sweep_row(
                        estimator="compact_c2_ramp",
                        gamma=gamma,
                        width_steps=width_steps,
                        width_physical=width,
                        evaluation=compact_evaluation,
                        blend_fields=compact_fields,
                    )
                )
        return sweep_rows

    @staticmethod
    def _compact_sweep_summary(
        rows: list[dict[str, float | int | str]],
    ) -> dict[str, Any]:
        compact_rows = [row for row in rows if row["estimator"] == "compact_c2_ramp"]
        if not compact_rows:
            return {}

        def identifying_fields(
            row: dict[str, float | int | str],
        ) -> dict[str, float | int | str]:
            return {
                "gamma": row["gamma"],
                "ramp_width_steps": row["ramp_width_steps"],
                "ramp_width_physical": row["ramp_width_physical"],
            }

        best_rel = min(
            compact_rows,
            key=lambda row: float(row["blend_rel_sol_mean"]),
        )
        best_transition = min(
            compact_rows,
            key=lambda row: float(row["blend_transition_error_rms_mean"]),
        )
        best_trace = min(
            compact_rows,
            key=lambda row: float(row["blend_transition_trace_error_jump_rms_mean"]),
        )
        optimal_gamma_by_width: list[dict[str, float]] = []
        for width in sorted({float(row["ramp_width_steps"]) for row in compact_rows}):
            representative = next(
                row
                for row in compact_rows
                if float(row["ramp_width_steps"]) == width and float(row["gamma"]) > 0.0
            )
            gamma = float(representative["gamma"])

            def quadratic_optimum(linear_key: str, quadratic_key: str) -> float:
                linear_coefficient = float(representative[linear_key]) / gamma
                quadratic_coefficient = float(representative[quadratic_key]) / gamma**2
                if quadratic_coefficient <= 0.0:
                    return 0.0
                return min(
                    1.0,
                    max(
                        0.0,
                        -linear_coefficient / (2.0 * quadratic_coefficient),
                    ),
                )

            optimal_gamma_by_width.append(
                {
                    "ramp_width_steps": width,
                    "global_mse_quadratic_optimum": quadratic_optimum(
                        "mse_change_linear_term_mean",
                        "mse_change_quadratic_term_mean",
                    ),
                    "transition_mse_quadratic_optimum": quadratic_optimum(
                        "transition_mse_change_linear_term_mean",
                        "transition_mse_change_quadratic_term_mean",
                    ),
                }
            )
        return {
            "exploratory_test_target_sweep": True,
            "selection_is_not_independent_confirmation": True,
            "compact_configuration_count": len(compact_rows),
            "best_mean_rel_sol": {
                **identifying_fields(best_rel),
                "value": best_rel["blend_rel_sol_mean"],
                "relative_change": best_rel["rel_sol_mean_relative_change"],
            },
            "best_transition_error_rms": {
                **identifying_fields(best_transition),
                "value": best_transition["blend_transition_error_rms_mean"],
                "relative_change": best_transition[
                    "transition_error_rms_mean_relative_change"
                ],
            },
            "best_transition_trace_error_jump": {
                **identifying_fields(best_trace),
                "value": best_trace["blend_transition_trace_error_jump_rms_mean"],
                "relative_change": best_trace[
                    "transition_trace_error_jump_mean_relative_change"
                ],
            },
            "target_derived_mse_optimal_gamma_by_width": optimal_gamma_by_width,
        }

    def _write_compact_sweep_figure(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> str:
        compact_rows = [row for row in rows if row["estimator"] == "compact_c2_ramp"]
        gammas = sorted({float(row["gamma"]) for row in compact_rows})
        widths = sorted({float(row["ramp_width_steps"]) for row in compact_rows})

        def matrix(key: str, *, percent: bool) -> np.ndarray:
            lookup = {
                (float(row["gamma"]), float(row["ramp_width_steps"])): float(row[key])
                for row in compact_rows
            }
            scale = 100.0 if percent else 1.0
            return np.asarray(
                [
                    [scale * lookup[(gamma, width)] for width in widths]
                    for gamma in gammas
                ],
                dtype=np.float64,
            )

        panels = (
            (
                "rel_sol_mean_relative_change",
                "Mean rel_sol change (%)",
                True,
                "RdBu",
            ),
            (
                "transition_error_rms_mean_relative_change",
                "Transition RMS change (%)",
                True,
                "RdBu",
            ),
            (
                "transition_trace_error_jump_mean_relative_change",
                "Transition trace-jump change (%)",
                True,
                "RdBu",
            ),
            (
                "w_neighbor_jump_max",
                "Maximum neighboring weight jump",
                False,
                "Viridis",
            ),
        )
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=tuple(panel[1] for panel in panels),
            horizontal_spacing=0.14,
            vertical_spacing=0.16,
        )
        for index, (key, _title, percent, colorscale) in enumerate(panels):
            values = matrix(key, percent=percent)
            symmetric_limit = (
                max(float(np.max(np.abs(values))), 1.0e-15)
                if colorscale == "RdBu"
                else None
            )
            figure.add_trace(
                go.Heatmap(
                    x=widths,
                    y=gammas,
                    z=values,
                    colorscale=colorscale,
                    zmin=None if symmetric_limit is None else -symmetric_limit,
                    zmax=symmetric_limit,
                    text=np.vectorize(lambda value: f"{value:.3g}")(values),
                    texttemplate="%{text}",
                    hovertemplate=(
                        "width=%{x}h<br>gamma=%{y}<br>value=%{z:.6g}<extra></extra>"
                    ),
                    colorbar={
                        "len": 0.35,
                        "x": 0.45 if index % 2 == 0 else 1.02,
                        "y": 0.79 if index < 2 else 0.21,
                    },
                ),
                row=index // 2 + 1,
                col=index % 2 + 1,
            )
        figure.update_xaxes(
            title_text="Ramp width (grid steps)",
            tickmode="array",
            tickvals=widths,
            ticktext=[f"{value:g}" for value in widths],
        )
        figure.update_yaxes(
            title_text="gamma",
            tickmode="array",
            tickvals=gammas,
            ticktext=[f"{value:g}" for value in gammas],
        )
        figure.update_layout(
            title="Exploratory compact C2 ramp sweep",
            template=self.request.theme,
            width=1280,
            height=960,
        )
        base = self.request.outdir / "figures" / "aggregate" / "compact_c2_ramp_sweep"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _select_samples(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> tuple[tuple[int, ...], dict[str, int]]:
        valid_ids = {int(row["sample_id"]) for row in rows}
        if self.request.selected_samples is not None:
            selected = tuple(dict.fromkeys(self.request.selected_samples))
            invalid = sorted(set(selected) - valid_ids)
            if invalid:
                raise IndexError(f"Selected sample indices are unavailable: {invalid}.")
            return selected, {}
        sorted_rows = sorted(rows, key=lambda row: float(row["baseline_rel_sol"]))
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
        return tuple(dict.fromkeys(roles.values())), roles

    @staticmethod
    def _write_csv(
        path: Path,
        rows: list[dict[str, float | int | str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_selected_npz(
        self,
        evaluation: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> None:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(evaluation.sample_ids.tolist())
        }
        offsets = [sample_to_offset[sample_id] for sample_id in selected]
        payload = {
            "selected_sample_ids": np.asarray(selected, dtype=np.int64),
            "selected_file_stems": np.asarray(
                [evaluation.file_stems[offset] for offset in offsets]
            ),
            "coords_valid": self._numpy(self.geometry.coords_valid),
            "j_phi_raw": self._numpy(self.blend_fields.j_phi_raw),
            "j_psi_raw": self._numpy(self.blend_fields.j_psi_raw),
            "j_phi": self._numpy(self.blend_fields.j_phi),
            "j_psi": self._numpy(self.blend_fields.j_psi),
            "rho_phi": self._numpy(self.blend_fields.rho_phi),
            "rho_psi": self._numpy(self.blend_fields.rho_psi),
            "distance_phi": self._numpy(self.blend_fields.distance_phi),
            "distance_psi": self._numpy(self.blend_fields.distance_psi),
            "influence_phi": self._numpy(self.blend_fields.influence_phi),
            "influence_psi": self._numpy(self.blend_fields.influence_psi),
            "theta": self._numpy(self.blend_fields.theta),
            "w_phi": self._numpy(self.blend_fields.w_phi),
            "w_psi": self._numpy(self.blend_fields.w_psi),
            "ramp_support_mask": self._numpy(self.blend_fields.ramp_support_mask),
            "phi_transition_coordinates": self._numpy(
                self.blend_fields.phi_transition_coordinates
            ),
            "psi_transition_coordinates": self._numpy(
                self.blend_fields.psi_transition_coordinates
            ),
            "transition_point_mask": self._numpy(
                self.blend_fields.transition_point_mask
            ),
            "sol": self._numpy(evaluation.sol[offsets]),
            "u_phi": self._numpy(evaluation.u_phi[offsets]),
            "u_psi": self._numpy(evaluation.u_psi[offsets]),
            "u_baseline": self._numpy(evaluation.baseline[offsets]),
            "u_blend": self._numpy(evaluation.blend[offsets]),
            "baseline_error": self._numpy(
                evaluation.baseline[offsets] - evaluation.sol[offsets]
            ),
            "blend_error": self._numpy(
                evaluation.blend[offsets] - evaluation.sol[offsets]
            ),
        }
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "selected_fixed_smooth_blend_arrays.npz",
            **payload,  # type: ignore[arg-type]
        )

    @staticmethod
    def _max_edge_jump(values: torch.Tensor, edges: torch.Tensor) -> float:
        if edges.numel() == 0:
            return 0.0
        return float((values[edges[:, 1]] - values[edges[:, 0]]).abs().max().item())

    def _max_edge_slope(self, values: torch.Tensor, edges: torch.Tensor) -> float:
        if edges.numel() == 0:
            return 0.0
        delta = (values[edges[:, 1]] - values[edges[:, 0]]).abs()
        spacing = torch.linalg.vector_norm(
            self.geometry.coords_valid[edges[:, 1]]
            - self.geometry.coords_valid[edges[:, 0]],
            dim=1,
        )
        return float((delta / spacing.clamp_min(1.0e-15)).max().item())

    def _geometry_statistics(
        self,
        blend_fields: FixedSmoothBlendGeometryFields | None = None,
    ) -> dict[str, Any]:
        fields = self.blend_fields if blend_fields is None else blend_fields
        all_edges = torch.cat((self.geometry.x_edges, self.geometry.y_edges), dim=0)
        outside_support = ~fields.ramp_support_mask
        return {
            "valid_point_count": self.geometry.num_points,
            "phi_transition_edge_count": int(
                fields.phi_transition_edge_mask.sum().item()
            ),
            "psi_transition_edge_count": int(
                fields.psi_transition_edge_mask.sum().item()
            ),
            "transition_zone_point_count": int(
                fields.transition_point_mask.sum().item()
            ),
            "weight_construction": fields.weight_construction,
            "j_phi_raw_max": float(fields.j_phi_raw.max().item()),
            "j_psi_raw_max": float(fields.j_psi_raw.max().item()),
            "j_phi_max": float(fields.j_phi.max().item()),
            "j_psi_max": float(fields.j_psi.max().item()),
            "w_phi_min": float(fields.w_phi.min().item()),
            "w_phi_max": float(fields.w_phi.max().item()),
            "w_psi_min": float(fields.w_psi.min().item()),
            "w_psi_max": float(fields.w_psi.max().item()),
            "weight_sum_max_abs_residual": float(
                (fields.w_phi + fields.w_psi - 1.0).abs().max().item()
            ),
            "w_phi_neighbor_jump_max": self._max_edge_jump(
                fields.w_phi,
                all_edges,
            ),
            "w_psi_neighbor_jump_max": self._max_edge_jump(
                fields.w_psi,
                all_edges,
            ),
            "w_phi_neighbor_slope_max": self._max_edge_slope(
                fields.w_phi,
                all_edges,
            ),
            "w_psi_neighbor_slope_max": self._max_edge_slope(
                fields.w_psi,
                all_edges,
            ),
            "ramp_support_point_count": int(fields.ramp_support_mask.sum().item()),
            "theta_max_abs": float(fields.theta.abs().max().item()),
            "outside_support_theta_max_abs": (
                0.0
                if not torch.any(outside_support)
                else float(fields.theta[outside_support].abs().max().item())
            ),
            "resolved_ramp_width": fields.resolved_ramp_width,
            "phi_transition_coordinates": [
                float(value)
                for value in fields.phi_transition_coordinates.detach().cpu().tolist()
            ],
            "psi_transition_coordinates": [
                float(value)
                for value in fields.psi_transition_coordinates.detach().cpu().tolist()
            ],
        }

    def _build_summary(
        self,
        *,
        configs: CouplingArtifactConfigs,
        dataset: ComplexCouplingDataset,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path | None,
        device: torch.device,
        rows: list[dict[str, float | int | str]],
        aggregate: dict[str, float | int | str],
        paired_bootstrap: dict[str, Any],
        sweep_summary: dict[str, Any],
        selected: tuple[int, ...],
        roles: dict[str, int],
        figure_paths: list[str],
    ) -> dict[str, Any]:
        del rows
        return {
            "diagnostic": "fixed_smooth_cross_axis_reconstruction_blend",
            "status": "post_hoc_diagnostic_only",
            "production_code_changed": False,
            "training_or_checkpoint_changed": False,
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": (
                None if coefficient_path is None else str(coefficient_path)
            ),
            "device": str(device),
            "dtype": str(configs.dataset.dtype).replace("torch.", ""),
            "num_samples": len(dataset),
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "baseline_formula": "u_baseline=0.5*(u_phi+u_psi)",
            "blend_formula": "u_blend=w_phi*u_phi+w_psi*u_psi",
            "weight_construction": {
                "sample_independent": True,
                "geometry_only": True,
                "uses_rhs": False,
                "uses_sol": False,
                "uses_flux_targets": False,
                "method": self.request.blend.weight_construction,
                "j_phi": "smooth_on_y_edges(abs(delta(log(Lx^2))))",
                "j_psi": "smooth_on_x_edges(abs(delta(log(Ly^2))))",
                "rho": (
                    "reliability_floor + exp(-alpha*J)"
                    if self.request.blend.weight_construction == "jump_smoothed"
                    else None
                ),
                "partition": (
                    "w_phi=rho_phi/(rho_phi+rho_psi); w_psi=1-w_phi"
                    if self.request.blend.weight_construction == "jump_smoothed"
                    else (
                        "theta=gamma*(B_psi-B_phi); "
                        "w_phi=(1+theta)/2; w_psi=(1-theta)/2"
                    )
                ),
                "compact_ramp": (
                    None
                    if self.request.blend.weight_construction == "jump_smoothed"
                    else {
                        "interface_detection": (
                            "midpoints between neighboring axial-line coordinates "
                            "whose connected-segment multiplicity changes"
                        ),
                        "distance": "absolute transverse-coordinate distance",
                        "bump": "1-10*s^3+15*s^4-6*s^5 for s<1; zero otherwise",
                        "resolved_width": self.blend_fields.resolved_ramp_width,
                    }
                ),
                "preset_selection": "recommended preset fixed before sweep",
                "config": asdict(self.request.blend),
            },
            "transition_diagnostic": {
                "edge_threshold": "abs(delta(log(L^2))) > transition_log_threshold",
                "zone": (
                    "transition edge endpoints dilated on existing x/y axial edges"
                ),
                "trace_jump": "RMS of one-edge signed field differences",
            },
            "geometry_statistics": self._geometry_statistics(),
            "aggregate_metrics": aggregate,
            "paired_bootstrap": paired_bootstrap,
            "compact_parameter_sweep": sweep_summary,
            "compact_parameter_sweep_csv": (
                "metrics/compact_c2_ramp_parameter_sweep.csv" if sweep_summary else None
            ),
            "metric_role": "evaluation_only_full_reference_test",
            "figure_count": len(figure_paths),
            "figure_paths": figure_paths,
            "raw_archive": (
                "data/selected_fixed_smooth_blend_arrays.npz"
                if self.request.save_generated_data
                else None
            ),
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        aggregate = summary["aggregate_metrics"]
        geometry = summary["geometry_statistics"]
        sweep = summary["compact_parameter_sweep"]
        bootstrap = summary["paired_bootstrap"]["metrics"]
        baseline_percent = 100.0 * float(aggregate["baseline_rel_sol_mean"])
        blend_percent = 100.0 * float(aggregate["blend_rel_sol_mean"])
        rel_change = 100.0 * float(aggregate["rel_sol_mean_relative_change"])
        transition_change = 100.0 * float(
            aggregate["transition_error_rms_mean_relative_change"]
        )
        trace_change = 100.0 * float(
            aggregate["transition_trace_error_jump_mean_relative_change"]
        )
        if self.request.blend.weight_construction == "compact_c2_ramp":
            fixed_preset = f"""- weight construction: `compact_c2_ramp`
- gamma: `{self.request.blend.ramp_gamma:.6g}`
- requested ramp width: `{self.request.blend.ramp_width}`
- resolved physical ramp width: `{float(geometry["resolved_ramp_width"]):.12g}`
- horizontal transition coordinates: `{geometry["phi_transition_coordinates"]}`
- vertical transition coordinates: `{geometry["psi_transition_coordinates"]}`
- transition log threshold: `{self.request.blend.transition_log_threshold:.12g}`
- transition dilation steps: `{self.request.blend.transition_dilation_steps}`

The recommended preset was fixed before the exploratory sweep. It uses only
axial-line multiplicity, transverse coordinates, and grid spacing."""
        else:
            fixed_preset = f"""- weight construction: `jump_smoothed`
- alpha: `{self.request.blend.alpha:.12g}`
- smoothing steps: `{self.request.blend.smoothing_steps}`
- smoothing relaxation: `{self.request.blend.smoothing_relaxation:.6g}`
- reliability floor: `{self.request.blend.reliability_floor:.6g}`
- transition log threshold: `{self.request.blend.transition_log_threshold:.12g}`
- transition dilation steps: `{self.request.blend.transition_dilation_steps}`

The preset was fixed before reading test reference fields. The weight maps use
only geometry segment lengths and existing axial edges."""
        if sweep:
            best_rel = sweep["best_mean_rel_sol"]
            best_transition = sweep["best_transition_error_rms"]
            best_trace = sweep["best_transition_trace_error_jump"]
            gamma_optima = sweep["target_derived_mse_optimal_gamma_by_width"]
            global_optima = [
                float(row["global_mse_quadratic_optimum"]) for row in gamma_optima
            ]
            transition_optima = [
                float(row["transition_mse_quadratic_optimum"]) for row in gamma_optima
            ]
            sweep_section = f"""
## Exploratory Parameter Sweep

The same frozen directional reconstructions were evaluated for
`{sweep["compact_configuration_count"]}` compact-ramp configurations. This
sweep reads the test solution and is exploratory; its selected optimum is not
an independent confirmation.

| Objective | gamma | Width | Relative change |
| --- | ---: | ---: | ---: |
| Best mean rel_sol | {float(best_rel["gamma"]):.3g} | {float(best_rel["ramp_width_steps"]):.3g}h | {100.0 * float(best_rel["relative_change"]):+.3f}% |
| Best transition RMS | {float(best_transition["gamma"]):.3g} | {float(best_transition["ramp_width_steps"]):.3g}h | {100.0 * float(best_transition["relative_change"]):+.3f}% |
| Best trace-error jump | {float(best_trace["gamma"]):.3g} | {float(best_trace["ramp_width_steps"]):.3g}h | {100.0 * float(best_trace["relative_change"]):+.3f}% |

The exact linear-plus-quadratic MSE decomposition estimates target-derived
optimal gamma ranges of
`[{min(global_optima):.3f}, {max(global_optima):.3f}]` globally and
`[{min(transition_optima):.3f}, {max(transition_optima):.3f}]` in the transition
zone. These are explanatory diagnostics, not production hyperparameters.
"""
        else:
            sweep_section = ""
        bootstrap_rows = []
        for label, key in (
            ("Mean rel_sol", "rel_sol"),
            ("Transition RMS", "transition_error_rms"),
            ("Transition trace-error jump", "transition_trace_error_jump_rms"),
        ):
            values = bootstrap[key]
            lower, upper = values["relative_change_ci95"]
            bootstrap_rows.append(
                f"| {label} | "
                f"[{100.0 * float(lower):+.3f}%, "
                f"{100.0 * float(upper):+.3f}%] | "
                f"{100.0 * float(values['bootstrap_probability_improvement']):.3f}% |"
            )
        bootstrap_table = "\n".join(bootstrap_rows)
        report = f"""# Fixed Smooth Cross-Axis Reconstruction Blend Diagnostic

## Verdict

`{aggregate["verdict"]}`

This is a post-hoc diagnostic. It does not alter CouplingNet, GreenNet,
projection, directional reconstructions, training loss, or the checkpoint.

## Fixed Preset

{fixed_preset}

## Full Test Comparison

| Diagnostic | Equal mean | Fixed blend | Relative change |
| --- | ---: | ---: | ---: |
| Mean rel_sol | {baseline_percent:.6f}% | {blend_percent:.6f}% | {rel_change:+.3f}% |
| Transition-zone error RMS | {float(aggregate["baseline_transition_error_rms_mean"]):.6e} | {float(aggregate["blend_transition_error_rms_mean"]):.6e} | {transition_change:+.3f}% |
| Transition trace-error jump RMS | {float(aggregate["baseline_transition_trace_error_jump_rms_mean"]):.6e} | {float(aggregate["blend_transition_trace_error_jump_rms_mean"]):.6e} | {trace_change:+.3f}% |

The fixed blend improves `rel_sol` on
`{aggregate["rel_sol_blend_win_count"]}/{aggregate["sample_count"]}` test
samples.

## Paired Bootstrap Audit

| Diagnostic | 95% CI for aggregate relative change | Improvement probability |
| --- | ---: | ---: |
{bootstrap_table}

The bootstrap treats the 50 source realizations as paired observations and
does not remove the exploratory test-set limitation.

## Weight Audit

- `w_phi` range: [{float(geometry["w_phi_min"]):.6f}, {float(geometry["w_phi_max"]):.6f}]
- `w_psi` range: [{float(geometry["w_psi_min"]):.6f}, {float(geometry["w_psi_max"]):.6f}]
- max partition residual: `{float(geometry["weight_sum_max_abs_residual"]):.6e}`
- max neighboring `w_phi` jump: `{float(geometry["w_phi_neighbor_jump_max"]):.6f}`
- max neighboring `w_psi` jump: `{float(geometry["w_psi_neighbor_jump_max"]):.6f}`
- max neighboring weight slope: `{max(float(geometry["w_phi_neighbor_slope_max"]), float(geometry["w_psi_neighbor_slope_max"])):.6f}`
- ramp-support points: `{geometry["ramp_support_point_count"]}`
- max `theta` outside compact support: `{float(geometry["outside_support_theta_max_abs"]):.6e}`
- max solution correction outside compact support: `{float(aggregate["blend_outside_support_correction_max_abs"]):.6e}`
- transition-zone points: `{geometry["transition_zone_point_count"]}`
- horizontal-chart transition edges: `{geometry["phi_transition_edge_count"]}`
- vertical-chart transition edges: `{geometry["psi_transition_edge_count"]}`

{sweep_section}

## Interpretation Boundary

The recommended preset is measured without training or checkpoint changes.
The optional sweep intentionally uses the same test references for sensitivity
analysis and cannot be presented as independent model selection evidence. The
blend cannot change directional source error, flux error, or either directional
reconstruction; it can only change their final scalar combination.
"""
        (self.request.outdir / "diagnosis_report.md").write_text(report)


def run_fixed_smooth_cross_axis_blend_diagnostic(
    request: FixedSmoothBlendDiagnosticRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the standalone fixed smooth cross-axis blend diagnostic."""

    return FixedSmoothCrossAxisBlendDiagnostic(request, logger=logger).run()
