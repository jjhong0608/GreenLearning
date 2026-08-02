from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_smooth_blend_diagnostics import (
    FixedSmoothBlendDiagnosticRequest,
    FixedSmoothBlendEvaluation,
    FixedSmoothCrossAxisBlendDiagnostic,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    load_coupling_artifact_configs,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class MismatchGradientBlendConfig:
    """Prediction-only mismatch-gradient blend parameters."""

    gamma: float = 0.5
    smoothing_steps: int = 2
    smoothing_relaxation: float = 0.5
    activation_lower: float = 0.15
    activation_upper: float = 0.35
    scale_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if (
            isinstance(self.smoothing_steps, bool)
            or not isinstance(self.smoothing_steps, int)
            or self.smoothing_steps < 0
        ):
            raise ValueError("smoothing_steps must be a non-negative integer.")
        if (
            not math.isfinite(self.smoothing_relaxation)
            or not 0.0 < self.smoothing_relaxation <= 1.0
        ):
            raise ValueError("smoothing_relaxation must be in (0, 1].")
        if not math.isfinite(self.activation_lower) or self.activation_lower < 0.0:
            raise ValueError("activation_lower must be finite and non-negative.")
        if (
            not math.isfinite(self.activation_upper)
            or self.activation_upper <= self.activation_lower
        ):
            raise ValueError(
                "activation_upper must be finite and greater than activation_lower."
            )
        if not math.isfinite(self.scale_eps) or self.scale_eps <= 0.0:
            raise ValueError("scale_eps must be finite and positive.")


@dataclass(frozen=True)
class MismatchSeamC2BlendConfig:
    """Prediction-only seam detection with a separate compact C2 weight profile."""

    gamma: float = 0.5
    ramp_width: float | None = None
    max_seams_per_axis: int = 2
    peak_relative_threshold: float = 0.25
    profile_smoothing_steps: int = 1
    minimum_separation: float | None = None
    scale_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if self.ramp_width is not None and (
            not math.isfinite(self.ramp_width) or self.ramp_width <= 0.0
        ):
            raise ValueError("ramp_width must be finite and positive when provided.")
        if (
            isinstance(self.max_seams_per_axis, bool)
            or not isinstance(self.max_seams_per_axis, int)
            or self.max_seams_per_axis < 1
        ):
            raise ValueError("max_seams_per_axis must be a positive integer.")
        if (
            not math.isfinite(self.peak_relative_threshold)
            or not 0.0 < self.peak_relative_threshold <= 1.0
        ):
            raise ValueError("peak_relative_threshold must be in (0, 1].")
        if (
            isinstance(self.profile_smoothing_steps, bool)
            or not isinstance(self.profile_smoothing_steps, int)
            or self.profile_smoothing_steps < 0
        ):
            raise ValueError("profile_smoothing_steps must be a non-negative integer.")
        if self.minimum_separation is not None and (
            not math.isfinite(self.minimum_separation) or self.minimum_separation <= 0.0
        ):
            raise ValueError(
                "minimum_separation must be finite and positive when provided."
            )
        if not math.isfinite(self.scale_eps) or self.scale_eps <= 0.0:
            raise ValueError("scale_eps must be finite and positive.")


@dataclass(frozen=True)
class CrossAxisBlendComparisonRequest(FixedSmoothBlendDiagnosticRequest):
    """Compare equal mean, geometry-only, and prediction-only estimators."""

    mismatch: MismatchGradientBlendConfig = MismatchGradientBlendConfig()
    seam_c2: MismatchSeamC2BlendConfig = MismatchSeamC2BlendConfig()
    seam_sweep: bool = False
    seam_sweep_gammas: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5)
    seam_sweep_width_steps: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0)
    seam_sweep_peak_thresholds: tuple[float, ...] = (0.15, 0.2, 0.25, 0.3)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.seam_sweep_gammas or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.seam_sweep_gammas
        ):
            raise ValueError("seam_sweep_gammas must contain finite values in [0, 1].")
        if not self.seam_sweep_width_steps or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.seam_sweep_width_steps
        ):
            raise ValueError(
                "seam_sweep_width_steps must contain finite positive values."
            )
        if not self.seam_sweep_peak_thresholds or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.seam_sweep_peak_thresholds
        ):
            raise ValueError(
                "seam_sweep_peak_thresholds must contain values in (0, 1]."
            )


@dataclass(frozen=True)
class MismatchGradientBlendFields:
    """Sample-dependent fields derived without reference targets."""

    mismatch: torch.Tensor
    mismatch_scale: torch.Tensor
    j_x_raw: torch.Tensor
    j_y_raw: torch.Tensor
    j_x: torch.Tensor
    j_y: torch.Tensor
    sensor_magnitude: torch.Tensor
    activation: torch.Tensor
    anisotropy: torch.Tensor
    theta: torch.Tensor
    w_phi: torch.Tensor
    w_psi: torch.Tensor
    support_mask: torch.Tensor


@dataclass(frozen=True)
class MismatchSeamC2BlendFields:
    """Sample-dependent seam locations and compact C2 blending fields."""

    x_edge_profile_raw: torch.Tensor
    y_edge_profile_raw: torch.Tensor
    x_edge_profile: torch.Tensor
    y_edge_profile: torch.Tensor
    x_edge_midpoints: torch.Tensor
    y_edge_midpoints: torch.Tensor
    x_seam_coordinates: torch.Tensor
    y_seam_coordinates: torch.Tensor
    x_seam_strengths: torch.Tensor
    y_seam_strengths: torch.Tensor
    x_seam_counts: torch.Tensor
    y_seam_counts: torch.Tensor
    distance_x: torch.Tensor
    distance_y: torch.Tensor
    influence_x: torch.Tensor
    influence_y: torch.Tensor
    theta: torch.Tensor
    w_phi: torch.Tensor
    w_psi: torch.Tensor
    support_mask: torch.Tensor
    resolved_ramp_width: float
    resolved_minimum_separation: float


class MismatchGradientBlendMixin:
    """Build a smooth directional sensor from u_phi - u_psi."""

    @staticmethod
    def _compact_c2_smoothstep(values: torch.Tensor) -> torch.Tensor:
        scaled = values.clamp(min=0.0, max=1.0)
        return 10.0 * scaled.pow(3) - 15.0 * scaled.pow(4) + 6.0 * scaled.pow(5)

    @staticmethod
    def _pointwise_normalized_edge_jump(
        values: torch.Tensor,
        *,
        scale: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        edges: torch.Tensor,
        reference_spacing: float,
    ) -> torch.Tensor:
        if values.dim() != 2:
            raise ValueError("values must have shape (B, P).")
        if scale.shape != (values.shape[0], 1):
            raise ValueError("scale must have shape (B, 1).")
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2).")
        if edges.numel() == 0:
            return torch.zeros_like(values)

        spacing = torch.linalg.vector_norm(
            geometry.coords_valid[edges[:, 1]] - geometry.coords_valid[edges[:, 0]],
            dim=1,
        ).clamp_min(torch.finfo(values.dtype).eps)
        edge_jump = (
            (values[:, edges[:, 1]] - values[:, edges[:, 0]]).abs()
            / scale
            * (reference_spacing / spacing).unsqueeze(0)
        )
        point_jump = torch.zeros_like(values)
        expanded_left = edges[:, 0].unsqueeze(0).expand(values.shape[0], -1)
        expanded_right = edges[:, 1].unsqueeze(0).expand(values.shape[0], -1)
        point_jump.scatter_reduce_(
            1,
            expanded_left,
            edge_jump,
            reduce="amax",
            include_self=True,
        )
        point_jump.scatter_reduce_(
            1,
            expanded_right,
            edge_jump,
            reduce="amax",
            include_self=True,
        )
        return point_jump

    @staticmethod
    def _normalized_edge_jump(
        values: torch.Tensor,
        *,
        scale: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        edges: torch.Tensor,
        reference_spacing: float,
    ) -> torch.Tensor:
        if values.dim() != 2:
            raise ValueError("values must have shape (B, P).")
        if scale.shape != (values.shape[0], 1):
            raise ValueError("scale must have shape (B, 1).")
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError("edges must have shape (E, 2).")
        if edges.numel() == 0:
            return values.new_zeros((values.shape[0], 0))
        spacing = torch.linalg.vector_norm(
            geometry.coords_valid[edges[:, 1]] - geometry.coords_valid[edges[:, 0]],
            dim=1,
        ).clamp_min(torch.finfo(values.dtype).eps)
        jump: torch.Tensor = (
            (values[:, edges[:, 1]] - values[:, edges[:, 0]]).abs()
            / scale
            * (reference_spacing / spacing).unsqueeze(0)
        )
        return jump

    @staticmethod
    def _edge_rms_profile(
        edge_jump: torch.Tensor,
        *,
        edge_interval_index: torch.Tensor,
        num_intervals: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_jump.dim() != 2:
            raise ValueError("edge_jump must have shape (B, E).")
        if edge_interval_index.shape != (edge_jump.shape[1],):
            raise ValueError("edge_interval_index must have shape (E,).")
        if num_intervals < 1:
            raise ValueError("num_intervals must be positive.")
        if torch.any(edge_interval_index < 0) or torch.any(
            edge_interval_index >= num_intervals
        ):
            raise ValueError("edge_interval_index is outside the profile range.")

        count = torch.zeros(
            num_intervals,
            dtype=edge_jump.dtype,
            device=edge_jump.device,
        )
        count.index_add_(
            0,
            edge_interval_index,
            torch.ones_like(edge_interval_index, dtype=edge_jump.dtype),
        )
        squared_sum = edge_jump.new_zeros((edge_jump.shape[0], num_intervals))
        expanded_index = edge_interval_index.unsqueeze(0).expand(edge_jump.shape[0], -1)
        squared_sum.scatter_add_(1, expanded_index, edge_jump.square())
        profile = torch.sqrt(squared_sum / count.clamp_min(1.0).unsqueeze(0))
        return profile, count > 0.0

    @staticmethod
    def _smooth_axis_profile(
        profile: torch.Tensor,
        occupied: torch.Tensor,
        *,
        steps: int,
    ) -> torch.Tensor:
        if profile.dim() != 2:
            raise ValueError("profile must have shape (B, N).")
        if occupied.shape != (profile.shape[1],) or occupied.dtype != torch.bool:
            raise ValueError("occupied must be boolean with shape (N,).")
        current = profile.clone()
        if steps == 0 or profile.shape[1] == 1:
            return current
        occupied_float = occupied.to(dtype=profile.dtype)
        for _ in range(steps):
            padded_values = torch.nn.functional.pad(current, (1, 1))
            padded_occupied = torch.nn.functional.pad(occupied_float, (1, 1))
            weighted_sum = (
                0.25 * padded_values[:, :-2]
                + 0.5 * padded_values[:, 1:-1]
                + 0.25 * padded_values[:, 2:]
            )
            weight_sum = (
                0.25 * padded_occupied[:-2]
                + 0.5 * padded_occupied[1:-1]
                + 0.25 * padded_occupied[2:]
            )
            smoothed = weighted_sum / weight_sum.clamp_min(1.0e-12).unsqueeze(0)
            current = torch.where(occupied.unsqueeze(0), smoothed, current)
        return current

    @staticmethod
    def _detect_axis_seams(
        profile: torch.Tensor,
        coordinates: torch.Tensor,
        occupied: torch.Tensor,
        *,
        max_seams: int,
        relative_threshold: float,
        minimum_separation: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if profile.dim() != 2:
            raise ValueError("profile must have shape (B, N).")
        if coordinates.shape != (profile.shape[1],):
            raise ValueError("coordinates must have shape (N,).")
        if occupied.shape != coordinates.shape or occupied.dtype != torch.bool:
            raise ValueError("occupied must be boolean with shape (N,).")

        seam_coordinates = torch.full(
            (profile.shape[0], max_seams),
            torch.nan,
            dtype=profile.dtype,
            device=profile.device,
        )
        seam_strengths = torch.zeros_like(seam_coordinates)
        seam_counts = torch.zeros(
            profile.shape[0],
            dtype=torch.long,
            device=profile.device,
        )
        occupied_indices = torch.nonzero(occupied, as_tuple=False).flatten()
        for sample_index in range(profile.shape[0]):
            if occupied_indices.numel() == 0:
                continue
            sample_profile = profile[sample_index]
            maximum = sample_profile[occupied_indices].max()
            if float(maximum.item()) <= eps:
                continue
            threshold = relative_threshold * maximum
            candidate_indices = occupied_indices[
                sample_profile[occupied_indices] >= threshold
            ]
            order = torch.argsort(
                sample_profile[candidate_indices],
                descending=True,
                stable=True,
            )
            selected: list[int] = []
            for candidate in candidate_indices[order].tolist():
                coordinate = float(coordinates[candidate].item())
                if all(
                    abs(coordinate - float(coordinates[index].item()))
                    > minimum_separation
                    for index in selected
                ):
                    selected.append(candidate)
                if len(selected) == max_seams:
                    break
            for output_index, candidate in enumerate(selected):
                seam_coordinates[sample_index, output_index] = coordinates[candidate]
                seam_strengths[sample_index, output_index] = sample_profile[candidate]
            seam_counts[sample_index] = len(selected)
        return seam_coordinates, seam_strengths, seam_counts

    @staticmethod
    def _compact_influence_from_seams(
        point_coordinates: torch.Tensor,
        seam_coordinates: torch.Tensor,
        *,
        width: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_coordinates.dim() != 1:
            raise ValueError("point_coordinates must be one-dimensional.")
        if seam_coordinates.dim() != 2:
            raise ValueError("seam_coordinates must have shape (B, K).")
        valid = torch.isfinite(seam_coordinates)
        distance = (
            point_coordinates.unsqueeze(0).unsqueeze(2)
            - torch.nan_to_num(seam_coordinates, nan=0.0).unsqueeze(1)
        ).abs()
        distance = torch.where(
            valid.unsqueeze(1),
            distance,
            torch.full_like(distance, torch.inf),
        )
        nearest = distance.amin(dim=2)
        influence = FixedSmoothCrossAxisBlendDiagnostic._compact_c2_bump(
            nearest,
            width=width,
        )
        return nearest, influence

    @classmethod
    def build_mismatch_gradient_fields(
        cls,
        geometry: ComplexGeometryMetadata,
        u_phi: torch.Tensor,
        u_psi: torch.Tensor,
        config: MismatchGradientBlendConfig,
    ) -> MismatchGradientBlendFields:
        if u_phi.shape != u_psi.shape or u_phi.dim() != 2:
            raise ValueError("u_phi and u_psi must share shape (B, P).")
        if u_phi.shape[1] != geometry.num_points:
            raise ValueError("Directional reconstruction point count is invalid.")
        if not torch.all(torch.isfinite(u_phi)) or not torch.all(torch.isfinite(u_psi)):
            raise ValueError("Directional reconstructions must be finite.")

        mismatch = u_phi - u_psi
        mismatch_scale = torch.sqrt(
            torch.mean(mismatch.square(), dim=1, keepdim=True)
        ).clamp_min(config.scale_eps)
        reference_spacing = max(float(geometry.hx.item()), float(geometry.hy.item()))
        j_x_raw = cls._pointwise_normalized_edge_jump(
            mismatch,
            scale=mismatch_scale,
            geometry=geometry,
            edges=geometry.x_edges,
            reference_spacing=reference_spacing,
        )
        j_y_raw = cls._pointwise_normalized_edge_jump(
            mismatch,
            scale=mismatch_scale,
            geometry=geometry,
            edges=geometry.y_edges,
            reference_spacing=reference_spacing,
        )
        all_edges = torch.cat((geometry.x_edges, geometry.y_edges), dim=0)
        j_x = torch.stack(
            [
                FixedSmoothCrossAxisBlendDiagnostic._smooth_on_edges(
                    sample,
                    all_edges,
                    steps=config.smoothing_steps,
                    relaxation=config.smoothing_relaxation,
                )
                for sample in j_x_raw
            ],
            dim=0,
        )
        j_y = torch.stack(
            [
                FixedSmoothCrossAxisBlendDiagnostic._smooth_on_edges(
                    sample,
                    all_edges,
                    steps=config.smoothing_steps,
                    relaxation=config.smoothing_relaxation,
                )
                for sample in j_y_raw
            ],
            dim=0,
        )
        sensor_magnitude = torch.sqrt(j_x.square() + j_y.square())
        normalized_activation = (sensor_magnitude - config.activation_lower) / (
            config.activation_upper - config.activation_lower
        )
        activation = cls._compact_c2_smoothstep(normalized_activation)
        anisotropy = (j_x - j_y) / (j_x + j_y + config.scale_eps)
        theta = config.gamma * activation * anisotropy
        w_phi = 0.5 * (1.0 + theta)
        w_psi = 0.5 * (1.0 - theta)
        support_mask = activation > 0.0

        fields = (
            mismatch,
            mismatch_scale,
            j_x_raw,
            j_y_raw,
            j_x,
            j_y,
            sensor_magnitude,
            activation,
            anisotropy,
            theta,
            w_phi,
            w_psi,
        )
        if not all(torch.all(torch.isfinite(field)) for field in fields):
            raise RuntimeError("Mismatch-gradient blend produced non-finite fields.")
        if torch.any(w_phi < 0.0) or torch.any(w_phi > 1.0):
            raise RuntimeError("Mismatch-gradient w_phi must be in [0, 1].")
        if torch.any(w_psi < 0.0) or torch.any(w_psi > 1.0):
            raise RuntimeError("Mismatch-gradient w_psi must be in [0, 1].")
        if not torch.allclose(
            w_phi + w_psi,
            torch.ones_like(w_phi),
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise RuntimeError("Mismatch-gradient weights must sum to one.")

        return MismatchGradientBlendFields(
            mismatch=mismatch,
            mismatch_scale=mismatch_scale,
            j_x_raw=j_x_raw,
            j_y_raw=j_y_raw,
            j_x=j_x,
            j_y=j_y,
            sensor_magnitude=sensor_magnitude,
            activation=activation,
            anisotropy=anisotropy,
            theta=theta,
            w_phi=w_phi,
            w_psi=w_psi,
            support_mask=support_mask,
        )

    @classmethod
    def build_mismatch_seam_c2_fields(
        cls,
        geometry: ComplexGeometryMetadata,
        mismatch_fields: MismatchGradientBlendFields,
        config: MismatchSeamC2BlendConfig,
    ) -> MismatchSeamC2BlendFields:
        """Detect seam coordinates from mismatch jumps, then build C2 weights."""

        mismatch = mismatch_fields.mismatch
        reference_spacing = max(float(geometry.hx.item()), float(geometry.hy.item()))
        resolved_ramp_width = config.ramp_width or 8.0 * reference_spacing
        resolved_minimum_separation = config.minimum_separation or (
            4.0 * resolved_ramp_width
        )
        num_x_intervals = int(
            round(
                float(
                    (
                        (geometry.y_transverse_max - geometry.y_transverse_min)
                        / geometry.hx
                    ).item()
                )
            )
        )
        num_y_intervals = int(
            round(
                float(
                    (
                        (geometry.x_transverse_max - geometry.x_transverse_min)
                        / geometry.hy
                    ).item()
                )
            )
        )
        if num_x_intervals < 1 or num_y_intervals < 1:
            raise ValueError("Geometry grid must have at least one interval per axis.")

        x_edge_jump = cls._normalized_edge_jump(
            mismatch,
            scale=mismatch_fields.mismatch_scale,
            geometry=geometry,
            edges=geometry.x_edges,
            reference_spacing=reference_spacing,
        )
        y_edge_jump = cls._normalized_edge_jump(
            mismatch,
            scale=mismatch_fields.mismatch_scale,
            geometry=geometry,
            edges=geometry.y_edges,
            reference_spacing=reference_spacing,
        )
        x_edge_interval_index = torch.minimum(
            geometry.valid_grid_x_index[geometry.x_edges[:, 0]],
            geometry.valid_grid_x_index[geometry.x_edges[:, 1]],
        )
        y_edge_interval_index = torch.minimum(
            geometry.valid_grid_y_index[geometry.y_edges[:, 0]],
            geometry.valid_grid_y_index[geometry.y_edges[:, 1]],
        )
        x_edge_profile_raw, x_occupied = cls._edge_rms_profile(
            x_edge_jump,
            edge_interval_index=x_edge_interval_index,
            num_intervals=num_x_intervals,
        )
        y_edge_profile_raw, y_occupied = cls._edge_rms_profile(
            y_edge_jump,
            edge_interval_index=y_edge_interval_index,
            num_intervals=num_y_intervals,
        )
        x_edge_profile = cls._smooth_axis_profile(
            x_edge_profile_raw,
            x_occupied,
            steps=config.profile_smoothing_steps,
        )
        y_edge_profile = cls._smooth_axis_profile(
            y_edge_profile_raw,
            y_occupied,
            steps=config.profile_smoothing_steps,
        )
        x_edge_midpoints = (
            geometry.y_transverse_min
            + (
                torch.arange(
                    num_x_intervals,
                    dtype=geometry.coords_valid.dtype,
                    device=geometry.coords_valid.device,
                )
                + 0.5
            )
            * geometry.hx
        )
        y_edge_midpoints = (
            geometry.x_transverse_min
            + (
                torch.arange(
                    num_y_intervals,
                    dtype=geometry.coords_valid.dtype,
                    device=geometry.coords_valid.device,
                )
                + 0.5
            )
            * geometry.hy
        )
        x_seam_coordinates, x_seam_strengths, x_seam_counts = cls._detect_axis_seams(
            x_edge_profile,
            x_edge_midpoints,
            x_occupied,
            max_seams=config.max_seams_per_axis,
            relative_threshold=config.peak_relative_threshold,
            minimum_separation=resolved_minimum_separation,
            eps=config.scale_eps,
        )
        y_seam_coordinates, y_seam_strengths, y_seam_counts = cls._detect_axis_seams(
            y_edge_profile,
            y_edge_midpoints,
            y_occupied,
            max_seams=config.max_seams_per_axis,
            relative_threshold=config.peak_relative_threshold,
            minimum_separation=resolved_minimum_separation,
            eps=config.scale_eps,
        )
        distance_x, influence_x = cls._compact_influence_from_seams(
            geometry.coords_valid[:, 0],
            x_seam_coordinates,
            width=resolved_ramp_width,
        )
        distance_y, influence_y = cls._compact_influence_from_seams(
            geometry.coords_valid[:, 1],
            y_seam_coordinates,
            width=resolved_ramp_width,
        )
        theta = config.gamma * (influence_x - influence_y)
        w_phi = 0.5 * (1.0 + theta)
        w_psi = 0.5 * (1.0 - theta)
        support_mask = (influence_x > 0.0) | (influence_y > 0.0)

        finite_fields = (
            x_edge_profile_raw,
            y_edge_profile_raw,
            x_edge_profile,
            y_edge_profile,
            x_seam_strengths,
            y_seam_strengths,
            influence_x,
            influence_y,
            theta,
            w_phi,
            w_psi,
        )
        if not all(torch.all(torch.isfinite(field)) for field in finite_fields):
            raise RuntimeError("Mismatch seam C2 blend produced non-finite fields.")
        FixedSmoothCrossAxisBlendDiagnostic._validate_partition_weights(
            w_phi,
            w_psi,
        )

        return MismatchSeamC2BlendFields(
            x_edge_profile_raw=x_edge_profile_raw,
            y_edge_profile_raw=y_edge_profile_raw,
            x_edge_profile=x_edge_profile,
            y_edge_profile=y_edge_profile,
            x_edge_midpoints=x_edge_midpoints,
            y_edge_midpoints=y_edge_midpoints,
            x_seam_coordinates=x_seam_coordinates,
            y_seam_coordinates=y_seam_coordinates,
            x_seam_strengths=x_seam_strengths,
            y_seam_strengths=y_seam_strengths,
            x_seam_counts=x_seam_counts,
            y_seam_counts=y_seam_counts,
            distance_x=distance_x,
            distance_y=distance_y,
            influence_x=influence_x,
            influence_y=influence_y,
            theta=theta,
            w_phi=w_phi,
            w_psi=w_psi,
            support_mask=support_mask,
            resolved_ramp_width=resolved_ramp_width,
            resolved_minimum_separation=resolved_minimum_separation,
        )


class CrossAxisBlendEstimatorComparison(
    MismatchGradientBlendMixin,
    FixedSmoothCrossAxisBlendDiagnostic,
):
    """Run a frozen-checkpoint comparison of four reconstruction estimators."""

    request: CrossAxisBlendComparisonRequest

    def __init__(
        self,
        request: CrossAxisBlendComparisonRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(request, logger=logger)
        self.mismatch_fields: MismatchGradientBlendFields
        self.seam_c2_fields: MismatchSeamC2BlendFields

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError(
                "Cross-axis estimator comparison requires geometry_mode='complex'."
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

        geometry_evaluation = self._evaluate_dataset(dataset, configs, device)
        self.mismatch_fields = self.build_mismatch_gradient_fields(
            self.geometry,
            geometry_evaluation.u_phi,
            geometry_evaluation.u_psi,
            self.request.mismatch,
        )
        mismatch_evaluation = self._evaluation_with_sample_weights(
            geometry_evaluation,
            self.mismatch_fields.w_phi,
            self.mismatch_fields.w_psi,
        )
        self.seam_c2_fields = self.build_mismatch_seam_c2_fields(
            self.geometry,
            self.mismatch_fields,
            self.request.seam_c2,
        )
        seam_c2_evaluation = self._evaluation_with_sample_weights(
            geometry_evaluation,
            self.seam_c2_fields.w_phi,
            self.seam_c2_fields.w_psi,
        )

        geometry_rows = self._sample_metric_rows(geometry_evaluation)
        mismatch_rows = self._sample_metric_rows(
            mismatch_evaluation,
            sample_support_masks=self.mismatch_fields.support_mask,
        )
        seam_c2_rows = self._sample_metric_rows(
            seam_c2_evaluation,
            sample_support_masks=self.seam_c2_fields.support_mask,
        )
        geometry_aggregate = self._aggregate_metrics(geometry_rows)
        mismatch_aggregate = self._aggregate_metrics(mismatch_rows)
        seam_c2_aggregate = self._aggregate_metrics(seam_c2_rows)
        comparison_rows = self._comparison_rows(
            geometry_rows,
            mismatch_rows,
            seam_c2_rows,
            self.mismatch_fields,
            self.seam_c2_fields,
        )
        selected, roles = self._select_samples(geometry_rows)

        metrics_dir = self.request.outdir / "metrics"
        self._write_csv(
            metrics_dir / "per_sample_estimator_comparison.csv",
            comparison_rows,
        )
        if self.request.save_generated_data:
            self._write_comparison_npz(
                geometry_evaluation,
                mismatch_evaluation,
                seam_c2_evaluation,
                selected,
            )

        figure_paths = [
            self._write_geometry_figure(),
            self._write_four_estimator_metric_figure(
                geometry_rows,
                mismatch_rows,
                seam_c2_rows,
            ),
        ]
        figure_paths.extend(
            self._write_estimator_selected_figures(
                geometry_evaluation,
                mismatch_evaluation,
                seam_c2_evaluation,
                selected,
            )
        )
        sweep_rows = (
            self._run_seam_c2_sweep(geometry_evaluation)
            if self.request.seam_sweep
            else []
        )
        if sweep_rows:
            self._write_csv(
                metrics_dir / "seam_c2_parameter_sweep.csv",
                sweep_rows,
            )
        summary = self._build_comparison_summary(
            configs=configs,
            dataset=dataset,
            geometry_path=Path(geometry_path),
            test_path=Path(test_path),
            coefficient_path=(
                None if coefficient_path is None else Path(coefficient_path)
            ),
            device=device,
            geometry_rows=geometry_rows,
            mismatch_rows=mismatch_rows,
            seam_c2_rows=seam_c2_rows,
            geometry_aggregate=geometry_aggregate,
            mismatch_aggregate=mismatch_aggregate,
            seam_c2_aggregate=seam_c2_aggregate,
            sweep_rows=sweep_rows,
            selected=selected,
            roles=roles,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_comparison_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Estimator comparison complete: equal_mean=%.6f "
                "geometry=%.6f mismatch=%.6f seam_c2=%.6f",
                geometry_aggregate["baseline_rel_sol_mean"],
                geometry_aggregate["blend_rel_sol_mean"],
                mismatch_aggregate["blend_rel_sol_mean"],
                seam_c2_aggregate["blend_rel_sol_mean"],
            )
        return summary

    @staticmethod
    def _evaluation_with_sample_weights(
        evaluation: FixedSmoothBlendEvaluation,
        w_phi: torch.Tensor,
        w_psi: torch.Tensor,
    ) -> FixedSmoothBlendEvaluation:
        if (
            w_phi.shape != evaluation.u_phi.shape
            or w_psi.shape != evaluation.u_psi.shape
        ):
            raise ValueError("Sample weights must share shape (B, P).")
        return replace(
            evaluation,
            blend=w_phi * evaluation.u_phi + w_psi * evaluation.u_psi,
        )

    @staticmethod
    def _comparison_rows(
        geometry_rows: list[dict[str, float | int | str]],
        mismatch_rows: list[dict[str, float | int | str]],
        seam_c2_rows: list[dict[str, float | int | str]],
        mismatch_fields: MismatchGradientBlendFields,
        seam_c2_fields: MismatchSeamC2BlendFields,
    ) -> list[dict[str, float | int | str]]:
        mismatch_by_id = {int(row["sample_id"]): row for row in mismatch_rows}
        seam_c2_by_id = {int(row["sample_id"]): row for row in seam_c2_rows}
        rows: list[dict[str, float | int | str]] = []
        for offset, geometry_row in enumerate(geometry_rows):
            sample_id = int(geometry_row["sample_id"])
            mismatch_row = mismatch_by_id[sample_id]
            seam_c2_row = seam_c2_by_id[sample_id]
            x_seams = seam_c2_fields.x_seam_coordinates[offset]
            y_seams = seam_c2_fields.y_seam_coordinates[offset]
            rows.append(
                {
                    "sample_id": sample_id,
                    "file_stem": geometry_row["file_stem"],
                    "equal_mean_rel_sol": geometry_row["baseline_rel_sol"],
                    "geometry_rel_sol": geometry_row["blend_rel_sol"],
                    "mismatch_rel_sol": mismatch_row["blend_rel_sol"],
                    "seam_c2_rel_sol": seam_c2_row["blend_rel_sol"],
                    "geometry_rel_sol_relative_change": geometry_row[
                        "rel_sol_relative_change"
                    ],
                    "mismatch_rel_sol_relative_change": mismatch_row[
                        "rel_sol_relative_change"
                    ],
                    "seam_c2_rel_sol_relative_change": seam_c2_row[
                        "rel_sol_relative_change"
                    ],
                    "equal_mean_transition_error_rms": geometry_row[
                        "baseline_transition_error_rms"
                    ],
                    "geometry_transition_error_rms": geometry_row[
                        "blend_transition_error_rms"
                    ],
                    "mismatch_transition_error_rms": mismatch_row[
                        "blend_transition_error_rms"
                    ],
                    "seam_c2_transition_error_rms": seam_c2_row[
                        "blend_transition_error_rms"
                    ],
                    "equal_mean_transition_trace_error_jump_rms": geometry_row[
                        "baseline_transition_trace_error_jump_rms"
                    ],
                    "geometry_transition_trace_error_jump_rms": geometry_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                    "mismatch_transition_trace_error_jump_rms": mismatch_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                    "seam_c2_transition_trace_error_jump_rms": seam_c2_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                    "geometry_correction_rms": geometry_row["blend_correction_rms"],
                    "mismatch_correction_rms": mismatch_row["blend_correction_rms"],
                    "seam_c2_correction_rms": seam_c2_row["blend_correction_rms"],
                    "mismatch_scale": float(
                        mismatch_fields.mismatch_scale[offset].item()
                    ),
                    "mismatch_sensor_max": float(
                        mismatch_fields.sensor_magnitude[offset].max().item()
                    ),
                    "mismatch_activation_mean": float(
                        mismatch_fields.activation[offset].mean().item()
                    ),
                    "mismatch_support_fraction": float(
                        mismatch_fields.support_mask[offset]
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                    "mismatch_theta_max_abs": float(
                        mismatch_fields.theta[offset].abs().max().item()
                    ),
                    "seam_c2_support_fraction": float(
                        seam_c2_fields.support_mask[offset]
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                    "seam_c2_theta_max_abs": float(
                        seam_c2_fields.theta[offset].abs().max().item()
                    ),
                    "seam_c2_x_count": int(seam_c2_fields.x_seam_counts[offset].item()),
                    "seam_c2_y_count": int(seam_c2_fields.y_seam_counts[offset].item()),
                    "seam_c2_x_coordinates": ";".join(
                        f"{float(value.item()):.10g}"
                        for value in x_seams[torch.isfinite(x_seams)]
                    ),
                    "seam_c2_y_coordinates": ";".join(
                        f"{float(value.item()):.10g}"
                        for value in y_seams[torch.isfinite(y_seams)]
                    ),
                }
            )
        return rows

    def _direct_estimator_comparison(
        self,
        geometry_rows: list[dict[str, float | int | str]],
        mismatch_rows: list[dict[str, float | int | str]],
    ) -> dict[str, Any]:
        mismatch_by_id = {int(row["sample_id"]): row for row in mismatch_rows}
        paired_rows: list[dict[str, float | int | str]] = []
        for geometry_row in geometry_rows:
            sample_id = int(geometry_row["sample_id"])
            mismatch_row = mismatch_by_id[sample_id]
            paired_rows.append(
                {
                    "baseline_rel_sol": geometry_row["blend_rel_sol"],
                    "blend_rel_sol": mismatch_row["blend_rel_sol"],
                    "baseline_transition_error_rms": geometry_row[
                        "blend_transition_error_rms"
                    ],
                    "blend_transition_error_rms": mismatch_row[
                        "blend_transition_error_rms"
                    ],
                    "baseline_transition_trace_error_jump_rms": geometry_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                    "blend_transition_trace_error_jump_rms": mismatch_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                }
            )
        return {
            "baseline": "geometry_only",
            "candidate": "mismatch_gradient",
            "sample_count": len(paired_rows),
            "mismatch_win_count": {
                "rel_sol": sum(
                    float(row["blend_rel_sol"]) < float(row["baseline_rel_sol"])
                    for row in paired_rows
                ),
                "transition_error_rms": sum(
                    float(row["blend_transition_error_rms"])
                    < float(row["baseline_transition_error_rms"])
                    for row in paired_rows
                ),
                "transition_trace_error_jump_rms": sum(
                    float(row["blend_transition_trace_error_jump_rms"])
                    < float(row["baseline_transition_trace_error_jump_rms"])
                    for row in paired_rows
                ),
            },
            "paired_bootstrap": self._paired_bootstrap_summary(paired_rows),
        }

    def _paired_estimator_comparison(
        self,
        baseline_rows: list[dict[str, float | int | str]],
        candidate_rows: list[dict[str, float | int | str]],
        *,
        baseline_name: str,
        candidate_name: str,
    ) -> dict[str, Any]:
        candidate_by_id = {int(row["sample_id"]): row for row in candidate_rows}
        paired_rows: list[dict[str, float | int | str]] = []
        for baseline_row in baseline_rows:
            sample_id = int(baseline_row["sample_id"])
            candidate_row = candidate_by_id[sample_id]
            paired_rows.append(
                {
                    "baseline_rel_sol": baseline_row["blend_rel_sol"],
                    "blend_rel_sol": candidate_row["blend_rel_sol"],
                    "baseline_transition_error_rms": baseline_row[
                        "blend_transition_error_rms"
                    ],
                    "blend_transition_error_rms": candidate_row[
                        "blend_transition_error_rms"
                    ],
                    "baseline_transition_trace_error_jump_rms": baseline_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                    "blend_transition_trace_error_jump_rms": candidate_row[
                        "blend_transition_trace_error_jump_rms"
                    ],
                }
            )
        return {
            "baseline": baseline_name,
            "candidate": candidate_name,
            "sample_count": len(paired_rows),
            "candidate_win_count": {
                "rel_sol": sum(
                    float(row["blend_rel_sol"]) < float(row["baseline_rel_sol"])
                    for row in paired_rows
                ),
                "transition_error_rms": sum(
                    float(row["blend_transition_error_rms"])
                    < float(row["baseline_transition_error_rms"])
                    for row in paired_rows
                ),
                "transition_trace_error_jump_rms": sum(
                    float(row["blend_transition_trace_error_jump_rms"])
                    < float(row["baseline_transition_trace_error_jump_rms"])
                    for row in paired_rows
                ),
            },
            "paired_bootstrap": self._paired_bootstrap_summary(paired_rows),
        }

    def _sensor_statistics(self) -> dict[str, Any]:
        transition = self.blend_fields.transition_point_mask
        regular = ~transition
        all_edges = torch.cat((self.geometry.x_edges, self.geometry.y_edges), dim=0)
        w_phi_jump = (
            self.mismatch_fields.w_phi[:, all_edges[:, 1]]
            - self.mismatch_fields.w_phi[:, all_edges[:, 0]]
        ).abs()
        sensor_energy = self.mismatch_fields.sensor_magnitude.square()
        transition_energy = sensor_energy[:, transition].sum(dim=1)
        total_energy = sensor_energy.sum(dim=1).clamp_min(
            self.request.mismatch.scale_eps
        )
        return {
            "sample_dependent": True,
            "uses_line_lengths": False,
            "uses_transition_coordinates": False,
            "uses_sol": False,
            "uses_flux_targets": False,
            "uses_axial_adjacency": True,
            "mismatch_scale_min": float(
                self.mismatch_fields.mismatch_scale.min().item()
            ),
            "mismatch_scale_mean": float(
                self.mismatch_fields.mismatch_scale.mean().item()
            ),
            "mismatch_scale_max": float(
                self.mismatch_fields.mismatch_scale.max().item()
            ),
            "activation_mean": float(self.mismatch_fields.activation.mean().item()),
            "activation_transition_mean": float(
                self.mismatch_fields.activation[:, transition].mean().item()
            ),
            "activation_regular_mean": float(
                self.mismatch_fields.activation[:, regular].mean().item()
            ),
            "support_fraction_mean": float(
                self.mismatch_fields.support_mask.to(torch.float64).mean().item()
            ),
            "transition_sensor_energy_share_mean": float(
                (transition_energy / total_energy).mean().item()
            ),
            "theta_max_abs": float(self.mismatch_fields.theta.abs().max().item()),
            "w_phi_min": float(self.mismatch_fields.w_phi.min().item()),
            "w_phi_max": float(self.mismatch_fields.w_phi.max().item()),
            "w_psi_min": float(self.mismatch_fields.w_psi.min().item()),
            "w_psi_max": float(self.mismatch_fields.w_psi.max().item()),
            "weight_sum_max_abs_residual": float(
                (self.mismatch_fields.w_phi + self.mismatch_fields.w_psi - 1.0)
                .abs()
                .max()
                .item()
            ),
            "weight_neighbor_jump_max": float(w_phi_jump.max().item()),
        }

    def _seam_c2_statistics(
        self,
        fields: MismatchSeamC2BlendFields | None = None,
    ) -> dict[str, Any]:
        active = self.seam_c2_fields if fields is None else fields
        transition = self.blend_fields.transition_point_mask
        regular = ~transition
        all_edges = torch.cat((self.geometry.x_edges, self.geometry.y_edges), dim=0)
        w_phi_jump = (
            active.w_phi[:, all_edges[:, 1]] - active.w_phi[:, all_edges[:, 0]]
        ).abs()

        expected_x = self.blend_fields.psi_transition_coordinates
        expected_y = self.blend_fields.phi_transition_coordinates
        spacing = max(float(self.geometry.hx.item()), float(self.geometry.hy.item()))

        def coordinate_audit(
            detected: torch.Tensor,
            expected: torch.Tensor,
        ) -> dict[str, float | int | None]:
            errors: list[float] = []
            hit_count = 0
            total_count = detected.shape[0] * expected.numel()
            if expected.numel() == 0:
                return {
                    "expected_count_per_sample": 0,
                    "mean_abs_error": None,
                    "max_abs_error": None,
                    "within_one_grid_spacing_fraction": None,
                }
            for sample_detected in detected:
                finite = sample_detected[torch.isfinite(sample_detected)]
                for expected_coordinate in expected:
                    if finite.numel() == 0:
                        errors.append(float("inf"))
                        continue
                    error = float((finite - expected_coordinate).abs().min().item())
                    errors.append(error)
                    hit_count += error <= spacing + 1.0e-12
            finite_errors = [value for value in errors if math.isfinite(value)]
            return {
                "expected_count_per_sample": int(expected.numel()),
                "mean_abs_error": (
                    None
                    if not finite_errors
                    else float(np.mean(np.asarray(finite_errors)))
                ),
                "max_abs_error": (
                    None if not finite_errors else float(max(finite_errors))
                ),
                "within_one_grid_spacing_fraction": (
                    None if total_count == 0 else hit_count / total_count
                ),
            }

        return {
            "sample_dependent": True,
            "uses_line_lengths": False,
            "uses_transition_coordinates": False,
            "uses_sol": False,
            "uses_flux_targets": False,
            "uses_axial_adjacency": True,
            "detector": "axis_edge_rms_profile_with_physical_nms",
            "weight_profile": "compact_c2_ramp_around_detected_seams",
            "resolved_ramp_width": active.resolved_ramp_width,
            "resolved_ramp_width_steps": active.resolved_ramp_width / spacing,
            "resolved_minimum_separation": active.resolved_minimum_separation,
            "x_seam_count_min": int(active.x_seam_counts.min().item()),
            "x_seam_count_mean": float(
                active.x_seam_counts.to(torch.float64).mean().item()
            ),
            "x_seam_count_max": int(active.x_seam_counts.max().item()),
            "y_seam_count_min": int(active.y_seam_counts.min().item()),
            "y_seam_count_mean": float(
                active.y_seam_counts.to(torch.float64).mean().item()
            ),
            "y_seam_count_max": int(active.y_seam_counts.max().item()),
            "support_fraction_mean": float(
                active.support_mask.to(torch.float64).mean().item()
            ),
            "influence_transition_mean": float(
                torch.maximum(
                    active.influence_x[:, transition],
                    active.influence_y[:, transition],
                )
                .mean()
                .item()
            ),
            "influence_regular_mean": float(
                torch.maximum(
                    active.influence_x[:, regular],
                    active.influence_y[:, regular],
                )
                .mean()
                .item()
            ),
            "theta_max_abs": float(active.theta.abs().max().item()),
            "w_phi_min": float(active.w_phi.min().item()),
            "w_phi_max": float(active.w_phi.max().item()),
            "w_psi_min": float(active.w_psi.min().item()),
            "w_psi_max": float(active.w_psi.max().item()),
            "weight_sum_max_abs_residual": float(
                (active.w_phi + active.w_psi - 1.0).abs().max().item()
            ),
            "weight_neighbor_jump_max": float(w_phi_jump.max().item()),
            "x_detection_audit": coordinate_audit(
                active.x_seam_coordinates,
                expected_x,
            ),
            "y_detection_audit": coordinate_audit(
                active.y_seam_coordinates,
                expected_y,
            ),
        }

    def _run_seam_c2_sweep(
        self,
        geometry_evaluation: FixedSmoothBlendEvaluation,
    ) -> list[dict[str, float | int | str]]:
        spacing = max(float(self.geometry.hx.item()), float(self.geometry.hy.item()))
        rows: list[dict[str, float | int | str]] = []
        for gamma in self.request.seam_sweep_gammas:
            for width_steps in self.request.seam_sweep_width_steps:
                for peak_threshold in self.request.seam_sweep_peak_thresholds:
                    config = replace(
                        self.request.seam_c2,
                        gamma=gamma,
                        ramp_width=width_steps * spacing,
                        peak_relative_threshold=peak_threshold,
                    )
                    fields = self.build_mismatch_seam_c2_fields(
                        self.geometry,
                        self.mismatch_fields,
                        config,
                    )
                    evaluation = self._evaluation_with_sample_weights(
                        geometry_evaluation,
                        fields.w_phi,
                        fields.w_psi,
                    )
                    metric_rows = self._sample_metric_rows(
                        evaluation,
                        sample_support_masks=fields.support_mask,
                    )
                    aggregate = self._aggregate_metrics(metric_rows)
                    statistics = self._seam_c2_statistics(fields)
                    rows.append(
                        {
                            "gamma": gamma,
                            "ramp_width_steps": width_steps,
                            "ramp_width": width_steps * spacing,
                            "peak_relative_threshold": peak_threshold,
                            "rel_sol_mean": aggregate["blend_rel_sol_mean"],
                            "rel_sol_mean_relative_change": aggregate[
                                "rel_sol_mean_relative_change"
                            ],
                            "rel_sol_win_count": aggregate["rel_sol_blend_win_count"],
                            "transition_error_rms_mean": aggregate[
                                "blend_transition_error_rms_mean"
                            ],
                            "transition_error_rms_mean_relative_change": aggregate[
                                "transition_error_rms_mean_relative_change"
                            ],
                            "transition_trace_error_jump_rms_mean": aggregate[
                                "blend_transition_trace_error_jump_rms_mean"
                            ],
                            "transition_trace_error_jump_mean_relative_change": (
                                aggregate[
                                    "transition_trace_error_jump_mean_relative_change"
                                ]
                            ),
                            "weight_neighbor_jump_max": statistics[
                                "weight_neighbor_jump_max"
                            ],
                            "support_fraction_mean": statistics[
                                "support_fraction_mean"
                            ],
                            "x_detection_mean_abs_error": statistics[
                                "x_detection_audit"
                            ]["mean_abs_error"],
                            "y_detection_mean_abs_error": statistics[
                                "y_detection_audit"
                            ]["mean_abs_error"],
                        }
                    )
        rows.sort(key=lambda row: float(row["rel_sol_mean"]))
        return rows

    def _write_four_estimator_metric_figure(
        self,
        geometry_rows: list[dict[str, float | int | str]],
        mismatch_rows: list[dict[str, float | int | str]],
        seam_c2_rows: list[dict[str, float | int | str]],
    ) -> str:
        baseline = np.asarray(
            [100.0 * float(row["baseline_rel_sol"]) for row in geometry_rows]
        )
        geometry = np.asarray(
            [100.0 * float(row["blend_rel_sol"]) for row in geometry_rows]
        )
        mismatch = np.asarray(
            [100.0 * float(row["blend_rel_sol"]) for row in mismatch_rows]
        )
        seam_c2 = np.asarray(
            [100.0 * float(row["blend_rel_sol"]) for row in seam_c2_rows]
        )
        lower = float(
            min(baseline.min(), geometry.min(), mismatch.min(), seam_c2.min())
        )
        upper = float(
            max(baseline.max(), geometry.max(), mismatch.max(), seam_c2.max())
        )
        padding = max(0.05 * (upper - lower), 1.0e-6)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=baseline,
                y=geometry,
                mode="markers",
                name="Geometry-only compact ramp",
                marker={"size": 9, "color": "#1f77b4"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=baseline,
                y=mismatch,
                mode="markers",
                name="Direct mismatch-gradient",
                marker={"size": 9, "color": "#d62728", "symbol": "diamond"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=baseline,
                y=seam_c2,
                mode="markers",
                name="Mismatch-detected C2 seam ramp",
                marker={"size": 9, "color": "#2ca02c", "symbol": "square"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[lower - padding, upper + padding],
                y=[lower - padding, upper + padding],
                mode="lines",
                name="No change",
                line={"color": "#444444", "dash": "dash"},
            )
        )
        figure.update_layout(
            title="Per-sample rel_sol: three post-hoc blends versus equal mean",
            xaxis_title="Equal-mean rel_sol (%)",
            yaxis_title="Blended rel_sol (%)",
            template=self.request.theme,
            width=880,
            height=760,
        )
        base = self.request.outdir / "figures" / "aggregate" / "four_estimator_rel_sol"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _write_estimator_selected_figures(
        self,
        geometry_evaluation: FixedSmoothBlendEvaluation,
        mismatch_evaluation: FixedSmoothBlendEvaluation,
        seam_c2_evaluation: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> list[str]:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(geometry_evaluation.sample_ids.tolist())
        }
        paths: list[str] = []
        for sample_id in selected:
            offset = sample_to_offset[sample_id]
            sol = self._numpy(geometry_evaluation.sol[offset])
            equal_error = self._numpy(geometry_evaluation.baseline[offset]) - sol
            geometry_error = self._numpy(geometry_evaluation.blend[offset]) - sol
            mismatch_error = self._numpy(mismatch_evaluation.blend[offset]) - sol
            seam_c2_error = self._numpy(seam_c2_evaluation.blend[offset]) - sol
            mismatch = self._numpy(self.mismatch_fields.mismatch[offset])
            j_x = self._numpy(self.mismatch_fields.j_x[offset])
            j_y = self._numpy(self.mismatch_fields.j_y[offset])
            activation = self._numpy(self.mismatch_fields.activation[offset])
            direct_theta = self._numpy(self.mismatch_fields.theta[offset])
            influence_x = self._numpy(self.seam_c2_fields.influence_x[offset])
            influence_y = self._numpy(self.seam_c2_fields.influence_y[offset])
            seam_theta = self._numpy(self.seam_c2_fields.theta[offset])
            error_limit = self._signed_limit(
                equal_error,
                geometry_error,
                mismatch_error,
                seam_c2_error,
            )
            mismatch_limit = self._signed_limit(mismatch)
            theta_limit = max(
                float(np.max(np.abs(direct_theta))),
                float(np.max(np.abs(seam_theta))),
                1.0e-12,
            )
            figure = make_subplots(
                rows=3,
                cols=4,
                subplot_titles=(
                    "u_phi - u_psi",
                    "J_x",
                    "J_y",
                    "direct activation",
                    "detected x-seam C2 influence",
                    "detected y-seam C2 influence",
                    "direct theta",
                    "seam C2 theta",
                    "equal mean - sol",
                    "geometry blend - sol",
                    "direct mismatch blend - sol",
                    "seam C2 blend - sol",
                ),
                horizontal_spacing=0.08,
                vertical_spacing=0.1,
            )
            panels = (
                (mismatch, "u_phi-u_psi", "RdBu", -mismatch_limit, mismatch_limit),
                (j_x, "J_x", "Viridis", 0.0, None),
                (j_y, "J_y", "Viridis", 0.0, None),
                (activation, "activation", "Viridis", 0.0, 1.0),
                (influence_x, "x influence", "Viridis", 0.0, 1.0),
                (influence_y, "y influence", "Viridis", 0.0, 1.0),
                (
                    direct_theta,
                    "direct mismatch theta",
                    "RdBu",
                    -theta_limit,
                    theta_limit,
                ),
                (
                    seam_theta,
                    "seam C2 theta",
                    "RdBu",
                    -theta_limit,
                    theta_limit,
                ),
                (
                    equal_error,
                    "equal mean error",
                    "RdBu",
                    -error_limit,
                    error_limit,
                ),
                (
                    geometry_error,
                    "geometry blend error",
                    "RdBu",
                    -error_limit,
                    error_limit,
                ),
                (
                    mismatch_error,
                    "direct mismatch blend error",
                    "RdBu",
                    -error_limit,
                    error_limit,
                ),
                (
                    seam_c2_error,
                    "seam C2 blend error",
                    "RdBu",
                    -error_limit,
                    error_limit,
                ),
            )
            for panel_index, (values, title, scale, cmin, cmax) in enumerate(panels):
                self._add_scatter(
                    figure,
                    row=panel_index // 4 + 1,
                    col=panel_index % 4 + 1,
                    values=values,
                    title=title,
                    colorscale=scale,
                    cmin=cmin,
                    cmax=cmax,
                    subplot_columns=4,
                    colorbar_column=4,
                    colorbar_y=(3.0 - (panel_index // 4 + 1) + 0.5) / 3.0,
                    colorbar_length=0.24,
                )
            figure.update_xaxes(scaleanchor="y", scaleratio=1)
            figure.update_layout(
                title=f"Sample {sample_id}: cross-axis estimator comparison",
                template=self.request.theme,
                width=1900,
                height=1320,
            )
            base = (
                self.request.outdir
                / "figures"
                / "selected"
                / f"sample_{sample_id:04d}_four_estimator_comparison"
            )
            save_plotly_figure(figure, base, logger=self.logger)
            paths.append(
                str(base.with_suffix(".html").relative_to(self.request.outdir))
            )
        return paths

    def _write_comparison_npz(
        self,
        geometry_evaluation: FixedSmoothBlendEvaluation,
        mismatch_evaluation: FixedSmoothBlendEvaluation,
        seam_c2_evaluation: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> None:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(geometry_evaluation.sample_ids.tolist())
        }
        offsets = [sample_to_offset[sample_id] for sample_id in selected]
        payload = {
            "selected_sample_ids": np.asarray(selected, dtype=np.int64),
            "selected_file_stems": np.asarray(
                [geometry_evaluation.file_stems[offset] for offset in offsets]
            ),
            "coords_valid": self._numpy(self.geometry.coords_valid),
            "geometry_theta": self._numpy(self.blend_fields.theta),
            "geometry_w_phi": self._numpy(self.blend_fields.w_phi),
            "geometry_w_psi": self._numpy(self.blend_fields.w_psi),
            "transition_point_mask": self._numpy(
                self.blend_fields.transition_point_mask
            ),
            "sol": self._numpy(geometry_evaluation.sol[offsets]),
            "u_phi": self._numpy(geometry_evaluation.u_phi[offsets]),
            "u_psi": self._numpy(geometry_evaluation.u_psi[offsets]),
            "u_equal_mean": self._numpy(geometry_evaluation.baseline[offsets]),
            "u_geometry_blend": self._numpy(geometry_evaluation.blend[offsets]),
            "u_mismatch_blend": self._numpy(mismatch_evaluation.blend[offsets]),
            "u_seam_c2_blend": self._numpy(seam_c2_evaluation.blend[offsets]),
            "mismatch": self._numpy(self.mismatch_fields.mismatch[offsets]),
            "mismatch_scale": self._numpy(self.mismatch_fields.mismatch_scale[offsets]),
            "mismatch_j_x_raw": self._numpy(self.mismatch_fields.j_x_raw[offsets]),
            "mismatch_j_y_raw": self._numpy(self.mismatch_fields.j_y_raw[offsets]),
            "mismatch_j_x": self._numpy(self.mismatch_fields.j_x[offsets]),
            "mismatch_j_y": self._numpy(self.mismatch_fields.j_y[offsets]),
            "mismatch_sensor_magnitude": self._numpy(
                self.mismatch_fields.sensor_magnitude[offsets]
            ),
            "mismatch_activation": self._numpy(
                self.mismatch_fields.activation[offsets]
            ),
            "mismatch_anisotropy": self._numpy(
                self.mismatch_fields.anisotropy[offsets]
            ),
            "mismatch_theta": self._numpy(self.mismatch_fields.theta[offsets]),
            "mismatch_w_phi": self._numpy(self.mismatch_fields.w_phi[offsets]),
            "mismatch_w_psi": self._numpy(self.mismatch_fields.w_psi[offsets]),
            "mismatch_support_mask": self._numpy(
                self.mismatch_fields.support_mask[offsets]
            ),
            "seam_c2_x_edge_profile_raw": self._numpy(
                self.seam_c2_fields.x_edge_profile_raw[offsets]
            ),
            "seam_c2_y_edge_profile_raw": self._numpy(
                self.seam_c2_fields.y_edge_profile_raw[offsets]
            ),
            "seam_c2_x_edge_profile": self._numpy(
                self.seam_c2_fields.x_edge_profile[offsets]
            ),
            "seam_c2_y_edge_profile": self._numpy(
                self.seam_c2_fields.y_edge_profile[offsets]
            ),
            "seam_c2_x_edge_midpoints": self._numpy(
                self.seam_c2_fields.x_edge_midpoints
            ),
            "seam_c2_y_edge_midpoints": self._numpy(
                self.seam_c2_fields.y_edge_midpoints
            ),
            "seam_c2_x_seam_coordinates": self._numpy(
                self.seam_c2_fields.x_seam_coordinates[offsets]
            ),
            "seam_c2_y_seam_coordinates": self._numpy(
                self.seam_c2_fields.y_seam_coordinates[offsets]
            ),
            "seam_c2_x_seam_strengths": self._numpy(
                self.seam_c2_fields.x_seam_strengths[offsets]
            ),
            "seam_c2_y_seam_strengths": self._numpy(
                self.seam_c2_fields.y_seam_strengths[offsets]
            ),
            "seam_c2_x_seam_counts": self._numpy(
                self.seam_c2_fields.x_seam_counts[offsets]
            ),
            "seam_c2_y_seam_counts": self._numpy(
                self.seam_c2_fields.y_seam_counts[offsets]
            ),
            "seam_c2_distance_x": self._numpy(self.seam_c2_fields.distance_x[offsets]),
            "seam_c2_distance_y": self._numpy(self.seam_c2_fields.distance_y[offsets]),
            "seam_c2_influence_x": self._numpy(
                self.seam_c2_fields.influence_x[offsets]
            ),
            "seam_c2_influence_y": self._numpy(
                self.seam_c2_fields.influence_y[offsets]
            ),
            "seam_c2_theta": self._numpy(self.seam_c2_fields.theta[offsets]),
            "seam_c2_w_phi": self._numpy(self.seam_c2_fields.w_phi[offsets]),
            "seam_c2_w_psi": self._numpy(self.seam_c2_fields.w_psi[offsets]),
            "seam_c2_support_mask": self._numpy(
                self.seam_c2_fields.support_mask[offsets]
            ),
        }
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "selected_cross_axis_blend_comparison_arrays.npz",
            **payload,  # type: ignore[arg-type]
        )

    def _build_comparison_summary(
        self,
        *,
        configs: CouplingArtifactConfigs,
        dataset: ComplexCouplingDataset,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path | None,
        device: torch.device,
        geometry_rows: list[dict[str, float | int | str]],
        mismatch_rows: list[dict[str, float | int | str]],
        seam_c2_rows: list[dict[str, float | int | str]],
        geometry_aggregate: dict[str, float | int | str],
        mismatch_aggregate: dict[str, float | int | str],
        seam_c2_aggregate: dict[str, float | int | str],
        sweep_rows: list[dict[str, float | int | str]],
        selected: tuple[int, ...],
        roles: dict[str, int],
        figure_paths: list[str],
    ) -> dict[str, Any]:
        return {
            "diagnostic": "cross_axis_reconstruction_blend_estimator_comparison",
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
            "estimators": {
                "equal_mean": {
                    "formula": "0.5*(u_phi+u_psi)",
                    "sample_dependent": True,
                },
                "geometry_only": {
                    "formula": "w_phi*u_phi+w_psi*u_psi",
                    "sample_dependent": False,
                    "uses_sol": False,
                    "method": self.request.blend.weight_construction,
                    "config": asdict(self.request.blend),
                },
                "mismatch_gradient": {
                    "formula": (
                        "theta=gamma*activation*(Jx-Jy)/(Jx+Jy+eps); "
                        "w_phi=(1+theta)/2; w_psi=1-w_phi"
                    ),
                    "sample_dependent": True,
                    "uses_sol": False,
                    "uses_flux_targets": False,
                    "uses_line_lengths": False,
                    "uses_transition_coordinates": False,
                    "uses_axial_adjacency": True,
                    "config": asdict(self.request.mismatch),
                },
                "mismatch_detected_seam_c2": {
                    "formula": (
                        "detect x/y seam coordinates from mismatch edge-RMS "
                        "profiles; theta=gamma*(B_x-B_y); "
                        "w_phi=(1+theta)/2; w_psi=1-w_phi"
                    ),
                    "sample_dependent": True,
                    "uses_sol": False,
                    "uses_flux_targets": False,
                    "uses_line_lengths": False,
                    "uses_transition_coordinates": False,
                    "uses_axial_adjacency": True,
                    "detector_and_weight_profile_separated": True,
                    "config": asdict(self.request.seam_c2),
                    "resolved_ramp_width": (self.seam_c2_fields.resolved_ramp_width),
                    "resolved_minimum_separation": (
                        self.seam_c2_fields.resolved_minimum_separation
                    ),
                },
            },
            "metric_role": "evaluation_only_full_reference_test",
            "transition_zone_role": (
                "geometry-derived audit region; not used by mismatch estimator"
            ),
            "aggregate_metrics": {
                "geometry_only": geometry_aggregate,
                "mismatch_gradient": mismatch_aggregate,
                "mismatch_detected_seam_c2": seam_c2_aggregate,
            },
            "paired_bootstrap": {
                "geometry_only": self._paired_bootstrap_summary(geometry_rows),
                "mismatch_gradient": self._paired_bootstrap_summary(mismatch_rows),
                "mismatch_detected_seam_c2": self._paired_bootstrap_summary(
                    seam_c2_rows
                ),
            },
            "geometry_vs_mismatch": self._direct_estimator_comparison(
                geometry_rows,
                mismatch_rows,
            ),
            "seam_c2_vs_mismatch": self._paired_estimator_comparison(
                mismatch_rows,
                seam_c2_rows,
                baseline_name="mismatch_gradient",
                candidate_name="mismatch_detected_seam_c2",
            ),
            "seam_c2_vs_geometry": self._paired_estimator_comparison(
                geometry_rows,
                seam_c2_rows,
                baseline_name="geometry_only",
                candidate_name="mismatch_detected_seam_c2",
            ),
            "geometry_statistics": self._geometry_statistics(),
            "mismatch_sensor_statistics": self._sensor_statistics(),
            "mismatch_detected_seam_c2_statistics": self._seam_c2_statistics(),
            "seam_c2_parameter_sweep": {
                "enabled": self.request.seam_sweep,
                "exploratory_test_target_sensitivity_only": True,
                "row_count": len(sweep_rows),
                "csv": ("metrics/seam_c2_parameter_sweep.csv" if sweep_rows else None),
                "best_rel_sol": (sweep_rows[0] if sweep_rows else None),
                "best_transition_error_rms": (
                    min(
                        sweep_rows,
                        key=lambda row: float(row["transition_error_rms_mean"]),
                    )
                    if sweep_rows
                    else None
                ),
                "best_transition_trace_error_jump": (
                    min(
                        sweep_rows,
                        key=lambda row: float(
                            row["transition_trace_error_jump_rms_mean"]
                        ),
                    )
                    if sweep_rows
                    else None
                ),
            },
            "per_sample_csv": "metrics/per_sample_estimator_comparison.csv",
            "raw_archive": (
                "data/selected_cross_axis_blend_comparison_arrays.npz"
                if self.request.save_generated_data
                else None
            ),
            "figure_count": len(figure_paths),
            "figure_paths": figure_paths,
        }

    def _write_comparison_report(self, summary: dict[str, Any]) -> None:
        geometry = summary["aggregate_metrics"]["geometry_only"]
        mismatch = summary["aggregate_metrics"]["mismatch_gradient"]
        seam_c2 = summary["aggregate_metrics"]["mismatch_detected_seam_c2"]
        sensor = summary["mismatch_sensor_statistics"]
        seam_sensor = summary["mismatch_detected_seam_c2_statistics"]
        seam_vs_mismatch = summary["seam_c2_vs_mismatch"]
        seam_vs_geometry = summary["seam_c2_vs_geometry"]
        seam_vs_mismatch_metrics = seam_vs_mismatch["paired_bootstrap"]["metrics"]
        seam_vs_geometry_metrics = seam_vs_geometry["paired_bootstrap"]["metrics"]
        baseline = float(geometry["baseline_rel_sol_mean"])
        sweep = summary["seam_c2_parameter_sweep"]

        def optional_percentage(value: float | None) -> str:
            return "n/a" if value is None else f"{100.0 * value:.3f}%"

        x_detection_rate = optional_percentage(
            seam_sensor["x_detection_audit"]["within_one_grid_spacing_fraction"]
        )
        y_detection_rate = optional_percentage(
            seam_sensor["y_detection_audit"]["within_one_grid_spacing_fraction"]
        )
        sweep_section = ""
        if sweep["enabled"]:
            best_rel = sweep["best_rel_sol"]
            best_transition = sweep["best_transition_error_rms"]
            best_trace = sweep["best_transition_trace_error_jump"]
            sweep_section = f"""
## Exploratory Parameter Sweep

The sweep is a test-target sensitivity analysis, not independent model
selection. It evaluates `{sweep["row_count"]}` combinations without rerunning
the checkpoint.

- best mean `rel_sol`: gamma `{best_rel["gamma"]}`, width
  `{best_rel["ramp_width_steps"]}h`, peak threshold
  `{best_rel["peak_relative_threshold"]}`, mean
  `{100.0 * float(best_rel["rel_sol_mean"]):.6f}%`
- best transition RMS: gamma `{best_transition["gamma"]}`, width
  `{best_transition["ramp_width_steps"]}h`, peak threshold
  `{best_transition["peak_relative_threshold"]}`
- best trace jump: gamma `{best_trace["gamma"]}`, width
  `{best_trace["ramp_width_steps"]}h`, peak threshold
  `{best_trace["peak_relative_threshold"]}`
"""

        report = f"""# Cross-Axis Reconstruction Blend Estimator Comparison

## Scope

This is a frozen-checkpoint post-hoc comparison. It does not change CouplingNet,
GreenNet, projection, directional reconstruction, training, or checkpoint
weights. Test `sol` is used only for evaluation metrics.

## Full Test Comparison

| Estimator | Mean rel_sol | Change vs equal mean | rel_sol wins | Transition RMS change | Trace-jump change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal mean | {100.0 * baseline:.6f}% | 0.000% | - | 0.000% | 0.000% |
| Geometry-only | {100.0 * float(geometry["blend_rel_sol_mean"]):.6f}% | {100.0 * float(geometry["rel_sol_mean_relative_change"]):+.3f}% | {geometry["rel_sol_blend_win_count"]}/{geometry["sample_count"]} | {100.0 * float(geometry["transition_error_rms_mean_relative_change"]):+.3f}% | {100.0 * float(geometry["transition_trace_error_jump_mean_relative_change"]):+.3f}% |
| Direct mismatch-gradient | {100.0 * float(mismatch["blend_rel_sol_mean"]):.6f}% | {100.0 * float(mismatch["rel_sol_mean_relative_change"]):+.3f}% | {mismatch["rel_sol_blend_win_count"]}/{mismatch["sample_count"]} | {100.0 * float(mismatch["transition_error_rms_mean_relative_change"]):+.3f}% | {100.0 * float(mismatch["transition_trace_error_jump_mean_relative_change"]):+.3f}% |
| Mismatch-detected seam C2 | {100.0 * float(seam_c2["blend_rel_sol_mean"]):.6f}% | {100.0 * float(seam_c2["rel_sol_mean_relative_change"]):+.3f}% | {seam_c2["rel_sol_blend_win_count"]}/{seam_c2["sample_count"]} | {100.0 * float(seam_c2["transition_error_rms_mean_relative_change"]):+.3f}% | {100.0 * float(seam_c2["transition_trace_error_jump_mean_relative_change"]):+.3f}% |

The geometry-only estimator uses the fixed compact topology-distance ramp. The
two mismatch estimators use only `u_phi-u_psi`, axial adjacency, and grid
spacing. The direct estimator maps the pointwise sensor directly to weights.
The seam C2 estimator instead compresses x/y edge jumps into axis profiles,
detects separated peaks, and places a wider compact C2 profile around those
detected coordinates. Neither reads line lengths, geometry transition
coordinates, `sol`, or flux targets.

## Direct Mismatch Sensor Audit

- mean support fraction: `{100.0 * float(sensor["support_fraction_mean"]):.3f}%`
- transition sensor-energy share: `{100.0 * float(sensor["transition_sensor_energy_share_mean"]):.3f}%`
- mean transition activation: `{float(sensor["activation_transition_mean"]):.6f}`
- mean regular activation: `{float(sensor["activation_regular_mean"]):.6f}`
- weight range: `[{float(sensor["w_phi_min"]):.6f}, {float(sensor["w_phi_max"]):.6f}]`
- maximum neighboring weight jump: `{float(sensor["weight_neighbor_jump_max"]):.6f}`
- partition residual: `{float(sensor["weight_sum_max_abs_residual"]):.6e}`

## Detected-Seam C2 Audit

- resolved width: `{float(seam_sensor["resolved_ramp_width"]):.8f}`
  (`{float(seam_sensor["resolved_ramp_width_steps"]):.3f}h`)
- mean x/y detected seam counts:
  `{float(seam_sensor["x_seam_count_mean"]):.3f}` /
  `{float(seam_sensor["y_seam_count_mean"]):.3f}`
- mean support fraction:
  `{100.0 * float(seam_sensor["support_fraction_mean"]):.3f}%`
- maximum neighboring weight jump:
  `{float(seam_sensor["weight_neighbor_jump_max"]):.6f}`
- x/y one-grid-spacing detection rates:
  `{x_detection_rate}` / `{y_detection_rate}`
- partition residual:
  `{float(seam_sensor["weight_sum_max_abs_residual"]):.6e}`

## Interpretation Boundary

The mismatch field can identify where the two axial reconstructions lose
compatibility, but it is not a reference error. The directional rule is
architecture-based: an x-directed mismatch jump favors `u_phi`, while a
y-directed jump favors `u_psi`. Common-mode errors with
`u_phi approximately equal to u_psi` remain invisible.

The selected activation thresholds were fixed from prediction-only scale
inspection. Any further threshold or gamma sweep on this test set must be
reported as exploratory rather than independent validation.

## Detected-Seam C2 Versus Direct Mismatch

| Metric | Seam C2 relative change | Paired bootstrap 95% CI | Seam C2 wins |
| --- | ---: | ---: | ---: |
| rel_sol | {100.0 * float(seam_vs_mismatch_metrics["rel_sol"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_mismatch_metrics["rel_sol"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_mismatch_metrics["rel_sol"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_mismatch["candidate_win_count"]["rel_sol"]}/{seam_vs_mismatch["sample_count"]} |
| Transition RMS | {100.0 * float(seam_vs_mismatch_metrics["transition_error_rms"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_mismatch_metrics["transition_error_rms"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_mismatch_metrics["transition_error_rms"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_mismatch["candidate_win_count"]["transition_error_rms"]}/{seam_vs_mismatch["sample_count"]} |
| Trace jump | {100.0 * float(seam_vs_mismatch_metrics["transition_trace_error_jump_rms"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_mismatch_metrics["transition_trace_error_jump_rms"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_mismatch_metrics["transition_trace_error_jump_rms"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_mismatch["candidate_win_count"]["transition_trace_error_jump_rms"]}/{seam_vs_mismatch["sample_count"]} |

## Detected-Seam C2 Versus Geometry Ramp

| Metric | Seam C2 relative change | Paired bootstrap 95% CI | Seam C2 wins |
| --- | ---: | ---: | ---: |
| rel_sol | {100.0 * float(seam_vs_geometry_metrics["rel_sol"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_geometry_metrics["rel_sol"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_geometry_metrics["rel_sol"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_geometry["candidate_win_count"]["rel_sol"]}/{seam_vs_geometry["sample_count"]} |
| Transition RMS | {100.0 * float(seam_vs_geometry_metrics["transition_error_rms"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_geometry_metrics["transition_error_rms"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_geometry_metrics["transition_error_rms"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_geometry["candidate_win_count"]["transition_error_rms"]}/{seam_vs_geometry["sample_count"]} |
| Trace jump | {100.0 * float(seam_vs_geometry_metrics["transition_trace_error_jump_rms"]["observed_relative_change"]):+.3f}% | [{100.0 * float(seam_vs_geometry_metrics["transition_trace_error_jump_rms"]["relative_change_ci95"][0]):+.3f}%, {100.0 * float(seam_vs_geometry_metrics["transition_trace_error_jump_rms"]["relative_change_ci95"][1]):+.3f}%] | {seam_vs_geometry["candidate_win_count"]["transition_trace_error_jump_rms"]}/{seam_vs_geometry["sample_count"]} |
{sweep_section}"""
        (self.request.outdir / "diagnosis_report.md").write_text(report)


def run_cross_axis_blend_estimator_comparison(
    request: CrossAxisBlendComparisonRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the four-estimator frozen-checkpoint comparison."""

    return CrossAxisBlendEstimatorComparison(request, logger=logger).run()
