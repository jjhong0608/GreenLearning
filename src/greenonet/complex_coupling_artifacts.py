from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
)
from greenonet.complex_tangent_projection import SymmetricTangentGreenResponseContext
from greenonet.complex_visualization_mesh import (
    ComplexVisualizationMesh,
    load_complex_visualization_mesh,
)
from greenonet.complex_pre_projection_fusion import (
    FINAL_LAYER_INITIALIZATION,
    FUSION_ARCHITECTURE,
    pre_projection_fusion_formula,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.coupling_optimizer import ComplexCouplingOptimizerFactory
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ColumnDiagonalGreenResponseProjectionConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexPreProjectionFusionConfig,
    ComplexPostLineSearchStationarityConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexResponseTrustConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingBestEnergyCheckpointConfig,
    CouplingBestPhysicsCheckpointConfig,
    CouplingCoefficientTermsConfig,
    CouplingTrainingConfig,
    GeometryKSelectionConfig,
    SymmetricTangentGreenResponseProjectionConfig,
)
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.plotly_io import save_plotly_figure
from greenonet.reproducibility import TrainingSeedContext


@dataclass(frozen=True)
class ComplexSelectedSample:
    sample_id: int
    file_stem: str
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ComplexArtifactColorRange:
    """Plotly range plus provenance for one shared scalar-field group."""

    group: str
    policy: str
    quantile: float
    cmin: float | None
    cmax: float | None

    def plotly_kwargs(self) -> dict[str, float]:
        if self.cmin is None or self.cmax is None:
            return {}
        return {"cmin": self.cmin, "cmax": self.cmax}

    def field_summary(self, values: np.ndarray) -> dict[str, float | int | str]:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Artifact color-range fields must contain finite values.")
        full_min = float(np.min(finite))
        full_max = float(np.max(finite))
        display_min = full_min if self.cmin is None else self.cmin
        display_max = full_max if self.cmax is None else self.cmax
        saturated = (finite < display_min) | (finite > display_max)
        saturated_count = int(np.count_nonzero(saturated))
        return {
            "group": self.group,
            "policy": self.policy,
            "quantile": self.quantile,
            "full_min": full_min,
            "full_max": full_max,
            "display_cmin": display_min,
            "display_cmax": display_max,
            "finite_point_count": int(finite.size),
            "saturated_point_count": saturated_count,
            "saturated_point_fraction": saturated_count / int(finite.size),
        }


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


@dataclass(frozen=True)
class ComplexCoefficientMeshFields:
    """Physical coefficients evaluated directly at visualization-mesh vertices."""

    coords: np.ndarray
    a: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    b_magnitude: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class ComplexDomainBoundaryOverlay:
    """Geometry-only boundary markers shared by every complex artifact figure."""

    coords: np.ndarray
    enabled: bool
    marker_color: str
    legend_bgcolor: str
    marker_size: float = 5.0

    @classmethod
    def from_endpoint_coords(
        cls,
        endpoint_coords: np.ndarray,
        *,
        enabled: bool,
        theme: str,
    ) -> ComplexDomainBoundaryOverlay:
        coords = np.asarray(endpoint_coords)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                "Domain boundary endpoint coordinates must have shape (N, 2)."
            )
        if coords.shape[0] == 0:
            raise ValueError("Domain boundary endpoint coordinates cannot be empty.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("Domain boundary endpoint coordinates must be finite.")
        unique_coords = np.unique(coords, axis=0)
        unique_coords.setflags(write=False)
        dark_theme = "dark" in theme.lower()
        return cls(
            coords=unique_coords,
            enabled=enabled,
            marker_color="#ECEFF1" if dark_theme else "#263238",
            legend_bgcolor=(
                "rgba(0, 0, 0, 0.55)" if dark_theme else "rgba(255, 255, 255, 0.72)"
            ),
        )

    @property
    def point_count(self) -> int:
        return int(self.coords.shape[0])

    def add_to_figure(self, figure: go.Figure) -> None:
        if not self.enabled:
            return
        for trace in figure.data:
            trace.update(showlegend=False)
        figure.add_trace(
            go.Scatter(
                x=self.coords[:, 0],
                y=self.coords[:, 1],
                mode="markers",
                marker={
                    "symbol": "circle-open",
                    "size": self.marker_size,
                    "color": self.marker_color,
                    "line": {"color": self.marker_color, "width": 1.2},
                },
                name="Domain boundary",
                showlegend=True,
                hovertemplate=(
                    "Domain boundary<br>x=%{x:.6g}<br>y=%{y:.6g}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            legend={
                "orientation": "h",
                "x": 0.01,
                "xanchor": "left",
                "y": 0.01,
                "yanchor": "bottom",
                "bgcolor": self.legend_bgcolor,
            }
        )

    def summary(self) -> dict[str, bool | int | str]:
        return {
            "enabled": self.enabled,
            "representation": "open_markers",
            "coordinate_source": ("canonical_boundary_energy_context.endpoint_coords"),
            "point_count": self.point_count,
            "scalar_values_included": False,
            "included_in_metrics": False,
        }


class ComplexMeshFigureLayoutMixin:
    """Shared top-down Plotly scene framing for scalar mesh artifacts."""

    MESH_SCENE_SCALE: ClassVar[float] = 1.5
    MESH_Z_ASPECT_RATIO: ClassVar[float] = 0.01

    @classmethod
    def _mesh_layout(
        cls,
        *,
        title: str,
        theme: str,
        vertices: np.ndarray,
    ) -> go.Layout:
        span = np.ptp(vertices, axis=0)
        max_span = max(float(np.max(span)), np.finfo(np.float64).eps)
        return go.Layout(
            template=theme,
            width=900,
            height=800,
            title=title,
            margin={"l": 10, "r": 70, "t": 65, "b": 10},
            scene={
                "xaxis": {"title": "x"},
                "yaxis": {"title": "y"},
                "zaxis": {"visible": False},
                "aspectmode": "manual",
                "aspectratio": {
                    "x": cls.MESH_SCENE_SCALE * max(float(span[0]) / max_span, 1.0e-3),
                    "y": cls.MESH_SCENE_SCALE * max(float(span[1]) / max_span, 1.0e-3),
                    "z": cls.MESH_Z_ASPECT_RATIO,
                },
                "camera": {
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "eye": {"x": 0.0, "y": 0.0, "z": 2.5},
                    "up": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "projection": {"type": "orthographic"},
                },
            },
        )


class ComplexScalarMeshArtifactMixin(ComplexMeshFigureLayoutMixin):
    """Add scalar mesh figures without changing valid-point diagnostics."""

    SOLUTION_MESH_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"sol", "u_pred", "u_pred_error"}
    )
    INTERIOR_SCALAR_MESH_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "rhs",
            "phi",
            "psi",
            "target_phi",
            "target_psi",
            "phi_error",
            "psi_error",
        }
    )
    MESH_FIGURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "sol",
        "u_pred",
        "u_pred_error",
        "rhs",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
    )
    FIGURE_TITLES: ClassVar[dict[str, str]]
    SIGNED_FIGURE_FIELDS: ClassVar[frozenset[str]]

    request: CouplingArtifactRequest
    logger: logging.Logger | None

    def _load_visualization_mesh(
        self,
        *,
        geometry_path: Any,
        geometry: ComplexGeometryMetadata,
    ) -> ComplexVisualizationMesh | None:
        if self.request.visualization_mesh is None:
            return None
        return load_complex_visualization_mesh(
            self.request.visualization_mesh,
            geometry_path=geometry_path,
            coords_valid=geometry.coords_valid.detach().cpu().numpy(),
        )

    def _write_scalar_mesh_figures(
        self,
        selected_samples: list[ComplexSelectedSample],
        visualization_mesh: ComplexVisualizationMesh | None,
        theme: str,
        boundary_overlay: ComplexDomainBoundaryOverlay,
        color_ranges_by_sample: dict[
            int,
            dict[str, ComplexArtifactColorRange],
        ],
    ) -> tuple[list[str], tuple[str, ...]]:
        if visualization_mesh is None:
            return [], ()
        paths: list[str] = []
        exported_fields: list[str] = []
        for sample in selected_samples:
            stem = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            color_ranges = color_ranges_by_sample[sample.sample_id]
            for field in self.MESH_FIGURE_FIELDS:
                if field not in sample.arrays:
                    continue
                figure = self._scalar_mesh_figure(
                    title=f"{stem} {self.FIGURE_TITLES[field]} mesh",
                    field=field,
                    visualization_mesh=visualization_mesh,
                    valid_values=sample.arrays[field],
                    theme=theme,
                    signed=field in self.SIGNED_FIGURE_FIELDS,
                    color_range=color_ranges.get(field),
                    boundary_overlay=boundary_overlay,
                )
                base_path = (
                    self.request.outdir
                    / "figures"
                    / "mesh"
                    / field
                    / f"{stem}_{field}_mesh"
                )
                save_plotly_figure(figure, base_path, logger=self.logger)
                paths.append(str(base_path.with_suffix(".json")))
                if field not in exported_fields:
                    exported_fields.append(field)
        return paths, tuple(exported_fields)

    def _copy_visualization_mesh(
        self,
        visualization_mesh: ComplexVisualizationMesh | None,
    ) -> str | None:
        if (
            visualization_mesh is None
            or not self.request.save_generated_data
            or self.request.visualization_mesh is None
        ):
            return None
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        destination = data_dir / "visualization_mesh.npz"
        source = self.request.visualization_mesh
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return "data/visualization_mesh.npz"

    @classmethod
    def _scalar_mesh_figure(
        cls,
        *,
        title: str,
        field: str,
        visualization_mesh: ComplexVisualizationMesh,
        valid_values: np.ndarray,
        theme: str,
        signed: bool,
        color_range: ComplexArtifactColorRange | None,
        boundary_overlay: ComplexDomainBoundaryOverlay,
    ) -> go.Figure:
        solution_field = field in ComplexScalarMeshArtifactMixin.SOLUTION_MESH_FIELDS
        if solution_field:
            values = visualization_mesh.transfer_solution(valid_values)
            intensity_mode = "vertex"
        else:
            values = visualization_mesh.transfer_interior_cell_values(valid_values)
            intensity_mode = "cell"
        vertices = visualization_mesh.vertices
        triangles = visualization_mesh.triangles
        traces: list[Any] = [
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=np.zeros(vertices.shape[0], dtype=np.float64),
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                intensity=values,
                intensitymode=intensity_mode,
                colorscale="RdBu" if signed else "Viridis",
                showscale=True,
                flatshading=not solution_field,
                lighting={
                    "ambient": 1.0,
                    "diffuse": 0.0,
                    "specular": 0.0,
                    "roughness": 1.0,
                    "fresnel": 0.0,
                },
                colorbar={"exponentformat": "power", "showexponent": "all"},
                hoverinfo="skip",
                name="Scalar mesh",
                showlegend=False,
                **({} if color_range is None else color_range.plotly_kwargs()),
            )
        ]
        hover_coords = vertices[visualization_mesh.valid_to_vertex]
        hover_values = np.asarray(valid_values)
        if solution_field:
            boundary_vertices = vertices[visualization_mesh.boundary_vertex_mask]
            hover_coords = np.concatenate((hover_coords, boundary_vertices), axis=0)
            hover_values = np.concatenate(
                (
                    hover_values,
                    np.zeros(boundary_vertices.shape[0], dtype=hover_values.dtype),
                )
            )
        traces.append(
            go.Scatter3d(
                x=hover_coords[:, 0],
                y=hover_coords[:, 1],
                z=np.zeros(hover_coords.shape[0], dtype=np.float64),
                customdata=hover_values.reshape(-1, 1),
                mode="markers",
                marker={"size": 8, "color": "rgba(0, 0, 0, 0.001)"},
                hovertemplate=(
                    "x=%{x:.6g}<br>y=%{y:.6g}<br>value=%{customdata[0]:.6e}"
                    "<extra></extra>"
                ),
                name="Exact scalar values",
                showlegend=False,
            )
        )
        if not solution_field and boundary_overlay.enabled:
            boundary_points = vertices[visualization_mesh.boundary_edges]
            separator = np.full(
                (boundary_points.shape[0], 1),
                np.nan,
                dtype=np.float64,
            )
            boundary_x = np.concatenate((boundary_points[:, :, 0], separator), axis=1)
            boundary_y = np.concatenate((boundary_points[:, :, 1], separator), axis=1)
            traces.append(
                go.Scatter3d(
                    x=boundary_x.reshape(-1),
                    y=boundary_y.reshape(-1),
                    z=np.zeros(boundary_x.size, dtype=np.float64),
                    mode="lines",
                    line={"color": boundary_overlay.marker_color, "width": 3.0},
                    hovertemplate=(
                        "Domain boundary<br>x=%{x:.6g}<br>y=%{y:.6g}<br>"
                        "scalar unavailable<extra></extra>"
                    ),
                    name="Domain boundary",
                    showlegend=False,
                )
            )

        figure = go.Figure(
            data=traces,
            layout=cls._mesh_layout(
                title=title,
                theme=theme,
                vertices=vertices,
            ),
        )
        return figure


class ComplexCoefficientArtifactMixin(ComplexMeshFigureLayoutMixin):
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
        coords_numpy, a_numpy, bx_numpy, by_numpy, c_numpy = (
            self._evaluate_coefficient_arrays_at_coords(coords, coeffs)
        )
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

    def _evaluate_coefficient_mesh_fields(
        self,
        visualization_mesh: ComplexVisualizationMesh | None,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
    ) -> ComplexCoefficientMeshFields | None:
        if visualization_mesh is None:
            return None
        coords = torch.as_tensor(
            visualization_mesh.vertices,
            dtype=geometry.coords_valid.dtype,
            device=geometry.coords_valid.device,
        )
        coords_numpy, a, bx, by, c = self._evaluate_coefficient_arrays_at_coords(
            coords,
            coeffs,
        )
        if coords_numpy.shape[0] != visualization_mesh.vertex_count:
            raise ValueError(
                "Coefficient mesh evaluation must return one value per "
                "visualization-mesh vertex."
            )
        return ComplexCoefficientMeshFields(
            coords=coords_numpy,
            a=a,
            bx=bx,
            by=by,
            b_magnitude=np.sqrt(np.square(bx) + np.square(by)),
            c=c,
        )

    def _evaluate_coefficient_arrays_at_coords(
        self,
        coords: torch.Tensor,
        coeffs: CoefficientFunctions,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Physical coefficient coordinates must have shape (N, 2).")
        if not torch.all(torch.isfinite(coords)):
            raise ValueError("Physical coefficient coordinates must be finite.")
        x = coords[:, 0]
        y = coords[:, 1]
        with torch.no_grad():
            a = self._evaluate_coefficient_function(coeffs.a_fun, x, y, "a")
            bx = self._evaluate_coefficient_function(coeffs.bx_fun, x, y, "bx")
            by = self._evaluate_coefficient_function(coeffs.by_fun, x, y, "by")
            c = self._evaluate_coefficient_function(coeffs.c_fun, x, y, "c")
        return (
            coords.detach().cpu().numpy(),
            a.detach().cpu().numpy(),
            bx.detach().cpu().numpy(),
            by.detach().cpu().numpy(),
            c.detach().cpu().numpy(),
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
        boundary_overlay: ComplexDomainBoundaryOverlay,
        color_reference_fields: ComplexCoefficientMeshFields | None = None,
    ) -> tuple[list[str], tuple[str, ...]]:
        figure_fields = self._coefficient_figure_fields(fields, terms)
        paths: list[str] = []
        scalar_specs = self._coefficient_scalar_specs(
            fields,
            color_reference_fields=color_reference_fields,
        )
        for name in figure_fields:
            if name == "convection_vector":
                figure = self._convection_vector_figure(
                    fields,
                    theme,
                    boundary_overlay,
                )
            else:
                values, color_values, title, label, signed = scalar_specs[name]
                figure = self._coefficient_scalar_figure(
                    title=title,
                    label=label,
                    coords=fields.coords_valid,
                    values=values,
                    color_reference_values=color_values,
                    theme=theme,
                    signed=signed,
                    boundary_overlay=boundary_overlay,
                )
            base_path = self.request.outdir / "figures" / "coefficients" / name
            save_plotly_figure(figure, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths, figure_fields

    def _write_coefficient_mesh_figures(
        self,
        visualization_mesh: ComplexVisualizationMesh | None,
        fields: ComplexCoefficientMeshFields | None,
        coefficient_figure_fields: tuple[str, ...],
        theme: str,
    ) -> tuple[list[str], tuple[str, ...]]:
        if visualization_mesh is None or fields is None:
            return [], ()
        mesh_figure_fields = tuple(
            name for name in coefficient_figure_fields if name != "convection_vector"
        )
        scalar_specs = self._coefficient_scalar_specs(fields)
        paths: list[str] = []
        for name in mesh_figure_fields:
            values, color_values, title, label, signed = scalar_specs[name]
            figure = self._coefficient_mesh_figure(
                title=f"{title} mesh",
                label=label,
                coords=fields.coords,
                triangles=visualization_mesh.triangles,
                values=values,
                color_reference_values=color_values,
                theme=theme,
                signed=signed,
            )
            base_path = (
                self.request.outdir
                / "figures"
                / "coefficients"
                / "mesh"
                / f"{name}_mesh"
            )
            save_plotly_figure(figure, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths, mesh_figure_fields

    @staticmethod
    def _coefficient_scalar_specs(
        fields: ComplexCoefficientFields | ComplexCoefficientMeshFields,
        *,
        color_reference_fields: ComplexCoefficientMeshFields | None = None,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, str, str, bool]]:
        reference = fields if color_reference_fields is None else color_reference_fields
        return {
            "diffusion_a": (
                fields.a,
                reference.a,
                "Diffusion coefficient a(x, y)",
                "a",
                False,
            ),
            "reaction_c": (
                fields.c,
                reference.c,
                "Reaction coefficient c(x, y)",
                "c",
                bool(np.min(reference.c) < 0.0 < np.max(reference.c)),
            ),
            "convection_bx": (
                fields.bx,
                reference.bx,
                "Convection coefficient b_x(x, y)",
                "b_x",
                True,
            ),
            "convection_by": (
                fields.by,
                reference.by,
                "Convection coefficient b_y(x, y)",
                "b_y",
                True,
            ),
            "convection_magnitude": (
                fields.b_magnitude,
                reference.b_magnitude,
                "Convection magnitude |b(x, y)|",
                "|b|",
                False,
            ),
        }

    def _coefficient_scalar_figure(
        self,
        *,
        title: str,
        label: str,
        coords: np.ndarray,
        values: np.ndarray,
        color_reference_values: np.ndarray | None = None,
        theme: str,
        signed: bool,
        boundary_overlay: ComplexDomainBoundaryOverlay,
    ) -> go.Figure:
        color_values = (
            values if color_reference_values is None else color_reference_values
        )
        color_style = self._coefficient_color_style(color_values, signed=signed)
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
                    "showscale": True,
                    "size": 6,
                    "colorbar": {
                        "title": label,
                        "exponentformat": "power",
                        "showexponent": "all",
                    },
                    **color_style,
                },
            ),
            layout=self._coefficient_layout(title, theme),
        )
        self._add_coefficient_constant_annotation(figure, values, label=label)
        boundary_overlay.add_to_figure(figure)
        return figure

    def _coefficient_mesh_figure(
        self,
        *,
        title: str,
        label: str,
        coords: np.ndarray,
        triangles: np.ndarray,
        values: np.ndarray,
        color_reference_values: np.ndarray,
        theme: str,
        signed: bool,
    ) -> go.Figure:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("Coefficient mesh coordinates must have shape (N, 2).")
        if values.shape != (coords.shape[0],):
            raise ValueError(
                "Coefficient mesh values must contain one scalar per mesh vertex."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Coefficient mesh values must be finite.")
        color_style = self._coefficient_color_style(
            color_reference_values,
            signed=signed,
        )
        figure = go.Figure(
            data=[
                go.Mesh3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=np.zeros(coords.shape[0], dtype=np.float64),
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    intensity=values,
                    intensitymode="vertex",
                    showscale=True,
                    flatshading=False,
                    lighting={
                        "ambient": 1.0,
                        "diffuse": 0.0,
                        "specular": 0.0,
                        "roughness": 1.0,
                        "fresnel": 0.0,
                    },
                    colorbar={
                        "title": label,
                        "exponentformat": "power",
                        "showexponent": "all",
                    },
                    hoverinfo="skip",
                    name="Physical coefficient mesh",
                    showlegend=False,
                    **color_style,
                ),
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=np.zeros(coords.shape[0], dtype=np.float64),
                    customdata=values.reshape(-1, 1),
                    mode="markers",
                    marker={"size": 8, "color": "rgba(0, 0, 0, 0.001)"},
                    hovertemplate=(
                        "x=%{x:.6g}<br>y=%{y:.6g}<br>"
                        f"{label}=%{{customdata[0]:.6g}}<extra></extra>"
                    ),
                    name="Direct physical coefficient values",
                    showlegend=False,
                ),
            ],
            layout=self._mesh_layout(
                title=title,
                theme=theme,
                vertices=coords,
            ),
        )
        self._add_coefficient_constant_annotation(figure, values, label=label)
        return figure

    @classmethod
    def _coefficient_color_style(
        cls,
        values: np.ndarray,
        *,
        signed: bool,
    ) -> dict[str, Any]:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("Coefficient color reference must contain finite values.")
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        max_abs = float(np.max(np.abs(finite)))
        tolerance = cls.COEFFICIENT_ZERO_TOLERANCE * max(1.0, max_abs)
        style: dict[str, Any] = {"colorscale": "RdBu" if signed else "Viridis"}
        if signed:
            if max_abs > 0.0:
                style.update(cmin=-max_abs, cmax=max_abs)
        elif maximum - minimum > tolerance:
            style.update(cmin=minimum, cmax=maximum)
        return style

    @classmethod
    def _add_coefficient_constant_annotation(
        cls,
        figure: go.Figure,
        values: np.ndarray,
        *,
        label: str,
    ) -> None:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        max_abs = float(np.max(np.abs(values)))
        tolerance = cls.COEFFICIENT_ZERO_TOLERANCE * max(1.0, max_abs)
        if maximum - minimum <= tolerance:
            figure.add_annotation(
                x=0.5,
                y=1.04,
                xref="paper",
                yref="paper",
                showarrow=False,
                text=f"Constant field: {label}={minimum:.6g}",
            )

    def _convection_vector_figure(
        self,
        fields: ComplexCoefficientFields,
        theme: str,
        boundary_overlay: ComplexDomainBoundaryOverlay,
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
        boundary_overlay.add_to_figure(figure)
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


class ComplexCouplingArtifactExporter(
    ComplexScalarMeshArtifactMixin,
    ComplexCoefficientArtifactMixin,
):
    """Export complex-geometry CouplingNet metrics, raw archives, and scatter plots."""

    DEFAULT_DIRECTIONAL_COLOR_QUANTILE: ClassVar[float] = 0.99
    COLOR_RANGE_POLICY: ClassVar[str] = (
        "solution_full_range_and_directional_robust_quantile"
    )
    COLOR_RANGE_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "solution": ("sol", "u_pred", "u_phi", "u_psi"),
        "phi": ("target_phi", "phi"),
        "psi": ("target_psi", "psi"),
    }
    DIRECTIONAL_ERROR_FIELDS: ClassVar[tuple[str, ...]] = (
        "phi_error",
        "psi_error",
    )
    DIRECTIONAL_COLOR_SUMMARY_FIELDS: ClassVar[tuple[str, ...]] = (
        "rhs",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
    )
    FIGURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "rhs",
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
        "u_pred_error",
        "u_equal_mean_error",
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
        "weak_reliability_eta_phi_squared",
        "weak_reliability_eta_psi_squared",
        "weak_reliability_theta",
        "weak_reliability_weight_phi",
        "split_mass_relative_contribution",
        "fusion_base_difference",
        "fusion_network_output_physical",
        "fusion_residual_physical",
        "fusion_fused_difference",
        "fusion_delta_from_base",
        "tangent_gradient",
        "tangent_delta",
        "tangent_mismatch_pre",
        "tangent_mismatch_k1",
        "tangent_mismatch_post",
        "tangent_direction_0",
        "tangent_direction_1",
        "tangent_response_direction_0",
        "tangent_response_direction_1",
        "tangent_residual_gradient_post",
        "tangent_stationarity_residual",
        "tangent_source_response_energy_density",
        "tangent_response_correction",
    )
    SIGNED_FIGURE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "u_pred_error",
            "u_equal_mean_error",
            "u_phi_error",
            "u_psi_error",
            "u_split_mismatch",
            "phi_error",
            "psi_error",
            "weak_residual_x",
            "weak_residual_y",
            "weak_reliability_theta",
            "fusion_base_difference",
            "fusion_network_output_physical",
            "fusion_residual_physical",
            "fusion_fused_difference",
            "fusion_delta_from_base",
            "tangent_gradient",
            "tangent_delta",
            "tangent_mismatch_pre",
            "tangent_mismatch_k1",
            "tangent_mismatch_post",
            "tangent_direction_0",
            "tangent_direction_1",
            "tangent_response_direction_0",
            "tangent_response_direction_1",
            "tangent_residual_gradient_post",
            "tangent_stationarity_residual",
            "tangent_response_correction",
        }
    )
    FIGURE_TITLES: ClassVar[dict[str, str]] = {
        "rhs": "Source rhs",
        "sol": "Exact solution sol",
        "u_pred": "Predicted solution u_pred",
        "u_phi": "Reconstructed solution u_phi",
        "u_psi": "Reconstructed solution u_psi",
        "u_pred_error": "Signed error u_pred - sol",
        "u_equal_mean_error": "Signed error equal mean - sol",
        "u_phi_error": "Signed error u_phi - sol",
        "u_psi_error": "Signed error u_psi - sol",
        "u_split_mismatch": "Mismatch u_phi - u_psi",
        "phi": "Projected phi",
        "psi": "Projected psi",
        "target_phi": "Target phi",
        "target_psi": "Target psi",
        "phi_error": "Signed error phi - target_phi",
        "psi_error": "Signed error psi - target_psi",
        "weak_residual_x": ("Training weak-closure residual Bx(u_equal_mean) - phi"),
        "weak_residual_y": ("Training weak-closure residual By(u_equal_mean) - psi"),
        "weak_reliability_eta_phi_squared": (
            "Local weak-residual indicator eta_phi squared"
        ),
        "weak_reliability_eta_psi_squared": (
            "Local weak-residual indicator eta_psi squared"
        ),
        "weak_reliability_theta": "Signed local weak reliability theta",
        "weak_reliability_weight_phi": "Local weak reliability weight w_phi",
        "split_mass_relative_contribution": (
            "Normalized pointwise split value mismatch"
        ),
        "fusion_base_difference": "Base physical difference p - q",
        "fusion_network_output_physical": "Physical MLP difference output",
        "fusion_residual_physical": "Nonlinear physical difference residual",
        "fusion_fused_difference": "Fused physical difference",
        "fusion_delta_from_base": "Fused difference minus base difference",
        "tangent_gradient": "Tangent response objective gradient",
        "tangent_delta": "Balance-preserving tangent source correction",
        "tangent_mismatch_pre": "Directional response mismatch before tangent step",
        "tangent_mismatch_k1": "Directional response mismatch after K=1",
        "tangent_mismatch_post": "Directional response mismatch after tangent step",
        "tangent_direction_0": "First Jacobi-preconditioned tangent direction",
        "tangent_direction_1": "Second response-orthogonal tangent direction",
        "tangent_response_direction_0": "Response of first tangent direction",
        "tangent_response_direction_1": "Response of second tangent direction",
        "tangent_residual_gradient_post": "Post-subspace tangent gradient residual",
        "tangent_stationarity_residual": "Tangent stationarity residual",
        "tangent_source_response_energy_density": (
            "Directional source-response energy density"
        ),
        "tangent_response_correction": "Applied tangent response correction S delta",
    }
    GREEN_RESPONSE_FIGURE_TITLES: ClassVar[dict[str, str]] = {
        "gamma_x_squared": "X source-column Green-response cost",
        "gamma_y_squared": "Y source-column Green-response cost",
        "correction_weight_phi": "Physical balance correction weight for phi",
        "tangent_preconditioner_base": "Tangent Jacobi preconditioner base",
        "tangent_denominator": "Tangent regularized denominator",
        "tangent_cross_axis_inner_product": "Cross-axis column Gram diagonal c",
        "tangent_normalized_correlation": "Normalized cross-axis correlation rho",
        "tangent_normalized_quadratic_cross_axis": (
            "Normalized quadratic cross-axis term q"
        ),
    }

    def __init__(
        self,
        request: CouplingArtifactRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.request.outdir.mkdir(parents=True, exist_ok=True)

    @property
    def directional_color_quantile(self) -> float:
        configured = self.request.directional_color_quantile
        return (
            self.DEFAULT_DIRECTIONAL_COLOR_QUANTILE
            if configured is None
            else float(configured)
        )

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
        visualization_mesh = self._load_visualization_mesh(
            geometry_path=configs.dataset.geometry_path,
            geometry=geometry,
        )
        coefficient_fields = self._evaluate_coefficient_fields(geometry, coeffs)
        coefficient_mesh_fields = self._evaluate_coefficient_mesh_fields(
            visualization_mesh,
            geometry,
            coeffs,
        )
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
            tangent_context_path=self.request.tangent_context,
            tangent_context_default_path=(
                self.request.coupling_checkpoint.parent
                / "tangent_response_context.safetensors"
            ),
        )
        boundary_context = evaluator.boundary_energy_context(geometry)
        boundary_overlay = ComplexDomainBoundaryOverlay.from_endpoint_coords(
            boundary_context.endpoint_coords.detach().cpu().numpy(),
            enabled=self.request.show_domain_boundary,
            theme=self.request.theme,
        )
        metric_rows = self._evaluate_rows(dataset, evaluator, configs)
        selected, roles, policy = self._select_sample_indices(metric_rows)
        selected_samples = self._evaluate_selected(dataset, evaluator, selected, device)
        color_ranges_by_sample = {
            sample.sample_id: self._color_ranges_for_sample(sample.arrays)
            for sample in selected_samples
        }
        self._write_metric_csv(metric_rows)
        self._write_selected_npz(selected_samples)
        self._write_coefficient_npz(coefficient_fields)
        self._write_green_response_context_npz(evaluator)
        self._write_tangent_response_context_npz(evaluator)
        figure_fields = self._figure_fields(selected_samples)
        sample_figure_paths = self._write_figures(
            selected_samples,
            self.request.theme,
            boundary_overlay,
            color_ranges_by_sample,
        )
        mesh_figure_paths, mesh_figure_fields = self._write_scalar_mesh_figures(
            selected_samples,
            visualization_mesh,
            self.request.theme,
            boundary_overlay,
            color_ranges_by_sample,
        )
        mesh_raw_archive = self._copy_visualization_mesh(visualization_mesh)
        coefficient_figure_paths, coefficient_figure_fields = (
            self._write_coefficient_figures(
                coefficient_fields,
                configs.coupling_model.coefficient_terms,
                self.request.theme,
                boundary_overlay,
                coefficient_mesh_fields,
            )
        )
        coefficient_mesh_figure_paths, coefficient_mesh_figure_fields = (
            self._write_coefficient_mesh_figures(
                visualization_mesh,
                coefficient_mesh_fields,
                coefficient_figure_fields,
                self.request.theme,
            )
        )
        column_projection_paths, column_projection_fields = (
            self._write_green_response_context_figures(
                evaluator.column_diagonal_green_response_context,
                geometry,
                self.request.theme,
                boundary_overlay,
            )
        )
        tangent_projection_paths, tangent_projection_fields = (
            self._write_tangent_response_context_figures(
                evaluator.symmetric_tangent_green_response_context,
                geometry,
                self.request.theme,
                boundary_overlay,
            )
        )
        projection_figure_paths = column_projection_paths + tangent_projection_paths
        projection_figure_fields = column_projection_fields + tangent_projection_fields
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
        cross_axis_reconstruction = ComplexCrossAxisReconstructionConfig.from_raw(
            configs.coupling_model.cross_axis_reconstruction
        )
        canonical_energy = ComplexCanonicalEnergyConfig.from_raw(
            configs.coupling_training.canonical_energy
        )
        relative_split = ComplexRelativeSplitConsistencyConfig.from_raw(
            configs.coupling_training.relative_split_consistency
        )
        weak_closure = ComplexWeakOperatorClosureConfig.from_raw(
            configs.coupling_training.weak_operator_closure
        )
        post_line_search_stationarity = (
            ComplexPostLineSearchStationarityConfig.from_raw(
                configs.coupling_training.post_line_search_stationarity
            )
        )
        response_trust = ComplexResponseTrustConfig.from_raw(
            configs.coupling_training.response_trust
        )
        tangent_projection = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            balance_projection.symmetric_tangent_green_response
        )
        tangent_subspace = (
            balance_projection.mode == "symmetric_tangent_green_response"
            and tangent_projection.subspace_dimension >= 2
        )
        tangent_dimension = tangent_projection.subspace_dimension
        best_energy = CouplingBestEnergyCheckpointConfig.from_raw(
            configs.coupling_training.best_energy_checkpoint
        )
        best_physics = CouplingBestPhysicsCheckpointConfig.from_raw(
            configs.coupling_training.best_physics_checkpoint
        )
        optimizer_provenance = ComplexCouplingOptimizerFactory(
            configs.coupling_training
        ).provenance()
        training_reproducibility = (
            {
                "available": False,
                "reason": "legacy_config_without_coupling_training_seed",
            }
            if configs.coupling_training.seed is None
            else {
                "available": True,
                **TrainingSeedContext(
                    stage="coupling",
                    base_seed=configs.coupling_training.seed,
                    deterministic_algorithms=(
                        configs.coupling_training.deterministic_algorithms
                    ),
                    device=configs.coupling_training.device,
                ).as_dict(),
                "source_seed_independent": True,
            }
        )
        projection_formula = self._projection_formula(balance_projection)
        green_response_context = evaluator.column_diagonal_green_response_context
        green_response_summary = self._green_response_context_summary(
            green_response_context,
            evaluator,
            balance_projection,
        )
        tangent_response_summary = self._tangent_response_context_summary(
            evaluator.symmetric_tangent_green_response_context,
            evaluator,
            balance_projection,
            configs.coupling_training,
            metric_rows,
        )
        architecture = coupling_model.architecture_provenance()
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generation_trigger": self.request.generation_trigger,
            "checkpoint_selector": self.request.checkpoint_selector,
            "config": str(self.request.config.resolve()),
            "coupling_checkpoint": str(self.request.coupling_checkpoint.resolve()),
            "green_checkpoint": str(self.request.green_checkpoint.resolve()),
            "outdir": str(self.request.outdir.resolve()),
            "geometry_mode": "complex",
            "device": str(device),
            "theme": self.request.theme,
            "tangent_context": (
                None
                if self.request.tangent_context is None
                else str(self.request.tangent_context.resolve())
            ),
            "coefficients": None if coeff_path is None else str(coeff_path),
            "geometry_path": str(configs.dataset.geometry_path),
            "test_path": str(configs.dataset.test_path),
            "training_source": asdict(configs.dataset.coupling_source),
            "reference_diagnostics": asdict(configs.dataset.reference_diagnostics),
            "training_reproducibility": training_reproducibility,
            "tangent_subspace_dimension_provenance": configs.raw.get(
                "tangent_subspace_dimension_provenance"
            ),
            "artifact_dataset_contract": "full_reference_test_npz",
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "selected_sample_policy": policy,
            "plot_workers": self.request.plot_workers,
            "save_generated_data": self.request.save_generated_data,
            "domain_boundary_overlay": boundary_overlay.summary(),
            "aggregate_metrics": aggregate,
            "figure_count": (
                len(sample_figure_paths)
                + len(mesh_figure_paths)
                + len(coefficient_figure_paths)
                + len(coefficient_mesh_figure_paths)
                + len(projection_figure_paths)
            ),
            "figure_fields": list(figure_fields),
            "coefficient_figure_count": len(coefficient_figure_paths),
            "coefficient_figure_fields": list(coefficient_figure_fields),
            "projection_figure_count": len(projection_figure_paths),
            "projection_figure_fields": list(projection_figure_fields),
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
            "solution_prediction": (
                "u_pred=w_phi*u_phi+(1-w_phi)*u_psi"
                if cross_axis_reconstruction.enabled
                else "u_pred=0.5*(u_phi+u_psi)"
            ),
            "raw_output_space": "reference_response",
            "output_contract_version": ComplexCouplingNet.OUTPUT_CONTRACT_VERSION,
            "optimizer": optimizer_provenance.as_dict(),
            "balance_projection": {
                "enabled": balance_projection.enabled,
                "mode": balance_projection.mode,
                "space": "physical_source",
                "formula": projection_formula,
                "constraint": "phi + psi = rhs",
                "raw_physical_difference_preserved": (
                    balance_projection.mode == "physical_symmetric"
                ),
                "pre_projection_scaling": "p=P_raw/Lx^2; q=Q_raw/Ly^2",
                "post_projection_pull_back": "Phi=Lx^2*phi; Psi=Ly^2*psi",
                "uses_reference_targets": False,
                "column_diagonal_green_response": green_response_summary,
                "symmetric_tangent_green_response": tangent_response_summary,
            },
            "pre_projection_fusion": {
                "enabled": pre_projection_fusion.enabled,
                "architecture": FUSION_ARCHITECTURE,
                "mode": pre_projection_fusion.mode,
                "space": "physical_directional_source",
                "input": [
                    "base_difference_over_safe_source_scale",
                    "rhs_over_safe_source_scale",
                ],
                "hidden_dim": pre_projection_fusion.hidden_dim,
                "depth": pre_projection_fusion.depth,
                "activation": configs.coupling_model.activation,
                "use_bias": configs.coupling_model.use_bias,
                "identity_skip": pre_projection_fusion.mode == "residual",
                "final_layer_initialization": FINAL_LAYER_INITIALIZATION,
                "final_layer_init_scale": (
                    pre_projection_fusion.final_layer_init_scale
                ),
                "explicit_geometry_features": False,
                "learned_linear_branch": False,
                "learned_gate": False,
                "source_scale": "sqrt((A_x^2+A_y^2)/2)",
                "formula": pre_projection_fusion_formula(pre_projection_fusion.mode),
                "pre_projection_balance_constructed": pre_projection_fusion.enabled,
                "uses_reference_targets": False,
            },
            "reconstruction_response_input": {
                "phi": "projected Phi is used directly",
                "psi": "projected Psi is used directly",
                "additional_length_scaling": False,
            },
            "cross_axis_reconstruction": {
                "enabled": cross_axis_reconstruction.enabled,
                "mode": (
                    cross_axis_reconstruction.mode
                    if cross_axis_reconstruction.enabled
                    else "equal_mean"
                ),
                "configured_mode": cross_axis_reconstruction.mode,
                "gamma": cross_axis_reconstruction.gamma,
                "smoothing_steps": cross_axis_reconstruction.smoothing_steps,
                "smoothing_relaxation": (
                    cross_axis_reconstruction.smoothing_relaxation
                ),
                "relative_floor": cross_axis_reconstruction.relative_floor,
                "eps": cross_axis_reconstruction.eps,
                "candidate_residual": ("R(v)=Rx(v;phi)+Ry(v;psi), v in {u_phi,u_psi}"),
                "raw_indicator": "eta_v^2=R(v)^2/(m_x+m_y+eps)",
                "graph": "geometry.x_edges union geometry.y_edges",
                "weight_formula": (
                    "theta=gamma*(eta_psi^2-eta_phi^2)/"
                    "(eta_phi^2+eta_psi^2+2*sample_floor); "
                    "w_phi=0.5*(1+theta); w_psi=1-w_phi"
                ),
                "prediction_formula": (
                    "u_pred=w_phi*u_phi+w_psi*u_psi"
                    if cross_axis_reconstruction.enabled
                    else "u_pred=0.5*(u_phi+u_psi)"
                ),
                "uses_reference_targets": False,
                "affects_training_objective": False,
                "uses_global_matrix_solve": False,
                "requires_global_matrix_solve": False,
                "geometry_only_and_mismatch_modes_available": False,
                "geometry_only_mode_available": False,
                "mismatch_detected_mode_available": False,
                "context_build_count": (
                    evaluator.cross_axis_reconstructor.context_build_count
                ),
            },
            "canonical_boundary_energy": {
                "enabled": True,
                "optimization_enabled": canonical_energy.boundary_weight > 0.0,
                "weight": canonical_energy.boundary_weight,
                "diagnostic_always_reported": True,
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
            "canonical_energy": {
                "enabled": True,
                "domain": "all_valid_same_segment_edges",
                "bulk_formula": (
                    "sum_edges arithmetic_mean(a)*(delta(u_phi-u_psi)/h_axis)^2*hx*hy"
                ),
                "boundary_included": True,
                "boundary_weight": canonical_energy.boundary_weight,
                "optimized_formula": "bulk + boundary_weight * boundary",
                "optimized_metric": "loss_energy_optimized",
                "unweighted_canonical_metric": "loss_energy_consistency",
                "transition_partition": False,
                "checkpoint_metric": "loss_energy_optimized",
                "uses_reference_targets": False,
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
                "trial_solution": "u_equal_mean=0.5*(u_phi+u_psi)",
                "test_space": "directional_segment_p1_nodal",
                "coefficient_evaluation": "direct_at_physical_element_midpoints",
                "reaction_split": "c/2_per_direction",
                "uses_reference_targets": False,
            },
            "post_line_search_stationarity": {
                "enabled": post_line_search_stationarity.enabled,
                "optimized": post_line_search_stationarity.enabled,
                "diagnostic_computed": (
                    post_line_search_stationarity.enabled or response_trust.enabled
                ),
                "weight": post_line_search_stationarity.weight,
                "eps": (
                    post_line_search_stationarity.eps
                    if post_line_search_stationarity.enabled
                    else response_trust.eps
                ),
                "objective": (
                    f"mean(r_K{tangent_dimension}^T*D^-1*r_K{tangent_dimension}/"
                    "(||H_x(f/2)||_M^2+||H_y(f/2)||_M^2+eps))"
                    if tangent_subspace
                    else (
                        "mean((g-eta_star*A*z)^T*D^-1*(g-eta_star*A*z)"
                        "/(||H_x(f/2)||_M^2+||H_y(f/2)||_M^2+eps))"
                    )
                ),
                "legacy_ratio_diagnostic": (
                    f"mean(r_K{tangent_dimension}^T*D^-1*r_K{tangent_dimension}/"
                    "(g0^T*D^-1*g0+eps))"
                    if tangent_subspace
                    else (
                        "mean((g-eta_star*A*z)^T*D^-1*"
                        "(g-eta_star*A*z)/(g^T*D^-1*g+eps))"
                    )
                ),
                "optimization_normalization": "source_response_energy",
                "legacy_ratio_optimized": False,
                "eta_source": (
                    "not_applicable" if tangent_subspace else "uncapped_eta_star"
                ),
                "forward_eta_source": (
                    "not_applicable" if tangent_subspace else "capped_eta_applied"
                ),
                "residual_source": (
                    f"post_k{tangent_dimension}_residual_gradient"
                    if tangent_subspace
                    else "uncapped_eta_star"
                ),
                "hessian_action": (
                    f"r_K{tangent_dimension}=(H_x+H_y)^T*M_Omega*"
                    f"m_K{tangent_dimension}; reused from K={tangent_dimension} "
                    "projection"
                    if tangent_subspace
                    else "A*z=(H_x+H_y)^T*M_Omega*(H_x+H_y)*z"
                ),
                "matrix_free": True,
                "extra_adjoint_actions_per_computed_batch": (
                    0 if tangent_subspace else 1
                ),
                "extra_adjoint_actions_per_enabled_batch": (
                    0 if tangent_subspace else 1
                ),
                "shared_source_response_forward_actions_per_computed_batch": 1,
                "joint_response_trust_enabled": (
                    post_line_search_stationarity.enabled and response_trust.enabled
                ),
                "global_response_matrix_materialized": False,
                "full_gram_solve": False,
                "uses_reference_targets": False,
            },
            "response_trust": {
                "enabled": response_trust.enabled,
                "weight": response_trust.weight,
                "trust_weight": response_trust.trust_weight,
                "eps": response_trust.eps,
                "objective": (
                    "mean((||m_post||_M^2 + trust_weight*||S*delta||_M^2)"
                    "/(||H_x(f/2)||_M^2+||H_y(f/2)||_M^2+eps))"
                ),
                "post_mismatch": "m_post=H_x*phi-H_y*psi",
                "correction_response": "S*delta=m_post-m_pre",
                "eta_source": (
                    "not_applicable" if tangent_subspace else "capped_eta_applied"
                ),
                "correction_source": (
                    f"unconstrained_k{tangent_dimension}_coefficients"
                    if tangent_subspace
                    else "capped_eta_applied"
                ),
                "source_normalization": "H_x(f/2)^2+H_y(f/2)^2",
                "stationarity_diagnostic_computed": response_trust.enabled,
                "stationarity_diagnostic_eta_source": (
                    "not_applicable" if tangent_subspace else "uncapped_eta_star"
                ),
                "joint_stationarity_optimized": (
                    response_trust.enabled and post_line_search_stationarity.enabled
                ),
                "source_response_shared_with_stationarity": (
                    response_trust.enabled and post_line_search_stationarity.enabled
                ),
                "matrix_free": True,
                "shared_source_response_forward_actions_per_enabled_batch": 1,
                "extra_forward_actions_per_enabled_batch": 1,
                "extra_adjoint_actions_per_enabled_batch": (
                    0 if tangent_subspace else 1
                ),
                "global_response_matrix_materialized": False,
                "full_gram_solve": False,
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
            "directional_color_range": self._directional_color_range_summary(
                selected_samples,
                color_ranges_by_sample,
            ),
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
            "complex_coupling_architecture": architecture,
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
        if coupling_model.branch_geometry is not None:
            summary["geometry_branch"] = {
                "enabled": True,
                "feature_count": coupling_model.geometry_feature_dim,
            }
        if coupling_model.branch_transverse is None:
            summary.pop("transverse_encoding")
        if not coupling_model.transverse_trunk_enabled:
            summary.pop("transverse_trunk")
        if visualization_mesh is not None:
            mesh_summary = visualization_mesh.summary(self.request.visualization_mesh)
            mesh_summary["raw_archive"] = mesh_raw_archive
            mesh_summary["figure_fields"] = list(mesh_figure_fields)
            mesh_summary["figure_count"] = len(mesh_figure_paths)
            mesh_summary["field_space"] = "physical_scalar"
            mesh_summary["color_range_policy"] = {
                "solution": "full_range_including_prescribed_zero",
                "rhs": "full_min_max",
                "directional_values": "shared_lower_upper_quantile",
                "directional_errors": "symmetric_absolute_quantile",
                "directional_quantile": self.directional_color_quantile,
            }
            mesh_summary["field_boundary_policy"] = {
                "sol/u_pred/u_pred_error": (
                    "prescribed_homogeneous_dirichlet_zero_without_outline"
                ),
                "rhs/phi/psi/target_phi/target_psi/phi_error/psi_error": (
                    "not_evaluated_black_outline"
                ),
            }
            mesh_summary["hover_policy"] = (
                "exact_valid_points_and_solution_boundary_zero_only"
            )
            mesh_summary["scene_scale"] = self.MESH_SCENE_SCALE
            summary["visualization_mesh"] = mesh_summary
            summary["mesh_figure_count"] = len(mesh_figure_paths)
            summary["mesh_figure_fields"] = list(mesh_figure_fields)
            summary["coefficient_mesh_figure_count"] = len(
                coefficient_mesh_figure_paths
            )
            summary["coefficient_mesh_figure_fields"] = list(
                coefficient_mesh_figure_fields
            )
            summary["coefficient_mesh_evaluation"] = (
                "direct_at_visualization_mesh_vertices"
            )
            summary["coefficient_mesh_boundary_value_source"] = (
                "direct_physical_coefficient_function"
            )
            summary["coefficient_mesh_intensity_mode"] = "vertex"
            summary["mesh_scene_scale"] = self.MESH_SCENE_SCALE
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
                    prediction.cross_axis_reconstruction.u_pred_valid[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                u_equal_mean = (
                    prediction.cross_axis_reconstruction.u_equal_mean_valid[0]
                    .detach()
                    .cpu()
                    .numpy()
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
                    "projection_balance_residual_before": (
                        prediction.projection.raw_response_constraint_residual[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "projection_correction_phi": (
                        prediction.projection.correction_phi[0].detach().cpu().numpy()
                    ),
                    "projection_correction_psi": (
                        prediction.projection.correction_psi[0].detach().cpu().numpy()
                    ),
                    "projection_correction_weight_phi": (
                        prediction.projection.correction_weight_phi[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "projection_correction_weight_psi": (
                        prediction.projection.correction_weight_psi[0]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    "projection_difference_update": (
                        prediction.projection.difference_update[0]
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
                tangent = prediction.projection.symmetric_tangent_diagnostics
                if tangent is not None:
                    arrays.update(
                        {
                            "symmetric_physical_phi": (
                                tangent.symmetric_physical[0, 0].detach().cpu().numpy()
                            ),
                            "symmetric_physical_psi": (
                                tangent.symmetric_physical[0, 1].detach().cpu().numpy()
                            ),
                            "symmetric_u_phi": (
                                tangent.symmetric_solution[0, 0].detach().cpu().numpy()
                            ),
                            "symmetric_u_psi": (
                                tangent.symmetric_solution[0, 1].detach().cpu().numpy()
                            ),
                            "tangent_mismatch_pre": (
                                tangent.mismatch_pre[0].detach().cpu().numpy()
                            ),
                            "tangent_gradient": (
                                tangent.gradient[0].detach().cpu().numpy()
                            ),
                            "tangent_preconditioner_base": (
                                tangent.preconditioner_base.detach().cpu().numpy()
                            ),
                            "tangent_denominator": (
                                tangent.denominator.detach().cpu().numpy()
                            ),
                            "tangent_delta": tangent.delta[0].detach().cpu().numpy(),
                            "tangent_mismatch_post": (
                                tangent.mismatch_post[0].detach().cpu().numpy()
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
                            or tangent.response_direction is None
                        ):
                            raise RuntimeError(
                                "Adaptive tangent diagnostics are incomplete."
                            )
                        arrays.update(
                            {
                                "tangent_response_direction": (
                                    tangent.response_direction[0].detach().cpu().numpy()
                                ),
                                "tangent_line_search_numerator": np.asarray(
                                    tangent.line_search_numerator[0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                ),
                                "tangent_line_search_denominator": np.asarray(
                                    tangent.line_search_denominator[0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                ),
                                "tangent_eta_star": np.asarray(
                                    tangent.eta_star[0].detach().cpu().item(),
                                    dtype=np.float64,
                                ),
                                "tangent_eta_applied": np.asarray(
                                    tangent.eta_applied[0].detach().cpu().item(),
                                    dtype=np.float64,
                                ),
                                "tangent_eta_cap": np.asarray(
                                    tangent.eta_cap,
                                    dtype=np.float64,
                                ),
                                "tangent_eta_capped": np.asarray(
                                    tangent.eta_capped[0].detach().cpu().item(),
                                    dtype=np.bool_,
                                ),
                            }
                        )
                    if tangent.subspace_dimension >= 2:
                        subspace = tangent.subspace_result
                        if subspace is None:
                            raise RuntimeError(
                                "K>=2 tangent artifact diagnostics are incomplete."
                            )
                        arrays.update(
                            {
                                "tangent_subspace_dimension": np.asarray(
                                    tangent.subspace_dimension,
                                    dtype=np.int64,
                                ),
                                "tangent_directions": (
                                    subspace.directions[:, 0].detach().cpu().numpy()
                                ),
                                "tangent_directional_responses": (
                                    subspace.directional_responses[:, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                "tangent_response_directions": (
                                    subspace.response_directions[:, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                "tangent_coefficients": (
                                    subspace.coefficients[:, 0].detach().cpu().numpy()
                                ),
                                "tangent_direction_active": (
                                    subspace.direction_active[:, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                "tangent_deltas_by_k": (
                                    subspace.deltas[:, 0].detach().cpu().numpy()
                                ),
                                "tangent_mismatches_by_k": (
                                    subspace.mismatches[:, 0].detach().cpu().numpy()
                                ),
                                "tangent_response_costs_by_k": (
                                    subspace.costs[:, 0].detach().cpu().numpy()
                                ),
                                "tangent_response_gram": (
                                    subspace.response_gram[0].detach().cpu().numpy()
                                ),
                                "tangent_response_orthogonality_max": (
                                    subspace.response_orthogonality_max[:, 0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                                "tangent_second_direction_active": np.asarray(
                                    subspace.direction_active[1, 0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.bool_,
                                ),
                                "tangent_mismatch_k1": (
                                    subspace.mismatches[0, 0].detach().cpu().numpy()
                                ),
                                "tangent_residual_gradient_post": (
                                    subspace.residual_gradient_post[0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                ),
                            }
                        )
                        for direction_index in range(tangent.subspace_dimension):
                            arrays[f"tangent_direction_{direction_index}"] = (
                                subspace.directions[direction_index, 0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            arrays[
                                f"tangent_directional_response_{direction_index}"
                            ] = (
                                subspace.directional_responses[direction_index, 0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            arrays[f"tangent_response_direction_{direction_index}"] = (
                                subspace.response_directions[direction_index, 0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            arrays[f"tangent_coefficient_{direction_index}"] = (
                                np.asarray(
                                    subspace.coefficients[direction_index, 0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                )
                            )
                            arrays[f"tangent_direction_{direction_index}_active"] = (
                                np.asarray(
                                    subspace.direction_active[direction_index, 0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.bool_,
                                )
                            )
                            arrays[f"tangent_mismatch_k{direction_index + 1}"] = (
                                subspace.mismatches[direction_index, 0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            arrays[f"tangent_response_cost_k{direction_index + 1}"] = (
                                np.asarray(
                                    subspace.costs[direction_index, 0]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                )
                            )
                stationarity = prediction.objective.post_line_search_stationarity
                if stationarity is not None:
                    source_response = stationarity.source_response[0]
                    arrays.update(
                        {
                            "tangent_hessian_direction": (
                                stationarity.hessian_direction[0].detach().cpu().numpy()
                            ),
                            "tangent_stationarity_residual": (
                                stationarity.stationarity_residual[0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "tangent_stationarity_ratio": np.asarray(
                                stationarity.relative_ratio_per_sample[0]
                                .detach()
                                .cpu()
                                .item(),
                                dtype=np.float64,
                            ),
                            "tangent_stationarity_source_normalized": np.asarray(
                                stationarity.loss_per_sample[0].detach().cpu().item(),
                                dtype=np.float64,
                            ),
                            "tangent_stationarity_initial_source_ratio": np.asarray(
                                stationarity.initial_source_ratio_per_sample[0]
                                .detach()
                                .cpu()
                                .item(),
                                dtype=np.float64,
                            ),
                            "tangent_stationarity_initial_preconditioned_energy": (
                                np.asarray(
                                    stationarity.initial_preconditioned_energy_per_sample[
                                        0
                                    ]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                )
                            ),
                            "tangent_stationarity_residual_preconditioned_energy": (
                                np.asarray(
                                    stationarity.residual_preconditioned_energy_per_sample[
                                        0
                                    ]
                                    .detach()
                                    .cpu()
                                    .item(),
                                    dtype=np.float64,
                                )
                            ),
                            "tangent_source_response_phi": (
                                source_response[0].detach().cpu().numpy()
                            ),
                            "tangent_source_response_psi": (
                                source_response[1].detach().cpu().numpy()
                            ),
                            "tangent_source_response_energy_density": (
                                source_response.square()
                                .sum(dim=0)
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "tangent_source_response_energy": np.asarray(
                                stationarity.source_response_energy_per_sample[0]
                                .detach()
                                .cpu()
                                .item(),
                                dtype=np.float64,
                            ),
                        }
                    )
                response_trust = prediction.objective.response_trust
                if response_trust is not None:
                    arrays.update(
                        {
                            "tangent_response_correction": (
                                response_trust.correction_response[0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "tangent_response_post_mismatch_ratio": np.asarray(
                                response_trust.post_mismatch_ratio_per_sample[0]
                                .detach()
                                .cpu()
                                .item(),
                                dtype=np.float64,
                            ),
                            "tangent_response_correction_ratio": np.asarray(
                                response_trust.correction_ratio_per_sample[0]
                                .detach()
                                .cpu()
                                .item(),
                                dtype=np.float64,
                            ),
                            "tangent_response_trust_ratio": np.asarray(
                                response_trust.loss_per_sample[0].detach().cpu().item(),
                                dtype=np.float64,
                            ),
                        }
                    )
                reliability = prediction.cross_axis_reconstruction.reliability
                if reliability is not None:
                    arrays.update(
                        {
                            "u_equal_mean": u_equal_mean,
                            "u_equal_mean_error": u_equal_mean - sol,
                            "weak_reliability_residual_phi_x": (
                                reliability.phi_x_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_residual_phi_y": (
                                reliability.phi_y_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_residual_phi_full": (
                                reliability.phi_full_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_residual_psi_x": (
                                reliability.psi_x_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_residual_psi_y": (
                                reliability.psi_y_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_residual_psi_full": (
                                reliability.psi_full_residual[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_nodal_mass": (
                                reliability.nodal_mass.detach().cpu().numpy()
                            ),
                            "weak_reliability_eta_phi_squared_raw": (
                                reliability.phi_indicator_raw[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_eta_psi_squared_raw": (
                                reliability.psi_indicator_raw[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_eta_phi_squared": (
                                reliability.phi_indicator[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_eta_psi_squared": (
                                reliability.psi_indicator[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_sample_floor": (
                                reliability.sample_floor[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_theta": (
                                reliability.theta[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_weight_phi": (
                                reliability.w_phi[0].detach().cpu().numpy()
                            ),
                            "weak_reliability_weight_psi": (
                                reliability.w_psi[0].detach().cpu().numpy()
                            ),
                        }
                    )
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
                            "fusion_base_physical_p": (
                                fusion.base_physical[0, 0].detach().cpu().numpy()
                            ),
                            "fusion_base_physical_q": (
                                fusion.base_physical[0, 1].detach().cpu().numpy()
                            ),
                            "fusion_base_difference": (
                                fusion.base_difference[0].detach().cpu().numpy()
                            ),
                            "fusion_normalized_difference": (
                                fusion.normalized_difference[0].detach().cpu().numpy()
                            ),
                            "fusion_normalized_rhs": (
                                fusion.normalized_rhs[0].detach().cpu().numpy()
                            ),
                            "fusion_network_output_normalized": (
                                fusion.normalized_network_output[0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                            "fusion_network_output_physical": (
                                fusion.physical_network_output[0].detach().cpu().numpy()
                            ),
                            "fusion_fused_difference": (
                                fusion.fused_difference[0].detach().cpu().numpy()
                            ),
                            "fusion_delta_from_base": (
                                fusion.difference_delta[0].detach().cpu().numpy()
                            ),
                            "fusion_pre_projection_phi": (
                                fusion.fused_physical[0, 0].detach().cpu().numpy()
                            ),
                            "fusion_pre_projection_psi": (
                                fusion.fused_physical[0, 1].detach().cpu().numpy()
                            ),
                            "fusion_source_scale": (
                                fusion.source_scale[0].detach().cpu().numpy()
                            ),
                            "fusion_safe_source_scale": (
                                fusion.safe_source_scale[0].detach().cpu().numpy()
                            ),
                            "fusion_pre_projection_balance_residual": (
                                fusion.pre_projection_balance_residual[0]
                                .detach()
                                .cpu()
                                .numpy()
                            ),
                        }
                    )
                    if fusion.mode == "residual":
                        arrays.update(
                            {
                                "fusion_residual_normalized": (
                                    fusion.normalized_residual[0].detach().cpu().numpy()
                                ),
                                "fusion_residual_physical": (
                                    fusion.physical_residual[0].detach().cpu().numpy()
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
                if evaluator.model.transverse_trunk_enabled:
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

    def _write_green_response_context_npz(
        self,
        evaluator: ComplexCouplingEvaluator,
    ) -> None:
        context = evaluator.column_diagonal_green_response_context
        if context is None or not self.request.save_generated_data:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "column_diagonal_green_response_fields.npz",
            gamma_x_squared=context.gamma_x_squared.detach().cpu().numpy(),
            gamma_y_squared=context.gamma_y_squared.detach().cpu().numpy(),
            regularized_gamma_x_squared=(
                context.regularized_gamma_x_squared.detach().cpu().numpy()
            ),
            regularized_gamma_y_squared=(
                context.regularized_gamma_y_squared.detach().cpu().numpy()
            ),
            correction_weight_phi=(
                context.correction_weight_phi.detach().cpu().numpy()
            ),
            correction_weight_psi=(
                context.correction_weight_psi.detach().cpu().numpy()
            ),
            gain_exponent=np.asarray(context.gain_exponent, dtype=np.float64),
        )

    def _write_tangent_response_context_npz(
        self,
        evaluator: ComplexCouplingEvaluator,
    ) -> None:
        context = evaluator.symmetric_tangent_green_response_context
        if context is None or not self.request.save_generated_data:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "gamma_x_squared": context.gamma_x_squared.detach().cpu().numpy(),
            "gamma_y_squared": context.gamma_y_squared.detach().cpu().numpy(),
            "cross_axis_inner_product": (
                context.cross_axis_inner_product.detach().cpu().numpy()
            ),
            "normalized_correlation": (
                context.normalized_correlation.detach().cpu().numpy()
            ),
            "normalized_quadratic_cross_axis": (
                context.normalized_quadratic_cross_axis.detach().cpu().numpy()
            ),
            "separable_preconditioner_base": (
                context.separable_preconditioner_base.detach().cpu().numpy()
            ),
            "exact_preconditioner_base": (
                context.exact_preconditioner_base.detach().cpu().numpy()
            ),
            "absolute_preconditioner_base": (
                context.absolute_preconditioner_base.detach().cpu().numpy()
            ),
            "quadratic_preconditioner_base": (
                context.quadratic_preconditioner_base.detach().cpu().numpy()
            ),
            "separable_denominator": (
                context.separable_denominator.detach().cpu().numpy()
            ),
            "exact_denominator": context.exact_denominator.detach().cpu().numpy(),
            "absolute_denominator": (
                context.absolute_denominator.detach().cpu().numpy()
            ),
            "quadratic_denominator": (
                context.quadratic_denominator.detach().cpu().numpy()
            ),
            "preconditioner_base": (context.preconditioner_base.detach().cpu().numpy()),
            "denominator": context.denominator.detach().cpu().numpy(),
            "preconditioner_variant": np.asarray(context.preconditioner_variant),
            "cross_axis_relative_eps": np.asarray(
                context.cross_axis_relative_eps,
                dtype=np.float64,
            ),
            "q_epsilon": context.q_epsilon.detach().cpu().numpy(),
            "damping": context.damping.detach().cpu().numpy(),
            "cauchy_violation": context.cauchy_violation.detach().cpu().numpy(),
            "cauchy_violation_max": (
                context.cauchy_violation_max.detach().cpu().numpy()
            ),
            "exact_roundoff_clamp_mask": (
                context.exact_roundoff_clamp_mask.detach().cpu().numpy()
            ),
            "exact_roundoff_clamp_count": np.asarray(
                context.exact_roundoff_clamp_count,
                dtype=np.int64,
            ),
            "point_mass": context.point_mass.detach().cpu().numpy(),
            "eta": np.asarray(context.eta, dtype=np.float64),
            "eta_strategy": np.asarray(context.eta_strategy),
            "line_search_relative_eps": np.asarray(
                context.line_search_relative_eps,
                dtype=np.float64,
            ),
            "relative_lambda": np.asarray(
                context.relative_lambda,
                dtype=np.float64,
            ),
            "denominator_relative_eps": np.asarray(
                context.denominator_relative_eps,
                dtype=np.float64,
            ),
        }
        if context.subspace_dimension >= 2:
            payload["subspace_dimension"] = np.asarray(
                context.subspace_dimension,
                dtype=np.int64,
            )
            payload["eta_applicability"] = np.asarray("k1_only_not_applied")
        np.savez(
            data_dir / "symmetric_tangent_green_response_fields.npz",
            **payload,  # type: ignore[arg-type]
        )

    def _write_green_response_context_figures(
        self,
        context: ColumnDiagonalGreenResponseContext | None,
        geometry: ComplexGeometryMetadata,
        theme: str,
        boundary_overlay: ComplexDomainBoundaryOverlay,
    ) -> tuple[list[str], tuple[str, ...]]:
        if context is None:
            return [], ()
        coords = geometry.coords_valid.detach().cpu().numpy()
        fields = {
            "gamma_x_squared": context.gamma_x_squared.detach().cpu().numpy(),
            "gamma_y_squared": context.gamma_y_squared.detach().cpu().numpy(),
            "correction_weight_phi": (
                context.correction_weight_phi.detach().cpu().numpy()
            ),
        }
        paths: list[str] = []
        for field, values in fields.items():
            figure = self._scatter_figure(
                title=(
                    f"{self.GREEN_RESPONSE_FIGURE_TITLES[field]} "
                    f"(fixed gain exponent={context.gain_exponent:g})"
                ),
                coords=coords,
                values=values,
                theme=theme,
                boundary_overlay=boundary_overlay,
            )
            base_path = self.request.outdir / "figures" / "balance_projection" / field
            save_plotly_figure(figure, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths, tuple(fields)

    def _write_tangent_response_context_figures(
        self,
        context: SymmetricTangentGreenResponseContext | None,
        geometry: ComplexGeometryMetadata,
        theme: str,
        boundary_overlay: ComplexDomainBoundaryOverlay,
    ) -> tuple[list[str], tuple[str, ...]]:
        if context is None:
            return [], ()
        coords = geometry.coords_valid.detach().cpu().numpy()
        fields = {
            "tangent_cross_axis_inner_product": (
                context.cross_axis_inner_product.detach().cpu().numpy()
            ),
            "tangent_normalized_correlation": (
                context.normalized_correlation.detach().cpu().numpy()
            ),
            "tangent_normalized_quadratic_cross_axis": (
                context.normalized_quadratic_cross_axis.detach().cpu().numpy()
            ),
            "tangent_preconditioner_base": (
                context.preconditioner_base.detach().cpu().numpy()
            ),
            "tangent_denominator": context.denominator.detach().cpu().numpy(),
        }
        paths: list[str] = []
        for field, values in fields.items():
            parameter_label = (
                f"K={context.subspace_dimension}, "
                f"variant={context.preconditioner_variant}"
                if context.subspace_dimension >= 2
                else f"eta={context.eta:g}, variant={context.preconditioner_variant}"
            )
            figure = self._scatter_figure(
                title=(
                    f"{self.GREEN_RESPONSE_FIGURE_TITLES[field]} ({parameter_label})"
                ),
                coords=coords,
                values=values,
                theme=theme,
                boundary_overlay=boundary_overlay,
            )
            base_path = self.request.outdir / "figures" / "balance_projection" / field
            save_plotly_figure(figure, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths, tuple(fields)

    @staticmethod
    def _projection_formula(projection: BalanceProjectionConfig) -> str:
        prefix = "p=P_raw/Lx^2; q=Q_raw/Ly^2; r=rhs-p-q; d=p-q; "
        suffix = "Phi=Lx^2*phi; Psi=Ly^2*psi"
        if projection.mode == "column_diagonal_green_response":
            column_config = ColumnDiagonalGreenResponseProjectionConfig.from_raw(
                projection.column_diagonal_green_response
            )
            return (
                prefix
                + "gx_bar=gamma_x^2+eps; gy_bar=gamma_y^2+eps; "
                + f"alpha={column_config.gain_exponent:g}; "
                + "w_phi=sigmoid(alpha*(log(gy_bar)-log(gx_bar))); "
                "d_star=d+(2*w_phi-1)*r; phi=(rhs+d_star)/2; "
                "psi=rhs-phi; " + suffix
            )
        if projection.mode == "symmetric_tangent_green_response":
            tangent_config = SymmetricTangentGreenResponseProjectionConfig.from_raw(
                projection.symmetric_tangent_green_response
            )
            if tangent_config.subspace_dimension >= 2:
                dimension = tangent_config.subspace_dimension
                update = (
                    "z0=D^-1*g; c0=argmin_c ||m0-c*S*z0||_M^2; "
                    "z1=orthogonalized(D^-1*(g-c0*A*z0)); "
                    "c1=argmin_c ||m0-c0*S*z0-c*S*z1||_M^2; "
                    + (
                        "for k>=2: zk=two_pass_response_MGS(D^-1*rk); "
                        "ck=argmin_c ||mk-c*S*zk||_M^2; "
                        if dimension >= 3
                        else ""
                    )
                    + f"delta=-sum(k=0..{dimension - 1}) ck*zk; "
                )
            elif tangent_config.eta_strategy == "closed_loop_exact_line_search":
                update = (
                    "z=g/D; v=(H_x+H_y)*z; "
                    "eta_star=(g^T*z)/(<v,v>_M+eps_relative); "
                    f"eta_applied=min(eta_star,{tangent_config.eta:g}); "
                    "delta=-eta_applied*z; "
                )
            else:
                update = f"delta=-{tangent_config.eta:g}*g/D; "
            return (
                prefix
                + "p_tilde=(rhs+d)/2; q_tilde=(rhs-d)/2; "
                + "m0=H_x*p_tilde-H_y*q_tilde; "
                + "g=(H_x+H_y)^T*M_Omega*m0; "
                + ComplexCouplingArtifactExporter._tangent_preconditioner_formula(
                    tangent_config
                )
                + "; "
                + update
                + "phi=p_tilde+delta; psi=q_tilde-delta; "
                + suffix
            )
        return prefix + "phi=(rhs+d)/2; psi=(rhs-d)/2; " + suffix

    @staticmethod
    def _tangent_preconditioner_formula(
        config: SymmetricTangentGreenResponseProjectionConfig,
    ) -> str:
        base = {
            "separable": "B=a+b",
            "exact_diagonal": "B=a+b+2*c",
            "absolute_cross_axis": "B=a+b+2*abs(c)",
            "normalized_quadratic_cross_axis": (
                "B=a+b+4*c^2/(a+b+cross_axis_relative_eps*mean(a+b))"
            ),
        }[config.preconditioner_variant]
        return (
            "a=diag(H_x^T*M*H_x); b=diag(H_y^T*M*H_y); "
            "c=diag(H_x^T*M*H_y); "
            + base
            + "; D=B+(relative_lambda+denominator_relative_eps)*mean(a+b)"
        )

    def _green_response_context_summary(
        self,
        context: ColumnDiagonalGreenResponseContext | None,
        evaluator: ComplexCouplingEvaluator,
        projection: BalanceProjectionConfig,
    ) -> dict[str, Any]:
        active = projection.mode == "column_diagonal_green_response"
        column_config = ColumnDiagonalGreenResponseProjectionConfig.from_raw(
            projection.column_diagonal_green_response
        )
        summary: dict[str, Any] = {
            "active": active,
            "gain_squared_eps": column_config.gain_squared_eps,
            "gain_exponent": column_config.gain_exponent,
            "fixed_exponent": True,
            "learnable_exponent": False,
            "weight_formula": (
                "w_phi=sigmoid(alpha*(log(gamma_y_squared+eps)-"
                "log(gamma_x_squared+eps))); w_psi=1-w_phi"
            ),
            "alpha_zero_endpoint": "physical_symmetric_correction",
            "alpha_one_endpoint": "legacy_column_diagonal_correction",
            "alpha_one_implementation": "direct_regularized_gain_ratio",
            "gain_definition": "diag(H_s^T M_Omega H_s)",
            "operator_definition": "H_s=K_s W_s L_s^2",
            "mass_definition": "M_Omega=(hx*hy)I_valid",
            "summation_axis": "output_rows_for_each_source_column",
            "row_norm_used": False,
            "full_gram_solve": False,
            "global_response_matrix_materialized": False,
            "context_build_count": (
                evaluator.column_diagonal_green_response_context_build_count
            ),
            "context_build_seconds": (
                evaluator.column_diagonal_green_response_context_build_seconds
            ),
            "raw_archive": (
                "data/column_diagonal_green_response_fields.npz"
                if active
                and evaluator.column_diagonal_green_response_context is not None
                and self.request.save_generated_data
                else None
            ),
        }
        if context is not None:
            summary["statistics"] = context.statistics()
            summary["point_mass"] = float(context.point_mass.item())
        return summary

    def _tangent_response_context_summary(
        self,
        context: SymmetricTangentGreenResponseContext | None,
        evaluator: ComplexCouplingEvaluator,
        projection: BalanceProjectionConfig,
        training_config: CouplingTrainingConfig,
        metric_rows: list[dict[str, float | int | str]],
    ) -> dict[str, Any]:
        active = projection.mode == "symmetric_tangent_green_response"
        config = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        geometry_k_selection = GeometryKSelectionConfig.from_raw(
            config.geometry_k_selection
        )
        subspace = config.subspace_dimension >= 2
        configured_lr_schedule = CouplingLearningRateSchedule.configured_config(
            training_config
        )
        adaptive = config.eta_strategy == "closed_loop_exact_line_search"
        if subspace:
            update = (
                "z0=D^-1*g0; c0=(g0^T*z0)/(<S*z0,S*z0>_M+eps0); "
                "r1=g0-c0*S^T*M*S*z0; z1_raw=D^-1*r1; "
                "z1=response_orthogonalize(z1_raw,z0); "
                "c1=(g0^T*z1)/(<S*z1,S*z1>_M+eps1); "
                + (
                    "for k>=2: zk=two_pass_response_MGS(D^-1*rk); "
                    "ck=(rk^T*zk)/(<S*zk,S*zk>_M+epsk); "
                    if config.subspace_dimension >= 3
                    else ""
                )
                + f"delta=-sum(k=0..{config.subspace_dimension - 1}) ck*zk; "
                "phi=p_tilde+delta; psi=q_tilde-delta"
            )
        elif adaptive:
            update = (
                "z=g/D; v=(H_x+H_y)z; "
                "eta_star=(g^T z)/(<v,v>_M+relative_eps); "
                "eta_applied=min(eta_star,eta_cap); "
                "delta=-eta_applied*z; phi=p_tilde+delta; psi=q_tilde-delta"
            )
        else:
            update = "delta=-eta*g/D; phi=p_tilde+delta; psi=q_tilde-delta"
        if subspace:
            eta_schedule_summary: dict[str, Any] = {
                "applicable": False,
                "kind": f"not_applicable_k{config.subspace_dimension}",
                "reason": (
                    f"K={config.subspace_dimension} uses unconstrained exact "
                    "subspace coefficients"
                ),
                "training_policy": "not_applied",
                "validation_policy": "not_applied",
                "evaluation_policy": "not_applied",
                "artifact_policy": "not_applied",
            }
        else:
            eta_schedule_enabled = bool(adaptive and configured_lr_schedule["enabled"])
            configured_warmup_steps = configured_lr_schedule["configured_warmup_steps"]
            raw_warmup_epochs = configured_lr_schedule["configured_warmup_epochs"]
            if not isinstance(raw_warmup_epochs, int) or isinstance(
                raw_warmup_epochs, bool
            ):
                raise RuntimeError(
                    "Configured CouplingNet warmup epochs must be an integer."
                )
            configured_warmup_epochs = raw_warmup_epochs
            has_configured_warmup = (
                isinstance(configured_warmup_steps, int) and configured_warmup_steps > 0
            ) or configured_warmup_epochs > 0
            eta_schedule_summary = {
                "applicable": True,
                "kind": (
                    "closed_loop_configured_half_cosine_warmup_hold"
                    if eta_schedule_enabled and has_configured_warmup
                    else ("closed_loop_final_cap" if adaptive else "fixed_eta")
                ),
                "final_eta": config.eta,
                "shared_with_lr_warmup": eta_schedule_enabled,
                "warmup_source": configured_lr_schedule["warmup_source"],
                "configured_warmup_epochs": configured_warmup_epochs,
                "configured_warmup_steps": configured_warmup_steps,
                "effective_warmup_steps": None,
                "resolution": "runtime_after_dataloader",
                "post_warmup_behavior": "hold_final_eta",
                "training_policy": "scheduled_cap",
                "validation_policy": "final_cap",
                "evaluation_policy": "final_cap",
                "artifact_policy": "final_cap",
            }
        summary: dict[str, Any] = {
            "active": active,
            "subspace_dimension": config.subspace_dimension,
            "max_subspace_dimension": config.max_subspace_dimension,
            "geometry_k_selection_enabled_at_runtime": (geometry_k_selection.enabled),
            "eta": config.eta,
            "eta_role": (
                "k1_only_not_applied"
                if subspace
                else ("final_safety_cap" if adaptive else "fixed_step")
            ),
            "eta_applicability": ("k1_only_not_applied" if subspace else "applied"),
            "eta_strategy": config.eta_strategy,
            "line_search_relative_eps": config.line_search_relative_eps,
            "relative_lambda": config.relative_lambda,
            "denominator_relative_eps": config.denominator_relative_eps,
            "preconditioner_variant": config.preconditioner_variant,
            "cross_axis_relative_eps": config.cross_axis_relative_eps,
            "fixed_parameters": not adaptive and not subspace,
            "sample_adaptive": adaptive or subspace,
            "batch_independent": True,
            "differentiable_eta": adaptive and not subspace,
            "differentiable_subspace_coefficients": subspace,
            "learnable_parameters": False,
            "reference_targets_used": False,
            "base_projection": "physical_symmetric",
            "objective": "0.5*||H_x*phi-H_y*psi||_M_Omega^2",
            "gradient": "g=(H_x+H_y)^T*M_Omega*m0",
            "update": update,
            "eta_cap_schedule": eta_schedule_summary,
            "preconditioner": self._tangent_preconditioner_formula(config),
            "preconditioner_suite": [
                "separable",
                "exact_diagonal",
                "absolute_cross_axis",
                "normalized_quadratic_cross_axis",
            ],
            "gain_definition": "diag(H_s^T M_Omega H_s)",
            "operator_definition": "H_s=K_s W_s L_s^2",
            "row_norm_used": False,
            "global_response_matrix_materialized": False,
            "full_gram_solve": False,
            "linear_solve_used": False,
            "direction_contract": (
                "two_jacobi_preconditioned_response_orthogonal_directions"
                if config.subspace_dimension == 2
                else (
                    f"{config.subspace_dimension}_jacobi_preconditioned_"
                    "response_orthogonal_directions"
                    if subspace
                    else "one_jacobi_preconditioned_direction"
                )
            ),
            "orthogonalization": (
                "existing_k2_seed_then_two_pass_modified_gram_schmidt"
                if config.subspace_dimension >= 3
                else ("existing_k2_response_orthogonalization" if subspace else None)
            ),
            "context_build_count": (
                evaluator.symmetric_tangent_green_response_context_build_count
            ),
            "context_build_seconds": (
                evaluator.symmetric_tangent_green_response_context_build_seconds
            ),
            "context_checkpoint": (
                evaluator.symmetric_tangent_green_response_context_telemetry
            ),
            "raw_archive": (
                "data/symmetric_tangent_green_response_fields.npz"
                if active and context is not None and self.request.save_generated_data
                else None
            ),
        }
        if context is not None:
            summary["statistics"] = context.statistics()
            summary["point_mass"] = float(context.point_mass.item())
        if subspace:
            metric_summaries: list[tuple[str, str]] = []
            for direction_index in range(config.subspace_dimension):
                metric_summaries.extend(
                    (
                        (
                            f"tangent_coefficient_{direction_index}",
                            f"coefficient_{direction_index}_statistics",
                        ),
                        (
                            f"tangent_response_cost_k{direction_index + 1}",
                            f"response_cost_k{direction_index + 1}_statistics",
                        ),
                    )
                )
                if direction_index > 0:
                    metric_summaries.append(
                        (
                            f"tangent_response_cost_k{direction_index + 1}_over_k"
                            f"{direction_index}",
                            f"response_cost_k{direction_index + 1}_over_k"
                            f"{direction_index}_statistics",
                        )
                    )
            metric_summaries.append(
                (
                    "tangent_response_orthogonality_max",
                    "response_orthogonality_max_statistics",
                )
            )
            for metric_key, summary_key in metric_summaries:
                values = np.asarray(
                    [
                        float(row[metric_key])
                        for row in metric_rows
                        if metric_key in row
                    ],
                    dtype=np.float64,
                )
                if values.size:
                    summary[summary_key] = self._scalar_distribution(values)
            active_values = np.asarray(
                [
                    float(row["tangent_second_direction_active"])
                    for row in metric_rows
                    if "tangent_second_direction_active" in row
                ],
                dtype=np.float64,
            )
            if active_values.size:
                summary["second_direction_active_fraction"] = float(
                    active_values.mean()
                )
            for direction_index in range(config.subspace_dimension):
                active_key = f"tangent_direction_{direction_index}_active"
                active_values = np.asarray(
                    [
                        float(row[active_key])
                        for row in metric_rows
                        if active_key in row
                    ],
                    dtype=np.float64,
                )
                if active_values.size:
                    summary[f"direction_{direction_index}_active_fraction"] = float(
                        active_values.mean()
                    )
            return summary
        eta_star = np.asarray(
            [
                float(row["tangent_eta_star"])
                for row in metric_rows
                if "tangent_eta_star" in row
            ],
            dtype=np.float64,
        )
        eta_applied = np.asarray(
            [
                float(row["tangent_eta_applied"])
                for row in metric_rows
                if "tangent_eta_applied" in row
            ],
            dtype=np.float64,
        )
        capped = np.asarray(
            [
                float(row["tangent_eta_capped"])
                for row in metric_rows
                if "tangent_eta_capped" in row
            ],
            dtype=np.float64,
        )
        if eta_star.size:
            summary["eta_star_statistics"] = self._scalar_distribution(eta_star)
        if eta_applied.size:
            summary["eta_applied_statistics"] = self._scalar_distribution(eta_applied)
        if capped.size:
            summary["eta_cap_hit_fraction"] = float(capped.mean())
        return summary

    @staticmethod
    def _scalar_distribution(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }

    def _write_figures(
        self,
        selected_samples: list[ComplexSelectedSample],
        theme: str,
        boundary_overlay: ComplexDomainBoundaryOverlay,
        color_ranges_by_sample: dict[
            int,
            dict[str, ComplexArtifactColorRange],
        ],
    ) -> list[str]:
        figure_paths: list[str] = []
        for sample in selected_samples:
            stem = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            color_ranges = color_ranges_by_sample[sample.sample_id]
            for field in self._figure_fields_for_sample(sample.arrays):
                fig = self._scatter_figure(
                    title=f"{stem} {self._figure_title(field)}",
                    coords=sample.arrays["coords_valid"],
                    values=sample.arrays[field],
                    theme=theme,
                    signed=self._figure_is_signed(field),
                    color_range=color_ranges.get(field),
                    boundary_overlay=boundary_overlay,
                )
                base_path = self.request.outdir / "figures" / field / f"{stem}_{field}"
                save_plotly_figure(fig, base_path, logger=self.logger)
                figure_paths.append(str(base_path.with_suffix(".json")))
        return figure_paths

    def _color_ranges_for_sample(
        self,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, ComplexArtifactColorRange]:
        ranges: dict[str, ComplexArtifactColorRange] = {}
        solution_range = self._shared_color_range(
            arrays,
            self.COLOR_RANGE_GROUPS["solution"],
            group="solution",
            policy="full_min_max_including_zero",
            quantile=1.0,
            include_zero=True,
        )
        if solution_range is not None:
            for field in self.COLOR_RANGE_GROUPS["solution"]:
                if field in arrays:
                    ranges[field] = solution_range

        quantile = self.directional_color_quantile
        for group in ("phi", "psi"):
            fields = self.COLOR_RANGE_GROUPS[group]
            color_range = self._shared_color_range(
                arrays,
                fields,
                group=group,
                policy="shared_lower_upper_quantile",
                quantile=quantile,
            )
            if color_range is not None:
                for field in fields:
                    if field in arrays:
                        ranges[field] = color_range

        for field in self.DIRECTIONAL_ERROR_FIELDS:
            color_range = self._shared_color_range(
                arrays,
                (field,),
                group=field,
                policy="symmetric_absolute_quantile",
                quantile=quantile,
                symmetric=True,
            )
            if color_range is not None:
                ranges[field] = color_range

        for field, symmetric in (("rhs", False), ("u_pred_error", True)):
            color_range = self._shared_color_range(
                arrays,
                (field,),
                group=field,
                policy="symmetric_full_range" if symmetric else "full_min_max",
                quantile=1.0,
                symmetric=symmetric,
            )
            if color_range is not None:
                ranges[field] = color_range
        return ranges

    @staticmethod
    def _shared_color_range(
        arrays: dict[str, np.ndarray],
        fields: tuple[str, ...],
        *,
        group: str,
        policy: str,
        quantile: float,
        symmetric: bool = False,
        include_zero: bool = False,
    ) -> ComplexArtifactColorRange | None:
        finite_values: list[np.ndarray] = []
        if include_zero:
            finite_values.append(np.asarray([0.0], dtype=np.float64))
        for field in fields:
            if field not in arrays:
                continue
            values = np.asarray(arrays[field])
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)
        if not finite_values:
            return None
        joined = np.concatenate(finite_values)
        full_min = float(np.min(joined))
        full_max = float(np.max(joined))
        if full_min == full_max:
            cmin: float | None = None
            cmax: float | None = None
        elif symmetric:
            maximum = float(np.quantile(np.abs(joined), quantile))
            if maximum == 0.0:
                maximum = max(abs(full_min), abs(full_max))
            cmin = -maximum
            cmax = maximum
        else:
            cmin = float(np.quantile(joined, 1.0 - quantile))
            cmax = float(np.quantile(joined, quantile))
            if cmin == cmax:
                cmin = full_min
                cmax = full_max
        return ComplexArtifactColorRange(
            group=group,
            policy=policy,
            quantile=quantile,
            cmin=cmin,
            cmax=cmax,
        )

    def _directional_color_range_summary(
        self,
        selected_samples: list[ComplexSelectedSample],
        color_ranges_by_sample: dict[
            int,
            dict[str, ComplexArtifactColorRange],
        ],
    ) -> dict[str, Any]:
        samples: dict[str, dict[str, dict[str, float | int | str]]] = {}
        for sample in selected_samples:
            stem = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            ranges = color_ranges_by_sample[sample.sample_id]
            sample_summary: dict[str, dict[str, float | int | str]] = {}
            for field in self.DIRECTIONAL_COLOR_SUMMARY_FIELDS:
                if field in sample.arrays and field in ranges:
                    sample_summary[field] = ranges[field].field_summary(
                        sample.arrays[field]
                    )
            samples[stem] = sample_summary
        return {
            "configured_quantile": self.request.directional_color_quantile,
            "resolved_quantile": self.directional_color_quantile,
            "value_policy": "shared_lower_upper_quantile",
            "error_policy": "symmetric_absolute_quantile",
            "source_policy": "full_min_max",
            "quantile_method": "numpy_linear",
            "input_points": "finite_coords_valid_only",
            "raw_values_modified": False,
            "samples": samples,
        }

    @classmethod
    def _figure_fields_for_sample(
        cls, arrays: dict[str, np.ndarray]
    ) -> tuple[str, ...]:
        configured = tuple(field for field in cls.FIGURE_FIELDS if field in arrays)
        dynamic = tuple(
            sorted(
                field
                for field in arrays
                if field not in configured
                and (
                    field.startswith("tangent_direction_")
                    or field.startswith("tangent_response_direction_")
                )
                and field.rsplit("_", maxsplit=1)[-1].isdigit()
            )
        )
        return configured + dynamic

    @classmethod
    def _figure_title(cls, field: str) -> str:
        if field in cls.FIGURE_TITLES:
            return cls.FIGURE_TITLES[field]
        if field.startswith("tangent_direction_"):
            index = int(field.rsplit("_", maxsplit=1)[-1])
            return f"Jacobi-preconditioned tangent direction {index}"
        if field.startswith("tangent_response_direction_"):
            index = int(field.rsplit("_", maxsplit=1)[-1])
            return f"Response of tangent direction {index}"
        raise KeyError(f"No figure title is defined for field '{field}'.")

    @classmethod
    def _figure_is_signed(cls, field: str) -> bool:
        return field in cls.SIGNED_FIGURE_FIELDS or field.startswith(
            ("tangent_direction_", "tangent_response_direction_")
        )

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
        color_range: ComplexArtifactColorRange | None = None,
        boundary_overlay: ComplexDomainBoundaryOverlay,
    ) -> go.Figure:
        finite_values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        marker_color_range = {} if color_range is None else color_range.plotly_kwargs()
        if color_range is None and signed and max_abs > 0.0:
            marker_color_range = {"cmin": -max_abs, "cmax": max_abs}
        figure = go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                customdata=np.asarray(values).reshape(-1, 1),
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": "RdBu" if signed else "Viridis",
                    "showscale": True,
                    "size": 6,
                    "colorbar": {"exponentformat": "power", "showexponent": "all"},
                    **marker_color_range,
                },
                hovertemplate=(
                    "x=%{x:.6g}<br>y=%{y:.6g}<br>value=%{customdata[0]:.6e}"
                    "<extra></extra>"
                ),
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
        boundary_overlay.add_to_figure(figure)
        return figure

    @staticmethod
    def _aggregate_metrics(
        metric_rows: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        for key in (
            "loss",
            "loss_energy_optimized",
            "loss_energy_consistency",
            "loss_energy_bulk",
            "loss_energy_boundary",
            "loss_energy_boundary_x",
            "loss_energy_boundary_y",
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
