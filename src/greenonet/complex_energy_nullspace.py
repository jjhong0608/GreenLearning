from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from greenonet.complex_geometry import (
    ComplexGeometryMetadata,
    load_complex_geometry,
)
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    build_boundary_energy_context,
)


@dataclass(frozen=True)
class ComplexEnergyNullspaceRequest:
    geometry: Path
    outdir: Path
    rank_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_tolerance) or self.rank_tolerance <= 0.0:
            raise ValueError("rank_tolerance must be finite and positive.")


@dataclass(frozen=True)
class ConstraintFamilySummary:
    name: str
    row_count: int
    component_reduced_rank: int
    component_imbalance_max: float
    annihilates_bulk_component_constants: bool


@dataclass(frozen=True)
class NullspaceStageSummary:
    name: str
    additional_families: tuple[str, ...]
    total_constraint_rows: int
    reduced_rank: int
    nullity: int


@dataclass(frozen=True)
class _ConstraintFamily:
    name: str
    matrix: sparse.csr_matrix


class ConstraintAssemblyMixin:
    geometry: ComplexGeometryMetadata
    boundary_context: ComplexBoundaryEnergyContext

    @staticmethod
    def _stencil_matrix(
        *,
        point_count: int,
        indices: np.ndarray,
        weights: np.ndarray,
    ) -> sparse.csr_matrix:
        if indices.shape != weights.shape:
            raise ValueError(
                "Constraint indices and weights must have matching shapes."
            )
        if indices.ndim != 2:
            raise ValueError("Constraint stencils must be rank-two arrays.")
        if indices.shape[0] == 0:
            return sparse.csr_matrix((0, point_count), dtype=np.float64)
        rows = np.repeat(np.arange(indices.shape[0]), indices.shape[1])
        return sparse.coo_matrix(
            (weights.reshape(-1), (rows, indices.reshape(-1))),
            shape=(indices.shape[0], point_count),
            dtype=np.float64,
        ).tocsr()

    def _bulk_constraints(self) -> _ConstraintFamily:
        x_edges = self.geometry.x_edges.detach().cpu().numpy().astype(np.int64)
        y_edges = self.geometry.y_edges.detach().cpu().numpy().astype(np.int64)
        edges = np.concatenate((x_edges, y_edges), axis=0)
        x_scale = math.sqrt(float((self.geometry.hy / self.geometry.hx).item()))
        y_scale = math.sqrt(float((self.geometry.hx / self.geometry.hy).item()))
        scales = np.concatenate(
            (
                np.full(x_edges.shape[0], x_scale, dtype=np.float64),
                np.full(y_edges.shape[0], y_scale, dtype=np.float64),
            )
        )
        matrix = self._stencil_matrix(
            point_count=self.geometry.num_points,
            indices=edges,
            weights=np.column_stack((-scales, scales)),
        )
        return _ConstraintFamily(name="bulk_edge_gradient", matrix=matrix)

    def _general_boundary_constraints(self) -> _ConstraintFamily:
        indices = (
            self.boundary_context.point_indices.detach()
            .cpu()
            .numpy()
            .astype(np.int64)[:, None]
        )
        scale = torch_to_numpy_sqrt_weight(self.boundary_context)[:, None]
        matrix = self._stencil_matrix(
            point_count=self.geometry.num_points,
            indices=indices,
            weights=scale,
        )
        return _ConstraintFamily(
            name="general_segment_boundary_anchor",
            matrix=matrix,
        )


def torch_to_numpy_sqrt_weight(
    context: ComplexBoundaryEnergyContext,
) -> np.ndarray:
    weight = context.transverse_measure / context.physical_distance
    return np.asarray(
        np.sqrt(weight.detach().cpu().numpy()),
        dtype=np.float64,
    )


class NullspaceReportMixin:
    request: ComplexEnergyNullspaceRequest

    @staticmethod
    def _write_stage_csv(
        path: Path,
        stages: tuple[NullspaceStageSummary, ...],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "name",
                    "additional_families",
                    "total_constraint_rows",
                    "reduced_rank",
                    "nullity",
                ),
            )
            writer.writeheader()
            for stage in stages:
                row = asdict(stage)
                row["additional_families"] = ",".join(stage.additional_families)
                writer.writerow(row)

    @staticmethod
    def _markdown_report(summary: dict[str, Any]) -> str:
        stage_rows = [
            (
                f"| `{stage['name']}` | {stage['total_constraint_rows']} | "
                f"{stage['reduced_rank']} | {stage['nullity']} |"
            )
            for stage in summary["stages"]
        ]
        conclusions = summary["conclusions"]
        return "\n".join(
            (
                "# Complex Canonical Energy Null-Space Analysis",
                "",
                f"- Geometry: `{summary['geometry_path']}`",
                f"- Valid-point DOFs: {summary['geometry']['num_points']}",
                f"- Bulk graph components: {summary['geometry']['bulk_components']}",
                (
                    "- General segment boundary anchors: "
                    f"{summary['geometry']['general_boundary_anchors']}"
                ),
                "",
                "## Stage Nullity",
                "",
                "| Stage | Constraint rows | Reduced rank | Nullity |",
                "|---|---:|---:|---:|",
                *stage_rows,
                "",
                "## Decisions",
                "",
                (
                    "- General segment boundary anchors required: "
                    f"`{conclusions['general_boundary_anchor_required']}`."
                ),
                (
                    "- General segment boundary anchors sufficient: "
                    f"`{conclusions['general_boundary_anchor_sufficient']}`."
                ),
                (
                    "- Carrier objective required: "
                    f"`{conclusions['carrier_objective_required']}`."
                ),
                "",
                "## Production Contract",
                "",
                (
                    "The canonical residual energy is the physical bulk edge energy "
                    "plus the endpoint-to-nearest-interior P1 edge energy for both "
                    "ends of every connected x/y segment. No self-trace or cross-axis "
                    "carrier objective is part of this coercivity contract."
                ),
                "",
            )
        )

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stages = tuple(NullspaceStageSummary(**stage) for stage in summary["stages"])
        self._write_stage_csv(self.request.outdir / "nullspace_stages.csv", stages)
        (self.request.outdir / "analysis_report.md").write_text(
            self._markdown_report(summary),
            encoding="utf-8",
        )


class ComplexEnergyNullspaceAnalyzer(
    ConstraintAssemblyMixin,
    NullspaceReportMixin,
):
    """Analyze the canonical bulk-plus-boundary residual energy null space."""

    def __init__(
        self,
        request: ComplexEnergyNullspaceRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger or logging.getLogger(__name__)
        if not request.geometry.exists():
            raise FileNotFoundError(f"Geometry file does not exist: {request.geometry}")
        self.geometry = load_complex_geometry(request.geometry)
        self.boundary_context = build_boundary_energy_context(self.geometry)

    def run(self) -> dict[str, Any]:
        bulk = self._bulk_constraints()
        boundary = self._general_boundary_constraints()
        families = {family.name: family for family in (bulk, boundary)}
        component_count, membership = self._bulk_component_membership()
        family_summaries = {
            name: self._family_summary(family, membership)
            for name, family in families.items()
        }
        stages = (
            self._stage_summary(
                name="bulk",
                family_names=(),
                families=families,
                membership=membership,
                component_count=component_count,
                bulk_row_count=bulk.matrix.shape[0],
            ),
            self._stage_summary(
                name="bulk_plus_general_boundary",
                family_names=(boundary.name,),
                families=families,
                membership=membership,
                component_count=component_count,
                bulk_row_count=bulk.matrix.shape[0],
            ),
        )
        stage_by_name = {stage.name: stage for stage in stages}
        bulk_nullity = stage_by_name["bulk"].nullity
        boundary_nullity = stage_by_name["bulk_plus_general_boundary"].nullity
        conclusions = {
            "general_boundary_anchor_required": (
                bulk_nullity > 0 and boundary_nullity < bulk_nullity
            ),
            "general_boundary_anchor_sufficient": boundary_nullity == 0,
            "carrier_objective_required": boundary_nullity > 0,
        }
        summary: dict[str, Any] = {
            "schema_version": 2,
            "geometry_path": str(self.request.geometry.resolve()),
            "geometry": {
                "num_points": self.geometry.num_points,
                "num_x_segments": self.geometry.num_x_segments,
                "num_y_segments": self.geometry.num_y_segments,
                "num_x_edges": int(self.geometry.x_edges.shape[0]),
                "num_y_edges": int(self.geometry.y_edges.shape[0]),
                "bulk_components": component_count,
                "general_boundary_anchors": self.boundary_context.total_anchors,
                "x_boundary_anchors": self.boundary_context.x_anchor_count,
                "y_boundary_anchors": self.boundary_context.y_anchor_count,
            },
            "rank_tolerance": self.request.rank_tolerance,
            "constraint_families": {
                name: asdict(family_summary)
                for name, family_summary in family_summaries.items()
            },
            "stages": [
                {
                    **asdict(stage),
                    "additional_families": list(stage.additional_families),
                }
                for stage in stages
            ],
            "conclusions": conclusions,
        }
        self._write_outputs(summary)
        self.logger.info(
            "Canonical null-space analysis components=%d bulk_nullity=%d "
            "boundary_nullity=%d anchors=%d",
            component_count,
            bulk_nullity,
            boundary_nullity,
            self.boundary_context.total_anchors,
        )
        return summary

    def _bulk_component_membership(
        self,
    ) -> tuple[int, sparse.csr_matrix]:
        edges = np.concatenate(
            (
                self.geometry.x_edges.detach().cpu().numpy(),
                self.geometry.y_edges.detach().cpu().numpy(),
            ),
            axis=0,
        ).astype(np.int64)
        point_count = self.geometry.num_points
        adjacency = sparse.coo_matrix(
            (
                np.ones(edges.shape[0] * 2, dtype=np.float64),
                (
                    np.concatenate((edges[:, 0], edges[:, 1])),
                    np.concatenate((edges[:, 1], edges[:, 0])),
                ),
            ),
            shape=(point_count, point_count),
        ).tocsr()
        component_count, labels = connected_components(
            adjacency,
            directed=False,
            return_labels=True,
        )
        membership = sparse.coo_matrix(
            (
                np.ones(point_count, dtype=np.float64),
                (np.arange(point_count), labels),
            ),
            shape=(point_count, component_count),
        ).tocsr()
        return int(component_count), membership

    def _family_summary(
        self,
        family: _ConstraintFamily,
        membership: sparse.csr_matrix,
    ) -> ConstraintFamilySummary:
        reduced = (family.matrix @ membership).toarray()
        rank = self._matrix_rank(reduced)
        imbalance = float(np.max(np.abs(reduced))) if reduced.size else 0.0
        return ConstraintFamilySummary(
            name=family.name,
            row_count=family.matrix.shape[0],
            component_reduced_rank=rank,
            component_imbalance_max=imbalance,
            annihilates_bulk_component_constants=(
                imbalance <= self.request.rank_tolerance
            ),
        )

    def _stage_summary(
        self,
        *,
        name: str,
        family_names: tuple[str, ...],
        families: dict[str, _ConstraintFamily],
        membership: sparse.csr_matrix,
        component_count: int,
        bulk_row_count: int,
    ) -> NullspaceStageSummary:
        additional = [families[family_name].matrix for family_name in family_names]
        if additional:
            reduced = (sparse.vstack(additional, format="csr") @ membership).toarray()
            reduced_rank = self._matrix_rank(reduced)
        else:
            reduced_rank = 0
        return NullspaceStageSummary(
            name=name,
            additional_families=family_names,
            total_constraint_rows=(
                bulk_row_count + sum(matrix.shape[0] for matrix in additional)
            ),
            reduced_rank=reduced_rank,
            nullity=component_count - reduced_rank,
        )

    def _matrix_rank(self, matrix: np.ndarray) -> int:
        if matrix.size == 0:
            return 0
        return int(np.linalg.matrix_rank(matrix, tol=self.request.rank_tolerance))


def analyze_complex_energy_nullspace(
    request: ComplexEnergyNullspaceRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexEnergyNullspaceAnalyzer(request, logger=logger).run()
