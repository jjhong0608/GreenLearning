from __future__ import annotations

import hashlib
import importlib
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from greenonet.fenicsx_samples.domain import (
    GmshDomainContext,
    GmshDomainParser,
    GmshScriptLoader,
)
from greenonet.fenicsx_samples.geometry import (
    GeometryGridLoader,
    RawComplexGeometryGrid,
)


@dataclass(frozen=True)
class ComplexVisualizationMesh:
    """Reusable conforming mesh and scalar-value transfer metadata."""

    SCHEMA_VERSION: ClassVar[int] = 1
    MAPPING_TOLERANCE: ClassVar[float] = 1.0e-10
    MIN_INTERPOLATION_NEIGHBORS: ClassVar[int] = 3
    MAX_INTERPOLATION_NEIGHBORS: ClassVar[int] = 8

    vertices: np.ndarray
    triangles: np.ndarray
    boundary_edges: np.ndarray
    valid_to_vertex: np.ndarray
    boundary_vertex_mask: np.ndarray
    auxiliary_vertex_mask: np.ndarray
    aux_interp_ptr: np.ndarray
    aux_interp_vertex_index: np.ndarray
    aux_interp_weight: np.ndarray
    geometry_sha256: str
    gmsh_script_sha256: str
    gmsh_version: str
    boundary_size_factor: float
    max_auxiliary_fraction: float
    schema_version: int = SCHEMA_VERSION

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def valid_vertex_count(self) -> int:
        return int(self.valid_to_vertex.size)

    @property
    def boundary_vertex_count(self) -> int:
        return int(np.count_nonzero(self.boundary_vertex_mask))

    @property
    def auxiliary_vertex_count(self) -> int:
        return int(np.count_nonzero(self.auxiliary_vertex_mask))

    @property
    def auxiliary_fraction(self) -> float:
        return self.auxiliary_vertex_count / max(self.vertex_count, 1)

    @property
    def auxiliary_vertices(self) -> np.ndarray:
        return np.flatnonzero(self.auxiliary_vertex_mask).astype(np.int64)

    def validate(self, coords_valid: np.ndarray | None = None) -> None:
        self._validate_provenance()
        self._validate_arrays()
        self._validate_partition()
        self._validate_interpolation()
        if coords_valid is not None:
            self._validate_valid_coordinates(coords_valid)

    def transfer_solution(self, values: np.ndarray) -> np.ndarray:
        """Map valid-point solution values onto the visualization mesh."""

        source = self._validate_source_values(values, label="Solution")

        transferred = np.zeros(
            (*source.shape[:-1], self.vertex_count),
            dtype=source.dtype,
        )
        transferred[..., self.valid_to_vertex] = source
        for aux_offset, vertex_index in enumerate(self.auxiliary_vertices):
            start = int(self.aux_interp_ptr[aux_offset])
            end = int(self.aux_interp_ptr[aux_offset + 1])
            stencil = self.aux_interp_vertex_index[start:end]
            weights = self.aux_interp_weight[start:end].astype(
                source.dtype,
                copy=False,
            )
            transferred[..., vertex_index] = np.sum(
                transferred[..., stencil] * weights,
                axis=-1,
            )
        return transferred

    def transfer_interior_cell_values(self, values: np.ndarray) -> np.ndarray:
        """Map interior scalar values to cells without inventing boundary values."""

        source = self._validate_source_values(values, label="Interior scalar")
        transferred = np.full(
            (*source.shape[:-1], self.vertex_count),
            np.nan,
            dtype=source.dtype,
        )
        transferred[..., self.valid_to_vertex] = source
        for aux_offset, vertex_index in enumerate(self.auxiliary_vertices):
            start = int(self.aux_interp_ptr[aux_offset])
            end = int(self.aux_interp_ptr[aux_offset + 1])
            stencil = self.aux_interp_vertex_index[start:end]
            weights = self.aux_interp_weight[start:end].astype(
                source.dtype,
                copy=False,
            )
            interior = ~self.boundary_vertex_mask[stencil]
            if not np.any(interior):
                raise ValueError(
                    "Auxiliary interpolation has no non-boundary source vertex."
                )
            stencil = stencil[interior]
            weights = weights[interior]
            weights = weights / np.sum(weights)
            transferred[..., vertex_index] = np.sum(
                transferred[..., stencil] * weights,
                axis=-1,
            )

        non_boundary = ~self.boundary_vertex_mask[self.triangles]
        counts = np.sum(non_boundary, axis=1)
        if np.any(counts == 0):
            count = int(np.count_nonzero(counts == 0))
            raise ValueError(
                "Interior scalar mesh contains "
                f"{count} triangle(s) with only boundary vertices; regenerate "
                "the visualization mesh with a finer boundary size."
            )
        cell_vertices = transferred[..., self.triangles]
        total = np.sum(
            np.where(non_boundary, cell_vertices, 0.0),
            axis=-1,
        )
        return np.asarray(
            total / counts.astype(source.dtype, copy=False),
            dtype=source.dtype,
        )

    def _validate_source_values(self, values: np.ndarray, *, label: str) -> np.ndarray:
        source = np.asarray(values)
        if source.ndim < 1 or source.shape[-1] != self.valid_vertex_count:
            raise ValueError(
                f"{label} values must have final dimension "
                f"{self.valid_vertex_count}, got {source.shape}."
            )
        if not np.issubdtype(source.dtype, np.floating):
            source = source.astype(np.float64)
        if not np.all(np.isfinite(source)):
            raise ValueError(f"{label} values must be finite.")
        return source

    def summary(self, source_path: Path | None = None) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": self.schema_version,
            "source_path": None if source_path is None else str(source_path),
            "geometry_sha256": self.geometry_sha256,
            "gmsh_script_sha256": self.gmsh_script_sha256,
            "gmsh_version": self.gmsh_version,
            "boundary_size_factor": self.boundary_size_factor,
            "max_auxiliary_fraction": self.max_auxiliary_fraction,
            "vertex_count": self.vertex_count,
            "valid_vertex_count": self.valid_vertex_count,
            "boundary_vertex_count": self.boundary_vertex_count,
            "auxiliary_vertex_count": self.auxiliary_vertex_count,
            "auxiliary_fraction": self.auxiliary_fraction,
            "triangle_count": self.triangle_count,
            "boundary_edge_count": int(self.boundary_edges.shape[0]),
            "valid_point_transfer": "exact_vertex_mapping",
            "boundary_value_source": "field_specific",
            "solution_boundary_value_source": ("prescribed_homogeneous_dirichlet"),
            "interior_scalar_boundary_value_source": "not_evaluated",
            "boundary_values_model_evaluated": False,
            "auxiliary_transfer": "mesh_adjacency_inverse_distance_stencil",
            "interior_scalar_cell_transfer": (
                "arithmetic_mean_of_non_boundary_vertices"
            ),
            "included_in_metrics": False,
        }

    def to_payload(self) -> dict[str, np.ndarray]:
        return {
            "vertices": self.vertices,
            "triangles": self.triangles,
            "boundary_edges": self.boundary_edges,
            "valid_to_vertex": self.valid_to_vertex,
            "boundary_vertex_mask": self.boundary_vertex_mask,
            "auxiliary_vertex_mask": self.auxiliary_vertex_mask,
            "aux_interp_ptr": self.aux_interp_ptr,
            "aux_interp_vertex_index": self.aux_interp_vertex_index,
            "aux_interp_weight": self.aux_interp_weight,
            "geometry_sha256": np.asarray(self.geometry_sha256),
            "gmsh_script_sha256": np.asarray(self.gmsh_script_sha256),
            "gmsh_version": np.asarray(self.gmsh_version),
            "boundary_size_factor": np.asarray(
                self.boundary_size_factor,
                dtype=np.float64,
            ),
            "max_auxiliary_fraction": np.asarray(
                self.max_auxiliary_fraction,
                dtype=np.float64,
            ),
            "schema_version": np.asarray(self.schema_version, dtype=np.int64),
        }

    def _validate_provenance(self) -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError(
                "Unsupported visualization mesh schema_version "
                f"{self.schema_version}; expected {self.SCHEMA_VERSION}."
            )
        for value, name in (
            (self.geometry_sha256, "geometry_sha256"),
            (self.gmsh_script_sha256, "gmsh_script_sha256"),
        ):
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        if not self.gmsh_version:
            raise ValueError("gmsh_version cannot be empty.")
        if (
            not math.isfinite(self.boundary_size_factor)
            or self.boundary_size_factor < 1.0
        ):
            raise ValueError("boundary_size_factor must be finite and at least 1.0.")
        if (
            not math.isfinite(self.max_auxiliary_fraction)
            or self.max_auxiliary_fraction < 0.0
            or self.max_auxiliary_fraction > 1.0
        ):
            raise ValueError("max_auxiliary_fraction must be in [0, 1].")

    def _validate_arrays(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (V, 2).")
        if self.vertex_count < 3 or not np.all(np.isfinite(self.vertices)):
            raise ValueError("vertices must contain at least three finite points.")
        self._validate_index_matrix(self.triangles, "triangles", columns=3)
        self._validate_index_matrix(self.boundary_edges, "boundary_edges", columns=2)
        if self.valid_to_vertex.ndim != 1 or self.valid_to_vertex.size == 0:
            raise ValueError("valid_to_vertex must be a non-empty 1D array.")
        self._validate_indices(self.valid_to_vertex, "valid_to_vertex")
        if np.unique(self.valid_to_vertex).size != self.valid_to_vertex.size:
            raise ValueError("valid_to_vertex must be one-to-one.")

        sorted_triangles = np.sort(self.triangles, axis=1)
        if np.any(np.diff(sorted_triangles, axis=1) == 0):
            raise ValueError("triangles cannot repeat a vertex within one cell.")
        if np.unique(sorted_triangles, axis=0).shape[0] != self.triangle_count:
            raise ValueError("triangles cannot contain duplicate cells.")
        points = self.vertices[self.triangles]
        cross = (points[:, 1, 0] - points[:, 0, 0]) * (
            points[:, 2, 1] - points[:, 0, 1]
        ) - (points[:, 1, 1] - points[:, 0, 1]) * (points[:, 2, 0] - points[:, 0, 0])
        extent = np.ptp(self.vertices, axis=0)
        area_tolerance = (
            32.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                float(np.prod(extent)),
            )
        )
        if np.any(np.abs(cross) <= area_tolerance):
            raise ValueError("triangles contain a degenerate cell.")

        sorted_edges = np.sort(self.boundary_edges, axis=1)
        if np.any(np.diff(sorted_edges, axis=1) == 0):
            raise ValueError("boundary_edges cannot contain zero-length edges.")
        if np.unique(sorted_edges, axis=0).shape[0] != self.boundary_edges.shape[0]:
            raise ValueError("boundary_edges cannot contain duplicate edges.")

    def _validate_partition(self) -> None:
        for values, name in (
            (self.boundary_vertex_mask, "boundary_vertex_mask"),
            (self.auxiliary_vertex_mask, "auxiliary_vertex_mask"),
        ):
            if values.shape != (self.vertex_count,) or values.dtype != np.bool_:
                raise ValueError(f"{name} must be a boolean array with shape (V,).")
        valid_mask = np.zeros(self.vertex_count, dtype=np.bool_)
        valid_mask[self.valid_to_vertex] = True
        if np.any(valid_mask & self.boundary_vertex_mask):
            raise ValueError("Valid and boundary mesh vertices must be disjoint.")
        if np.any(valid_mask & self.auxiliary_vertex_mask):
            raise ValueError("Valid and auxiliary mesh vertices must be disjoint.")
        if np.any(self.boundary_vertex_mask & self.auxiliary_vertex_mask):
            raise ValueError("Boundary and auxiliary mesh vertices must be disjoint.")
        if not np.all(
            valid_mask | self.boundary_vertex_mask | self.auxiliary_vertex_mask
        ):
            raise ValueError("Every mesh vertex must be valid, boundary, or auxiliary.")
        if self.boundary_vertex_count == 0:
            raise ValueError("Visualization mesh must contain boundary vertices.")
        if not np.all(self.boundary_vertex_mask[self.boundary_edges]):
            raise ValueError("boundary_edges must reference only boundary vertices.")
        covered_boundary = np.unique(self.boundary_edges)
        if covered_boundary.size != self.boundary_vertex_count or not np.all(
            self.boundary_vertex_mask[covered_boundary]
        ):
            raise ValueError("Every boundary vertex must belong to a boundary edge.")
        if self.auxiliary_fraction > self.max_auxiliary_fraction + 1.0e-15:
            raise ValueError(
                "Visualization mesh auxiliary fraction exceeds its configured "
                f"maximum: {self.auxiliary_fraction:.6e} > "
                f"{self.max_auxiliary_fraction:.6e}."
            )

    def _validate_interpolation(self) -> None:
        auxiliary_count = self.auxiliary_vertex_count
        if self.aux_interp_ptr.shape != (auxiliary_count + 1,):
            raise ValueError("aux_interp_ptr must have shape (A + 1,).")
        if self.aux_interp_ptr.dtype.kind not in "iu":
            raise ValueError("aux_interp_ptr must contain integer indices.")
        if self.aux_interp_ptr[0] != 0 or np.any(np.diff(self.aux_interp_ptr) < 0):
            raise ValueError("aux_interp_ptr must be a monotone CSR pointer.")
        if self.aux_interp_vertex_index.ndim != 1:
            raise ValueError("aux_interp_vertex_index must be one-dimensional.")
        if self.aux_interp_weight.ndim != 1:
            raise ValueError("aux_interp_weight must be one-dimensional.")
        nnz = int(self.aux_interp_ptr[-1])
        if (
            nnz != self.aux_interp_vertex_index.size
            or nnz != self.aux_interp_weight.size
        ):
            raise ValueError(
                "Auxiliary interpolation CSR arrays have inconsistent sizes."
            )
        if nnz:
            self._validate_indices(
                self.aux_interp_vertex_index,
                "aux_interp_vertex_index",
            )
            if not np.all(np.isfinite(self.aux_interp_weight)) or np.any(
                self.aux_interp_weight < 0.0
            ):
                raise ValueError(
                    "Auxiliary interpolation weights must be finite and nonnegative."
                )
            if np.any(self.auxiliary_vertex_mask[self.aux_interp_vertex_index]):
                raise ValueError(
                    "Auxiliary interpolation cannot reference auxiliary vertices."
                )
        for offset in range(auxiliary_count):
            start = int(self.aux_interp_ptr[offset])
            end = int(self.aux_interp_ptr[offset + 1])
            count = end - start
            if (
                count < self.MIN_INTERPOLATION_NEIGHBORS
                or count > self.MAX_INTERPOLATION_NEIGHBORS
            ):
                raise ValueError(
                    "Each auxiliary interpolation row must contain 3 to 8 vertices."
                )
            if not math.isclose(
                float(np.sum(self.aux_interp_weight[start:end])),
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Auxiliary interpolation weights must sum to one.")

    def _validate_valid_coordinates(self, coords_valid: np.ndarray) -> None:
        coords = np.asarray(coords_valid, dtype=np.float64)
        if coords.shape != (self.valid_vertex_count, 2):
            raise ValueError(
                "coords_valid shape does not match visualization mesh mapping: "
                f"{coords.shape} != ({self.valid_vertex_count}, 2)."
            )
        if not np.allclose(
            self.vertices[self.valid_to_vertex],
            coords,
            rtol=0.0,
            atol=self.MAPPING_TOLERANCE,
        ):
            maximum = float(
                np.max(
                    np.linalg.norm(self.vertices[self.valid_to_vertex] - coords, axis=1)
                )
            )
            raise ValueError(
                "Visualization mesh valid-point mapping does not match coords_valid; "
                f"maximum distance is {maximum:.6e}."
            )

    def _validate_index_matrix(
        self,
        values: np.ndarray,
        field_name: str,
        *,
        columns: int,
    ) -> None:
        if values.ndim != 2 or values.shape[1] != columns or values.shape[0] == 0:
            raise ValueError(f"{field_name} must have non-empty shape (N, {columns}).")
        self._validate_indices(values, field_name)

    def _validate_indices(self, values: np.ndarray, field_name: str) -> None:
        if values.dtype.kind not in "iu":
            raise ValueError(f"{field_name} must contain integer indices.")
        if np.any(values < 0) or np.any(values >= self.vertex_count):
            raise ValueError(f"{field_name} contains an out-of-range vertex index.")


@dataclass(frozen=True)
class ComplexVisualizationMeshConfig:
    geometry: Path
    gmsh_script: Path
    out: Path
    boundary_size_factor: float = 3.0
    max_auxiliary_fraction: float = 1.0e-3
    overwrite: bool = False

    def __post_init__(self) -> None:
        if not self.geometry.is_file():
            raise FileNotFoundError(f"Geometry file does not exist: {self.geometry}")
        if not self.gmsh_script.is_file():
            raise FileNotFoundError(f"Gmsh script does not exist: {self.gmsh_script}")
        if (
            not math.isfinite(self.boundary_size_factor)
            or self.boundary_size_factor < 1.0
        ):
            raise ValueError("boundary_size_factor must be finite and at least 1.0.")
        if (
            not math.isfinite(self.max_auxiliary_fraction)
            or self.max_auxiliary_fraction < 0.0
            or self.max_auxiliary_fraction > 1.0
        ):
            raise ValueError("max_auxiliary_fraction must be in [0, 1].")


class MeshAdjacencyInterpolationMixin:
    """Build deterministic local interpolation stencils on mesh adjacency."""

    @classmethod
    def build_auxiliary_stencils(
        cls,
        *,
        vertices: np.ndarray,
        triangles: np.ndarray,
        known_mask: np.ndarray,
        auxiliary_mask: np.ndarray,
        h_reference: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        adjacency = cls._build_adjacency(vertices.shape[0], triangles)
        pointers = [0]
        stencil_indices: list[int] = []
        stencil_weights: list[float] = []
        for auxiliary_vertex in np.flatnonzero(auxiliary_mask):
            candidates = cls._nearest_known_graph_ring(
                int(auxiliary_vertex),
                adjacency,
                known_mask,
            )
            ordered = sorted(
                candidates,
                key=lambda index: (
                    float(np.linalg.norm(vertices[index] - vertices[auxiliary_vertex])),
                    index,
                ),
            )[: ComplexVisualizationMesh.MAX_INTERPOLATION_NEIGHBORS]
            distances_squared = np.sum(
                (vertices[np.asarray(ordered)] - vertices[auxiliary_vertex]) ** 2,
                axis=1,
            )
            raw_weights = 1.0 / (
                distances_squared + 1.0e-12 * h_reference * h_reference
            )
            weights = raw_weights / np.sum(raw_weights)
            stencil_indices.extend(ordered)
            stencil_weights.extend(float(value) for value in weights)
            pointers.append(len(stencil_indices))
        return (
            np.asarray(pointers, dtype=np.int64),
            np.asarray(stencil_indices, dtype=np.int64),
            np.asarray(stencil_weights, dtype=np.float64),
        )

    @staticmethod
    def _build_adjacency(
        vertex_count: int,
        triangles: np.ndarray,
    ) -> tuple[frozenset[int], ...]:
        adjacency: list[set[int]] = [set() for _ in range(vertex_count)]
        for first, second, third in triangles.tolist():
            adjacency[first].update((second, third))
            adjacency[second].update((first, third))
            adjacency[third].update((first, second))
        return tuple(frozenset(neighbors) for neighbors in adjacency)

    @staticmethod
    def _nearest_known_graph_ring(
        start: int,
        adjacency: tuple[frozenset[int], ...],
        known_mask: np.ndarray,
    ) -> set[int]:
        visited = {start}
        frontier: deque[int] = deque([start])
        candidates: set[int] = set()
        while (
            frontier
            and len(candidates) < ComplexVisualizationMesh.MIN_INTERPOLATION_NEIGHBORS
        ):
            next_frontier: deque[int] = deque()
            while frontier:
                vertex = frontier.popleft()
                for neighbor in sorted(adjacency[vertex]):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    if known_mask[neighbor]:
                        candidates.add(neighbor)
                    else:
                        next_frontier.append(neighbor)
            frontier = next_frontier
        if len(candidates) < ComplexVisualizationMesh.MIN_INTERPOLATION_NEIGHBORS:
            raise ValueError(
                "Auxiliary mesh vertex is not connected to at least three known "
                "valid or boundary vertices."
            )
        return candidates


class ComplexVisualizationMeshGenerator(MeshAdjacencyInterpolationMixin):
    """Generate a reusable conforming mesh without adding Gmsh to main runtime."""

    def __init__(
        self,
        config: ComplexVisualizationMeshConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger

    def write(self) -> Path:
        if self.config.out.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Visualization mesh already exists: {self.config.out}. "
                "Pass --overwrite to replace it."
            )
        mesh = self.build()
        self.config.out.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = mesh.to_payload()
        np.savez_compressed(
            self.config.out,
            **payload,
        )
        if self.logger is not None:
            self.logger.info(
                "Wrote visualization mesh to %s (vertices=%d, triangles=%d, "
                "boundary=%d, auxiliary=%d)",
                self.config.out,
                mesh.vertex_count,
                mesh.triangle_count,
                mesh.boundary_vertex_count,
                mesh.auxiliary_vertex_count,
            )
        return self.config.out

    def build(self) -> ComplexVisualizationMesh:
        geometry = GeometryGridLoader().load(self.config.geometry)
        h_reference = min(
            float(np.min(np.diff(geometry.grid_x))),
            float(np.min(np.diff(geometry.grid_y))),
        )
        boundary_size = self.config.boundary_size_factor * h_reference
        gmsh = self._load_gmsh()
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("greenonet_visualization_mesh")
            tags = self._build_domain(gmsh, geometry, boundary_size)
            valid_point_tags = self._embed_valid_points(gmsh, geometry, tags)
            self._configure_meshing(gmsh, boundary_size)
            gmsh.model.mesh.generate(2)
            (
                vertices,
                triangles,
                boundary_edges,
                valid_to_vertex,
            ) = self._extract_mesh(gmsh, tags.surface_tags, valid_point_tags)
            gmsh_version = str(getattr(gmsh, "__version__", "unknown"))
        finally:
            gmsh.finalize()

        boundary_mask = np.zeros(vertices.shape[0], dtype=np.bool_)
        boundary_mask[np.unique(boundary_edges)] = True
        valid_mask = np.zeros(vertices.shape[0], dtype=np.bool_)
        valid_mask[valid_to_vertex] = True
        if np.any(boundary_mask & valid_mask):
            raise ValueError(
                "Strict interior coords_valid unexpectedly overlap Gmsh boundary nodes."
            )
        auxiliary_mask = ~(valid_mask | boundary_mask)
        auxiliary_fraction = float(np.count_nonzero(auxiliary_mask)) / vertices.shape[0]
        if auxiliary_fraction > self.config.max_auxiliary_fraction + 1.0e-15:
            raise ValueError(
                "Generated visualization mesh auxiliary fraction exceeds the "
                f"configured maximum: {auxiliary_fraction:.6e} > "
                f"{self.config.max_auxiliary_fraction:.6e}."
            )
        ptr, indices, weights = self.build_auxiliary_stencils(
            vertices=vertices,
            triangles=triangles,
            known_mask=valid_mask | boundary_mask,
            auxiliary_mask=auxiliary_mask,
            h_reference=h_reference,
        )
        mesh = ComplexVisualizationMesh(
            vertices=vertices,
            triangles=triangles,
            boundary_edges=boundary_edges,
            valid_to_vertex=valid_to_vertex,
            boundary_vertex_mask=boundary_mask,
            auxiliary_vertex_mask=auxiliary_mask,
            aux_interp_ptr=ptr,
            aux_interp_vertex_index=indices,
            aux_interp_weight=weights,
            geometry_sha256=file_sha256(self.config.geometry),
            gmsh_script_sha256=file_sha256(self.config.gmsh_script),
            gmsh_version=gmsh_version,
            boundary_size_factor=self.config.boundary_size_factor,
            max_auxiliary_fraction=self.config.max_auxiliary_fraction,
        )
        mesh.validate(geometry.coords_valid)
        return mesh

    @staticmethod
    def _load_gmsh() -> Any:
        try:
            return importlib.import_module("gmsh")
        except ImportError as exc:
            raise RuntimeError(
                "Visualization mesh generation requires the optional gmsh package. "
                "Run this CLI in the green_fenicsx environment."
            ) from exc

    def _build_domain(
        self,
        gmsh: Any,
        geometry: RawComplexGeometryGrid,
        boundary_size: float,
    ) -> Any:
        module = GmshScriptLoader.load(self.config.gmsh_script)
        context = GmshDomainContext(
            geometry_path=geometry.path,
            grid_x=geometry.grid_x,
            grid_y=geometry.grid_y,
            coords_valid=geometry.coords_valid,
            mesh_size=boundary_size,
        )
        raw_tags = module.build_domain(gmsh, context)
        gmsh.model.occ.synchronize()
        gmsh.model.geo.synchronize()
        return GmshDomainParser.parse(
            raw_tags,
            num_valid_points=geometry.num_valid_points,
        )

    @staticmethod
    def _embed_valid_points(
        gmsh: Any,
        geometry: RawComplexGeometryGrid,
        tags: Any,
    ) -> list[int]:
        point_tags = [
            int(gmsh.model.occ.addPoint(float(x), float(y), 0.0, 0.0))
            for x, y in geometry.coords_valid
        ]
        gmsh.model.occ.synchronize()
        if tags.point_surface_tags is None:
            surface_groups = {int(tags.surface_tags[0]): point_tags}
        else:
            surface_groups = {int(tag): [] for tag in tags.surface_tags}
            for point_tag, surface_tag in zip(point_tags, tags.point_surface_tags):
                surface_groups[int(surface_tag)].append(point_tag)
        for surface_tag, embedded_points in surface_groups.items():
            if embedded_points:
                gmsh.model.mesh.embed(0, embedded_points, 2, surface_tag)
        gmsh.model.occ.synchronize()
        return point_tags

    @staticmethod
    def _configure_meshing(gmsh: Any, boundary_size: float) -> None:
        options = {
            "Mesh.Algorithm": 5,
            "Mesh.ElementOrder": 1,
            "Mesh.RecombineAll": 0,
            "Mesh.MeshSizeFromPoints": 0,
            "Mesh.MeshSizeFromCurvature": 0,
            "Mesh.MeshSizeExtendFromBoundary": 0,
            "Mesh.CharacteristicLengthMax": boundary_size,
        }
        for name, value in options.items():
            gmsh.option.setNumber(name, value)

    @classmethod
    def _extract_mesh(
        cls,
        gmsh: Any,
        surface_tags: tuple[int, ...],
        valid_point_tags: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        tags_array = np.asarray(node_tags, dtype=np.int64)
        vertices = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        if tags_array.size != vertices.shape[0]:
            raise ValueError("Gmsh node tags and coordinates have inconsistent sizes.")
        tag_to_index = {
            int(tag): index for index, tag in enumerate(tags_array.tolist())
        }

        triangles = cls._extract_elements(
            gmsh,
            entity_dim=2,
            entity_tags=surface_tags,
            expected_element_type=2,
            nodes_per_element=3,
            tag_to_index=tag_to_index,
        )
        boundary_entities = gmsh.model.getBoundary(
            [(2, int(tag)) for tag in surface_tags],
            combined=True,
            oriented=False,
            recursive=False,
        )
        boundary_curve_tags = tuple(
            int(tag) for dim, tag in boundary_entities if int(dim) == 1
        )
        boundary_edges = cls._extract_elements(
            gmsh,
            entity_dim=1,
            entity_tags=boundary_curve_tags,
            expected_element_type=1,
            nodes_per_element=2,
            tag_to_index=tag_to_index,
        )
        valid_to_vertex = np.empty(len(valid_point_tags), dtype=np.int64)
        for offset, point_tag in enumerate(valid_point_tags):
            point_nodes, _, _ = gmsh.model.mesh.getNodes(
                0,
                int(point_tag),
                includeBoundary=True,
            )
            if len(point_nodes) != 1:
                raise ValueError(
                    "Every embedded valid point must produce exactly one Gmsh node."
                )
            valid_to_vertex[offset] = tag_to_index[int(point_nodes[0])]
        return (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64),
            np.asarray(boundary_edges, dtype=np.int64),
            valid_to_vertex,
        )

    @staticmethod
    def _extract_elements(
        gmsh: Any,
        *,
        entity_dim: int,
        entity_tags: tuple[int, ...],
        expected_element_type: int,
        nodes_per_element: int,
        tag_to_index: dict[int, int],
    ) -> np.ndarray:
        elements: list[list[int]] = []
        for entity_tag in entity_tags:
            element_types, _, element_node_tags = gmsh.model.mesh.getElements(
                entity_dim,
                int(entity_tag),
            )
            for element_type, raw_nodes in zip(element_types, element_node_tags):
                if int(element_type) != expected_element_type:
                    raise ValueError(
                        "Visualization mesh supports only first-order line and "
                        "triangle elements."
                    )
                nodes = np.asarray(raw_nodes, dtype=np.int64).reshape(
                    -1,
                    nodes_per_element,
                )
                for row in nodes:
                    elements.append([tag_to_index[int(tag)] for tag in row])
        if not elements:
            raise ValueError("Gmsh visualization mesh contains no expected elements.")
        values = np.asarray(elements, dtype=np.int64)
        canonical = np.sort(values, axis=1)
        _, unique_indices = np.unique(canonical, axis=0, return_index=True)
        return values[np.sort(unique_indices)]


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_complex_visualization_mesh(
    path: Path | str,
    *,
    geometry_path: Path | str | None = None,
    coords_valid: np.ndarray | None = None,
) -> ComplexVisualizationMesh:
    mesh_path = Path(path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Visualization mesh does not exist: {mesh_path}")
    required = {
        "vertices",
        "triangles",
        "boundary_edges",
        "valid_to_vertex",
        "boundary_vertex_mask",
        "auxiliary_vertex_mask",
        "aux_interp_ptr",
        "aux_interp_vertex_index",
        "aux_interp_weight",
        "geometry_sha256",
        "gmsh_script_sha256",
        "gmsh_version",
        "boundary_size_factor",
        "max_auxiliary_fraction",
        "schema_version",
    }
    with np.load(mesh_path, allow_pickle=False) as raw:
        missing = sorted(required - set(raw.files))
        if missing:
            raise KeyError(
                "Visualization mesh NPZ is missing required keys: "
                f"{', '.join(missing)}."
            )
        mesh = ComplexVisualizationMesh(
            vertices=np.asarray(raw["vertices"], dtype=np.float64),
            triangles=np.asarray(raw["triangles"], dtype=np.int64),
            boundary_edges=np.asarray(raw["boundary_edges"], dtype=np.int64),
            valid_to_vertex=np.asarray(raw["valid_to_vertex"], dtype=np.int64),
            boundary_vertex_mask=np.asarray(
                raw["boundary_vertex_mask"],
                dtype=np.bool_,
            ),
            auxiliary_vertex_mask=np.asarray(
                raw["auxiliary_vertex_mask"],
                dtype=np.bool_,
            ),
            aux_interp_ptr=np.asarray(raw["aux_interp_ptr"], dtype=np.int64),
            aux_interp_vertex_index=np.asarray(
                raw["aux_interp_vertex_index"],
                dtype=np.int64,
            ),
            aux_interp_weight=np.asarray(raw["aux_interp_weight"], dtype=np.float64),
            geometry_sha256=str(np.asarray(raw["geometry_sha256"]).item()),
            gmsh_script_sha256=str(np.asarray(raw["gmsh_script_sha256"]).item()),
            gmsh_version=str(np.asarray(raw["gmsh_version"]).item()),
            boundary_size_factor=float(
                np.asarray(raw["boundary_size_factor"], dtype=np.float64).item()
            ),
            max_auxiliary_fraction=float(
                np.asarray(raw["max_auxiliary_fraction"], dtype=np.float64).item()
            ),
            schema_version=int(
                np.asarray(raw["schema_version"], dtype=np.int64).item()
            ),
        )
    mesh.validate(coords_valid)
    if geometry_path is not None:
        actual_digest = file_sha256(geometry_path)
        if actual_digest != mesh.geometry_sha256:
            raise ValueError(
                "Visualization mesh geometry SHA-256 does not match the configured "
                f"geometry: {mesh.geometry_sha256} != {actual_digest}."
            )
    return mesh
