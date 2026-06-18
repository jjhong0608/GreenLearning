from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from greenonet.fenicsx_samples.config import FenicsxSampleConfig
from greenonet.fenicsx_samples.fenicsx_runtime import FenicsxRuntime
from greenonet.fenicsx_samples.geometry import RawComplexGeometryGrid


@dataclass(frozen=True)
class GmshDomainContext:
    """Context object passed into user-provided Gmsh scripts."""

    geometry_path: Path
    grid_x: np.ndarray
    grid_y: np.ndarray
    coords_valid: np.ndarray
    mesh_size: float | None


@dataclass(frozen=True)
class GmshDomainTags:
    """Surface tags returned by a user Gmsh script."""

    surface_tags: tuple[int, ...]
    boundary_tags: tuple[int, ...] = ()
    point_surface_tags: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.surface_tags:
            raise ValueError("Gmsh script must return at least one surface tag.")
        if self.point_surface_tags is not None:
            invalid = set(self.point_surface_tags) - set(self.surface_tags)
            if invalid:
                raise ValueError(
                    "point_surface_tags contains tags not listed in surface_tags: "
                    f"{sorted(invalid)}."
                )


@dataclass(frozen=True)
class FenicsxMeshBundle:
    """DOLFINx mesh plus optional mesh tags."""

    domain: Any
    cell_tags: Any
    facet_tags: Any
    vertex_coverage_max_distance: float | None


class GmshScriptLoader:
    """Load a Python file that defines build_domain(gmsh, context)."""

    @staticmethod
    def load(path: Path) -> ModuleType:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Gmsh script does not exist: {path}")
        spec = importlib.util.spec_from_file_location("greenonet_user_gmsh", resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import Gmsh script: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "build_domain", None)):
            raise ValueError(
                "Gmsh script must define callable build_domain(gmsh, context)."
            )
        return module


class GmshDomainParser:
    """Normalize user Gmsh return values to GmshDomainTags."""

    @staticmethod
    def parse(result: Any, *, num_valid_points: int) -> GmshDomainTags:
        if isinstance(result, GmshDomainTags):
            GmshDomainParser._validate_point_tags(result, num_valid_points)
            return result
        if isinstance(result, int):
            return GmshDomainTags(surface_tags=(int(result),))
        if isinstance(result, Mapping):
            tags = GmshDomainTags(
                surface_tags=GmshDomainParser._as_int_tuple(result.get("surface_tags")),
                boundary_tags=GmshDomainParser._as_int_tuple(
                    result.get("boundary_tags", ())
                ),
                point_surface_tags=GmshDomainParser._optional_int_tuple(
                    result.get("point_surface_tags")
                ),
            )
            GmshDomainParser._validate_point_tags(tags, num_valid_points)
            return tags
        if isinstance(result, Sequence) and not isinstance(result, str):
            values = tuple(result)
            if len(values) not in {2, 3}:
                raise ValueError(
                    "Gmsh tuple returns must be (surface_tags, boundary_tags) "
                    "or (surface_tags, boundary_tags, point_surface_tags)."
                )
            tags = GmshDomainTags(
                surface_tags=GmshDomainParser._as_int_tuple(values[0]),
                boundary_tags=GmshDomainParser._as_int_tuple(values[1]),
                point_surface_tags=(
                    None
                    if len(values) == 2
                    else GmshDomainParser._optional_int_tuple(values[2])
                ),
            )
            GmshDomainParser._validate_point_tags(tags, num_valid_points)
            return tags
        raise TypeError(
            "build_domain must return an int, mapping, tuple, or GmshDomainTags."
        )

    @staticmethod
    def _as_int_tuple(value: Any) -> tuple[int, ...]:
        if value is None:
            raise ValueError("surface_tags must be provided.")
        if isinstance(value, int):
            return (int(value),)
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(int(item) for item in value)
        raise TypeError("Gmsh tags must be an int or a sequence of ints.")

    @staticmethod
    def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
        if value is None:
            return None
        return GmshDomainParser._as_int_tuple(value)

    @staticmethod
    def _validate_point_tags(
        tags: GmshDomainTags,
        num_valid_points: int,
    ) -> None:
        if tags.point_surface_tags is not None and (
            len(tags.point_surface_tags) != num_valid_points
        ):
            raise ValueError(
                "point_surface_tags must have one entry per valid geometry point."
            )
        if len(tags.surface_tags) > 1 and tags.point_surface_tags is None:
            raise ValueError(
                "Multiple Gmsh surfaces require point_surface_tags so valid "
                "points are embedded in the correct connected component."
            )


class MeshCoverageMixin:
    """Check whether all valid grid points are present as mesh vertices."""

    @staticmethod
    def vertex_coverage_distance(domain: Any, points: np.ndarray) -> float:
        from scipy.spatial import cKDTree

        vertices = np.asarray(domain.geometry.x, dtype=np.float64)[:, :2]
        distances, _ = cKDTree(vertices).query(points[:, :2], k=1)
        return float(np.max(distances))

    @classmethod
    def require_vertex_coverage(
        cls,
        domain: Any,
        points: np.ndarray,
        *,
        tolerance: float = 1.0e-10,
    ) -> float:
        max_distance = cls.vertex_coverage_distance(domain, points)
        if max_distance > tolerance:
            raise ValueError(
                "Valid geometry points are not all mesh vertices; max nearest "
                f"vertex distance is {max_distance:.6e}."
            )
        return max_distance


class FenicsxDomainBuilder(MeshCoverageMixin):
    """Build or load a DOLFINx mesh from Gmsh inputs."""

    def __init__(
        self,
        runtime: FenicsxRuntime,
        config: FenicsxSampleConfig,
    ) -> None:
        self.runtime = runtime
        self.config = config

    def build(self, geometry: RawComplexGeometryGrid) -> FenicsxMeshBundle:
        if self.config.gmsh_script is not None:
            bundle = self._from_gmsh_script(geometry)
        elif self.config.msh is not None:
            bundle = self._from_msh()
        else:
            raise ValueError("Domain source is not configured.")

        max_distance: float | None = None
        if self.config.require_valid_points_in_mesh:
            max_distance = self.require_vertex_coverage(
                bundle.domain,
                geometry.coords_valid,
            )
        return FenicsxMeshBundle(
            domain=bundle.domain,
            cell_tags=bundle.cell_tags,
            facet_tags=bundle.facet_tags,
            vertex_coverage_max_distance=max_distance,
        )

    def _from_gmsh_script(
        self,
        geometry: RawComplexGeometryGrid,
    ) -> FenicsxMeshBundle:
        gmsh = self.runtime.gmsh
        comm = self.runtime.mpi.COMM_WORLD
        module = GmshScriptLoader.load(self.config.gmsh_script or Path())
        gmsh.initialize()
        try:
            gmsh.model.add("greenonet_complex_domain")
            context = GmshDomainContext(
                geometry_path=geometry.path,
                grid_x=geometry.grid_x,
                grid_y=geometry.grid_y,
                coords_valid=geometry.coords_valid,
                mesh_size=self.config.mesh_size,
            )
            raw_tags = module.build_domain(gmsh, context)
            self._synchronize_gmsh(gmsh)
            tags = GmshDomainParser.parse(
                raw_tags,
                num_valid_points=geometry.num_valid_points,
            )
            self._ensure_physical_groups(gmsh, tags)
            if self.config.embed_valid_points:
                self._embed_valid_points(gmsh, geometry, tags)
            if self.config.mesh_size is not None:
                gmsh.option.setNumber(
                    "Mesh.CharacteristicLengthMax", self.config.mesh_size
                )
            gmsh.model.mesh.generate(2)
            domain, cell_tags, facet_tags = self.runtime.gmshio.model_to_mesh(
                gmsh.model,
                comm,
                0,
                gdim=2,
            )
        finally:
            gmsh.finalize()
        return FenicsxMeshBundle(
            domain=domain,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            vertex_coverage_max_distance=None,
        )

    def _from_msh(self) -> FenicsxMeshBundle:
        if self.config.msh is None:
            raise ValueError("--msh is not configured.")
        if not self.config.msh.is_file():
            raise FileNotFoundError(f"Msh file does not exist: {self.config.msh}")
        comm = self.runtime.mpi.COMM_WORLD
        domain, cell_tags, facet_tags = self.runtime.gmshio.read_from_msh(
            str(self.config.msh),
            comm,
            0,
            gdim=2,
        )
        return FenicsxMeshBundle(
            domain=domain,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            vertex_coverage_max_distance=None,
        )

    @staticmethod
    def _synchronize_gmsh(gmsh: Any) -> None:
        gmsh.model.occ.synchronize()
        gmsh.model.geo.synchronize()

    @staticmethod
    def _ensure_physical_groups(gmsh: Any, tags: GmshDomainTags) -> None:
        gmsh.model.addPhysicalGroup(2, list(tags.surface_tags))
        if tags.boundary_tags:
            gmsh.model.addPhysicalGroup(1, list(tags.boundary_tags))

    def _embed_valid_points(
        self,
        gmsh: Any,
        geometry: RawComplexGeometryGrid,
        tags: GmshDomainTags,
    ) -> None:
        point_tags: list[int] = []
        mesh_size = 0.0 if self.config.mesh_size is None else self.config.mesh_size
        for x_coord, y_coord in geometry.coords_valid:
            point_tags.append(
                int(
                    gmsh.model.occ.addPoint(
                        float(x_coord), float(y_coord), 0.0, mesh_size
                    )
                )
            )
        gmsh.model.occ.synchronize()

        surface_groups: dict[int, list[int]]
        if tags.point_surface_tags is None:
            surface_groups = {tags.surface_tags[0]: point_tags}
        else:
            surface_groups = {tag: [] for tag in tags.surface_tags}
            for point_tag, surface_tag in zip(point_tags, tags.point_surface_tags):
                surface_groups[surface_tag].append(point_tag)

        for surface_tag, embedded_points in surface_groups.items():
            if embedded_points:
                gmsh.model.mesh.embed(0, embedded_points, 2, int(surface_tag))
        gmsh.model.occ.synchronize()
