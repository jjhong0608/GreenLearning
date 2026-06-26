from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class AnnulusGmshMixin:
    """Helpers for the centered annulus Gmsh example domain."""

    CENTER_X = 0.0
    CENTER_Y = 0.0
    CENTER_Z = 0.0

    @classmethod
    def add_annulus_surface(
        cls,
        gmsh: Any,
        *,
        inner_radius: float,
        outer_radius: float,
    ) -> int:
        cls._validate_radii(inner_radius, outer_radius)
        outer_disk = int(
            gmsh.model.occ.addDisk(
                cls.CENTER_X,
                cls.CENTER_Y,
                cls.CENTER_Z,
                outer_radius,
                outer_radius,
            )
        )
        inner_disk = int(
            gmsh.model.occ.addDisk(
                cls.CENTER_X,
                cls.CENTER_Y,
                cls.CENTER_Z,
                inner_radius,
                inner_radius,
            )
        )
        result, _mapping = gmsh.model.occ.cut(
            [(2, outer_disk)],
            [(2, inner_disk)],
            removeObject=True,
            removeTool=True,
        )
        surface_tags = [int(tag) for dim, tag in result if int(dim) == 2]
        if len(surface_tags) != 1:
            raise RuntimeError(
                f"Expected one annulus surface after Gmsh cut; received {surface_tags}."
            )
        return surface_tags[0]

    @classmethod
    def radii_from_context(cls, context: Any) -> tuple[float, float]:
        geometry_path = getattr(context, "geometry_path", None)
        if geometry_path is None:
            raise ValueError("Annulus Gmsh script requires context.geometry_path.")
        with np.load(Path(geometry_path), allow_pickle=False) as raw:
            missing = sorted({"inner_radius", "outer_radius"} - set(raw.files))
            if missing:
                raise KeyError(
                    "Annulus geometry metadata is missing required keys: "
                    f"{', '.join(missing)}."
                )
            inner_radius = float(
                np.asarray(raw["inner_radius"], dtype=np.float64).item()
            )
            outer_radius = float(
                np.asarray(raw["outer_radius"], dtype=np.float64).item()
            )
        cls._validate_radii(inner_radius, outer_radius)
        return inner_radius, outer_radius

    @staticmethod
    def _validate_radii(inner_radius: float, outer_radius: float) -> None:
        if not np.isfinite(inner_radius) or not np.isfinite(outer_radius):
            raise ValueError("Annulus radii metadata must be finite.")
        if inner_radius <= 0.0:
            raise ValueError("Annulus inner_radius metadata must be positive.")
        if outer_radius <= 0.0:
            raise ValueError("Annulus outer_radius metadata must be positive.")
        if outer_radius <= inner_radius:
            raise ValueError(
                "Annulus outer_radius metadata must be greater than inner_radius."
            )


class AnnulusDomainBuilder(AnnulusGmshMixin):
    """Build a single Gmsh surface with an inner hole for annulus samples."""

    def build(self, gmsh: Any, context: Any) -> dict[str, list[int]]:
        inner_radius, outer_radius = self.radii_from_context(context)
        surface_tag = self.add_annulus_surface(
            gmsh,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )
        gmsh.model.occ.synchronize()
        if context.mesh_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", context.mesh_size)
        return {"surface_tags": [surface_tag]}


def build_domain(gmsh: Any, context: Any) -> dict[str, list[int]]:
    return AnnulusDomainBuilder().build(gmsh, context)
