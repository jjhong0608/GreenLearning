from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class UnitCircleGmshMixin:
    """Helpers for the default unit-circle, radius-aware Gmsh example domain."""

    CENTER_X = 0.0
    CENTER_Y = 0.0
    CENTER_Z = 0.0
    DEFAULT_RADIUS = 1.0

    @classmethod
    def add_disk(cls, gmsh: Any, radius: float) -> int:
        if radius <= 0.0:
            raise ValueError("Circle radius must be positive.")
        return int(
            gmsh.model.occ.addDisk(
                cls.CENTER_X,
                cls.CENTER_Y,
                cls.CENTER_Z,
                radius,
                radius,
            )
        )

    @classmethod
    def radius_from_context(cls, context: Any) -> float:
        geometry_path = getattr(context, "geometry_path", None)
        if geometry_path is None:
            return cls.DEFAULT_RADIUS
        with np.load(Path(geometry_path), allow_pickle=False) as raw:
            if "radius" not in raw.files:
                return cls.DEFAULT_RADIUS
            radius = float(np.asarray(raw["radius"], dtype=np.float64).item())
        if radius <= 0.0:
            raise ValueError("Geometry radius metadata must be positive.")
        return radius


class UnitCircleDomainBuilder(UnitCircleGmshMixin):
    """Build a single-surface disk consumed by make_fenicsx_samples.py."""

    def build(self, gmsh: Any, context: Any) -> dict[str, list[int]]:
        surface_tag = self.add_disk(gmsh, self.radius_from_context(context))
        gmsh.model.occ.synchronize()
        if context.mesh_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", context.mesh_size)
        return {"surface_tags": [surface_tag]}


def build_domain(gmsh: Any, context: Any) -> dict[str, list[int]]:
    return UnitCircleDomainBuilder().build(gmsh, context)
