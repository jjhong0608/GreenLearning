from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class RectangleDomainMetadata:
    """Validated NPZ metadata used as the Gmsh boundary source of truth."""

    def __init__(
        self,
        *,
        boundary_vertices: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        self.boundary_vertices = boundary_vertices
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max


class RectangleGmshMixin:
    """Metadata validation and OCC helpers for an axis-aligned rectangle."""

    DOMAIN_TYPE = "rectangle"
    NUM_BOUNDARY_VERTICES = 4
    REQUIRED_METADATA = frozenset(
        {
            "domain_type",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "width",
            "height",
            "bounds",
            "center",
            "boundary_vertices",
            "has_hole",
        }
    )

    @classmethod
    def metadata_from_context(cls, context: Any) -> RectangleDomainMetadata:
        geometry_path = getattr(context, "geometry_path", None)
        if geometry_path is None:
            raise ValueError("Rectangle Gmsh script requires context.geometry_path.")

        with np.load(Path(geometry_path), allow_pickle=False) as raw:
            missing = sorted(cls.REQUIRED_METADATA - set(raw.files))
            if missing:
                raise KeyError(
                    "Rectangle geometry metadata is missing required keys: "
                    f"{', '.join(missing)}."
                )
            domain_type = str(np.asarray(raw["domain_type"]).item())
            has_hole = bool(np.asarray(raw["has_hole"]).item())
            x_min = float(np.asarray(raw["x_min"], dtype=np.float64).item())
            x_max = float(np.asarray(raw["x_max"], dtype=np.float64).item())
            y_min = float(np.asarray(raw["y_min"], dtype=np.float64).item())
            y_max = float(np.asarray(raw["y_max"], dtype=np.float64).item())
            width = float(np.asarray(raw["width"], dtype=np.float64).item())
            height = float(np.asarray(raw["height"], dtype=np.float64).item())
            bounds = np.asarray(raw["bounds"], dtype=np.float64)
            center = np.asarray(raw["center"], dtype=np.float64)
            boundary_vertices = np.asarray(
                raw["boundary_vertices"],
                dtype=np.float64,
            )

        cls._validate_metadata_contract(
            domain_type=domain_type,
            has_hole=has_hole,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            width=width,
            height=height,
            bounds=bounds,
            center=center,
            boundary_vertices=boundary_vertices,
        )
        return RectangleDomainMetadata(
            boundary_vertices=boundary_vertices,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

    @classmethod
    def _validate_metadata_contract(
        cls,
        *,
        domain_type: str,
        has_hole: bool,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        width: float,
        height: float,
        bounds: np.ndarray,
        center: np.ndarray,
        boundary_vertices: np.ndarray,
    ) -> None:
        if domain_type != cls.DOMAIN_TYPE:
            raise ValueError(
                "Rectangle geometry domain_type must be "
                f"'{cls.DOMAIN_TYPE}', got '{domain_type}'."
            )
        if has_hole:
            raise ValueError("Rectangle has_hole metadata must be false.")

        scalar_values = (x_min, x_max, y_min, y_max, width, height)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("Rectangle bounds, width, and height must be finite.")
        if x_max <= x_min:
            raise ValueError("Rectangle x_max metadata must be greater than x_min.")
        if y_max <= y_min:
            raise ValueError("Rectangle y_max metadata must be greater than y_min.")

        expected_width = x_max - x_min
        expected_height = y_max - y_min
        tolerance = 1.0e-10 * max(
            1.0,
            abs(x_min),
            abs(x_max),
            abs(y_min),
            abs(y_max),
            expected_width,
            expected_height,
        )
        if not math.isclose(width, expected_width, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("Rectangle width metadata does not match x bounds.")
        if not math.isclose(height, expected_height, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError("Rectangle height metadata does not match y bounds.")

        expected_bounds = np.array(
            [[x_min, x_max], [y_min, y_max]],
            dtype=np.float64,
        )
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all():
            raise ValueError("Rectangle bounds metadata must have finite shape (2, 2).")
        if not np.allclose(bounds, expected_bounds, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Rectangle bounds metadata is inconsistent with x/y bounds."
            )

        expected_center = np.array(
            [0.5 * (x_min + x_max), 0.5 * (y_min + y_max)],
            dtype=np.float64,
        )
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Rectangle center metadata must have finite shape (2,).")
        if not np.allclose(center, expected_center, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Rectangle center metadata is inconsistent with its bounds."
            )

        cls._validate_boundary_vertices(
            boundary_vertices,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            tolerance=tolerance,
        )

    @classmethod
    def _validate_boundary_vertices(
        cls,
        vertices: np.ndarray,
        *,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        tolerance: float,
    ) -> None:
        if vertices.shape != (cls.NUM_BOUNDARY_VERTICES, 2):
            raise ValueError("Rectangle boundary_vertices must have shape (4, 2).")
        if not np.isfinite(vertices).all():
            raise ValueError("Rectangle boundary_vertices must be finite.")

        expected_vertices = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ],
            dtype=np.float64,
        )
        if not np.allclose(vertices, expected_vertices, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Rectangle boundary_vertices must be lower-left, lower-right, "
                "upper-right, upper-left in counter-clockwise order."
            )

    @staticmethod
    def add_rectangle_surface(
        gmsh: Any,
        boundary_vertices: np.ndarray,
    ) -> tuple[int, list[int]]:
        point_tags = [
            int(gmsh.model.occ.addPoint(float(x), float(y), 0.0))
            for x, y in boundary_vertices
        ]
        boundary_tags = [
            int(
                gmsh.model.occ.addLine(
                    point_tags[index],
                    point_tags[(index + 1) % len(point_tags)],
                )
            )
            for index in range(len(point_tags))
        ]
        curve_loop = int(gmsh.model.occ.addCurveLoop(boundary_tags))
        surface_tag = int(gmsh.model.occ.addPlaneSurface([curve_loop]))
        return surface_tag, boundary_tags


class RectangleDomainBuilder(RectangleGmshMixin):
    """Build one rectangular Gmsh surface from saved geometry vertices."""

    def build(self, gmsh: Any, context: Any) -> dict[str, list[int]]:
        metadata = self.metadata_from_context(context)
        surface_tag, boundary_tags = self.add_rectangle_surface(
            gmsh,
            metadata.boundary_vertices,
        )
        gmsh.model.occ.synchronize()
        if context.mesh_size is not None:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", context.mesh_size)
        return {
            "surface_tags": [surface_tag],
            "boundary_tags": boundary_tags,
        }


def build_domain(gmsh: Any, context: Any) -> dict[str, list[int]]:
    return RectangleDomainBuilder().build(gmsh, context)
