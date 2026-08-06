from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


class PentagramDomainMetadata:
    """Validated NPZ metadata used as the Gmsh boundary source of truth."""

    def __init__(
        self,
        *,
        boundary_vertices: np.ndarray,
        outer_radius: float,
        inner_radius: float,
    ) -> None:
        self.boundary_vertices = boundary_vertices
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius


class PentagramGmshMixin:
    """Metadata validation and OCC helpers for the regular pentagram domain."""

    GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5.0))
    ORIENTATION_ANGLE = math.pi / 2.0
    DOMAIN_TYPE = "regular_pentagram"
    FILL_RULE = "filled_simple_decagon"
    NUM_BOUNDARY_VERTICES = 10
    REQUIRED_METADATA = frozenset(
        {
            "domain_type",
            "outer_radius",
            "inner_radius",
            "center",
            "orientation_angle",
            "fill_rule",
            "has_hole",
            "boundary_vertices",
        }
    )

    @classmethod
    def metadata_from_context(cls, context: Any) -> PentagramDomainMetadata:
        geometry_path = getattr(context, "geometry_path", None)
        if geometry_path is None:
            raise ValueError("Pentagram Gmsh script requires context.geometry_path.")
        with np.load(Path(geometry_path), allow_pickle=False) as raw:
            missing = sorted(cls.REQUIRED_METADATA - set(raw.files))
            if missing:
                raise KeyError(
                    "Pentagram geometry metadata is missing required keys: "
                    f"{', '.join(missing)}."
                )
            domain_type = str(np.asarray(raw["domain_type"]).item())
            fill_rule = str(np.asarray(raw["fill_rule"]).item())
            has_hole = bool(np.asarray(raw["has_hole"]).item())
            outer_radius = float(
                np.asarray(raw["outer_radius"], dtype=np.float64).item()
            )
            inner_radius = float(
                np.asarray(raw["inner_radius"], dtype=np.float64).item()
            )
            center = np.asarray(raw["center"], dtype=np.float64)
            orientation_angle = float(
                np.asarray(raw["orientation_angle"], dtype=np.float64).item()
            )
            boundary_vertices = np.asarray(
                raw["boundary_vertices"],
                dtype=np.float64,
            )

        cls._validate_metadata_contract(
            domain_type=domain_type,
            fill_rule=fill_rule,
            has_hole=has_hole,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            center=center,
            orientation_angle=orientation_angle,
            boundary_vertices=boundary_vertices,
        )
        return PentagramDomainMetadata(
            boundary_vertices=boundary_vertices,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
        )

    @classmethod
    def _validate_metadata_contract(
        cls,
        *,
        domain_type: str,
        fill_rule: str,
        has_hole: bool,
        outer_radius: float,
        inner_radius: float,
        center: np.ndarray,
        orientation_angle: float,
        boundary_vertices: np.ndarray,
    ) -> None:
        if domain_type != cls.DOMAIN_TYPE:
            raise ValueError(
                "Pentagram geometry domain_type must be "
                f"'{cls.DOMAIN_TYPE}', got '{domain_type}'."
            )
        if fill_rule != cls.FILL_RULE:
            raise ValueError(
                f"Pentagram fill_rule must be '{cls.FILL_RULE}', got '{fill_rule}'."
            )
        if has_hole:
            raise ValueError("Pentagram has_hole metadata must be false.")
        if not math.isfinite(outer_radius) or outer_radius <= 0.0:
            raise ValueError(
                "Pentagram outer_radius metadata must be finite and positive."
            )
        if not math.isfinite(inner_radius) or inner_radius <= 0.0:
            raise ValueError(
                "Pentagram inner_radius metadata must be finite and positive."
            )
        expected_inner = outer_radius / cls.GOLDEN_RATIO**2
        tolerance = 1.0e-10 * max(1.0, outer_radius)
        if not math.isclose(inner_radius, expected_inner, abs_tol=tolerance):
            raise ValueError(
                "Pentagram inner_radius metadata must equal outer_radius / phi^2."
            )
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Pentagram center metadata must have finite shape (2,).")
        if not np.allclose(center, 0.0, rtol=0.0, atol=tolerance):
            raise ValueError("Pentagram center metadata must be [0, 0].")
        if not math.isfinite(orientation_angle) or not math.isclose(
            orientation_angle,
            cls.ORIENTATION_ANGLE,
            abs_tol=tolerance,
        ):
            raise ValueError("Pentagram orientation_angle metadata must be pi / 2.")
        cls._validate_boundary_vertices(
            boundary_vertices,
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            tolerance=tolerance,
        )

    @classmethod
    def _validate_boundary_vertices(
        cls,
        vertices: np.ndarray,
        *,
        outer_radius: float,
        inner_radius: float,
        tolerance: float,
    ) -> None:
        if vertices.shape != (cls.NUM_BOUNDARY_VERTICES, 2):
            raise ValueError("Pentagram boundary_vertices must have shape (10, 2).")
        if not np.isfinite(vertices).all():
            raise ValueError("Pentagram boundary_vertices must be finite.")
        expected_top = np.array([0.0, outer_radius], dtype=np.float64)
        if not np.allclose(vertices[0], expected_top, rtol=0.0, atol=tolerance):
            raise ValueError(
                "Pentagram first boundary vertex must be (0, outer_radius)."
            )
        if not np.allclose(
            np.linalg.norm(vertices[0::2], axis=1),
            outer_radius,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Pentagram outer boundary vertices have invalid radii.")
        if not np.allclose(
            np.linalg.norm(vertices[1::2], axis=1),
            inner_radius,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("Pentagram inner boundary vertices have invalid radii.")
        edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
        if np.any(edge_lengths <= tolerance):
            raise ValueError("Adjacent pentagram boundary vertices must be distinct.")
        if cls._signed_area(vertices) <= tolerance:
            raise ValueError("Pentagram boundary_vertices must be counter-clockwise.")
        if cls._has_self_intersection(vertices, tolerance):
            raise ValueError("Pentagram boundary_vertices must not self-intersect.")

    @staticmethod
    def _signed_area(vertices: np.ndarray) -> float:
        following = np.roll(vertices, -1, axis=0)
        return 0.5 * float(
            np.sum(vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0])
        )

    @classmethod
    def _has_self_intersection(cls, vertices: np.ndarray, tolerance: float) -> bool:
        count = int(vertices.shape[0])
        for first_index in range(count):
            first_next = (first_index + 1) % count
            for second_index in range(first_index + 1, count):
                second_next = (second_index + 1) % count
                if first_next == second_index or second_next == first_index:
                    continue
                if cls._segments_intersect(
                    vertices[first_index],
                    vertices[first_next],
                    vertices[second_index],
                    vertices[second_next],
                    tolerance,
                ):
                    return True
        return False

    @classmethod
    def _segments_intersect(
        cls,
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
        tolerance: float,
    ) -> bool:
        orientations = (
            cls._cross(first_start, first_end, second_start),
            cls._cross(first_start, first_end, second_end),
            cls._cross(second_start, second_end, first_start),
            cls._cross(second_start, second_end, first_end),
        )
        if (
            orientations[0] * orientations[1] < -(tolerance**2)
            and orientations[2] * orientations[3] < -(tolerance**2)
        ):
            return True
        for value, point, start, end in (
            (orientations[0], second_start, first_start, first_end),
            (orientations[1], second_end, first_start, first_end),
            (orientations[2], first_start, second_start, second_end),
            (orientations[3], first_end, second_start, second_end),
        ):
            if abs(value) <= tolerance and cls._point_on_segment(
                point, start, end, tolerance
            ):
                return True
        return False

    @staticmethod
    def _cross(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    @staticmethod
    def _point_on_segment(
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        tolerance: float,
    ) -> bool:
        return bool(
            np.all(point >= np.minimum(start, end) - tolerance)
            and np.all(point <= np.maximum(start, end) + tolerance)
        )

    @staticmethod
    def add_pentagram_surface(
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


class PentagramDomainBuilder(PentagramGmshMixin):
    """Build one filled regular-pentagram Gmsh surface from saved vertices."""

    def build(self, gmsh: Any, context: Any) -> dict[str, list[int]]:
        metadata = self.metadata_from_context(context)
        surface_tag, boundary_tags = self.add_pentagram_surface(
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
    return PentagramDomainBuilder().build(gmsh, context)
