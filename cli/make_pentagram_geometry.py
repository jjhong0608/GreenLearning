from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.logging import RichHandler

from greenonet.complex_geometry import load_complex_geometry


logger = logging.getLogger(__name__)

GOLDEN_RATIO = 0.5 * (1.0 + math.sqrt(5.0))


@dataclass(frozen=True)
class PentagramGeometryConfig:
    """Configuration for centered regular-pentagram geometry generation."""

    step_size: float
    out: Path
    outer_radius: float
    boundary_tol: float = 1e-12
    overwrite: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.step_size):
            raise ValueError("--step-size must be finite.")
        if self.step_size <= 0.0:
            raise ValueError("--step-size must be positive.")
        if not math.isfinite(self.outer_radius):
            raise ValueError("--outer-radius must be finite.")
        if self.outer_radius <= 0.0:
            raise ValueError("--outer-radius must be positive.")
        if not math.isfinite(self.boundary_tol):
            raise ValueError("--boundary-tol must be finite.")
        if self.boundary_tol < 0.0:
            raise ValueError("--boundary-tol must be non-negative.")
        if self.boundary_tol >= self.inner_radius:
            raise ValueError(
                "--boundary-tol must be smaller than the derived inner radius."
            )

    @property
    def inner_radius(self) -> float:
        return self.outer_radius / GOLDEN_RATIO**2


@dataclass(frozen=True)
class AxisSegment:
    """One connected axial interval and its valid-point indices."""

    grid_index: int
    fixed_coordinate: float
    start: float
    end: float
    point_indices: tuple[int, ...]

    @property
    def length(self) -> float:
        return self.end - self.start


class GeometryValidationMixin:
    """Fail-fast validation helpers shared by pentagram builder steps."""

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _validate_local_coordinates(values: np.ndarray, field_name: str) -> None:
        if np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError(f"{field_name} values must be strictly inside (0, 1).")


class PentagramPolygonMixin(GeometryValidationMixin):
    """Regular-pentagram vertices and robust polygon scanline operations."""

    GOLDEN_RATIO = GOLDEN_RATIO
    ORIENTATION_ANGLE = math.pi / 2.0
    CENTER = np.array([0.0, 0.0], dtype=np.float64)
    NUM_BOUNDARY_VERTICES = 10

    @property
    def _numeric_tol(self) -> float:
        scale = max(1.0, self.config.outer_radius)
        return 128.0 * np.finfo(np.float64).eps * scale

    def boundary_vertices(self) -> np.ndarray:
        indices = np.arange(self.NUM_BOUNDARY_VERTICES, dtype=np.float64)
        angles = self.ORIENTATION_ANGLE + indices * math.pi / 5.0
        radii = np.where(
            np.arange(self.NUM_BOUNDARY_VERTICES) % 2 == 0,
            self.config.outer_radius,
            self.config.inner_radius,
        )
        vertices = np.column_stack(
            (radii * np.cos(angles), radii * np.sin(angles))
        ).astype(np.float64, copy=False)
        vertices[np.abs(vertices) <= self._numeric_tol] = 0.0
        vertices[0] = (0.0, self.config.outer_radius)
        self._validate_polygon(vertices)
        return vertices

    def _validate_polygon(self, vertices: np.ndarray) -> None:
        if vertices.shape != (self.NUM_BOUNDARY_VERTICES, 2):
            raise ValueError("Pentagram boundary vertices must have shape (10, 2).")
        if not np.isfinite(vertices).all():
            raise ValueError("Pentagram boundary vertices must be finite.")
        edge_vectors = np.roll(vertices, -1, axis=0) - vertices
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        if np.any(edge_lengths <= self._numeric_tol):
            raise ValueError("Adjacent pentagram boundary vertices must be distinct.")
        if self._signed_area(vertices) <= self._numeric_tol:
            raise ValueError("Pentagram boundary vertices must be counter-clockwise.")
        if self._has_self_intersection(vertices):
            raise ValueError("Pentagram boundary polygon must not self-intersect.")

    @staticmethod
    def _signed_area(vertices: np.ndarray) -> float:
        next_vertices = np.roll(vertices, -1, axis=0)
        return 0.5 * float(
            np.sum(
                vertices[:, 0] * next_vertices[:, 1]
                - vertices[:, 1] * next_vertices[:, 0]
            )
        )

    def _has_self_intersection(self, vertices: np.ndarray) -> bool:
        count = int(vertices.shape[0])
        for first_index in range(count):
            first_next = (first_index + 1) % count
            for second_index in range(first_index + 1, count):
                second_next = (second_index + 1) % count
                if (
                    first_index == second_index
                    or first_next == second_index
                    or second_next == first_index
                ):
                    continue
                if self._segments_intersect(
                    vertices[first_index],
                    vertices[first_next],
                    vertices[second_index],
                    vertices[second_next],
                ):
                    return True
        return False

    def _segments_intersect(
        self,
        first_start: np.ndarray,
        first_end: np.ndarray,
        second_start: np.ndarray,
        second_end: np.ndarray,
    ) -> bool:
        orientations = (
            self._cross(first_start, first_end, second_start),
            self._cross(first_start, first_end, second_end),
            self._cross(second_start, second_end, first_start),
            self._cross(second_start, second_end, first_end),
        )
        first_opposite = orientations[0] * orientations[1] < -(self._numeric_tol**2)
        second_opposite = orientations[2] * orientations[3] < -(self._numeric_tol**2)
        if first_opposite and second_opposite:
            return True
        for value, point, start, end in (
            (orientations[0], second_start, first_start, first_end),
            (orientations[1], second_end, first_start, first_end),
            (orientations[2], first_start, second_start, second_end),
            (orientations[3], first_end, second_start, second_end),
        ):
            if abs(value) <= self._numeric_tol and self._point_on_segment(
                point, start, end
            ):
                return True
        return False

    @staticmethod
    def _cross(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
        first = end - start
        second = point - start
        return float(first[0] * second[1] - first[1] * second[0])

    def _point_on_segment(
        self,
        point: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> bool:
        minimum = np.minimum(start, end) - self._numeric_tol
        maximum = np.maximum(start, end) + self._numeric_tol
        return bool(np.all(point >= minimum) and np.all(point <= maximum))

    def _axis_intervals(
        self,
        vertices: np.ndarray,
        *,
        fixed_coordinate: float,
        coordinate_axis: int,
    ) -> list[tuple[float, float]]:
        if coordinate_axis not in (0, 1):
            raise ValueError("coordinate_axis must be 0 or 1.")
        fixed_axis = 1 - coordinate_axis
        intersections: list[float] = []
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            start_fixed = float(start[fixed_axis])
            end_fixed = float(end[fixed_axis])
            if abs(end_fixed - start_fixed) <= self._numeric_tol:
                continue

            scan_value = float(fixed_coordinate)
            if abs(scan_value - start_fixed) <= self._numeric_tol:
                scan_value = start_fixed
            elif abs(scan_value - end_fixed) <= self._numeric_tol:
                scan_value = end_fixed

            crosses = (start_fixed <= scan_value < end_fixed) or (
                end_fixed <= scan_value < start_fixed
            )
            if not crosses:
                continue
            fraction = (scan_value - start_fixed) / (end_fixed - start_fixed)
            intersection = float(
                start[coordinate_axis]
                + fraction * (end[coordinate_axis] - start[coordinate_axis])
            )
            if abs(intersection) <= self._numeric_tol:
                intersection = 0.0
            intersections.append(intersection)

        fixed_values = vertices[:, fixed_axis]
        for vertex_index, vertex_fixed in enumerate(fixed_values):
            if abs(float(vertex_fixed) - fixed_coordinate) > self._numeric_tol:
                continue
            previous_delta = float(
                fixed_values[(vertex_index - 1) % len(vertices)] - vertex_fixed
            )
            following_delta = float(
                fixed_values[(vertex_index + 1) % len(vertices)] - vertex_fixed
            )
            if (
                abs(previous_delta) <= self._numeric_tol
                or abs(following_delta) <= self._numeric_tol
                or previous_delta * following_delta <= 0.0
            ):
                continue
            tangent_coordinate = float(vertices[vertex_index, coordinate_axis])
            intersections.extend((tangent_coordinate, tangent_coordinate))
        return self._pair_intersections(intersections)

    def _pair_intersections(
        self,
        intersections: Sequence[float],
    ) -> list[tuple[float, float]]:
        values = sorted(float(value) for value in intersections)
        normalized: list[float] = []
        index = 0
        while index < len(values):
            end = index + 1
            while (
                end < len(values)
                and abs(values[end] - values[index]) <= self._numeric_tol
            ):
                end += 1
            multiplicity = end - index
            representative = float(np.mean(values[index:end]))
            normalized.extend([representative] * (2 if multiplicity % 2 == 0 else 1))
            index = end
        if len(normalized) % 2 != 0:
            raise ValueError(
                "Polygon scanline must contain an even number of intersections."
            )
        intervals: list[tuple[float, float]] = []
        for start, end in zip(normalized[::2], normalized[1::2]):
            if end - start > self._numeric_tol:
                intervals.append((start, end))
        return intervals

    @staticmethod
    def _distance_to_boundary(
        points: np.ndarray,
        vertices: np.ndarray,
    ) -> np.ndarray:
        point_array = np.asarray(points, dtype=np.float64)
        vertex_array = np.asarray(vertices, dtype=np.float64)
        starts = vertex_array
        ends = np.roll(vertex_array, -1, axis=0)
        vectors = ends - starts
        length_squared = np.sum(vectors * vectors, axis=1)
        offsets = point_array[:, None, :] - starts[None, :, :]
        fractions = np.sum(offsets * vectors[None, :, :], axis=2) / length_squared
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        distances = np.linalg.norm(point_array[:, None, :] - closest, axis=2)
        return np.min(distances, axis=1)


class PentagramGeometryBuilder(PentagramPolygonMixin):
    """Build complex-geometry metadata for a filled regular pentagram."""

    INTEGER_TOL = 1e-10

    def __init__(
        self,
        config: PentagramGeometryConfig,
        build_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = build_logger if build_logger is not None else logger

    def write(self) -> Path:
        if self.config.out.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Output file already exists: {self.config.out}. "
                "Pass --overwrite to replace it."
            )
        self.config.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.config.out, **self.build())
        self.logger.info("Wrote pentagram geometry metadata to %s", self.config.out)
        if self.config.validate:
            load_complex_geometry(self.config.out)
            self.logger.info("Validated geometry metadata with load_complex_geometry")
        return self.config.out

    def build(self) -> dict[str, np.ndarray]:
        grid = self._build_grid()
        vertices = self.boundary_vertices()
        x_intervals = [
            self._axis_intervals(
                vertices,
                fixed_coordinate=float(value),
                coordinate_axis=0,
            )
            for value in grid
        ]
        y_intervals = [
            self._axis_intervals(
                vertices,
                fixed_coordinate=float(value),
                coordinate_axis=1,
            )
            for value in grid
        ]
        coords_valid, valid_y, valid_x, point_index_by_grid = self._build_valid_points(
            grid, vertices, x_intervals
        )
        x_segments = self._build_x_segments(
            grid,
            point_index_by_grid,
            coords_valid,
            x_intervals,
        )
        y_segments = self._build_y_segments(
            grid,
            point_index_by_grid,
            coords_valid,
            y_intervals,
        )
        self._require(x_segments, "Pentagram geometry has no x-axis segments.")
        self._require(y_segments, "Pentagram geometry has no y-axis segments.")

        x_segment_id = np.full(coords_valid.shape[0], -1, dtype=np.int64)
        y_segment_id = np.full(coords_valid.shape[0], -1, dtype=np.int64)
        x_local_t = np.zeros(coords_valid.shape[0], dtype=np.float64)
        y_local_t = np.zeros(coords_valid.shape[0], dtype=np.float64)
        self._assign_segment_coordinates(
            segments=x_segments,
            coords_valid=coords_valid,
            segment_id=x_segment_id,
            local_t=x_local_t,
            coordinate_axis=0,
            field_name="x_local_t",
        )
        self._assign_segment_coordinates(
            segments=y_segments,
            coords_valid=coords_valid,
            segment_id=y_segment_id,
            local_t=y_local_t,
            coordinate_axis=1,
            field_name="y_local_t",
        )
        self._require(
            bool(np.all(x_segment_id >= 0) and np.all(y_segment_id >= 0)),
            "Every valid point must belong to one x-segment and one y-segment.",
        )

        x_recon_ptr, x_recon_t, x_recon_weight, x_recon_valid_index = (
            self._build_reconstruction_arrays(x_segments, x_local_t)
        )
        y_recon_ptr, y_recon_t, y_recon_weight, y_recon_valid_index = (
            self._build_reconstruction_arrays(y_segments, y_local_t)
        )
        payload: dict[str, np.ndarray] = {
            "coords_valid": coords_valid,
            "valid_grid_y_index": valid_y,
            "valid_grid_x_index": valid_x,
            "x_segment_id": x_segment_id,
            "y_segment_id": y_segment_id,
            "x_local_t": x_local_t,
            "y_local_t": y_local_t,
            "x_segment_left": self._segment_array(x_segments, "start"),
            "x_segment_right": self._segment_array(x_segments, "end"),
            "x_segment_y": np.array(
                [segment.fixed_coordinate for segment in x_segments],
                dtype=np.float64,
            ),
            "x_segment_length": self._segment_lengths(x_segments),
            "y_segment_bottom": self._segment_array(y_segments, "start"),
            "y_segment_top": self._segment_array(y_segments, "end"),
            "y_segment_x": np.array(
                [segment.fixed_coordinate for segment in y_segments],
                dtype=np.float64,
            ),
            "y_segment_length": self._segment_lengths(y_segments),
            "x_recon_ptr": x_recon_ptr,
            "x_recon_t": x_recon_t,
            "x_recon_weight": x_recon_weight,
            "x_recon_valid_index": x_recon_valid_index,
            "y_recon_ptr": y_recon_ptr,
            "y_recon_t": y_recon_t,
            "y_recon_weight": y_recon_weight,
            "y_recon_valid_index": y_recon_valid_index,
            "x_edges": self._build_edges(x_segments, x_local_t),
            "y_edges": self._build_edges(y_segments, y_local_t),
            "hx": np.array(self.config.step_size, dtype=np.float64),
            "hy": np.array(self.config.step_size, dtype=np.float64),
            "domain_type": np.array("regular_pentagram"),
            "outer_radius": np.array(self.config.outer_radius, dtype=np.float64),
            "inner_radius": np.array(self.config.inner_radius, dtype=np.float64),
            "center": self.CENTER.copy(),
            "orientation_angle": np.array(
                self.ORIENTATION_ANGLE,
                dtype=np.float64,
            ),
            "fill_rule": np.array("filled_simple_decagon"),
            "has_hole": np.array(False),
            "boundary_vertices": vertices,
            "step_size": np.array(self.config.step_size, dtype=np.float64),
            "boundary_tol": np.array(self.config.boundary_tol, dtype=np.float64),
            "grid_x": grid.copy(),
            "grid_y": grid.copy(),
        }
        self.logger.info(
            "Built regular pentagram with outer_radius=%s, inner_radius=%s, "
            "%d valid points, %d x-segments, %d y-segments",
            self.config.outer_radius,
            self.config.inner_radius,
            coords_valid.shape[0],
            len(x_segments),
            len(y_segments),
        )
        return payload

    def _build_grid(self) -> np.ndarray:
        diameter = 2.0 * self.config.outer_radius
        n_intervals = round(diameter / self.config.step_size)
        if abs(n_intervals * self.config.step_size - diameter) > self.INTEGER_TOL:
            raise ValueError(
                "--step-size must divide the interval [-outer-radius, outer-radius]; "
                "2 * outer_radius / step_size must be an integer."
            )
        grid = np.linspace(
            -self.config.outer_radius,
            self.config.outer_radius,
            n_intervals + 1,
            dtype=np.float64,
        )
        if grid.size < 3:
            raise ValueError(
                "--step-size is too large to produce interior grid points."
            )
        self.logger.info(
            "Generated %d by %d pentagram grid with outer_radius=%s and step_size=%s",
            grid.size,
            grid.size,
            self.config.outer_radius,
            self.config.step_size,
        )
        return grid

    def _build_valid_points(
        self,
        grid: np.ndarray,
        vertices: np.ndarray,
        x_intervals: Sequence[Sequence[tuple[float, float]]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
        coords: list[tuple[float, float]] = []
        valid_y: list[int] = []
        valid_x: list[int] = []
        point_index_by_grid: dict[tuple[int, int], int] = {}
        exclusion_tol = max(self.config.boundary_tol, self._numeric_tol)
        for row_index, (y_value, intervals) in enumerate(zip(grid, x_intervals)):
            for col_index, x_value in enumerate(grid):
                if not any(start < x_value < end for start, end in intervals):
                    continue
                point = np.array([[x_value, y_value]], dtype=np.float64)
                if self._distance_to_boundary(point, vertices)[0] <= exclusion_tol:
                    continue
                point_index = len(coords)
                coords.append((float(x_value), float(y_value)))
                valid_y.append(row_index)
                valid_x.append(col_index)
                point_index_by_grid[(row_index, col_index)] = point_index
        self._require(coords, "Pentagram geometry contains no valid interior points.")
        return (
            np.array(coords, dtype=np.float64),
            np.array(valid_y, dtype=np.int64),
            np.array(valid_x, dtype=np.int64),
            point_index_by_grid,
        )

    def _build_x_segments(
        self,
        grid: np.ndarray,
        point_index_by_grid: dict[tuple[int, int], int],
        coords_valid: np.ndarray,
        intervals_by_row: Sequence[Sequence[tuple[float, float]]],
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for row_index, (y_value, intervals) in enumerate(zip(grid, intervals_by_row)):
            point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for col_index in range(grid.size)
                if (row_index, col_index) in point_index_by_grid
            )
            segments.extend(
                self._segments_for_intervals(
                    intervals=intervals,
                    point_indices=point_indices,
                    coords_valid=coords_valid,
                    coordinate_axis=0,
                    grid_index=row_index,
                    fixed_coordinate=float(y_value),
                )
            )
        return segments

    def _build_y_segments(
        self,
        grid: np.ndarray,
        point_index_by_grid: dict[tuple[int, int], int],
        coords_valid: np.ndarray,
        intervals_by_column: Sequence[Sequence[tuple[float, float]]],
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for col_index, (x_value, intervals) in enumerate(
            zip(grid, intervals_by_column)
        ):
            point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for row_index in range(grid.size)
                if (row_index, col_index) in point_index_by_grid
            )
            segments.extend(
                self._segments_for_intervals(
                    intervals=intervals,
                    point_indices=point_indices,
                    coords_valid=coords_valid,
                    coordinate_axis=1,
                    grid_index=col_index,
                    fixed_coordinate=float(x_value),
                )
            )
        return segments

    @staticmethod
    def _segments_for_intervals(
        *,
        intervals: Sequence[tuple[float, float]],
        point_indices: Sequence[int],
        coords_valid: np.ndarray,
        coordinate_axis: int,
        grid_index: int,
        fixed_coordinate: float,
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for start, end in intervals:
            included = tuple(
                point_index
                for point_index in point_indices
                if start < coords_valid[point_index, coordinate_axis] < end
            )
            if included:
                segments.append(
                    AxisSegment(
                        grid_index=grid_index,
                        fixed_coordinate=fixed_coordinate,
                        start=float(start),
                        end=float(end),
                        point_indices=included,
                    )
                )
        return segments

    def _assign_segment_coordinates(
        self,
        *,
        segments: Sequence[AxisSegment],
        coords_valid: np.ndarray,
        segment_id: np.ndarray,
        local_t: np.ndarray,
        coordinate_axis: int,
        field_name: str,
    ) -> None:
        for segment_index, segment in enumerate(segments):
            self._require(segment.length > 0.0, "Segment length must be positive.")
            values = coords_valid[list(segment.point_indices), coordinate_axis]
            local_values = (values - segment.start) / segment.length
            self._validate_local_coordinates(local_values, field_name)
            segment_id[list(segment.point_indices)] = segment_index
            local_t[list(segment.point_indices)] = local_values

    @staticmethod
    def _segment_array(
        segments: Sequence[AxisSegment],
        attr_name: str,
    ) -> np.ndarray:
        return np.array(
            [float(getattr(segment, attr_name)) for segment in segments],
            dtype=np.float64,
        )

    @staticmethod
    def _segment_lengths(segments: Sequence[AxisSegment]) -> np.ndarray:
        return np.array([segment.length for segment in segments], dtype=np.float64)

    @staticmethod
    def _build_reconstruction_arrays(
        segments: Sequence[AxisSegment],
        local_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ptr = [0]
        recon_t: list[float] = []
        recon_weight: list[float] = []
        recon_valid_index: list[int] = []
        for segment in segments:
            interior_indices = sorted(
                segment.point_indices,
                key=lambda point_index: float(local_t[point_index]),
            )
            segment_t = np.array(
                [0.0, *(float(local_t[index]) for index in interior_indices), 1.0],
                dtype=np.float64,
            )
            weights = PentagramGeometryBuilder._trapezoid_weights(segment_t)
            recon_t.extend(float(value) for value in segment_t)
            recon_weight.extend(float(value) for value in weights)
            recon_valid_index.extend([-1, *interior_indices, -1])
            ptr.append(len(recon_t))
        return (
            np.array(ptr, dtype=np.int64),
            np.array(recon_t, dtype=np.float64),
            np.array(recon_weight, dtype=np.float64),
            np.array(recon_valid_index, dtype=np.int64),
        )

    @staticmethod
    def _trapezoid_weights(t_values: np.ndarray) -> np.ndarray:
        if t_values.size < 2:
            raise ValueError("A reconstruction segment needs at least two nodes.")
        if np.any(np.diff(t_values) <= 0.0):
            raise ValueError("Reconstruction nodes must be strictly increasing.")
        weights = np.empty_like(t_values, dtype=np.float64)
        weights[0] = 0.5 * (t_values[1] - t_values[0])
        weights[-1] = 0.5 * (t_values[-1] - t_values[-2])
        if t_values.size > 2:
            weights[1:-1] = 0.5 * (t_values[2:] - t_values[:-2])
        return weights

    @staticmethod
    def _build_edges(
        segments: Sequence[AxisSegment],
        local_t: np.ndarray,
    ) -> np.ndarray:
        edges: list[tuple[int, int]] = []
        for segment in segments:
            interior_indices = sorted(
                segment.point_indices,
                key=lambda point_index: float(local_t[point_index]),
            )
            edges.extend(zip(interior_indices[:-1], interior_indices[1:]))
        if not edges:
            return np.empty((0, 2), dtype=np.int64)
        return np.array(edges, dtype=np.int64)


class MakePentagramGeometryCLI:
    """Command-line surface for regular-pentagram geometry generation."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Generate a centered regular-pentagram geometry NPZ file."
        )
        parser.add_argument(
            "--step-size",
            type=float,
            required=True,
            help=(
                "Positive grid spacing; 2 * outer_radius / step_size "
                "must be an integer."
            ),
        )
        parser.add_argument(
            "--outer-radius",
            type=float,
            required=True,
            help="Positive circumradius of the regular pentagram.",
        )
        parser.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Path to the geometry NPZ file to create.",
        )
        parser.add_argument(
            "--boundary-tol",
            type=float,
            default=1e-12,
            help="Physical-distance tolerance used to exclude boundary points.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite the output file if it already exists.",
        )
        parser.add_argument(
            "--validate",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Validate the saved NPZ with load_complex_geometry after writing.",
        )
        self.parser = parser

    @staticmethod
    def _build_logger(out_dir: Path) -> logging.Logger:
        out_dir.mkdir(parents=True, exist_ok=True)
        build_logger = logging.getLogger("MakePentagramGeometry")
        build_logger.handlers.clear()
        build_logger.propagate = False
        build_logger.setLevel(logging.INFO)
        logging.root.handlers.clear()

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(
            out_dir / "make_pentagram_geometry.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        build_logger.addHandler(rich_handler)
        build_logger.addHandler(file_handler)
        return build_logger

    def run(self, argv: Sequence[str] | None = None) -> Path:
        args = self.parser.parse_args(argv)
        config = PentagramGeometryConfig(
            step_size=float(args.step_size),
            out=args.out,
            outer_radius=float(args.outer_radius),
            boundary_tol=float(args.boundary_tol),
            overwrite=bool(args.overwrite),
            validate=bool(args.validate),
        )
        build_logger = self._build_logger(config.out.parent)
        output_path = PentagramGeometryBuilder(config, build_logger).write()
        build_logger.info("Completed pentagram geometry generation")
        return output_path


def main() -> None:
    MakePentagramGeometryCLI().run()


if __name__ == "__main__":
    main()
