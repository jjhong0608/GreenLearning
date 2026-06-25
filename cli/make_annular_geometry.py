from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from rich.logging import RichHandler

from greenonet.complex_geometry import load_complex_geometry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnularGeometryConfig:
    """Configuration for centered annular complex-geometry generation."""

    step_size: float
    out: Path
    inner_radius: float
    outer_radius: float
    boundary_tol: float = 1e-12
    overwrite: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        if self.step_size <= 0.0:
            raise ValueError("--step-size must be positive.")
        if self.inner_radius <= 0.0:
            raise ValueError("--inner-radius must be positive.")
        if self.outer_radius <= 0.0:
            raise ValueError("--outer-radius must be positive.")
        if self.outer_radius <= self.inner_radius:
            raise ValueError("--outer-radius must be greater than --inner-radius.")
        if self.boundary_tol < 0.0:
            raise ValueError("--boundary-tol must be non-negative.")
        if self.inner_radius * self.inner_radius <= self.boundary_tol:
            raise ValueError(
                "--boundary-tol must be smaller than inner radius squared."
            )
        if self.outer_radius * self.outer_radius <= self.boundary_tol:
            raise ValueError(
                "--boundary-tol must be smaller than outer radius squared."
            )


@dataclass(frozen=True)
class AxisSegment:
    """One connected axial interval and the valid-point indices lying on it."""

    grid_index: int
    fixed_coordinate: float
    start: float
    end: float
    point_indices: tuple[int, ...]

    @property
    def length(self) -> float:
        return self.end - self.start


class GeometryValidationMixin:
    """Small fail-fast validation helpers shared by the builder steps."""

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _validate_local_coordinates(values: np.ndarray, field_name: str) -> None:
        if np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError(f"{field_name} values must be strictly inside (0, 1).")


class AnnularGeometryBuilder(GeometryValidationMixin):
    """Build and write complex-geometry metadata for a centered annulus."""

    INTEGER_TOL = 1e-10
    CENTER = np.array([0.0, 0.0], dtype=np.float64)

    def __init__(
        self,
        config: AnnularGeometryConfig,
        build_logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = build_logger if build_logger is not None else logger

    def write(self) -> Path:
        """Write the geometry NPZ and optionally validate it with the runtime loader."""

        if self.config.out.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Output file already exists: {self.config.out}. "
                "Pass --overwrite to replace it."
            )
        self.config.out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.build()
        np.savez(self.config.out, **payload)
        self.logger.info("Wrote annular geometry metadata to %s", self.config.out)
        if self.config.validate:
            load_complex_geometry(self.config.out)
            self.logger.info("Validated geometry metadata with load_complex_geometry")
        return self.config.out

    def build(self) -> dict[str, np.ndarray]:
        grid = self._build_grid()
        coords_valid, valid_y, valid_x, point_index_by_grid = self._build_valid_points(
            grid
        )
        x_segments = self._build_x_segments(grid, point_index_by_grid, coords_valid)
        y_segments = self._build_y_segments(grid, point_index_by_grid, coords_valid)
        self._require(x_segments, "Annular geometry has no x-axis segments.")
        self._require(y_segments, "Annular geometry has no y-axis segments.")

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
            "domain_type": np.array("annulus"),
            "inner_radius": np.array(self.config.inner_radius, dtype=np.float64),
            "outer_radius": np.array(self.config.outer_radius, dtype=np.float64),
            "center": self.CENTER.copy(),
            "step_size": np.array(self.config.step_size, dtype=np.float64),
            "boundary_tol": np.array(self.config.boundary_tol, dtype=np.float64),
            "grid_x": grid.copy(),
            "grid_y": grid.copy(),
        }
        self.logger.info(
            "Built annular geometry with inner_radius=%s, outer_radius=%s, "
            "%d valid points, %d x-segments, %d y-segments",
            self.config.inner_radius,
            self.config.outer_radius,
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
            "Generated %d by %d annular grid with outer_radius=%s and step_size=%s",
            grid.size,
            grid.size,
            self.config.outer_radius,
            self.config.step_size,
        )
        return grid

    def _build_valid_points(
        self,
        grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
        coords: list[tuple[float, float]] = []
        valid_y: list[int] = []
        valid_x: list[int] = []
        point_index_by_grid: dict[tuple[int, int], int] = {}
        inner_threshold = self.config.inner_radius**2 + self.config.boundary_tol
        outer_threshold = self.config.outer_radius**2 - self.config.boundary_tol
        for row_index, y_value in enumerate(grid):
            for col_index, x_value in enumerate(grid):
                radius_sq = x_value * x_value + y_value * y_value
                if inner_threshold < radius_sq < outer_threshold:
                    point_index = len(coords)
                    coords.append((float(x_value), float(y_value)))
                    valid_y.append(row_index)
                    valid_x.append(col_index)
                    point_index_by_grid[(row_index, col_index)] = point_index
        self._require(coords, "Annular geometry contains no valid interior points.")
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
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for row_index, y_value in enumerate(grid):
            row_point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for col_index in range(grid.size)
                if (row_index, col_index) in point_index_by_grid
            )
            for start, end in self._intervals_for_fixed(float(y_value)):
                point_indices = self._points_in_interval(
                    row_point_indices,
                    coords_valid,
                    coordinate_axis=0,
                    start=start,
                    end=end,
                )
                if point_indices:
                    segments.append(
                        AxisSegment(
                            grid_index=row_index,
                            fixed_coordinate=float(y_value),
                            start=start,
                            end=end,
                            point_indices=point_indices,
                        )
                    )
        return segments

    def _build_y_segments(
        self,
        grid: np.ndarray,
        point_index_by_grid: dict[tuple[int, int], int],
        coords_valid: np.ndarray,
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for col_index, x_value in enumerate(grid):
            col_point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for row_index in range(grid.size)
                if (row_index, col_index) in point_index_by_grid
            )
            for start, end in self._intervals_for_fixed(float(x_value)):
                point_indices = self._points_in_interval(
                    col_point_indices,
                    coords_valid,
                    coordinate_axis=1,
                    start=start,
                    end=end,
                )
                if point_indices:
                    segments.append(
                        AxisSegment(
                            grid_index=col_index,
                            fixed_coordinate=float(x_value),
                            start=start,
                            end=end,
                            point_indices=point_indices,
                        )
                    )
        return segments

    def _intervals_for_fixed(
        self, fixed_coordinate: float
    ) -> list[tuple[float, float]]:
        abs_fixed = abs(fixed_coordinate)
        if abs_fixed >= self.config.outer_radius - self.config.boundary_tol:
            return []
        outer_half = float(
            np.sqrt(max(self.config.outer_radius**2 - fixed_coordinate**2, 0.0))
        )
        if abs_fixed <= self.config.inner_radius + self.config.boundary_tol:
            inner_half = float(
                np.sqrt(max(self.config.inner_radius**2 - fixed_coordinate**2, 0.0))
            )
            return [(-outer_half, -inner_half), (inner_half, outer_half)]
        return [(-outer_half, outer_half)]

    @staticmethod
    def _points_in_interval(
        point_indices: Sequence[int],
        coords_valid: np.ndarray,
        *,
        coordinate_axis: int,
        start: float,
        end: float,
    ) -> tuple[int, ...]:
        return tuple(
            point_index
            for point_index in point_indices
            if start < float(coords_valid[point_index, coordinate_axis]) < end
        )

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
            weights = AnnularGeometryBuilder._trapezoid_weights(segment_t)
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


class MakeAnnularGeometryCLI:
    """Command-line surface for centered annular geometry generation."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Generate a centered annular complex-geometry NPZ file."
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
            "--inner-radius",
            type=float,
            required=True,
            help="Positive inner radius for the annular hole.",
        )
        parser.add_argument(
            "--outer-radius",
            type=float,
            required=True,
            help="Positive outer radius; must be greater than inner radius.",
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
            help="Tolerance used to exclude inner and outer boundary points.",
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
        build_logger = logging.getLogger("MakeAnnularGeometry")
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
            out_dir / "make_annular_geometry.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        build_logger.addHandler(rich_handler)
        build_logger.addHandler(file_handler)
        return build_logger

    def run(self, argv: Sequence[str] | None = None) -> Path:
        args = self.parser.parse_args(argv)
        config = AnnularGeometryConfig(
            step_size=float(args.step_size),
            out=args.out,
            inner_radius=float(args.inner_radius),
            outer_radius=float(args.outer_radius),
            boundary_tol=float(args.boundary_tol),
            overwrite=bool(args.overwrite),
            validate=bool(args.validate),
        )
        build_logger = self._build_logger(config.out.parent)
        output_path = AnnularGeometryBuilder(config, build_logger).write()
        build_logger.info("Completed annular geometry generation")
        return output_path


def main() -> None:
    MakeAnnularGeometryCLI().run()


if __name__ == "__main__":
    main()
