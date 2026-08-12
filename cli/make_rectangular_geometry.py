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


@dataclass(frozen=True)
class RectangularGeometryConfig:
    """Configuration for axis-aligned rectangular geometry generation."""

    step_size: float
    out: Path
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    boundary_tol: float = 1.0e-12
    overwrite: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "step_size",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "boundary_tol",
        ):
            if not math.isfinite(float(getattr(self, field_name))):
                option_name = field_name.replace("_", "-")
                raise ValueError(f"--{option_name} must be finite.")
        if self.step_size <= 0.0:
            raise ValueError("--step-size must be positive.")
        if self.x_max <= self.x_min:
            raise ValueError("--x-max must be greater than --x-min.")
        if self.y_max <= self.y_min:
            raise ValueError("--y-max must be greater than --y-min.")
        if self.boundary_tol < 0.0:
            raise ValueError("--boundary-tol must be non-negative.")
        if self.boundary_tol >= self.step_size:
            raise ValueError("--boundary-tol must be smaller than --step-size.")


@dataclass(frozen=True)
class AxisSegment:
    """One full-width or full-height axial segment."""

    grid_index: int
    fixed_coordinate: float
    start: float
    end: float
    point_indices: tuple[int, ...]

    @property
    def length(self) -> float:
        return self.end - self.start


class GeometryValidationMixin:
    """Fail-fast validation helpers shared by the builder steps."""

    @staticmethod
    def _require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    @staticmethod
    def _validate_local_coordinates(values: np.ndarray, field_name: str) -> None:
        if np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ValueError(f"{field_name} values must be strictly inside (0, 1).")


class RectangularGeometryBuilder(GeometryValidationMixin):
    """Build complex-geometry metadata for an axis-aligned rectangle."""

    INTEGER_TOL = 1.0e-10

    def __init__(
        self,
        config: RectangularGeometryConfig,
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
        np.savez(self.config.out, **self.build())  # type: ignore[arg-type]
        self.logger.info("Wrote geometry metadata to %s", self.config.out)
        if self.config.validate:
            load_complex_geometry(self.config.out)
            self.logger.info("Validated geometry metadata with load_complex_geometry")
        return self.config.out

    def build(self) -> dict[str, np.ndarray]:
        """Construct the complete complex-geometry NPZ payload."""

        grid_x = self._build_axis_grid(
            self.config.x_min,
            self.config.x_max,
            axis_name="x",
        )
        grid_y = self._build_axis_grid(
            self.config.y_min,
            self.config.y_max,
            axis_name="y",
        )
        coords_valid, valid_y, valid_x, point_index_by_grid = self._build_valid_points(
            grid_x, grid_y
        )
        x_segments = self._build_x_segments(
            grid_x,
            grid_y,
            point_index_by_grid,
        )
        y_segments = self._build_y_segments(
            grid_x,
            grid_y,
            point_index_by_grid,
        )
        self._require(
            bool(x_segments),
            "Rectangular geometry has no x-axis segments.",
        )
        self._require(
            bool(y_segments),
            "Rectangular geometry has no y-axis segments.",
        )

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
        width = self.config.x_max - self.config.x_min
        height = self.config.y_max - self.config.y_min
        center = np.array(
            [
                0.5 * (self.config.x_min + self.config.x_max),
                0.5 * (self.config.y_min + self.config.y_max),
            ],
            dtype=np.float64,
        )
        boundary_vertices = np.array(
            [
                [self.config.x_min, self.config.y_min],
                [self.config.x_max, self.config.y_min],
                [self.config.x_max, self.config.y_max],
                [self.config.x_min, self.config.y_max],
            ],
            dtype=np.float64,
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
            "domain_type": np.array("rectangle"),
            "x_min": np.array(self.config.x_min, dtype=np.float64),
            "x_max": np.array(self.config.x_max, dtype=np.float64),
            "y_min": np.array(self.config.y_min, dtype=np.float64),
            "y_max": np.array(self.config.y_max, dtype=np.float64),
            "width": np.array(width, dtype=np.float64),
            "height": np.array(height, dtype=np.float64),
            "bounds": np.array(
                [
                    [self.config.x_min, self.config.x_max],
                    [self.config.y_min, self.config.y_max],
                ],
                dtype=np.float64,
            ),
            "center": center,
            "boundary_vertices": boundary_vertices,
            "has_hole": np.array(False, dtype=np.bool_),
            "step_size": np.array(self.config.step_size, dtype=np.float64),
            "boundary_tol": np.array(self.config.boundary_tol, dtype=np.float64),
            "grid_x": grid_x,
            "grid_y": grid_y,
        }
        self.logger.info(
            "Built rectangular geometry on [%s, %s] x [%s, %s] with "
            "%d valid points, %d x-segments, %d y-segments",
            self.config.x_min,
            self.config.x_max,
            self.config.y_min,
            self.config.y_max,
            coords_valid.shape[0],
            len(x_segments),
            len(y_segments),
        )
        return payload

    def _build_axis_grid(
        self,
        minimum: float,
        maximum: float,
        *,
        axis_name: str,
    ) -> np.ndarray:
        length = maximum - minimum
        n_intervals = round(length / self.config.step_size)
        tolerance = self.INTEGER_TOL * max(1.0, abs(minimum), abs(maximum), length)
        if abs(n_intervals * self.config.step_size - length) > tolerance:
            raise ValueError(
                f"--step-size must divide the {axis_name}-interval exactly; "
                f"({axis_name}_max - {axis_name}_min) / step_size must be an integer."
            )
        if n_intervals < 2:
            raise ValueError(
                f"--step-size is too large to produce interior {axis_name}-grid points."
            )
        grid = np.linspace(
            minimum,
            maximum,
            n_intervals + 1,
            dtype=np.float64,
        )
        self.logger.info(
            "Generated %s-grid with %d points on [%s, %s]",
            axis_name,
            grid.size,
            minimum,
            maximum,
        )
        return grid

    def _build_valid_points(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
        coords: list[tuple[float, float]] = []
        valid_y: list[int] = []
        valid_x: list[int] = []
        point_index_by_grid: dict[tuple[int, int], int] = {}
        x_lower = self.config.x_min + self.config.boundary_tol
        x_upper = self.config.x_max - self.config.boundary_tol
        y_lower = self.config.y_min + self.config.boundary_tol
        y_upper = self.config.y_max - self.config.boundary_tol
        for row_index, y_value in enumerate(grid_y):
            for col_index, x_value in enumerate(grid_x):
                if x_lower < x_value < x_upper and y_lower < y_value < y_upper:
                    point_index = len(coords)
                    coords.append((float(x_value), float(y_value)))
                    valid_y.append(row_index)
                    valid_x.append(col_index)
                    point_index_by_grid[(row_index, col_index)] = point_index
        self._require(
            bool(coords),
            "Rectangular geometry contains no valid interior points.",
        )
        return (
            np.array(coords, dtype=np.float64),
            np.array(valid_y, dtype=np.int64),
            np.array(valid_x, dtype=np.int64),
            point_index_by_grid,
        )

    def _build_x_segments(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        point_index_by_grid: dict[tuple[int, int], int],
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for row_index, y_value in enumerate(grid_y):
            point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for col_index in range(grid_x.size)
                if (row_index, col_index) in point_index_by_grid
            )
            if point_indices:
                segments.append(
                    AxisSegment(
                        grid_index=row_index,
                        fixed_coordinate=float(y_value),
                        start=self.config.x_min,
                        end=self.config.x_max,
                        point_indices=point_indices,
                    )
                )
        return segments

    def _build_y_segments(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        point_index_by_grid: dict[tuple[int, int], int],
    ) -> list[AxisSegment]:
        segments: list[AxisSegment] = []
        for col_index, x_value in enumerate(grid_x):
            point_indices = tuple(
                point_index_by_grid[(row_index, col_index)]
                for row_index in range(grid_y.size)
                if (row_index, col_index) in point_index_by_grid
            )
            if point_indices:
                segments.append(
                    AxisSegment(
                        grid_index=col_index,
                        fixed_coordinate=float(x_value),
                        start=self.config.y_min,
                        end=self.config.y_max,
                        point_indices=point_indices,
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
            point_indices = list(segment.point_indices)
            values = coords_valid[point_indices, coordinate_axis]
            local_values = (values - segment.start) / segment.length
            self._validate_local_coordinates(local_values, field_name)
            segment_id[point_indices] = segment_index
            local_t[point_indices] = local_values

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
            weights = RectangularGeometryBuilder._trapezoid_weights(segment_t)
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


class MakeRectangularGeometryCLI:
    """Command-line surface for axis-aligned rectangular geometry generation."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Generate an axis-aligned rectangular complex-geometry NPZ."
        )
        parser.add_argument(
            "--step-size",
            type=float,
            required=True,
            help=("Positive uniform grid spacing; it must divide both side lengths."),
        )
        parser.add_argument("--x-min", type=float, default=0.0)
        parser.add_argument("--x-max", type=float, default=1.0)
        parser.add_argument("--y-min", type=float, default=0.0)
        parser.add_argument("--y-max", type=float, default=1.0)
        parser.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Path to the geometry NPZ file to create.",
        )
        parser.add_argument(
            "--boundary-tol",
            type=float,
            default=1.0e-12,
            help="Non-negative tolerance used to exclude boundary grid points.",
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
        build_logger = logging.getLogger("MakeRectangularGeometry")
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
            out_dir / "make_rectangular_geometry.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        build_logger.addHandler(rich_handler)
        build_logger.addHandler(file_handler)
        return build_logger

    def run(self, argv: Sequence[str] | None = None) -> Path:
        args = self.parser.parse_args(argv)
        config = RectangularGeometryConfig(
            step_size=float(args.step_size),
            out=args.out,
            x_min=float(args.x_min),
            x_max=float(args.x_max),
            y_min=float(args.y_min),
            y_max=float(args.y_max),
            boundary_tol=float(args.boundary_tol),
            overwrite=bool(args.overwrite),
            validate=bool(args.validate),
        )
        build_logger = self._build_logger(config.out.parent)
        output_path = RectangularGeometryBuilder(config, build_logger).write()
        build_logger.info("Completed rectangular geometry generation")
        return output_path


def main() -> None:
    MakeRectangularGeometryCLI().run()


if __name__ == "__main__":
    main()
