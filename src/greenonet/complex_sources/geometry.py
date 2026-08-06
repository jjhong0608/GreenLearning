from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from greenonet.complex_geometry import load_complex_geometry


@dataclass(frozen=True)
class RawComplexGeometryGrid:
    """Full-grid metadata needed for deterministic complex source generation."""

    path: Path
    grid_x: np.ndarray
    grid_y: np.ndarray
    coords_valid: np.ndarray
    valid_grid_y_index: np.ndarray
    valid_grid_x_index: np.ndarray
    metadata: dict[str, Any]

    @property
    def full_grid_shape(self) -> tuple[int, int]:
        return (int(self.grid_y.size), int(self.grid_x.size))

    @property
    def num_valid_points(self) -> int:
        return int(self.coords_valid.shape[0])

    def valid_values_to_full_grid(self, values: np.ndarray) -> np.ndarray:
        if values.shape != (self.num_valid_points,):
            raise ValueError(
                "Valid-point values must have shape "
                f"({self.num_valid_points},), got {values.shape}."
            )
        full = np.zeros(self.full_grid_shape, dtype=np.float64)
        full[self.valid_grid_y_index, self.valid_grid_x_index] = values
        return full

    def mask_full_grid(self, values: np.ndarray) -> np.ndarray:
        if values.shape != self.full_grid_shape:
            raise ValueError(
                f"Full-grid values must have shape {self.full_grid_shape}, "
                f"got {values.shape}."
            )
        return self.valid_values_to_full_grid(
            np.asarray(
                values[self.valid_grid_y_index, self.valid_grid_x_index],
                dtype=np.float64,
            )
        )


class GeometryGridLoader:
    """Load full-grid metadata while reusing complex geometry validation."""

    REQUIRED_GRID_KEYS: tuple[str, str] = ("grid_x", "grid_y")
    OPTIONAL_METADATA_KEYS: tuple[str, ...] = (
        "domain_type",
        "radius",
        "inner_radius",
        "outer_radius",
        "center",
        "orientation_angle",
        "fill_rule",
        "has_hole",
        "boundary_vertices",
        "step_size",
        "boundary_tol",
    )

    def load(self, path: Path | str) -> RawComplexGeometryGrid:
        geometry_path = Path(path)
        if not geometry_path.is_file():
            raise FileNotFoundError(f"Geometry file does not exist: {geometry_path}")

        with np.load(geometry_path, allow_pickle=False) as raw:
            missing = sorted(set(self.REQUIRED_GRID_KEYS) - set(raw.files))
            if missing:
                raise KeyError(
                    "Complex source generation requires geometry NPZ keys: "
                    f"{', '.join(missing)}."
                )
            grid_x = self._load_grid(raw["grid_x"], "grid_x")
            grid_y = self._load_grid(raw["grid_y"], "grid_y")
            coords_valid = self._load_coords(raw["coords_valid"])
            valid_y = self._load_index(raw["valid_grid_y_index"], "valid_grid_y_index")
            valid_x = self._load_index(raw["valid_grid_x_index"], "valid_grid_x_index")
            metadata = {
                key: raw[key].tolist()
                for key in self.OPTIONAL_METADATA_KEYS
                if key in raw.files
            }

        load_complex_geometry(geometry_path)
        self._validate_indices(valid_y, valid_x, grid_y.size, grid_x.size)
        if (
            coords_valid.shape[0] != valid_y.shape[0]
            or coords_valid.shape[0] != valid_x.shape[0]
        ):
            raise ValueError(
                "coords_valid, valid_grid_y_index, and valid_grid_x_index must "
                "have matching first dimensions."
            )
        return RawComplexGeometryGrid(
            path=geometry_path,
            grid_x=grid_x,
            grid_y=grid_y,
            coords_valid=coords_valid,
            valid_grid_y_index=valid_y,
            valid_grid_x_index=valid_x,
            metadata=metadata,
        )

    @staticmethod
    def _load_grid(value: np.ndarray, field_name: str) -> np.ndarray:
        grid = np.asarray(value, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError(f"{field_name} must be a one-dimensional grid array.")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"{field_name} must be strictly increasing.")
        return grid

    @staticmethod
    def _load_coords(value: np.ndarray) -> np.ndarray:
        coords = np.asarray(value, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords_valid must have shape (P, 2).")
        return coords

    @staticmethod
    def _load_index(value: np.ndarray, field_name: str) -> np.ndarray:
        index = np.asarray(value, dtype=np.int64)
        if index.ndim != 1:
            raise ValueError(f"{field_name} must be one-dimensional.")
        return index

    @staticmethod
    def _validate_indices(
        valid_y: np.ndarray,
        valid_x: np.ndarray,
        grid_y_size: int,
        grid_x_size: int,
    ) -> None:
        if valid_y.size == 0:
            raise ValueError("Geometry must contain at least one valid grid point.")
        if np.any(valid_y < 0) or np.any(valid_y >= grid_y_size):
            raise ValueError("valid_grid_y_index is out of range for grid_y.")
        if np.any(valid_x < 0) or np.any(valid_x >= grid_x_size):
            raise ValueError("valid_grid_x_index is out of range for grid_x.")
