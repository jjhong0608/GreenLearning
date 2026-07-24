from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from greenonet.complex_sources.geometry import (
    GeometryGridLoader,
    RawComplexGeometryGrid,
)
from greenonet.complex_sources.providers import (
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
)


@dataclass(frozen=True)
class ComplexSourceGenerationConfig:
    """Configuration for deterministic source-only NPZ generation."""

    geometry: Path
    out: Path
    num_train: int
    num_valid: int
    lengthscale: float = 0.2
    amplitude: float = 1.0
    mean: float = 0.0
    seed: int = 0
    overwrite: bool = False
    validate: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "geometry", Path(self.geometry))
        object.__setattr__(self, "out", Path(self.out))
        for field_name, value, minimum in (
            ("num_train", self.num_train, 1),
            ("num_valid", self.num_valid, 0),
            ("seed", self.seed, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer.")
            if value < minimum:
                raise ValueError(f"{field_name} must be >= {minimum}.")
        for field_name, numeric_value in (
            ("lengthscale", self.lengthscale),
            ("amplitude", self.amplitude),
            ("mean", self.mean),
        ):
            if not isinstance(numeric_value, (int, float)) or isinstance(
                numeric_value, bool
            ):
                raise TypeError(f"{field_name} must be numeric.")
            if not math.isfinite(float(numeric_value)):
                raise ValueError(f"{field_name} must be finite.")
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative.")
        if not isinstance(self.overwrite, bool):
            raise TypeError("overwrite must be a boolean.")
        if not isinstance(self.validate, bool):
            raise TypeError("validate must be a boolean.")

    @property
    def parameters(self) -> IndexedGpParameters:
        return IndexedGpParameters(
            seed=self.seed,
            lengthscale=self.lengthscale,
            amplitude=self.amplitude,
            mean=self.mean,
        )


@dataclass(frozen=True)
class SourceOnlySampleWriter:
    """Atomically write one full-grid RHS array."""

    out: Path
    full_grid_shape: tuple[int, int]
    overwrite: bool = False

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)

    def sample_path(self, split: str, index: int) -> Path:
        return self.out / split / f"sample_{index:06d}.npz"

    def write_sample(self, split: str, index: int, rhs: np.ndarray) -> Path:
        if rhs.shape != self.full_grid_shape:
            raise ValueError(
                f"rhs must have shape {self.full_grid_shape}, got {rhs.shape}."
            )
        path = self.sample_path(split, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Sample already exists: {path}")
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp_path.open("wb") as file:
            np.savez(file, rhs=np.asarray(rhs, dtype=np.float64))
        if path.exists() and not self.overwrite:
            tmp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Sample already exists: {path}")
        os.replace(tmp_path, path)
        return path


class ComplexSourceGenerator:
    """Generate deterministic source-only train/validation NPZ splits."""

    def __init__(
        self,
        config: ComplexSourceGenerationConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def run(self) -> dict[str, object]:
        self.config.out.mkdir(parents=True, exist_ok=True)
        geometry = GeometryGridLoader().load(self.config.geometry)
        writer = SourceOnlySampleWriter(
            self.config.out,
            geometry.full_grid_shape,
            overwrite=self.config.overwrite,
        )
        samples: list[dict[str, object]] = []
        split_counts: tuple[tuple[Literal["train", "valid"], int], ...] = (
            ("train", self.config.num_train),
            ("valid", self.config.num_valid),
        )
        for split, count in split_counts:
            provider = IndexedGpComplexSourceProvider(
                geometry,
                split=split,
                sample_count=count,
                parameters=self.config.parameters,
            )
            for index in range(count):
                sample = provider[index]
                path = writer.write_sample(split, index, sample.rhs)
                if self.config.validate:
                    self._validate_sample(path, geometry)
                samples.append(
                    {
                        "split": split,
                        "index": index,
                        "name": sample.sample_name,
                        "path": str(path),
                    }
                )
                self.logger.info("wrote %s source %06d", split, index)

        summary = self._summary(geometry, samples)
        summary_path = self.config.out / "generation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self.logger.info("wrote generation summary to %s", summary_path)
        return summary

    def _validate_sample(
        self,
        path: Path,
        geometry: RawComplexGeometryGrid,
    ) -> None:
        with np.load(path, allow_pickle=False) as raw:
            if raw.files != ["rhs"]:
                raise ValueError(f"{path} must contain only the rhs array.")
            rhs = np.asarray(raw["rhs"])
        if rhs.shape != geometry.full_grid_shape:
            raise ValueError(
                f"{path}:rhs must have shape {geometry.full_grid_shape}, "
                f"got {rhs.shape}."
            )
        if rhs.dtype != np.float64:
            raise TypeError(f"{path}:rhs must use float64, got {rhs.dtype}.")
        if not np.isfinite(rhs).all():
            raise ValueError(f"{path}:rhs contains non-finite values.")
        mask = np.zeros(geometry.full_grid_shape, dtype=bool)
        mask[geometry.valid_grid_y_index, geometry.valid_grid_x_index] = True
        if np.any(rhs[~mask] != 0.0):
            raise ValueError(f"{path}:rhs must be zero outside the domain.")

    def _summary(
        self,
        geometry: RawComplexGeometryGrid,
        samples: list[dict[str, object]],
    ) -> dict[str, object]:
        raw_config = asdict(self.config)
        config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in raw_config.items()
        }
        return {
            "config": config,
            "geometry_path": str(geometry.path),
            "geometry_metadata": geometry.metadata,
            "full_grid_shape": list(geometry.full_grid_shape),
            "num_valid_points": geometry.num_valid_points,
            "sample_counts": {
                "train": self.config.num_train,
                "valid": self.config.num_valid,
            },
            "seed_policy": "indexed",
            "split_ids": {"train": 0, "valid": 1},
            "outside_domain": 0.0,
            "sample_schema": ["rhs"],
            "samples": samples,
        }
