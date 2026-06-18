from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleWriter:
    """Write full-grid Coupling sample arrays to split directories."""

    out: Path
    full_grid_shape: tuple[int, int]
    overwrite: bool = False

    REQUIRED_KEYS: tuple[str, ...] = ("rhs", "sol", "phi", "psi")

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)

    def write_sample(
        self,
        split: str,
        index: int,
        *,
        rhs: np.ndarray,
        sol: np.ndarray,
        phi: np.ndarray,
        psi: np.ndarray,
    ) -> Path:
        arrays = {"rhs": rhs, "sol": sol, "phi": phi, "psi": psi}
        for key, value in arrays.items():
            if value.shape != self.full_grid_shape:
                raise ValueError(
                    f"{key} must have shape {self.full_grid_shape}, got {value.shape}."
                )
        path = self.sample_path(split, index)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not self.overwrite:
            raise FileExistsError(f"Sample already exists: {path}")
        tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        with tmp_path.open("wb") as file:
            np.savez(
                file,
                rhs=arrays["rhs"],
                sol=arrays["sol"],
                phi=arrays["phi"],
                psi=arrays["psi"],
            )
        if path.exists() and not self.overwrite:
            tmp_path.unlink(missing_ok=True)
            raise FileExistsError(f"Sample already exists: {path}")
        os.replace(tmp_path, path)
        return path

    def sample_path(self, split: str, index: int) -> Path:
        return self.out / split / f"sample_{index:06d}.npz"
