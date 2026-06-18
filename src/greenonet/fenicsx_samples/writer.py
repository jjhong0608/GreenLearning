from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampleWriter:
    """Write full-grid Coupling sample arrays to split directories."""

    out: Path
    full_grid_shape: tuple[int, int]

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
        split_dir = self.out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        path = split_dir / f"sample_{index:06d}.npz"
        np.savez(
            path,
            rhs=arrays["rhs"],
            sol=arrays["sol"],
            phi=arrays["phi"],
            psi=arrays["psi"],
        )
        return path
