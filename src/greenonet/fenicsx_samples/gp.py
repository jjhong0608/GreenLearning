from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def squared_exponential_kernel(coords: np.ndarray, lengthscale: float) -> np.ndarray:
    diff = coords[:, None] - coords[None, :]
    return np.asarray(np.exp(-0.5 * (diff / lengthscale) ** 2), dtype=np.float64)


def stable_symmetric_factor(matrix: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    clipped = np.clip(eigenvalues, eps, None)
    return eigenvectors @ np.diag(np.sqrt(clipped))


@dataclass
class GaussianProcessSourceSampler:
    """Separable squared-exponential GP sampler on a full Cartesian grid."""

    grid_x: np.ndarray
    grid_y: np.ndarray
    lengthscale: float = 0.2
    amplitude: float = 1.0
    mean: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative.")
        self._rng = np.random.default_rng(self.seed)
        self._factor_x = stable_symmetric_factor(
            squared_exponential_kernel(self.grid_x, self.lengthscale)
        )
        self._factor_y = stable_symmetric_factor(
            squared_exponential_kernel(self.grid_y, self.lengthscale)
        )

    def sample(self) -> np.ndarray:
        latent = self._rng.standard_normal((self.grid_y.size, self.grid_x.size))
        field = self.mean + self.amplitude * (
            self._factor_y @ latent @ self._factor_x.T
        )
        return np.asarray(field, dtype=np.float64)
