from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from greenonet.complex_sources.geometry import RawComplexGeometryGrid
from greenonet.complex_sources.gp import GaussianProcessSourceSampler
from greenonet.complex_sources.seeding import SPLIT_IDS, derive_indexed_seed


@dataclass(frozen=True)
class ComplexSourceSample:
    """One full-grid source with optional evaluation-only references."""

    rhs: np.ndarray
    sample_index: int
    sample_name: str
    sol: np.ndarray | None = None
    flux: tuple[np.ndarray, np.ndarray] | None = None


class ComplexSourceProvider(ABC):
    """Index-stable source provider consumed by ComplexCouplingDataset."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> ComplexSourceSample:
        raise NotImplementedError

    @property
    def files(self) -> tuple[Path, ...]:
        return ()

    @property
    def data_dir(self) -> Path | None:
        return None


@dataclass(frozen=True)
class IndexedGpParameters:
    """Parameters shared by offline and runtime indexed GP sources."""

    seed: int = 0
    lengthscale: float = 0.2
    amplitude: float = 1.0
    mean: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")
        for field_name, value in (
            ("lengthscale", self.lengthscale),
            ("amplitude", self.amplitude),
            ("mean", self.mean),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite.")
        if self.lengthscale <= 0.0:
            raise ValueError("lengthscale must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative.")


class NpzComplexSourceProvider(ComplexSourceProvider):
    """Read deterministic full-grid sources and optional references from NPZ."""

    def __init__(
        self,
        data_dir: Path | str,
        *,
        reference_diagnostics: bool,
    ) -> None:
        if not isinstance(reference_diagnostics, bool):
            raise TypeError("reference_diagnostics must be a boolean.")
        self._data_dir = Path(data_dir)
        self._files = tuple(sorted(self._data_dir.glob("*.npz")))
        if not self._files:
            raise FileNotFoundError(f"No npz files found in {self._data_dir}")
        self.reference_diagnostics = reference_diagnostics

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> ComplexSourceSample:
        path = self._files[index]
        with np.load(path, allow_pickle=False) as raw:
            required = {"rhs"}
            if self.reference_diagnostics:
                required.add("sol")
            missing = sorted(required - set(raw.files))
            if missing:
                raise KeyError(f"{path} is missing required keys: {', '.join(missing)}")
            rhs = np.asarray(raw["rhs"], dtype=np.float64)
            sol = (
                np.asarray(raw["sol"], dtype=np.float64)
                if self.reference_diagnostics
                else None
            )
            flux = self._load_optional_flux(raw) if self.reference_diagnostics else None
        return ComplexSourceSample(
            rhs=rhs,
            sol=sol,
            flux=flux,
            sample_index=index,
            sample_name=path.stem,
        )

    @staticmethod
    def _load_optional_flux(
        raw: np.lib.npyio.NpzFile,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if {"phi", "psi"}.issubset(raw.files):
            return (
                np.asarray(raw["phi"], dtype=np.float64),
                np.asarray(raw["psi"], dtype=np.float64),
            )
        if {"uxx", "uyy"}.issubset(raw.files):
            return (
                np.asarray(raw["uxx"], dtype=np.float64),
                np.asarray(raw["uyy"], dtype=np.float64),
            )
        return None

    @property
    def files(self) -> tuple[Path, ...]:
        return self._files

    @property
    def data_dir(self) -> Path:
        return self._data_dir


def generate_fixed_rhs(
    geometry: RawComplexGeometryGrid,
    sampler: GaussianProcessSourceSampler,
    *,
    base_seed: int,
    split: str,
    sample_index: int,
) -> np.ndarray:
    """Generate the fixed masked RHS for one stable sample identity."""

    seed = derive_indexed_seed(base_seed, split, sample_index)
    return geometry.mask_full_grid(sampler.sample_with_seed(seed))


class IndexedGpComplexSourceProvider(ComplexSourceProvider):
    """Regenerate fixed GP sources by stable split/index identity."""

    def __init__(
        self,
        geometry: RawComplexGeometryGrid,
        *,
        split: Literal["train", "valid", "test"],
        sample_count: int,
        parameters: IndexedGpParameters,
    ) -> None:
        if split not in SPLIT_IDS:
            raise ValueError(f"Unknown split: {split}")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise TypeError("sample_count must be an integer.")
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative.")
        self.geometry = geometry
        self.split = split
        self.sample_count = sample_count
        self.parameters = parameters
        self.sampler = GaussianProcessSourceSampler(
            geometry.grid_x,
            geometry.grid_y,
            lengthscale=parameters.lengthscale,
            amplitude=parameters.amplitude,
            mean=parameters.mean,
            seed=parameters.seed,
        )

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> ComplexSourceSample:
        if index < 0:
            index += self.sample_count
        if index < 0 or index >= self.sample_count:
            raise IndexError(index)
        rhs = generate_fixed_rhs(
            self.geometry,
            self.sampler,
            base_seed=self.parameters.seed,
            split=self.split,
            sample_index=index,
        )
        return ComplexSourceSample(
            rhs=rhs,
            sample_index=index,
            sample_name=f"sample_{index:06d}",
        )
