from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SampleSeedPolicy = Literal["sequential", "indexed"]


@dataclass(frozen=True)
class FenicsxSampleConfig:
    """Configuration for FEniCSx complex Coupling sample generation."""

    geometry: Path
    out: Path
    gmsh_script: Path | None
    msh: Path | None
    num_train: int
    num_valid: int
    num_test: int
    lengthscale: float = 0.2
    amplitude: float = 1.0
    mean: float = 0.0
    seed: int = 0
    solution_degree: int = 2
    target_degree: int = 1
    mesh_size: float | None = None
    embed_valid_points: bool = True
    require_valid_points_in_mesh: bool = True
    coefficients: Path | None = None
    num_workers: int = 1
    sample_seed_policy: SampleSeedPolicy = "sequential"
    overwrite: bool = False
    skip_existing: bool = False

    def __post_init__(self) -> None:
        if (self.gmsh_script is None) == (self.msh is None):
            raise ValueError("Specify exactly one of --gmsh-script or --msh.")
        for field_name in ("num_train", "num_valid", "num_test"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"--{field_name.replace('_', '-')} must be >= 0.")
        if self.num_train + self.num_valid + self.num_test <= 0:
            raise ValueError("At least one sample must be requested.")
        if self.lengthscale <= 0.0:
            raise ValueError("--lengthscale must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("--amplitude must be non-negative.")
        if self.solution_degree < 1:
            raise ValueError("--solution-degree must be positive.")
        if self.target_degree < 1:
            raise ValueError("--target-degree must be positive.")
        if self.mesh_size is not None and self.mesh_size <= 0.0:
            raise ValueError("--mesh-size must be positive when provided.")
        if self.geometry.suffix != ".npz":
            raise ValueError("--geometry must point to a .npz geometry file.")
        if self.num_workers < 1:
            raise ValueError("--num-workers must be >= 1.")
        if self.sample_seed_policy not in ("sequential", "indexed"):
            raise ValueError(
                "--sample-seed-policy must be one of: sequential, indexed."
            )
        if self.num_workers > 1 and self.sample_seed_policy != "indexed":
            raise ValueError(
                '--sample-seed-policy must be "indexed" when --num-workers > 1.'
            )
        if self.overwrite and self.skip_existing:
            raise ValueError("--overwrite and --skip-existing cannot both be set.")

    @property
    def split_counts(self) -> tuple[tuple[str, int], ...]:
        return (
            ("train", self.num_train),
            ("valid", self.num_valid),
            ("test", self.num_test),
        )

    @property
    def domain_source(self) -> Path:
        if self.gmsh_script is not None:
            return self.gmsh_script
        if self.msh is not None:
            return self.msh
        raise ValueError("Domain source is not configured.")
