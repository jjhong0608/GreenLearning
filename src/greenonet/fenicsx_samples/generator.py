from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from greenonet.coefficients import load_coefficient_functions
from greenonet.fenicsx_samples.config import FenicsxSampleConfig
from greenonet.fenicsx_samples.domain import FenicsxDomainBuilder
from greenonet.fenicsx_samples.fenicsx_runtime import FenicsxImportMixin
from greenonet.fenicsx_samples.geometry import GeometryGridLoader
from greenonet.fenicsx_samples.gp import GaussianProcessSourceSampler
from greenonet.fenicsx_samples.solver import FenicsxPdeSolver
from greenonet.fenicsx_samples.writer import SampleWriter


@dataclass(frozen=True)
class GeneratedSampleSummary:
    split: str
    index: int
    path: str
    balance_relative_residual: float


class FenicsxSampleGenerator(FenicsxImportMixin):
    """Generate complex Coupling training/validation/test samples with FEniCSx."""

    def __init__(
        self,
        config: FenicsxSampleConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def run(self) -> dict[str, object]:
        self.config.out.mkdir(parents=True, exist_ok=True)
        geometry = GeometryGridLoader().load(self.config.geometry)
        runtime = self.load_runtime()
        mesh_bundle = FenicsxDomainBuilder(runtime, self.config).build(geometry)
        coeffs = load_coefficient_functions(self.config.coefficients)
        solver = FenicsxPdeSolver(
            runtime,
            mesh_bundle.domain,
            geometry,
            coeffs,
            solution_degree=self.config.solution_degree,
            target_degree=self.config.target_degree,
        )
        sampler = GaussianProcessSourceSampler(
            geometry.grid_x,
            geometry.grid_y,
            lengthscale=self.config.lengthscale,
            amplitude=self.config.amplitude,
            mean=self.config.mean,
            seed=self.config.seed,
        )
        writer = SampleWriter(self.config.out, geometry.full_grid_shape)
        samples: list[GeneratedSampleSummary] = []
        for split, count in self.config.split_counts:
            for index in range(count):
                raw_rhs = sampler.sample()
                rhs = geometry.valid_values_to_full_grid(
                    raw_rhs[
                        geometry.valid_grid_y_index,
                        geometry.valid_grid_x_index,
                    ]
                )
                solve_result = solver.solve(rhs)
                path = writer.write_sample(
                    split,
                    index,
                    rhs=rhs,
                    sol=solve_result.sol,
                    phi=solve_result.phi,
                    psi=solve_result.psi,
                )
                summary = GeneratedSampleSummary(
                    split=split,
                    index=index,
                    path=str(path),
                    balance_relative_residual=solve_result.balance_relative_residual,
                )
                samples.append(summary)
                self.logger.info(
                    "wrote %s sample %06d with balance residual %.6e",
                    split,
                    index,
                    solve_result.balance_relative_residual,
                )
        payload = self._summary_payload(
            geometry_metadata=geometry.metadata,
            vertex_coverage_max_distance=mesh_bundle.vertex_coverage_max_distance,
            samples=samples,
        )
        summary_path = self.config.out / "generation_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("wrote generation summary to %s", summary_path)
        return payload

    def _summary_payload(
        self,
        *,
        geometry_metadata: dict[str, object],
        vertex_coverage_max_distance: float | None,
        samples: list[GeneratedSampleSummary],
    ) -> dict[str, object]:
        residuals = [sample.balance_relative_residual for sample in samples]
        return {
            "config": self._serializable_config(),
            "geometry_metadata": geometry_metadata,
            "domain_source": str(self.config.domain_source),
            "vertex_coverage_max_distance": vertex_coverage_max_distance,
            "num_samples": len(samples),
            "sample_counts": {
                "train": self.config.num_train,
                "valid": self.config.num_valid,
                "test": self.config.num_test,
            },
            "balance_relative_residual_mean": (
                None if not residuals else sum(residuals) / len(residuals)
            ),
            "balance_relative_residual_max": (
                None if not residuals else max(residuals)
            ),
            "samples": [asdict(sample) for sample in samples],
        }

    def _serializable_config(self) -> dict[str, object]:
        raw = asdict(self.config)
        return {
            key: (None if value is None else str(value))
            if isinstance(value, Path) or value is None
            else value
            for key, value in raw.items()
        }
