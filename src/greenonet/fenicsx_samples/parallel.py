from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_sources.seeding import derive_indexed_seed
from greenonet.fenicsx_samples.config import FenicsxSampleConfig
from greenonet.fenicsx_samples.domain import FenicsxDomainBuilder
from greenonet.fenicsx_samples.fenicsx_runtime import FenicsxImportMixin
from greenonet.fenicsx_samples.geometry import GeometryGridLoader
from greenonet.fenicsx_samples.gp import GaussianProcessSourceSampler
from greenonet.fenicsx_samples.solver import FenicsxPdeSolver
from greenonet.fenicsx_samples.writer import SampleWriter


@dataclass(frozen=True)
class SampleTask:
    split: str
    index: int
    global_ordinal: int
    seed: int | None


@dataclass(frozen=True)
class GeneratedSampleSummary:
    split: str
    index: int
    path: str
    balance_relative_residual: float | None
    global_ordinal: int
    seed: int | None
    skipped: bool = False


@dataclass(frozen=True)
class WorkerBatchResult:
    worker_id: int
    samples: list[GeneratedSampleSummary]
    vertex_coverage_max_distance: float | None
    generated_count: int
    skipped_count: int


def build_sample_tasks(config: FenicsxSampleConfig) -> list[SampleTask]:
    tasks: list[SampleTask] = []
    global_ordinal = 0
    for split, count in config.split_counts:
        for index in range(count):
            seed = (
                derive_indexed_seed(config.seed, split, index)
                if config.sample_seed_policy == "indexed"
                else None
            )
            tasks.append(
                SampleTask(
                    split=split,
                    index=index,
                    global_ordinal=global_ordinal,
                    seed=seed,
                )
            )
            global_ordinal += 1
    return tasks


def partition_tasks(
    tasks: list[SampleTask], num_workers: int
) -> list[tuple[int, list[SampleTask]]]:
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1.")
    batches: list[tuple[int, list[SampleTask]]] = [
        (worker_id, []) for worker_id in range(num_workers)
    ]
    for offset, task in enumerate(tasks):
        batches[offset % num_workers][1].append(task)
    return [(worker_id, batch) for worker_id, batch in batches if batch]


class FenicsxSampleWorker(FenicsxImportMixin):
    """Generate an assigned batch of independent FEniCSx samples."""

    def __init__(self, config: FenicsxSampleConfig, worker_id: int = 0) -> None:
        self.config = config
        self.worker_id = worker_id

    def run(self, tasks: list[SampleTask]) -> WorkerBatchResult:
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
        writer = SampleWriter(
            self.config.out,
            geometry.full_grid_shape,
            overwrite=self.config.overwrite,
        )
        samples: list[GeneratedSampleSummary] = []
        skipped_count = 0
        for task in tasks:
            path = writer.sample_path(task.split, task.index)
            if self.config.skip_existing and path.exists():
                if self.config.sample_seed_policy == "sequential":
                    sampler.sample()
                samples.append(self._skipped_summary(task, path))
                skipped_count += 1
                continue
            if path.exists() and not self.config.overwrite:
                raise FileExistsError(f"Sample already exists: {path}")
            raw_rhs = self._sample_rhs(sampler, task)
            rhs = geometry.valid_values_to_full_grid(
                raw_rhs[
                    geometry.valid_grid_y_index,
                    geometry.valid_grid_x_index,
                ]
            )
            solve_result = solver.solve(rhs)
            written_path = writer.write_sample(
                task.split,
                task.index,
                rhs=rhs,
                sol=solve_result.sol,
                phi=solve_result.phi,
                psi=solve_result.psi,
            )
            samples.append(
                GeneratedSampleSummary(
                    split=task.split,
                    index=task.index,
                    path=str(written_path),
                    balance_relative_residual=solve_result.balance_relative_residual,
                    global_ordinal=task.global_ordinal,
                    seed=task.seed,
                    skipped=False,
                )
            )
        return WorkerBatchResult(
            worker_id=self.worker_id,
            samples=samples,
            vertex_coverage_max_distance=mesh_bundle.vertex_coverage_max_distance,
            generated_count=len(samples) - skipped_count,
            skipped_count=skipped_count,
        )

    def _sample_rhs(
        self,
        sampler: GaussianProcessSourceSampler,
        task: SampleTask,
    ) -> np.ndarray:
        if self.config.sample_seed_policy == "indexed":
            if task.seed is None:
                raise ValueError("Indexed sample task is missing its seed.")
            return sampler.sample_with_seed(task.seed)
        return sampler.sample()

    @staticmethod
    def _skipped_summary(
        task: SampleTask,
        path: Path,
    ) -> GeneratedSampleSummary:
        return GeneratedSampleSummary(
            split=task.split,
            index=task.index,
            path=str(path),
            balance_relative_residual=None,
            global_ordinal=task.global_ordinal,
            seed=task.seed,
            skipped=True,
        )


def run_worker_batch(
    config: FenicsxSampleConfig,
    worker_id: int,
    tasks: list[SampleTask],
) -> WorkerBatchResult:
    return FenicsxSampleWorker(config, worker_id=worker_id).run(tasks)


def run_parallel_batches(
    config: FenicsxSampleConfig,
    tasks: list[SampleTask],
) -> list[WorkerBatchResult]:
    context = mp.get_context("spawn")
    batches = partition_tasks(tasks, config.num_workers)
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(run_worker_batch, config, worker_id, batch)
            for worker_id, batch in batches
        ]
        return [future.result() for future in futures]
