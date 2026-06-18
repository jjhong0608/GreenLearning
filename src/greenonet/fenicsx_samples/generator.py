from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from greenonet.fenicsx_samples.config import FenicsxSampleConfig
from greenonet.fenicsx_samples.fenicsx_runtime import FenicsxImportMixin
from greenonet.fenicsx_samples.geometry import GeometryGridLoader
from greenonet.fenicsx_samples.parallel import (
    FenicsxSampleWorker,
    GeneratedSampleSummary,
    SampleTask,
    WorkerBatchResult,
    build_sample_tasks,
    run_parallel_batches,
)


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
        tasks = build_sample_tasks(self.config)
        results = self._run_batches(tasks)
        samples = sorted(
            (sample for result in results for sample in result.samples),
            key=lambda sample: sample.global_ordinal,
        )
        self._log_samples(samples)
        payload = self._summary_payload(
            geometry_metadata=geometry.metadata,
            vertex_coverage_max_distance=self._max_vertex_coverage(results),
            samples=samples,
            task_count=len(tasks),
            results=results,
        )
        summary_path = self.config.out / "generation_summary.json"
        summary_path.write_text(json.dumps(payload, indent=2))
        self.logger.info("wrote generation summary to %s", summary_path)
        return payload

    def _run_batches(self, tasks: list[SampleTask]) -> list[WorkerBatchResult]:
        if self.config.num_workers == 1:
            return [FenicsxSampleWorker(self.config, worker_id=0).run(tasks)]
        return run_parallel_batches(self.config, tasks)

    def _log_samples(self, samples: list[GeneratedSampleSummary]) -> None:
        for sample in samples:
            if sample.skipped:
                self.logger.info(
                    "skipped existing %s sample %06d",
                    sample.split,
                    sample.index,
                )
                continue
            if sample.balance_relative_residual is None:
                raise ValueError("Generated sample summary is missing residual.")
            self.logger.info(
                "wrote %s sample %06d with balance residual %.6e",
                sample.split,
                sample.index,
                sample.balance_relative_residual,
            )

    def _summary_payload(
        self,
        *,
        geometry_metadata: dict[str, object],
        vertex_coverage_max_distance: float | None,
        samples: list[GeneratedSampleSummary],
        task_count: int,
        results: list[WorkerBatchResult],
    ) -> dict[str, object]:
        residuals = [
            sample.balance_relative_residual
            for sample in samples
            if sample.balance_relative_residual is not None
        ]
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
            "parallel": self._parallel_payload(
                task_count=task_count,
                results=results,
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

    def _parallel_payload(
        self,
        *,
        task_count: int,
        results: list[WorkerBatchResult],
    ) -> dict[str, object]:
        return {
            "num_workers": self.config.num_workers,
            "sample_seed_policy": self.config.sample_seed_policy,
            "task_count": task_count,
            "skipped_count": sum(result.skipped_count for result in results),
            "worker_generated_counts": {
                str(result.worker_id): result.generated_count for result in results
            },
            "worker_skipped_counts": {
                str(result.worker_id): result.skipped_count for result in results
            },
        }

    @staticmethod
    def _max_vertex_coverage(
        results: list[WorkerBatchResult],
    ) -> float | None:
        distances = [
            result.vertex_coverage_max_distance
            for result in results
            if result.vertex_coverage_max_distance is not None
        ]
        return None if not distances else max(distances)
