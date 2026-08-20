from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from greenonet.sequential_experiment_queue import (
    SequentialExperimentQueue,
    SequentialExperimentQueueConfig,
    configure_queue_logging,
)


@dataclass(frozen=True)
class DiskExperiment:
    """One objective and seed combination in the Disk experiment matrix."""

    objective: str
    seed: int

    @property
    def name(self) -> str:
        return f"coupling_{self.objective}_seed{self.seed}"

    @property
    def config_filename(self) -> str:
        return f"disk_{self.objective}_seed{self.seed}.json"

    def status_fields(self) -> dict[str, object]:
        return {"objective": self.objective, "seed": self.seed}


@dataclass(frozen=True)
class DiskExperimentQueueConfig(SequentialExperimentQueueConfig):
    """Runtime paths and resource limits for the Disk queue."""


class DiskExperimentQueue(SequentialExperimentQueue):
    """Run the fixed 2-by-2 objective matrix over four seeds in sequence."""

    OBJECTIVES = (
        "energy_only",
        "energy_response_trust",
        "energy_stationarity",
        "energy_response_trust_stationarity",
    )
    SEEDS = (0, 1, 2, 3)
    QUEUE_ID: str | None = None

    def __init__(
        self,
        config: DiskExperimentQueueConfig,
        queue_logger: logging.Logger | None = None,
    ) -> None:
        experiments = tuple(
            DiskExperiment(objective=objective, seed=seed)
            for objective in self.OBJECTIVES
            for seed in self.SEEDS
        )
        super().__init__(
            config=config,
            experiments=experiments,
            config_directory=Path("numerical_examples/disk"),
            queue_logger=queue_logger,
        )
        if self.QUEUE_ID is not None:
            self.status_path = (
                self.config.output_root / f"queue_{self.QUEUE_ID}_status.json"
            )
            self.pid_path = self.config.output_root / f"queue_{self.QUEUE_ID}.pid"


def configure_logging(
    log_path: Path,
    *,
    logger_name: str = "disk_experiment_queue",
) -> logging.Logger:
    return configure_queue_logging(log_path, logger_name=logger_name)


def build_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=description
        or (
            "Run the 16 Disk variable-diffusion objective and seed experiments "
            "sequentially."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root containing cli/train.py and numerical_examples/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("checkpoints/numerical_examples/disk"),
        help="Root directory containing one work directory per experiment.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for each training process.",
    )
    parser.add_argument(
        "--max-cpu-threads",
        type=int,
        default=4,
        help="Thread limit exported to common CPU numerical backends.",
    )
    return parser


def run_disk_queue(
    argv: Sequence[str] | None = None,
    *,
    queue_type: type[DiskExperimentQueue] = DiskExperimentQueue,
    description: str | None = None,
    log_filename: str = "queue.log",
    logger_name: str = "disk_experiment_queue",
) -> int:
    """Resolve common CLI arguments and run one Disk queue variant."""

    args = build_parser(description=description).parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queue_logger = configure_logging(
        output_root / log_filename,
        logger_name=logger_name,
    )
    queue = queue_type(
        DiskExperimentQueueConfig(
            repo_root=repo_root,
            output_root=output_root,
            python_executable=args.python_executable.expanduser().resolve(),
            max_cpu_threads=args.max_cpu_threads,
        ),
        queue_logger=queue_logger,
    )
    return queue.run()


def main(argv: Sequence[str] | None = None) -> int:
    return run_disk_queue(argv)


if __name__ == "__main__":
    raise SystemExit(main())
