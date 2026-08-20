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
class UnitSquareExperiment:
    """One source-count and seed combination in the paper experiment matrix."""

    train_count: int
    seed: int

    @property
    def name(self) -> str:
        return f"coupling_train{self.train_count}_seed{self.seed}"

    @property
    def config_filename(self) -> str:
        return f"unit_square_train{self.train_count}_seed{self.seed}.json"

    def status_fields(self) -> dict[str, object]:
        return {"train_count": self.train_count, "seed": self.seed}


@dataclass(frozen=True)
class UnitSquareExperimentQueueConfig(SequentialExperimentQueueConfig):
    """Runtime paths and resource limits for the Unit-square queue."""


class UnitSquareExperimentQueue(SequentialExperimentQueue):
    """Run the fixed 4-by-4 Unit-square Poisson experiment matrix in sequence."""

    TRAIN_COUNTS = (600, 1200, 2400, 4800)
    SEEDS = (0, 1, 2, 3)

    def __init__(
        self,
        config: UnitSquareExperimentQueueConfig,
        queue_logger: logging.Logger | None = None,
    ) -> None:
        experiments = tuple(
            UnitSquareExperiment(train_count=train_count, seed=seed)
            for train_count in self.TRAIN_COUNTS
            for seed in self.SEEDS
        )
        super().__init__(
            config=config,
            experiments=experiments,
            config_directory=Path("numerical_examples/unit_square"),
            queue_logger=queue_logger,
        )


def configure_logging(log_path: Path) -> logging.Logger:
    return configure_queue_logging(
        log_path,
        logger_name="unit_square_experiment_queue",
    )


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run the 16 Unit-square Poisson source-count and seed experiments "
            "sequentially."
        )
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
        default=Path("checkpoints/numerical_examples/unit_square_poisson"),
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    queue_logger = configure_logging(output_root / "queue.log")
    queue = UnitSquareExperimentQueue(
        UnitSquareExperimentQueueConfig(
            repo_root=repo_root,
            output_root=output_root,
            python_executable=args.python_executable.expanduser().resolve(),
            max_cpu_threads=args.max_cpu_threads,
        ),
        queue_logger=queue_logger,
    )
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
