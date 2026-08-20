from __future__ import annotations

from typing import Sequence

if __package__:
    from .run_disk_experiment_queue import DiskExperimentQueue, run_disk_queue
else:
    from run_disk_experiment_queue import DiskExperimentQueue, run_disk_queue


class DiskSeed23ExperimentQueue(DiskExperimentQueue):
    """Run all four Disk objectives for seeds two and three."""

    SEEDS = (2, 3)
    QUEUE_ID = "seed23"


def main(argv: Sequence[str] | None = None) -> int:
    return run_disk_queue(
        argv,
        queue_type=DiskSeed23ExperimentQueue,
        description=(
            "Run the eight Disk variable-diffusion experiments for seeds 2 and 3 "
            "sequentially."
        ),
        log_filename="queue_seed23.log",
        logger_name="disk_experiment_queue_seed23",
    )


if __name__ == "__main__":
    raise SystemExit(main())
