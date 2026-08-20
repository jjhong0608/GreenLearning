from __future__ import annotations

from typing import Sequence

if __package__:
    from .run_disk_experiment_queue import DiskExperimentQueue, run_disk_queue
else:
    from run_disk_experiment_queue import DiskExperimentQueue, run_disk_queue


class DiskSeed01ExperimentQueue(DiskExperimentQueue):
    """Run all four Disk objectives for seeds zero and one."""

    SEEDS = (0, 1)
    QUEUE_ID = "seed01"


def main(argv: Sequence[str] | None = None) -> int:
    return run_disk_queue(
        argv,
        queue_type=DiskSeed01ExperimentQueue,
        description=(
            "Run the eight Disk variable-diffusion experiments for seeds 0 and 1 "
            "sequentially."
        ),
        log_filename="queue_seed01.log",
        logger_name="disk_experiment_queue_seed01",
    )


if __name__ == "__main__":
    raise SystemExit(main())
