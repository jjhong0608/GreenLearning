from __future__ import annotations

import json
from pathlib import Path

from cli.run_unit_square_experiment_queue import (
    UnitSquareExperimentQueue,
    UnitSquareExperimentQueueConfig,
)


def _build_fake_repo(tmp_path: Path, *, failing_config: str | None = None) -> Path:
    repo_root = tmp_path / "repo"
    config_root = repo_root / "numerical_examples" / "unit_square"
    cli_root = repo_root / "cli"
    config_root.mkdir(parents=True)
    cli_root.mkdir(parents=True)
    for train_count in UnitSquareExperimentQueue.TRAIN_COUNTS:
        for seed in UnitSquareExperimentQueue.SEEDS:
            filename = f"unit_square_train{train_count}_seed{seed}.json"
            payload = {"fail": filename == failing_config}
            (config_root / filename).write_text(json.dumps(payload), encoding="utf-8")
    (cli_root / "train.py").write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--work-dir", type=Path, required=True)
args = parser.parse_args()
payload = json.loads(args.config.read_text(encoding="utf-8"))
(args.work_dir / "fake_training.txt").write_text(
    args.config.name,
    encoding="utf-8",
)
raise SystemExit(7 if payload["fail"] else 0)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return repo_root


def _queue(tmp_path: Path, repo_root: Path) -> UnitSquareExperimentQueue:
    return UnitSquareExperimentQueue(
        UnitSquareExperimentQueueConfig(
            repo_root=repo_root,
            output_root=tmp_path / "output",
            python_executable=Path(__import__("sys").executable),
            max_cpu_threads=2,
        )
    )


def test_queue_builds_expected_experiment_order(tmp_path):
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))

    assert len(queue.experiments) == 16
    assert queue.experiments[0].name == "coupling_train600_seed0"
    assert queue.experiments[-1].name == "coupling_train4800_seed3"


def test_queue_runs_all_experiments_sequentially(tmp_path):
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))

    assert queue.run() == 0

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert len(status["completed"]) == 16
    for experiment in queue.experiments:
        work_dir = queue.config.output_root / experiment.name
        assert (work_dir / "_SUCCESS").is_file()
        assert (work_dir / "fake_training.txt").is_file()


def test_queue_stops_at_first_training_failure(tmp_path):
    failing_config = "unit_square_train600_seed1.json"
    queue = _queue(
        tmp_path,
        _build_fake_repo(tmp_path, failing_config=failing_config),
    )

    assert queue.run() == 7

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["completed"] == ["coupling_train600_seed0"]
    assert status["failed"]["name"] == "coupling_train600_seed1"
    assert not (queue.config.output_root / "coupling_train600_seed2").exists()


def test_queue_refuses_incomplete_nonempty_work_directory(tmp_path):
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))
    incomplete_dir = queue.config.output_root / "coupling_train600_seed0"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "partial.log").write_text("partial", encoding="utf-8")

    assert queue.run() == 2

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["failed"]["name"] == "coupling_train600_seed0"
