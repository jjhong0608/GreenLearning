from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cli.run_disk_experiment_queue import (
    DiskExperimentQueue,
    DiskExperimentQueueConfig,
)
from cli.run_disk_experiment_queue_seed01 import (
    DiskSeed01ExperimentQueue,
    main as seed01_main,
)
from cli.run_disk_experiment_queue_seed23 import (
    DiskSeed23ExperimentQueue,
    main as seed23_main,
)


def _build_fake_repo(tmp_path: Path, *, failing_config: str | None = None) -> Path:
    repo_root = tmp_path / "repo"
    config_root = repo_root / "numerical_examples" / "disk"
    cli_root = repo_root / "cli"
    config_root.mkdir(parents=True)
    cli_root.mkdir(parents=True)
    for objective in DiskExperimentQueue.OBJECTIVES:
        for seed in DiskExperimentQueue.SEEDS:
            filename = f"disk_{objective}_seed{seed}.json"
            payload = {"fail": filename == failing_config}
            (config_root / filename).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
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


def _queue(tmp_path: Path, repo_root: Path) -> DiskExperimentQueue:
    return DiskExperimentQueue(
        DiskExperimentQueueConfig(
            repo_root=repo_root,
            output_root=tmp_path / "output",
            python_executable=Path(sys.executable),
            max_cpu_threads=2,
        )
    )


def test_disk_queue_builds_expected_objective_and_seed_order(tmp_path: Path) -> None:
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))

    assert len(queue.experiments) == 16
    assert queue.experiments[0].name == "coupling_energy_only_seed0"
    assert queue.experiments[3].name == "coupling_energy_only_seed3"
    assert queue.experiments[4].name == "coupling_energy_response_trust_seed0"
    assert queue.experiments[-1].name == (
        "coupling_energy_response_trust_stationarity_seed3"
    )


def test_disk_queue_runs_all_experiments_sequentially(tmp_path: Path) -> None:
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))

    assert queue.run() == 0

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert len(status["completed"]) == 16
    assert status["skipped"] == []
    for experiment in queue.experiments:
        work_dir = queue.config.output_root / experiment.name
        assert (work_dir / "_SUCCESS").is_file()
        assert (work_dir / "fake_training.txt").read_text(encoding="utf-8") == (
            experiment.config_filename
        )


def test_disk_queue_stops_at_first_training_failure(tmp_path: Path) -> None:
    failing_config = "disk_energy_response_trust_seed0.json"
    queue = _queue(
        tmp_path,
        _build_fake_repo(tmp_path, failing_config=failing_config),
    )

    assert queue.run() == 7

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["completed"] == [
        f"coupling_energy_only_seed{seed}" for seed in DiskExperimentQueue.SEEDS
    ]
    assert status["failed"]["name"] == "coupling_energy_response_trust_seed0"
    assert status["current"]["objective"] == "energy_response_trust"
    assert status["current"]["seed"] == 0
    assert not (
        queue.config.output_root / "coupling_energy_response_trust_seed1"
    ).exists()


def test_disk_queue_refuses_incomplete_nonempty_work_directory(
    tmp_path: Path,
) -> None:
    queue = _queue(tmp_path, _build_fake_repo(tmp_path))
    incomplete_dir = queue.config.output_root / "coupling_energy_only_seed0"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "partial.log").write_text("partial", encoding="utf-8")

    assert queue.run() == 2

    status = json.loads(queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "failed"
    assert status["failed"]["name"] == "coupling_energy_only_seed0"


def test_disk_queue_restart_skips_successful_runs(tmp_path: Path) -> None:
    repo_root = _build_fake_repo(tmp_path)
    first_queue = _queue(tmp_path, repo_root)
    assert first_queue.run() == 0

    second_queue = _queue(tmp_path, repo_root)
    assert second_queue.run() == 0

    status = json.loads(second_queue.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "completed"
    assert status["completed"] == []
    assert status["skipped"] == [
        experiment.name for experiment in second_queue.experiments
    ]


def test_split_disk_queues_form_disjoint_complete_partition(tmp_path: Path) -> None:
    repo_root = _build_fake_repo(tmp_path)
    config = DiskExperimentQueueConfig(
        repo_root=repo_root,
        output_root=tmp_path / "output",
        python_executable=Path(sys.executable),
        max_cpu_threads=2,
    )
    full_queue = DiskExperimentQueue(config)
    seed01_queue = DiskSeed01ExperimentQueue(config)
    seed23_queue = DiskSeed23ExperimentQueue(config)

    full_names = {experiment.name for experiment in full_queue.experiments}
    seed01_names = {experiment.name for experiment in seed01_queue.experiments}
    seed23_names = {experiment.name for experiment in seed23_queue.experiments}

    assert len(seed01_names) == 8
    assert len(seed23_names) == 8
    assert seed01_names.isdisjoint(seed23_names)
    assert seed01_names | seed23_names == full_names
    assert {experiment.seed for experiment in seed01_queue.experiments} == {0, 1}
    assert {experiment.seed for experiment in seed23_queue.experiments} == {2, 3}
    assert seed01_queue.status_path.name == "queue_seed01_status.json"
    assert seed01_queue.pid_path.name == "queue_seed01.pid"
    assert seed23_queue.status_path.name == "queue_seed23_status.json"
    assert seed23_queue.pid_path.name == "queue_seed23.pid"
    assert full_queue.status_path.name == "queue_status.json"
    assert full_queue.pid_path.name == "queue.pid"


def test_split_disk_cli_files_share_output_root_without_metadata_collision(
    tmp_path: Path,
) -> None:
    repo_root = _build_fake_repo(tmp_path)
    output_root = tmp_path / "output"
    common_arguments = [
        "--repo-root",
        str(repo_root),
        "--output-root",
        str(output_root),
        "--python-executable",
        sys.executable,
        "--max-cpu-threads",
        "2",
    ]

    assert seed01_main(common_arguments) == 0
    assert seed23_main(common_arguments) == 0

    for queue_id in ("seed01", "seed23"):
        status = json.loads(
            (output_root / f"queue_{queue_id}_status.json").read_text(encoding="utf-8")
        )
        assert status["state"] == "completed"
        assert len(status["completed"]) == 8
        assert (output_root / f"queue_{queue_id}.pid").is_file()
        assert (output_root / f"queue_{queue_id}.log").is_file()

    assert not (output_root / "queue_status.json").exists()
    assert not (output_root / "queue.pid").exists()
    for objective in DiskExperimentQueue.OBJECTIVES:
        for seed in DiskExperimentQueue.SEEDS:
            work_dir = output_root / f"coupling_{objective}_seed{seed}"
            assert (work_dir / "_SUCCESS").is_file()


@pytest.mark.parametrize(
    "script_name",
    (
        "run_disk_experiment_queue_seed01.py",
        "run_disk_experiment_queue_seed23.py",
    ),
)
def test_split_disk_cli_files_support_direct_script_execution(
    script_name: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [sys.executable, f"cli/{script_name}", "--help"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--max-cpu-threads" in completed.stdout
