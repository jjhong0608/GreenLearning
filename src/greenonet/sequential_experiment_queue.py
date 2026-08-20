from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence

from rich.logging import RichHandler


logger = logging.getLogger(__name__)


class SequentialExperiment(Protocol):
    """Description required by the fixed sequential experiment runner."""

    @property
    def name(self) -> str: ...

    @property
    def config_filename(self) -> str: ...

    def status_fields(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class SequentialExperimentQueueConfig:
    """Runtime paths and resource limits for a sequential experiment queue."""

    repo_root: Path
    output_root: Path
    python_executable: Path
    max_cpu_threads: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.max_cpu_threads, int) or isinstance(
            self.max_cpu_threads, bool
        ):
            raise TypeError("max_cpu_threads must be an integer.")
        if self.max_cpu_threads < 1:
            raise ValueError("max_cpu_threads must be positive.")


class SequentialExperimentQueue:
    """Run a fixed config matrix sequentially and stop at the first failure."""

    def __init__(
        self,
        *,
        config: SequentialExperimentQueueConfig,
        experiments: Sequence[SequentialExperiment],
        config_directory: Path,
        queue_logger: logging.Logger | None = None,
    ) -> None:
        if not experiments:
            raise ValueError("experiments must not be empty.")
        if config_directory.is_absolute():
            raise ValueError("config_directory must be relative to repo_root.")
        self.config = config
        self.logger = queue_logger if queue_logger is not None else logger
        self.experiments = tuple(experiments)
        self.config_directory = config_directory
        self.status_path = self.config.output_root / "queue_status.json"
        self.pid_path = self.config.output_root / "queue.pid"
        self.status: dict[str, Any] = {
            "state": "pending",
            "total": len(self.experiments),
            "completed": [],
            "skipped": [],
            "failed": None,
            "current": None,
        }

    def run(self) -> int:
        """Run every experiment sequentially and stop at the first failure."""

        try:
            self._validate_inputs()
            self._claim_queue_pid()
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
            self.logger.error("Queue preflight failed: %s", exc)
            return 2

        self.status.update(state="running", started_at=self._timestamp())
        self._write_status()
        self.logger.info("Starting queue with %d experiments", len(self.experiments))

        for index, experiment in enumerate(self.experiments, start=1):
            return_code = self._run_experiment(experiment=experiment, index=index)
            if return_code != 0:
                return return_code

        self.status.update(
            state="completed",
            current=None,
            finished_at=self._timestamp(),
        )
        self._write_status()
        self.logger.info("All %d experiments completed", len(self.experiments))
        return 0

    def _run_experiment(
        self,
        *,
        experiment: SequentialExperiment,
        index: int,
    ) -> int:
        config_path = self._config_path(experiment)
        work_dir = self.config.output_root / experiment.name
        success_marker = work_dir / "_SUCCESS"
        training_log = work_dir / "popen_stdout_stderr.log"

        current: dict[str, object] = {
            "index": index,
            "name": experiment.name,
            **experiment.status_fields(),
            "config": str(config_path),
            "work_dir": str(work_dir),
        }
        self.status["current"] = current
        self._write_status()

        if success_marker.is_file():
            self.logger.info(
                "[%d/%d] Skipping completed run %s",
                index,
                len(self.experiments),
                experiment.name,
            )
            self._status_list("skipped").append(experiment.name)
            self.status["current"] = None
            self._write_status()
            return 0

        if work_dir.exists() and any(work_dir.iterdir()):
            message = f"refusing to reuse incomplete directory: {work_dir}"
            self._record_failure(experiment, message=message, return_code=2)
            return 2

        work_dir.mkdir(parents=True, exist_ok=True)
        command = self._training_command(config_path=config_path, work_dir=work_dir)
        self.logger.info(
            "[%d/%d] Starting %s",
            index,
            len(self.experiments),
            experiment.name,
        )
        self.logger.debug("Command: %s", " ".join(command))

        with training_log.open("ab", buffering=0) as handle:
            completed = subprocess.run(
                command,
                cwd=self.config.repo_root,
                env=self._training_environment(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                check=False,
            )

        if completed.returncode != 0:
            message = f"training exited with code {completed.returncode}"
            self._record_failure(
                experiment,
                message=message,
                return_code=completed.returncode,
                training_log=training_log,
            )
            return completed.returncode

        success_marker.write_text(self._timestamp() + "\n", encoding="utf-8")
        self._status_list("completed").append(experiment.name)
        self.status["current"] = None
        self._write_status()
        self.logger.info(
            "[%d/%d] Completed %s",
            index,
            len(self.experiments),
            experiment.name,
        )
        return 0

    def _validate_inputs(self) -> None:
        if not self.config.repo_root.is_dir():
            raise FileNotFoundError(
                f"repository does not exist: {self.config.repo_root}"
            )
        if not self.config.python_executable.is_file():
            raise FileNotFoundError(
                f"Python executable does not exist: {self.config.python_executable}"
            )
        train_cli = self.config.repo_root / "cli" / "train.py"
        if not train_cli.is_file():
            raise FileNotFoundError(f"training CLI does not exist: {train_cli}")
        for experiment in self.experiments:
            config_path = self._config_path(experiment)
            if not config_path.is_file():
                raise FileNotFoundError(f"experiment config is missing: {config_path}")

    def _claim_queue_pid(self) -> None:
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        current_pid = os.getpid()
        if self.pid_path.is_file():
            raw_pid = self.pid_path.read_text(encoding="utf-8").strip()
            if raw_pid:
                try:
                    previous_pid = int(raw_pid)
                except ValueError as exc:
                    raise ValueError(
                        f"queue PID file is invalid: {self.pid_path}"
                    ) from exc
                if previous_pid != current_pid and self._process_exists(previous_pid):
                    raise RuntimeError(
                        f"queue supervisor PID {previous_pid} is already running"
                    )
        self.pid_path.write_text(f"{current_pid}\n", encoding="utf-8")

    def _training_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = "src"
        thread_count = str(self.config.max_cpu_threads)
        for name in (
            "ACCELERATE_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            environment[name] = thread_count
        return environment

    def _training_command(self, *, config_path: Path, work_dir: Path) -> list[str]:
        return [
            str(self.config.python_executable),
            "cli/train.py",
            "--config",
            str(config_path.relative_to(self.config.repo_root)),
            "--work-dir",
            str(work_dir),
        ]

    def _config_path(self, experiment: SequentialExperiment) -> Path:
        return (
            self.config.repo_root / self.config_directory / experiment.config_filename
        )

    def _record_failure(
        self,
        experiment: SequentialExperiment,
        *,
        message: str,
        return_code: int,
        training_log: Path | None = None,
    ) -> None:
        self.logger.error("%s: %s", experiment.name, message)
        failure: dict[str, Any] = {
            "name": experiment.name,
            "message": message,
            "return_code": return_code,
        }
        if training_log is not None:
            failure["training_log"] = str(training_log)
        self.status.update(
            state="failed",
            failed=failure,
            finished_at=self._timestamp(),
        )
        self._write_status()

    def _write_status(self) -> None:
        self.status["updated_at"] = self._timestamp()
        temporary_path = self.status_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(self.status, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.status_path)

    def _status_list(self, field_name: str) -> list[str]:
        value = self.status[field_name]
        if not isinstance(value, list):
            raise RuntimeError(f"status field {field_name} is not a list")
        return value

    @staticmethod
    def _process_exists(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().astimezone().isoformat()


def configure_queue_logging(log_path: Path, *, logger_name: str) -> logging.Logger:
    """Configure Rich console output and an equivalent persistent queue log."""

    queue_logger = logging.getLogger(logger_name)
    queue_logger.handlers.clear()
    queue_logger.propagate = False
    queue_logger.setLevel(logging.DEBUG)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        omit_repeated_times=False,
    )
    formatter = logging.Formatter("%(funcName)s - %(message)s")
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(logging.DEBUG)
    queue_logger.addHandler(rich_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(funcName)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    queue_logger.addHandler(file_handler)
    logging.root.handlers.clear()
    return queue_logger
