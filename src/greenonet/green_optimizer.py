from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from torch import optim

from greenonet.config import GreenOptimizerConfig, SoapOptimizerConfig, TrainingConfig
from greenonet.green_lr_scheduler import GreenLearningRateSchedule
from greenonet.optimizer_support import (
    OptimizerProvenance,
    build_adamw_or_soap,
    build_optimizer_provenance,
)
from greenonet.training_step_schedule import StepValidationSchedule


class GreenOptimizerFactory:
    """Build AdamW or SOAP for either GreenNet geometry path."""

    def __init__(self, config: TrainingConfig) -> None:
        self.training_config = config
        self.optimizer_config = GreenOptimizerConfig.from_raw(config.optimizer)
        self._validate_shared_settings()

    def _validate_shared_settings(self) -> None:
        for field_name, value, positive in (
            ("learning_rate", self.training_config.learning_rate, True),
            ("weight_decay", self.training_config.weight_decay, False),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"training.{field_name} must be numeric.")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"training.{field_name} must be finite.")
            if positive and converted <= 0.0:
                raise ValueError(f"training.{field_name} must be positive.")
            if not positive and converted < 0.0:
                raise ValueError(f"training.{field_name} must be non-negative.")

    def build(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> optim.Optimizer:
        config = self.optimizer_config
        soap = SoapOptimizerConfig.from_raw(config.soap)
        return build_adamw_or_soap(
            parameters,
            name=config.name,
            learning_rate=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
            betas=(float(config.betas[0]), float(config.betas[1])),
            eps=config.eps,
            soap=soap,
        )

    def provenance(self) -> OptimizerProvenance:
        config = self.optimizer_config
        soap = SoapOptimizerConfig.from_raw(config.soap)
        return build_optimizer_provenance(
            name=config.name,
            learning_rate=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
            betas=(float(config.betas[0]), float(config.betas[1])),
            eps=config.eps,
            profile_step_time=config.profile_step_time,
            soap=soap,
        )

    def resolved_config(self) -> dict[str, object]:
        """Return an executable optimizer block with all defaults materialized."""

        return asdict(self.optimizer_config)


class GreenTrainingRecorder:
    """Persist first-stage and LBFGS GreenNet training provenance and metrics."""

    def __init__(
        self,
        *,
        work_dir: Path,
        logger: logging.Logger,
        provenance: OptimizerProvenance,
    ) -> None:
        self.work_dir = work_dir
        self.logger = logger
        self.provenance = provenance
        self.rows: list[dict[str, float | int | str]] = []

    def log_startup(
        self,
        schedule: GreenLearningRateSchedule,
        validation_schedule: StepValidationSchedule | None,
    ) -> None:
        provenance = self.provenance
        self.logger.info(
            "Green optimizer name=%s implementation=%s base_lr=%.6e "
            "weight_decay=%.6e betas=%s eps=%.6e profile_step_time=%s "
            "checkpoint_policy=%s",
            provenance.name,
            provenance.implementation,
            provenance.learning_rate,
            provenance.weight_decay,
            provenance.betas,
            provenance.eps,
            provenance.profile_step_time,
            provenance.checkpoint_policy,
        )
        if provenance.soap is not None:
            self.logger.info(
                "Green SOAP upstream_commit=%s settings=%s "
                "frequency_unit=optimizer_step "
                "first_step_initializes_preconditioner=true",
                provenance.upstream_commit,
                provenance.soap,
            )
        self.logger.info(
            "Green learning-rate schedule enabled=%s kind=%s base_lr=%.6e "
            "min_lr=%.6e warmup_source=%s configured_warmup_epochs=%d "
            "configured_warmup_steps=%d effective_warmup_steps=%d "
            "steps_per_epoch=%d total_epochs=%d total_optimizer_steps=%d "
            "applies_to=first_stage_only",
            schedule.enabled,
            schedule.kind,
            schedule.base_learning_rate,
            schedule.min_learning_rate,
            schedule.warmup_source,
            schedule.configured_warmup_epochs,
            schedule.configured_warmup_steps,
            schedule.effective_warmup_steps,
            schedule.steps_per_epoch,
            schedule.total_epochs,
            schedule.total_optimizer_steps,
        )
        if validation_schedule is None:
            self.logger.info("Green validation schedule active=false")
        else:
            self.logger.info(
                "Green validation schedule active=true "
                "frequency_unit=optimizer_step every_steps=%d "
                "total_optimizer_steps=%d expected_events=%d "
                "final_step_mandatory=true",
                validation_schedule.every_steps,
                validation_schedule.total_optimizer_steps,
                validation_schedule.expected_event_count,
            )

    def write_provenance(
        self,
        schedule: GreenLearningRateSchedule,
        validation_schedule: StepValidationSchedule | None,
    ) -> None:
        payload = {
            "optimizer": self.provenance.as_dict(),
            "learning_rate_schedule": schedule.as_dict(),
            "validation_schedule": (
                {"active": False}
                if validation_schedule is None
                else validation_schedule.as_dict()
            ),
            "lbfgs_scheduler": "disabled",
        }
        path = self.work_dir / "green_optimizer_provenance.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")

    def record(
        self,
        *,
        phase: str,
        epoch: int,
        learning_rate: float,
        loss: float,
        split: str = "train",
        global_step: int | None = None,
        step_in_epoch: int | None = None,
        validation_index: int | None = None,
        learning_rate_first: float | None = None,
        learning_rate_last: float | None = None,
        rel_sol: float | None = None,
        val_rel_sol: float | None = None,
        rel_green: float | None = None,
        telemetry: dict[str, float] | None = None,
    ) -> None:
        row: dict[str, float | int | str] = {
            "phase": phase,
            "epoch": epoch,
            "split": split,
            "learning_rate": learning_rate,
            "loss": loss,
        }
        if global_step is not None:
            row["global_step"] = global_step
        if step_in_epoch is not None:
            row["step_in_epoch"] = step_in_epoch
        if validation_index is not None:
            row["validation_index"] = validation_index
        if learning_rate_first is not None:
            row["learning_rate_first"] = learning_rate_first
        if learning_rate_last is not None:
            row["learning_rate_last"] = learning_rate_last
        if rel_sol is not None:
            row["rel_sol"] = rel_sol
        if val_rel_sol is not None:
            row["val_rel_sol"] = val_rel_sol
        if rel_green is not None:
            row["rel_green"] = rel_green
        if telemetry is not None:
            row.update(telemetry)
        self.rows.append(row)

    def write_csv(self) -> None:
        if not self.rows:
            return
        fieldnames = list(self.rows[0])
        for row in self.rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        path = self.work_dir / "green_training_metrics.csv"
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
