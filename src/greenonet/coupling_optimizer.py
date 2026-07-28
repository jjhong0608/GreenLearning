from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import torch
from torch import optim

from greenonet.config import (
    CouplingOptimizerConfig,
    CouplingTrainingConfig,
    SoapOptimizerConfig,
)
from greenonet.optimizer_support import (
    OptimizerProvenance,
    OptimizerStepProfiler as OptimizerStepProfiler,
    SOAP_UPSTREAM_COMMIT as SOAP_UPSTREAM_COMMIT,
    SOAP_UPSTREAM_REPOSITORY as SOAP_UPSTREAM_REPOSITORY,
    build_adamw_or_soap,
    build_optimizer_provenance,
)

CouplingOptimizerProvenance = OptimizerProvenance


class ComplexCouplingOptimizerFactory:
    """Build the selected optimizer without changing unit-square behavior."""

    def __init__(self, config: CouplingTrainingConfig) -> None:
        self.training_config = config
        self.optimizer_config = CouplingOptimizerConfig.from_raw(config.optimizer)
        self._validate_shared_settings()

    def _validate_shared_settings(self) -> None:
        for field_name, value, positive in (
            ("learning_rate", self.training_config.learning_rate, True),
            ("weight_decay", self.training_config.weight_decay, False),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"coupling_training.{field_name} must be numeric.")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"coupling_training.{field_name} must be finite.")
            if positive and converted <= 0.0:
                raise ValueError(f"coupling_training.{field_name} must be positive.")
            if not positive and converted < 0.0:
                raise ValueError(
                    f"coupling_training.{field_name} must be non-negative."
                )

    def build(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> optim.Optimizer:
        config = self.optimizer_config
        betas = (float(config.betas[0]), float(config.betas[1]))
        soap = SoapOptimizerConfig.from_raw(config.soap)
        return build_adamw_or_soap(
            parameters,
            name=config.name,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=betas,
            eps=config.eps,
            soap=soap,
        )

    def provenance(self) -> CouplingOptimizerProvenance:
        config = self.optimizer_config
        betas = (float(config.betas[0]), float(config.betas[1]))
        soap = SoapOptimizerConfig.from_raw(config.soap)
        return build_optimizer_provenance(
            name=config.name,
            learning_rate=float(self.training_config.learning_rate),
            weight_decay=float(self.training_config.weight_decay),
            betas=betas,
            eps=config.eps,
            profile_step_time=config.profile_step_time,
            soap=soap,
        )

    def resolved_config(self) -> dict[str, object]:
        """Return an executable optimizer block with all defaults materialized."""

        return asdict(self.optimizer_config)
