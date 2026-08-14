from __future__ import annotations

from greenonet.config import CouplingTrainingConfig
from greenonet.learning_rate_scheduler import LinearWarmupCosineSchedule


class CouplingLearningRateSchedule(LinearWarmupCosineSchedule):
    """Validated linear-warmup and cosine-decay schedule for CouplingNet."""

    enabled: bool
    base_learning_rate: float
    min_learning_rate: float

    @classmethod
    def from_config(
        cls,
        config: CouplingTrainingConfig,
        *,
        steps_per_epoch: int,
    ) -> CouplingLearningRateSchedule:
        return cls.from_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            field_prefix="coupling_training",
        )

    @classmethod
    def validate_config(cls, config: CouplingTrainingConfig) -> None:
        cls.validate_config_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            field_prefix="coupling_training",
        )

    @classmethod
    def configured_config(cls, config: CouplingTrainingConfig) -> dict[str, object]:
        return cls.configured_dict(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            field_prefix="coupling_training",
        )
