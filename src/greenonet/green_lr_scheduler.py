from __future__ import annotations

from greenonet.config import TrainingConfig
from greenonet.learning_rate_scheduler import LinearWarmupCosineSchedule


class GreenLearningRateSchedule(LinearWarmupCosineSchedule):
    """Linear-warmup and cosine-decay schedule for GreenNet's first stage."""

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        *,
        steps_per_epoch: int,
    ) -> GreenLearningRateSchedule:
        return cls.from_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            field_prefix="training",
        )

    @classmethod
    def validate_config(cls, config: TrainingConfig) -> None:
        cls.validate_config_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            field_prefix="training",
        )

    @classmethod
    def configured_config(cls, config: TrainingConfig) -> dict[str, object]:
        return cls.configured_dict(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            total_epochs=config.epochs,
            field_prefix="training",
        )
