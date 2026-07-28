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
        total_epochs: int,
    ) -> GreenLearningRateSchedule:
        return cls.from_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            total_epochs=total_epochs,
            field_prefix="training",
        )
