from __future__ import annotations

from greenonet.config import CouplingTrainingConfig
from greenonet.learning_rate_scheduler import LinearWarmupCosineSchedule


class CouplingLearningRateSchedule(LinearWarmupCosineSchedule):
    """Validated linear-warmup and cosine-decay schedule for CouplingNet."""

    enabled: bool
    base_learning_rate: float
    min_learning_rate: float
    configured_warmup_epochs: int
    effective_warmup_epochs: int
    total_epochs: int

    @classmethod
    def from_config(
        cls,
        config: CouplingTrainingConfig,
        *,
        total_epochs: int,
    ) -> CouplingLearningRateSchedule:
        return cls.from_values(
            enabled=config.use_lr_schedule,
            base_learning_rate=config.learning_rate,
            min_learning_rate=config.min_lr,
            warmup_epochs=config.warmup_epochs,
            total_epochs=total_epochs,
            field_prefix="coupling_training",
        )
