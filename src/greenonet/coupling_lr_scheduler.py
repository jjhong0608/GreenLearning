from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, pi

from torch import optim

from greenonet.config import CouplingTrainingConfig


@dataclass(frozen=True)
class CouplingLearningRateSchedule:
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
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool):
            raise TypeError("coupling_training.epochs must be an integer.")
        if total_epochs < 1:
            raise ValueError("coupling_training.epochs must be at least 1.")
        if not isinstance(config.use_lr_schedule, bool):
            raise TypeError("coupling_training.use_lr_schedule must be a boolean.")

        base_learning_rate = cls._finite_float(
            "learning_rate",
            config.learning_rate,
        )
        if base_learning_rate <= 0.0:
            raise ValueError("coupling_training.learning_rate must be positive.")

        if not config.use_lr_schedule:
            return cls(
                enabled=False,
                base_learning_rate=base_learning_rate,
                min_learning_rate=base_learning_rate,
                configured_warmup_epochs=0,
                effective_warmup_epochs=0,
                total_epochs=total_epochs,
            )

        if not isinstance(config.warmup_epochs, int) or isinstance(
            config.warmup_epochs,
            bool,
        ):
            raise TypeError("coupling_training.warmup_epochs must be an integer.")
        if config.warmup_epochs < 0:
            raise ValueError("coupling_training.warmup_epochs must be non-negative.")
        min_learning_rate = cls._finite_float("min_lr", config.min_lr)
        if min_learning_rate < 0.0:
            raise ValueError("coupling_training.min_lr must be non-negative.")
        if min_learning_rate > base_learning_rate:
            raise ValueError("coupling_training.min_lr must not exceed learning_rate.")

        effective_warmup_epochs = min(config.warmup_epochs, total_epochs - 1)
        return cls(
            enabled=True,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            configured_warmup_epochs=config.warmup_epochs,
            effective_warmup_epochs=effective_warmup_epochs,
            total_epochs=total_epochs,
        )

    @staticmethod
    def _finite_float(field_name: str, value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"coupling_training.{field_name} must be numeric.")
        converted = float(value)
        if not isfinite(converted):
            raise ValueError(f"coupling_training.{field_name} must be finite.")
        return converted

    @property
    def kind(self) -> str:
        return "linear_warmup_cosine_decay" if self.enabled else "fixed"

    def multiplier(self, epoch_index: int) -> float:
        """Return the LR multiplier used by the zero-based training epoch."""

        if not self.enabled:
            return 1.0
        if epoch_index < 0:
            raise ValueError("epoch_index must be non-negative.")

        warmup_epochs = self.effective_warmup_epochs
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        if self.total_epochs <= warmup_epochs + 1:
            return self.min_learning_rate / self.base_learning_rate

        progress = float(epoch_index - warmup_epochs) / float(
            self.total_epochs - warmup_epochs - 1
        )
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        min_ratio = self.min_learning_rate / self.base_learning_rate
        return float(min_ratio + (1.0 - min_ratio) * cosine)

    def build(
        self,
        optimizer: optim.Optimizer,
    ) -> optim.lr_scheduler.LambdaLR | None:
        if not self.enabled:
            return None
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=self.multiplier,
        )
