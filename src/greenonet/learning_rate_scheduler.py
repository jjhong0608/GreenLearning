from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, isfinite, pi
from typing_extensions import Self

from torch import optim


@dataclass(frozen=True)
class LinearWarmupCosineSchedule:
    """Validated epoch-wise linear-warmup and cosine-decay schedule."""

    enabled: bool
    base_learning_rate: float
    min_learning_rate: float
    configured_warmup_epochs: int
    effective_warmup_epochs: int
    total_epochs: int

    @classmethod
    def from_values(
        cls,
        *,
        enabled: object,
        base_learning_rate: object,
        min_learning_rate: object,
        warmup_epochs: object,
        total_epochs: object,
        field_prefix: str,
    ) -> Self:
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool):
            raise TypeError(f"{field_prefix}.epochs must be an integer.")
        if total_epochs < 1:
            raise ValueError(f"{field_prefix}.epochs must be at least 1.")
        if not isinstance(enabled, bool):
            raise TypeError(f"{field_prefix}.use_lr_schedule must be a boolean.")

        base_lr = cls._finite_float(
            f"{field_prefix}.learning_rate",
            base_learning_rate,
        )
        if base_lr <= 0.0:
            raise ValueError(f"{field_prefix}.learning_rate must be positive.")

        if not enabled:
            return cls(
                enabled=False,
                base_learning_rate=base_lr,
                min_learning_rate=base_lr,
                configured_warmup_epochs=0,
                effective_warmup_epochs=0,
                total_epochs=total_epochs,
            )

        if not isinstance(warmup_epochs, int) or isinstance(warmup_epochs, bool):
            raise TypeError(f"{field_prefix}.warmup_epochs must be an integer.")
        if warmup_epochs < 0:
            raise ValueError(f"{field_prefix}.warmup_epochs must be non-negative.")
        min_lr = cls._finite_float(
            f"{field_prefix}.min_lr",
            min_learning_rate,
        )
        if min_lr < 0.0:
            raise ValueError(f"{field_prefix}.min_lr must be non-negative.")
        if min_lr > base_lr:
            raise ValueError(f"{field_prefix}.min_lr must not exceed learning_rate.")

        effective_warmup_epochs = min(warmup_epochs, total_epochs - 1)
        return cls(
            enabled=True,
            base_learning_rate=base_lr,
            min_learning_rate=min_lr,
            configured_warmup_epochs=warmup_epochs,
            effective_warmup_epochs=effective_warmup_epochs,
            total_epochs=total_epochs,
        )

    @staticmethod
    def _finite_float(field_name: str, value: object) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{field_name} must be numeric.")
        converted = float(value)
        if not isfinite(converted):
            raise ValueError(f"{field_name} must be finite.")
        return converted

    @property
    def kind(self) -> str:
        return "linear_warmup_cosine_decay" if self.enabled else "fixed"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["kind"] = self.kind
        return payload

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
