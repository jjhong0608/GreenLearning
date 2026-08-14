from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, isfinite, pi
from typing import Literal
from typing_extensions import Self

from torch import optim


WarmupSource = Literal["warmup_steps", "legacy_warmup_epochs", "disabled"]


@dataclass(frozen=True)
class LinearWarmupCosineSchedule:
    """Validated optimizer-step linear-warmup and cosine-decay schedule."""

    enabled: bool
    base_learning_rate: float
    min_learning_rate: float
    configured_warmup_epochs: int
    configured_warmup_steps: int
    effective_warmup_steps: int
    warmup_source: WarmupSource
    steps_per_epoch: int
    total_epochs: int
    total_optimizer_steps: int

    @classmethod
    def validate_config_values(
        cls,
        *,
        enabled: object,
        base_learning_rate: object,
        min_learning_rate: object,
        warmup_epochs: object,
        warmup_steps: object,
        total_epochs: object,
        field_prefix: str,
    ) -> None:
        cls._validated_config_values(
            enabled=enabled,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_epochs=warmup_epochs,
            warmup_steps=warmup_steps,
            total_epochs=total_epochs,
            field_prefix=field_prefix,
        )

    @classmethod
    def configured_dict(
        cls,
        *,
        enabled: object,
        base_learning_rate: object,
        min_learning_rate: object,
        warmup_epochs: object,
        warmup_steps: object,
        total_epochs: object,
        field_prefix: str,
    ) -> dict[str, object]:
        (
            schedule_enabled,
            base_lr,
            min_lr,
            configured_warmup_epochs,
            configured_warmup_steps,
        ) = cls._validated_config_values(
            enabled=enabled,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_epochs=warmup_epochs,
            warmup_steps=warmup_steps,
            total_epochs=total_epochs,
            field_prefix=field_prefix,
        )
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool):
            raise AssertionError("validated total_epochs must be an integer")
        return {
            "enabled": schedule_enabled,
            "kind": "linear_warmup_cosine_decay" if schedule_enabled else "fixed",
            "base_learning_rate": base_lr,
            "min_learning_rate": min_lr,
            "configured_warmup_epochs": configured_warmup_epochs,
            "configured_warmup_steps": configured_warmup_steps,
            "warmup_source": (
                "disabled"
                if not schedule_enabled
                else (
                    "warmup_steps"
                    if configured_warmup_steps is not None
                    else "legacy_warmup_epochs"
                )
            ),
            "total_epochs": total_epochs,
            "resolution": "runtime_after_dataloader",
        }

    @classmethod
    def from_values(
        cls,
        *,
        enabled: object,
        base_learning_rate: object,
        min_learning_rate: object,
        warmup_epochs: object,
        warmup_steps: object,
        total_epochs: object,
        steps_per_epoch: object,
        field_prefix: str,
    ) -> Self:
        (
            schedule_enabled,
            base_lr,
            min_lr,
            configured_warmup_epochs,
            explicit_warmup_steps,
        ) = cls._validated_config_values(
            enabled=enabled,
            base_learning_rate=base_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_epochs=warmup_epochs,
            warmup_steps=warmup_steps,
            total_epochs=total_epochs,
            field_prefix=field_prefix,
        )
        if not isinstance(steps_per_epoch, int) or isinstance(steps_per_epoch, bool):
            raise TypeError(f"{field_prefix}.steps_per_epoch must be an integer.")
        if steps_per_epoch < 1:
            raise ValueError(f"{field_prefix}.steps_per_epoch must be at least 1.")
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool):
            raise AssertionError("validated total_epochs must be an integer")

        resolved_total_epochs = total_epochs
        total_optimizer_steps = resolved_total_epochs * steps_per_epoch
        if not schedule_enabled:
            return cls(
                enabled=False,
                base_learning_rate=base_lr,
                min_learning_rate=base_lr,
                configured_warmup_epochs=configured_warmup_epochs,
                configured_warmup_steps=0,
                effective_warmup_steps=0,
                warmup_source="disabled",
                steps_per_epoch=steps_per_epoch,
                total_epochs=resolved_total_epochs,
                total_optimizer_steps=total_optimizer_steps,
            )

        if explicit_warmup_steps is None:
            configured_steps = configured_warmup_epochs * steps_per_epoch
            warmup_source: WarmupSource = "legacy_warmup_epochs"
        else:
            configured_steps = explicit_warmup_steps
            warmup_source = "warmup_steps"
        effective_warmup_steps = min(configured_steps, total_optimizer_steps - 1)
        return cls(
            enabled=True,
            base_learning_rate=base_lr,
            min_learning_rate=min_lr,
            configured_warmup_epochs=configured_warmup_epochs,
            configured_warmup_steps=configured_steps,
            effective_warmup_steps=effective_warmup_steps,
            warmup_source=warmup_source,
            steps_per_epoch=steps_per_epoch,
            total_epochs=resolved_total_epochs,
            total_optimizer_steps=total_optimizer_steps,
        )

    @classmethod
    def _validated_config_values(
        cls,
        *,
        enabled: object,
        base_learning_rate: object,
        min_learning_rate: object,
        warmup_epochs: object,
        warmup_steps: object,
        total_epochs: object,
        field_prefix: str,
    ) -> tuple[bool, float, float, int, int | None]:
        if not isinstance(total_epochs, int) or isinstance(total_epochs, bool):
            raise TypeError(f"{field_prefix}.epochs must be an integer.")
        if total_epochs < 1:
            raise ValueError(f"{field_prefix}.epochs must be at least 1.")
        if not isinstance(enabled, bool):
            raise TypeError(f"{field_prefix}.use_lr_schedule must be a boolean.")
        if not isinstance(warmup_epochs, int) or isinstance(warmup_epochs, bool):
            raise TypeError(f"{field_prefix}.warmup_epochs must be an integer.")
        if warmup_epochs < 0:
            raise ValueError(f"{field_prefix}.warmup_epochs must be non-negative.")
        resolved_warmup_steps: int | None
        if warmup_steps is None:
            resolved_warmup_steps = None
        else:
            if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool):
                raise TypeError(
                    f"{field_prefix}.warmup_steps must be an integer or null."
                )
            if warmup_steps < 0:
                raise ValueError(f"{field_prefix}.warmup_steps must be non-negative.")
            if warmup_epochs > 0:
                raise ValueError(
                    f"{field_prefix}.warmup_steps and a positive warmup_epochs "
                    "cannot be configured together."
                )
            resolved_warmup_steps = warmup_steps

        base_lr = cls._finite_float(
            f"{field_prefix}.learning_rate",
            base_learning_rate,
        )
        if base_lr <= 0.0:
            raise ValueError(f"{field_prefix}.learning_rate must be positive.")
        if not enabled:
            return False, base_lr, base_lr, warmup_epochs, resolved_warmup_steps

        min_lr = cls._finite_float(
            f"{field_prefix}.min_lr",
            min_learning_rate,
        )
        if min_lr < 0.0:
            raise ValueError(f"{field_prefix}.min_lr must be non-negative.")
        if min_lr > base_lr:
            raise ValueError(f"{field_prefix}.min_lr must not exceed learning_rate.")
        return True, base_lr, min_lr, warmup_epochs, resolved_warmup_steps

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
        payload["resolution"] = "runtime_resolved"
        return payload

    def multiplier(self, step_index: int) -> float:
        """Return the LR multiplier used by a zero-based optimizer call."""

        if not self.enabled:
            return 1.0
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise TypeError("step_index must be an integer.")
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")

        warmup_steps = self.effective_warmup_steps
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)
        if self.total_optimizer_steps <= warmup_steps + 1:
            return self.min_learning_rate / self.base_learning_rate

        progress = float(step_index - warmup_steps) / float(
            self.total_optimizer_steps - warmup_steps - 1
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
