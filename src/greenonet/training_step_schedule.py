from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepValidationSchedule:
    """Fixed optimizer-step validation cadence with a mandatory final event."""

    every_steps: int
    total_optimizer_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.every_steps, int) or isinstance(self.every_steps, bool):
            raise TypeError("validation_every_steps must be an integer.")
        if self.every_steps <= 0:
            raise ValueError("validation_every_steps must be positive.")
        if not isinstance(self.total_optimizer_steps, int) or isinstance(
            self.total_optimizer_steps, bool
        ):
            raise TypeError("total_optimizer_steps must be an integer.")
        if self.total_optimizer_steps < 1:
            raise ValueError("total_optimizer_steps must be at least 1.")

    @classmethod
    def for_validation(
        cls,
        *,
        validation_every_steps: int | None,
        total_optimizer_steps: int,
        field_prefix: str,
    ) -> StepValidationSchedule:
        if validation_every_steps is None:
            raise ValueError(
                f"{field_prefix}.validation_every_steps is required when a "
                "validation dataset is active."
            )
        return cls(
            every_steps=validation_every_steps,
            total_optimizer_steps=total_optimizer_steps,
        )

    @property
    def expected_event_count(self) -> int:
        quotient, remainder = divmod(self.total_optimizer_steps, self.every_steps)
        return quotient + int(remainder > 0)

    def is_due(self, global_step: int) -> bool:
        if not isinstance(global_step, int) or isinstance(global_step, bool):
            raise TypeError("global_step must be an integer.")
        if global_step < 1 or global_step > self.total_optimizer_steps:
            raise ValueError(
                f"global_step must be in [1, {self.total_optimizer_steps}]."
            )
        return (
            global_step % self.every_steps == 0
            or global_step == self.total_optimizer_steps
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "kind": "fixed_optimizer_step_interval_with_final",
            "every_steps": self.every_steps,
            "total_optimizer_steps": self.total_optimizer_steps,
            "expected_event_count": self.expected_event_count,
        }
