from __future__ import annotations

import pytest

from greenonet.training_step_schedule import StepValidationSchedule


def test_step_validation_schedule_uses_interval_and_unique_final_step():
    divisible = StepValidationSchedule(every_steps=4, total_optimizer_steps=12)
    remainder = StepValidationSchedule(every_steps=4, total_optimizer_steps=10)

    assert [step for step in range(1, 13) if divisible.is_due(step)] == [4, 8, 12]
    assert [step for step in range(1, 11) if remainder.is_due(step)] == [4, 8, 10]
    assert divisible.expected_event_count == 3
    assert remainder.expected_event_count == 3


def test_step_validation_schedule_requires_interval_for_active_validation():
    with pytest.raises(ValueError, match="validation_every_steps is required"):
        StepValidationSchedule.for_validation(
            validation_every_steps=None,
            total_optimizer_steps=10,
            field_prefix="training",
        )


@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_step_validation_schedule_rejects_invalid_intervals(value):
    with pytest.raises((TypeError, ValueError)):
        StepValidationSchedule(every_steps=value, total_optimizer_steps=10)
