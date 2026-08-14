from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from greenonet.config import TrainingConfig
from greenonet.green_lr_scheduler import GreenLearningRateSchedule


def _learning_rates(
    config: TrainingConfig,
    *,
    steps_per_epoch: int,
    group_learning_rates: tuple[float, ...] | None = None,
) -> tuple[GreenLearningRateSchedule, list[tuple[float, ...]]]:
    parameters = [
        torch.nn.Parameter(torch.tensor(float(index)))
        for index in range(len(group_learning_rates or (config.learning_rate,)))
    ]
    optimizer = torch.optim.AdamW(
        (
            parameters
            if group_learning_rates is None
            else [
                {"params": [parameter], "lr": learning_rate}
                for parameter, learning_rate in zip(
                    parameters,
                    group_learning_rates,
                    strict=True,
                )
            ]
        ),
        lr=config.learning_rate,
    )
    schedule = GreenLearningRateSchedule.from_config(
        config,
        steps_per_epoch=steps_per_epoch,
    )
    scheduler = schedule.build(optimizer)
    values: list[tuple[float, ...]] = []
    for _step in range(schedule.total_optimizer_steps):
        values.append(tuple(float(group["lr"]) for group in optimizer.param_groups))
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return schedule, values


def test_green_step_schedule_and_legacy_epoch_fallback():
    explicit = TrainingConfig(
        learning_rate=6.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_steps=3,
        min_lr=1.0e-3,
    )
    legacy = TrainingConfig(
        learning_rate=6.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_epochs=1,
        min_lr=1.0e-3,
    )

    explicit_schedule, explicit_values = _learning_rates(
        explicit,
        steps_per_epoch=2,
    )
    legacy_schedule, _legacy_values = _learning_rates(legacy, steps_per_epoch=2)

    assert explicit_schedule.total_optimizer_steps == 6
    assert [value[0] for value in explicit_values] == pytest.approx(
        [2.0e-3, 4.0e-3, 6.0e-3, 6.0e-3, 3.5e-3, 1.0e-3]
    )
    assert legacy_schedule.warmup_source == "legacy_warmup_epochs"
    assert legacy_schedule.configured_warmup_steps == 2


def test_green_zero_warmup_disabled_schedule_and_parameter_group_ratio():
    scheduled = TrainingConfig(
        learning_rate=6.0e-3,
        epochs=2,
        use_lr_schedule=True,
        warmup_steps=0,
        min_lr=1.0e-3,
    )
    fixed = TrainingConfig(
        learning_rate=2.5e-3,
        epochs=2,
        use_lr_schedule=False,
    )

    _schedule, values = _learning_rates(
        scheduled,
        steps_per_epoch=2,
        group_learning_rates=(6.0e-3, 3.0e-3),
    )
    _fixed_schedule, fixed_values = _learning_rates(fixed, steps_per_epoch=2)

    for main_lr, secondary_lr in values:
        assert main_lr / secondary_lr == pytest.approx(2.0)
    assert values[-1] == pytest.approx((1.0e-3, 5.0e-4))
    assert [value[0] for value in fixed_values] == pytest.approx([2.5e-3] * 4)


def test_green_explicit_warmup_steps_conflicts_with_positive_warmup_epochs():
    with pytest.raises(ValueError, match="cannot be configured together"):
        TrainingConfig(warmup_epochs=1, warmup_steps=10)


@pytest.mark.parametrize(
    ("updates", "error_type", "message"),
    (
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": float("inf")}, ValueError, "finite"),
        ({"epochs": 0}, ValueError, "epochs"),
        ({"min_lr": -1.0e-6}, ValueError, "min_lr"),
        ({"learning_rate": 1.0e-3, "min_lr": 2.0e-3}, ValueError, "exceed"),
    ),
)
def test_green_schedule_rejects_invalid_enabled_config(
    updates,
    error_type,
    message,
):
    config = replace(
        TrainingConfig(
            use_lr_schedule=True,
            learning_rate=1.0e-3,
            epochs=3,
            warmup_steps=1,
            min_lr=1.0e-5,
        ),
        **updates,
    )

    with pytest.raises(error_type, match=message):
        GreenLearningRateSchedule.from_config(config, steps_per_epoch=1)


def test_green_config_rejects_negative_warmup_steps():
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(warmup_steps=-1)
