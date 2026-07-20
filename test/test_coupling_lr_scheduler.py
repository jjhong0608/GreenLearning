from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from greenonet.config import CouplingTrainingConfig
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule


def _learning_rates(
    config: CouplingTrainingConfig,
    *,
    group_learning_rates: tuple[float, ...] | None = None,
) -> list[tuple[float, ...]]:
    parameters = [
        torch.nn.Parameter(torch.tensor(float(index)))
        for index in range(len(group_learning_rates or (config.learning_rate,)))
    ]
    if group_learning_rates is None:
        optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(
            [
                {"params": [parameter], "lr": learning_rate}
                for parameter, learning_rate in zip(
                    parameters,
                    group_learning_rates,
                    strict=True,
                )
            ],
            lr=config.learning_rate,
        )
    schedule = CouplingLearningRateSchedule.from_config(
        config,
        total_epochs=config.epochs,
    )
    scheduler = schedule.build(optimizer)
    values: list[tuple[float, ...]] = []
    for _epoch in range(config.epochs):
        values.append(tuple(float(group["lr"]) for group in optimizer.param_groups))
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return values


def test_warmup_cosine_schedule_matches_expected_epoch_learning_rates():
    config = CouplingTrainingConfig(
        learning_rate=6.0e-3,
        epochs=6,
        use_lr_schedule=True,
        warmup_epochs=3,
        min_lr=1.0e-3,
    )

    values = _learning_rates(config)

    assert [value[0] for value in values] == pytest.approx(
        [2.0e-3, 4.0e-3, 6.0e-3, 6.0e-3, 3.5e-3, 1.0e-3]
    )


def test_zero_warmup_starts_at_base_lr_and_ends_at_min_lr():
    config = CouplingTrainingConfig(
        learning_rate=5.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_epochs=0,
        min_lr=1.0e-3,
    )

    values = _learning_rates(config)

    assert [value[0] for value in values] == pytest.approx([5.0e-3, 3.0e-3, 1.0e-3])


def test_disabled_schedule_keeps_fixed_learning_rate():
    config = CouplingTrainingConfig(
        learning_rate=2.5e-3,
        epochs=4,
        use_lr_schedule=False,
    )

    schedule = CouplingLearningRateSchedule.from_config(
        config,
        total_epochs=config.epochs,
    )
    values = _learning_rates(config)

    assert schedule.enabled is False
    assert (
        schedule.build(
            torch.optim.AdamW(
                [torch.nn.Parameter(torch.tensor(0.0))],
                lr=config.learning_rate,
            )
        )
        is None
    )
    assert [value[0] for value in values] == pytest.approx([2.5e-3] * 4)


def test_schedule_preserves_parameter_group_learning_rate_ratios():
    config = CouplingTrainingConfig(
        learning_rate=6.0e-3,
        epochs=4,
        use_lr_schedule=True,
        warmup_epochs=1,
        min_lr=1.0e-3,
    )

    values = _learning_rates(
        config,
        group_learning_rates=(6.0e-3, 3.0e-3),
    )

    for main_lr, secondary_lr in values:
        assert main_lr / secondary_lr == pytest.approx(2.0)
    assert values[-1] == pytest.approx((1.0e-3, 5.0e-4))


def test_warmup_longer_than_training_is_clamped_to_epochs_minus_one():
    config = CouplingTrainingConfig(
        learning_rate=4.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_epochs=9,
        min_lr=1.0e-3,
    )

    schedule = CouplingLearningRateSchedule.from_config(
        config,
        total_epochs=config.epochs,
    )
    values = _learning_rates(config)

    assert schedule.configured_warmup_epochs == 9
    assert schedule.effective_warmup_epochs == 2
    assert [value[0] for value in values] == pytest.approx([2.0e-3, 4.0e-3, 1.0e-3])


@pytest.mark.parametrize(
    ("updates", "error_type", "message"),
    (
        ({"learning_rate": 0.0}, ValueError, "learning_rate"),
        ({"learning_rate": float("inf")}, ValueError, "finite"),
        ({"epochs": 0}, ValueError, "epochs"),
        ({"warmup_epochs": -1}, ValueError, "warmup_epochs"),
        ({"min_lr": -1.0e-6}, ValueError, "min_lr"),
        ({"learning_rate": 1.0e-3, "min_lr": 2.0e-3}, ValueError, "exceed"),
    ),
)
def test_schedule_rejects_invalid_enabled_config(updates, error_type, message):
    config = replace(
        CouplingTrainingConfig(
            use_lr_schedule=True,
            learning_rate=1.0e-3,
            epochs=3,
            warmup_epochs=1,
            min_lr=1.0e-5,
        ),
        **updates,
    )

    with pytest.raises(error_type, match=message):
        CouplingLearningRateSchedule.from_config(
            config,
            total_epochs=config.epochs,
        )
