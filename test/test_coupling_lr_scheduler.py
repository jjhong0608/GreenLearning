from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from greenonet.config import CouplingTrainingConfig
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule


def _learning_rates(
    config: CouplingTrainingConfig,
    *,
    steps_per_epoch: int,
    group_learning_rates: tuple[float, ...] | None = None,
) -> tuple[CouplingLearningRateSchedule, list[tuple[float, ...]]]:
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


def test_step_warmup_cosine_schedule_matches_expected_learning_rates():
    config = CouplingTrainingConfig(
        learning_rate=6.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_steps=3,
        min_lr=1.0e-3,
    )

    schedule, values = _learning_rates(config, steps_per_epoch=2)

    assert schedule.total_optimizer_steps == 6
    assert schedule.warmup_source == "warmup_steps"
    assert [value[0] for value in values] == pytest.approx(
        [2.0e-3, 4.0e-3, 6.0e-3, 6.0e-3, 3.5e-3, 1.0e-3]
    )


def test_legacy_warmup_epochs_resolve_from_steps_per_epoch():
    config = CouplingTrainingConfig(
        learning_rate=4.0e-3,
        epochs=4,
        use_lr_schedule=True,
        warmup_epochs=2,
        min_lr=1.0e-3,
    )

    schedule, _values = _learning_rates(config, steps_per_epoch=3)

    assert schedule.warmup_source == "legacy_warmup_epochs"
    assert schedule.configured_warmup_epochs == 2
    assert schedule.configured_warmup_steps == 6
    assert schedule.effective_warmup_steps == 6


def test_zero_warmup_and_disabled_schedule():
    scheduled = CouplingTrainingConfig(
        learning_rate=5.0e-3,
        epochs=3,
        use_lr_schedule=True,
        warmup_steps=0,
        min_lr=1.0e-3,
    )
    fixed = CouplingTrainingConfig(
        learning_rate=2.5e-3,
        epochs=2,
        use_lr_schedule=False,
    )

    _schedule, scheduled_values = _learning_rates(scheduled, steps_per_epoch=1)
    fixed_schedule, fixed_values = _learning_rates(fixed, steps_per_epoch=2)

    assert [value[0] for value in scheduled_values] == pytest.approx(
        [5.0e-3, 3.0e-3, 1.0e-3]
    )
    assert fixed_schedule.enabled is False
    assert [value[0] for value in fixed_values] == pytest.approx([2.5e-3] * 4)


def test_schedule_preserves_parameter_group_learning_rate_ratios():
    config = CouplingTrainingConfig(
        learning_rate=6.0e-3,
        epochs=2,
        use_lr_schedule=True,
        warmup_steps=1,
        min_lr=1.0e-3,
    )

    _schedule, values = _learning_rates(
        config,
        steps_per_epoch=2,
        group_learning_rates=(6.0e-3, 3.0e-3),
    )

    for main_lr, secondary_lr in values:
        assert main_lr / secondary_lr == pytest.approx(2.0)
    assert values[-1] == pytest.approx((1.0e-3, 5.0e-4))


def test_warmup_longer_than_training_is_clamped_to_total_steps_minus_one():
    config = CouplingTrainingConfig(
        learning_rate=4.0e-3,
        epochs=2,
        use_lr_schedule=True,
        warmup_steps=99,
        min_lr=1.0e-3,
    )

    schedule, values = _learning_rates(config, steps_per_epoch=2)

    assert schedule.configured_warmup_steps == 99
    assert schedule.effective_warmup_steps == 3
    assert [value[0] for value in values] == pytest.approx(
        [4.0e-3 / 3.0, 8.0e-3 / 3.0, 4.0e-3, 1.0e-3]
    )


def test_equal_step_budgets_have_identical_learning_rate_sequences():
    sequences: list[list[float]] = []
    for steps_per_epoch, epochs in ((3, 800), (6, 400), (12, 200), (24, 100)):
        config = CouplingTrainingConfig(
            learning_rate=2.0e-3,
            epochs=epochs,
            use_lr_schedule=True,
            warmup_steps=240,
            min_lr=1.0e-5,
        )
        schedule, values = _learning_rates(config, steps_per_epoch=steps_per_epoch)
        assert schedule.total_optimizer_steps == 2400
        sequences.append([value[0] for value in values])

    for sequence in sequences[1:]:
        assert sequence == pytest.approx(sequences[0])


def test_single_step_schedule_uses_final_minimum_learning_rate():
    config = CouplingTrainingConfig(
        learning_rate=2.0e-3,
        epochs=1,
        use_lr_schedule=True,
        warmup_steps=0,
        min_lr=1.0e-5,
    )

    _schedule, values = _learning_rates(config, steps_per_epoch=1)

    assert values == pytest.approx([(1.0e-5,)])


def test_explicit_warmup_steps_conflicts_with_positive_warmup_epochs():
    with pytest.raises(ValueError, match="cannot be configured together"):
        CouplingTrainingConfig(warmup_epochs=1, warmup_steps=10)


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
def test_schedule_rejects_invalid_enabled_config(updates, error_type, message):
    config = replace(
        CouplingTrainingConfig(
            use_lr_schedule=True,
            learning_rate=1.0e-3,
            epochs=3,
            warmup_steps=1,
            min_lr=1.0e-5,
        ),
        **updates,
    )

    with pytest.raises(error_type, match=message):
        CouplingLearningRateSchedule.from_config(config, steps_per_epoch=1)


def test_config_rejects_negative_warmup_steps():
    with pytest.raises(ValueError, match="warmup_steps"):
        CouplingTrainingConfig(warmup_steps=-1)
