from __future__ import annotations

import math

import pytest

from greenonet.unit_square_training_size_analysis import (
    SizeSummary,
    _mean_sd_ci95,
    choose_training_size,
)


def _summary(train_size: int, rel_sol_mean: float) -> SizeSummary:
    return SizeSummary(
        train_size=train_size,
        seed_count=4,
        rel_sol_mean=rel_sol_mean,
        rel_sol_sd=0.0,
        rel_sol_ci95_low=rel_sol_mean,
        rel_sol_ci95_high=rel_sol_mean,
        rel_sol_median_mean=rel_sol_mean,
        rel_sol_p90_mean=rel_sol_mean,
        rel_sol_max_mean=rel_sol_mean,
        rel_flux_mean=rel_sol_mean,
        rel_flux_sd=0.0,
        rel_flux_ci95_low=rel_sol_mean,
        rel_flux_ci95_high=rel_sol_mean,
        rel_flux_p90_mean=rel_sol_mean,
        rel_flux_max_mean=rel_sol_mean,
        rel_sol_equal_mean=rel_sol_mean * 1.1,
        weak_blend_gain=1.0 - 1.0 / 1.1,
        energy_optimized_mean=1e-5,
        best_step_mean=2000.0,
        best_step_fraction=2000.0 / 2400.0,
        final_over_best_validation_energy_mean=1.0,
        final_model_rel_sol_mean=rel_sol_mean,
        best_checkpoint_gain_over_final=0.0,
        relative_gap_to_best=0.0,
        within_saturation_tolerance=True,
    )


def test_choose_training_size_uses_smallest_candidate_within_tolerance() -> None:
    summaries = (
        _summary(1200, 0.0040),
        _summary(2400, 0.00369),
        _summary(4800, 0.00350),
    )

    decision = choose_training_size(summaries, saturation_tolerance=0.06)

    assert decision.best_observed_num_train == 4800
    assert decision.smallest_within_tolerance == 2400
    assert decision.recommended_num_train == 2400


def test_choose_training_size_rejects_candidate_outside_five_percent() -> None:
    summaries = (
        _summary(2400, 0.003695946),
        _summary(4800, 0.003505000),
    )

    decision = choose_training_size(summaries, saturation_tolerance=0.05)

    assert decision.recommended_num_train == 4800


def test_mean_sd_ci95_uses_seed_level_sample_standard_deviation() -> None:
    center, spread, low, high = _mean_sd_ci95((1.0, 2.0, 3.0, 4.0))

    assert center == 2.5
    assert spread == pytest.approx(math.sqrt(5.0 / 3.0))
    assert low < center < high
    assert center - low == pytest.approx(high - center)


@pytest.mark.parametrize("tolerance", (-0.1, 1.0, math.inf, math.nan))
def test_choose_training_size_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="saturation_tolerance"):
        choose_training_size((_summary(4800, 0.0035),), tolerance)
