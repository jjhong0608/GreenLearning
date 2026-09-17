import numpy as np
import pytest

from docs.analysis.source_objective_comparison.build_report import objective_stats


def test_half_squared_norm_convention_and_quantiles():
    result = objective_stats([2.0, 4.0, 6.0])
    assert result["J_mean"] == 2.0
    assert result["J_max"] == 3.0
    assert result["J_p95"] == pytest.approx(2.9)
    tiny = objective_stats([2e-20, 4e-20, 6e-20])
    np.testing.assert_allclose(
        list(tiny.values()), np.array(list(result.values())) * 1e-20, rtol=1e-15, atol=0
    )


@pytest.mark.parametrize("values", [[], [-1.0], [float("nan")], [float("inf")]])
def test_invalid_objective_values_are_rejected(values):
    with pytest.raises(ValueError):
        objective_stats(values)
