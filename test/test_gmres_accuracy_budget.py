import numpy as np
import pytest
import torch

from cli.audit_gmres_accuracy_budget import restart_for_budget, first_match


def test_budget_contract():
    assert restart_for_budget(3) == 3
    assert restart_for_budget(100) == 100
    assert restart_for_budget(700) == 100
    for budget in (-1, 0, 101, 150):
        with pytest.raises(ValueError):
            restart_for_budget(budget)


def test_first_crossing_is_not_global_minimum():
    rows = [
        dict(budget=0, mean=4.0, p95=5.0),
        dict(budget=1, mean=1.0, p95=3.0),
        dict(budget=2, mean=2.0, p95=4.0),
        dict(budget=3, mean=0.5, p95=1.0),
    ]
    assert first_match(rows, 1.5, 2.0, False) == 1
    assert first_match(rows, 1.5, 2.0, True) == 3
    assert first_match(rows, 0.1, 0.1, True) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cupy_truncated_cycle_matches_scipy():
    cp = pytest.importorskip("cupy")
    from cupyx.scipy.sparse.linalg import gmres
    from scipy.sparse.linalg import gmres as scipy_gmres

    rng = np.random.default_rng(91)
    a = rng.normal(size=(12, 12)) + 5 * np.eye(12)
    b = rng.normal(size=12)
    with cp.cuda.Device(1):
        for budget in (1, 2, 4, 7):
            history = []
            actual, status = gmres(
                cp.asarray(a),
                cp.asarray(b),
                restart=restart_for_budget(budget),
                maxiter=budget,
                rtol=1e-14,
                callback=history.append,
                callback_type="pr_norm",
            )
            expected, _ = scipy_gmres(a, b, restart=budget, maxiter=1, rtol=1e-14)
            np.testing.assert_allclose(
                cp.asnumpy(actual), expected, rtol=1e-10, atol=1e-12
            )
            assert len(history) == 1
            assert status == budget
