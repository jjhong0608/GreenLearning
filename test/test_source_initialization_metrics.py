import numpy as np
import pytest
import torch

from greenonet.source_initialization_metrics import (
    match_accuracy,
    physical_equal_split,
    relative_norm,
    summarize,
    symmetric_source_balance,
)


def test_balance_preserves_input_and_pair_correction_norm():
    pair = torch.tensor([[[1.0, 3.0], [2.0, 5.0]]], dtype=torch.float64)
    rhs = torch.tensor([[2.0, 4.0]], dtype=torch.float64)
    original = pair.clone()
    balanced, residual = symmetric_source_balance(pair, rhs)
    assert torch.equal(pair, original)
    torch.testing.assert_close(balanced.sum(1), rhs)
    torch.testing.assert_close((balanced - pair).norm(), residual.norm() / np.sqrt(2))
    torch.testing.assert_close(physical_equal_split(rhs).sum(1), rhs)


def test_relative_zero_is_missing():
    assert relative_norm(torch.ones(2), torch.zeros(2)) == (
        None,
        "target_norm_at_or_below_eps",
    )


def test_first_crossing_tail_and_reversal():
    target = dict(
        example="square",
        run_id="s0",
        seed=0,
        fingerprint="a",
        evaluation_k=2,
        rel_sol_mean=1.0,
        rel_sol_p95=2.0,
        rel_sol_max=3.0,
    )
    curve = [
        dict(evaluation_k=k, rel_sol_mean=m, rel_sol_p95=p, rel_sol_max=4.0)
        for k, (m, p) in enumerate(((2.0, 4.0), (0.9, 3.0), (1.1, 2.0), (0.8, 1.9)))
    ]
    matched = match_accuracy(target, curve)
    assert matched["mean_k"] == 1
    assert matched["mean_p95_k"] == 3
    assert matched["mean_later_nonattainment"]
    assert not matched["mean_p95_later_nonattainment"]
    target["rel_sol_mean"] = 0.1
    assert match_accuracy(target, curve)["mean_status"] == "not_reached"
    with pytest.raises(ValueError, match="complete"):
        match_accuracy(target, curve[1:])
    curve[0]["rel_sol_mean"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        match_accuracy(target, curve)


def test_summary_linear_quantile_and_missing():
    rows = [
        dict(example="x", run_id="a", rel_sol=float(i), balance_relative=None)
        for i in range(10)
    ]
    result = summarize(rows)[0]
    assert result["rel_sol_p95"] == pytest.approx(8.55)
    assert "balance_relative_mean" not in result
