import pytest
import torch

from greenonet.reference_green_reconstruction import DiffusionReferenceBuilder
from greenonet.greens import ExactGreenFunction


def test_reference_matches_existing_on_nonuniform_grid():
    t = torch.tensor([0.0, 0.07, 0.4, 0.9, 1.0], dtype=torch.float64)
    a = 1 + t
    actual = DiffusionReferenceBuilder.kernel(t, a, 1)
    torch.testing.assert_close(actual, ExactGreenFunction(t, a)(), rtol=0, atol=0)


@pytest.mark.parametrize("factor", [1, 4, 32])
def test_constant_diffusion_poisson_and_boundary(factor):
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    fine = DiffusionReferenceBuilder.refine(t, factor)
    actual = DiffusionReferenceBuilder.kernel(fine, torch.ones_like(fine), factor)
    expected = torch.minimum(t[:, None], t[None, :]) * (
        1 - torch.maximum(t[:, None], t[None, :])
    )
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-15)
    assert torch.count_nonzero(actual[[0, -1]]) == 0
    torch.testing.assert_close(actual, actual.T)


def test_variable_diffusion_converges_to_log_formula():
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    r = torch.log1p(t)
    expected = (
        torch.minimum(r[:, None], r[None, :])
        * (r[-1] - torch.maximum(r[:, None], r[None, :]))
        / r[-1]
    )
    errors = []
    for factor in (1, 2, 4, 8):
        fine = DiffusionReferenceBuilder.refine(t, factor)
        actual = DiffusionReferenceBuilder.kernel(fine, 1 + fine, factor)
        errors.append(float((actual - expected).norm()))
    assert all(a / b > 3.9 for a, b in zip(errors[:-1], errors[1:]))


def test_kernel_rejects_nonpositive_diffusion():
    t = torch.linspace(0, 1, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive"):
        DiffusionReferenceBuilder.kernel(t, torch.zeros_like(t), 1)


def test_physical_response_scale_is_length_squared():
    t = torch.linspace(0, 1, 9, dtype=torch.float64)
    g = DiffusionReferenceBuilder.kernel(t, torch.ones_like(t), 1)
    weights = torch.ones_like(t) / 8
    weights[[0, -1]] /= 2
    indices = torch.arange(1, 8)
    unit = DiffusionReferenceBuilder.response_matrix(g, weights, indices, 1.0)
    short = DiffusionReferenceBuilder.response_matrix(g, weights, indices, 0.25)
    torch.testing.assert_close(short, unit / 16)


def test_refined_reference_matches_existing_dense_subset():
    base = torch.tensor([0.0, 0.07, 0.4, 0.9, 1.0], dtype=torch.float64)
    fine = DiffusionReferenceBuilder.refine(base, 4)
    a = 1 + 0.4 * torch.sin(7 * fine)
    actual = DiffusionReferenceBuilder.kernel(fine, a, 4)
    expected = ExactGreenFunction(fine, a)()[::4, ::4]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_summary_does_not_merge_reference_resolutions():
    from cli.audit_reference_green_reconstruction import METRICS, summarize

    rows = []
    for factor in (1, 4):
        for i in range(3):
            rows.append(
                dict(
                    example="disk",
                    run_id="shared",
                    seed=None,
                    source_condition="reference_raw",
                    evaluation_k=0,
                    kernel="reference",
                    factor=factor,
                    **{metric: float(i + factor) for metric in METRICS},
                )
            )
    summary = summarize(rows)
    assert len(summary) == 2
    assert summary[0]["J_mean"] == 2
    assert summary[1]["J_mean"] == 5
    assert all(r["sample_count"] == 3 for r in summary)
