from dataclasses import replace

import pytest
import torch

from cli.audit_reference_green_reoptimization import optimize_sources, reference_context
from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_tangent_projection import SymmetricTangentGreenResponseContext
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig


def make_operator(x, y):
    indices = torch.arange(len(x))
    return FrozenBidirectionalResponseOperator(
        *[
            FrozenAxialResponseOperator(
                axis,
                len(x),
                (
                    AxialResponseBlock(
                        indices, torch.diag(torch.tensor(values, dtype=torch.float64))
                    ),
                ),
            )
            for axis, values in (("x", x), ("y", y))
        ]
    )


def test_reoptimization_rebuilds_preconditioner_and_solves_reference_objective():
    config = SymmetricTangentGreenResponseProjectionConfig(
        eta_strategy="closed_loop_exact_line_search",
        subspace_dimension=2,
        max_subspace_dimension=2,
        direction_normalization="response",
        line_search_relative_eps=1e-24,
        eta_cap_enabled=False,
    )
    learned = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=make_operator([0.5, 0.25], [1.0, 1.0]),
        point_mass=0.2,
        config=config,
    )
    operator = make_operator([1.0, 2.0], [2.0, 3.0])
    reference = reference_context(operator, learned.point_mass, config, 2)
    assert reference.response_operator is operator
    assert not torch.equal(reference.denominator, learned.denominator)
    torch.testing.assert_close(
        reference.gamma_x_squared, 0.2 * torch.tensor([1.0, 4.0], dtype=torch.float64)
    )
    pair = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], dtype=torch.float64)
    original = pair.clone()
    result = optimize_sources(reference, pair, 2)
    corrected = pair + torch.stack((result.final_delta, -result.final_delta), 1)
    torch.testing.assert_close(pair, original, rtol=0, atol=0)
    torch.testing.assert_close(corrected.sum(1), pair.sum(1), rtol=1e-14, atol=1e-14)
    expected_phi = torch.tensor([[2 / 3, 3 / 5]], dtype=torch.float64)
    torch.testing.assert_close(corrected[:, 0], expected_phi, rtol=1e-12, atol=1e-12)
    response = operator.forward_pair(corrected)
    assert float((response[:, 0] - response[:, 1]).norm()) < 1e-12


@pytest.mark.parametrize("normalization", ["legacy", "response"])
def test_reference_gradient_is_adjoint_of_reference_not_learned(normalization):
    config = SymmetricTangentGreenResponseProjectionConfig(
        eta_strategy="closed_loop_exact_line_search",
        direction_normalization=normalization,
        eta_cap_enabled=False,
    )
    operator = make_operator([1.0, 2.0], [2.0, 3.0])
    context = reference_context(operator, 0.2, config, 4)
    pair = torch.tensor([[[1.0, -0.5], [0.1, 0.7]]], dtype=torch.float64)
    delta = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    response = operator.forward_pair(pair + torch.stack((delta, -delta), 1))
    mismatch = response[:, 0] - response[:, 1]
    objective = 0.5 * 0.2 * mismatch.square().sum()
    expected = torch.autograd.grad(objective, delta)[0]
    torch.testing.assert_close(context.tangent_gradient(mismatch.detach()), expected)
    result = optimize_sources(context, pair, 4)
    assert torch.isfinite(result.deltas).all()
    assert float(result.final_cost.detach()) < 0.2 * float(
        mismatch.detach().square().sum()
    )


def test_reference_context_preserves_numerical_settings():
    config = SymmetricTangentGreenResponseProjectionConfig(
        eta_strategy="closed_loop_exact_line_search",
        direction_normalization="response",
        direction_independence_relative_eps=1e-10,
        relative_lambda=0.02,
        line_search_relative_eps=1e-15,
        eta_cap_enabled=False,
    )
    context = reference_context(make_operator([1.0, 2.0], [3.0, 4.0]), 0.2, config, 64)
    assert context.subspace_dimension == 64
    assert context.relative_lambda == config.relative_lambda
    assert context.direction_normalization == config.direction_normalization
    assert (
        context.direction_independence_relative_eps
        == config.direction_independence_relative_eps
    )
    assert context.line_search_relative_eps == config.line_search_relative_eps
    assert (
        replace(
            config, subspace_dimension=64, max_subspace_dimension=64
        ).preconditioner_variant
        == context.preconditioner_variant
    )
