from __future__ import annotations

import pytest
import torch

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
    TangentColumnGramTerms,
)
from greenonet.complex_tangent_preconditioner import (
    TANGENT_PRECONDITIONER_VARIANTS,
    build_tangent_preconditioner_terms,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)


def _operator(axis: str, blocks: list[tuple[list[int], torch.Tensor]]):
    return FrozenAxialResponseOperator(
        axis=axis,
        point_count=sum(len(indices) for indices, _matrix in blocks),
        blocks=tuple(
            AxialResponseBlock(
                valid_indices=torch.tensor(indices, dtype=torch.long),
                matrix=matrix.to(torch.float64),
            )
            for indices, matrix in blocks
        ),
    )


def _dense(operator: FrozenAxialResponseOperator) -> torch.Tensor:
    dense = torch.zeros(
        (operator.point_count, operator.point_count),
        dtype=operator.dtype,
    )
    for block in operator.blocks:
        rows = block.valid_indices[:, None].expand(-1, block.valid_indices.numel())
        cols = block.valid_indices[None, :].expand(block.valid_indices.numel(), -1)
        dense[rows, cols] = block.matrix
    return dense


@pytest.mark.parametrize("overlap", ["singleton", "general"])
def test_tangent_column_gram_terms_match_dense_diagonal(overlap: str) -> None:
    if overlap == "singleton":
        x = _operator(
            "x",
            [
                ([0, 1], torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
                ([2, 3], torch.tensor([[0.5, -1.0], [2.0, 1.5]])),
            ],
        )
        y = _operator(
            "y",
            [
                ([0, 2], torch.tensor([[2.0, 1.0], [-0.5, 3.0]])),
                ([1, 3], torch.tensor([[1.5, -2.0], [0.25, 1.0]])),
            ],
        )
    else:
        x = _operator(
            "x",
            [
                (
                    [0, 1, 2],
                    torch.tensor([[1.0, 2.0, 0.5], [3.0, 4.0, -1.0], [0.2, 0.3, 2.0]]),
                )
            ],
        )
        y = _operator(
            "y",
            [
                (
                    [0, 1, 2],
                    torch.tensor(
                        [[2.0, 1.0, -0.5], [-0.5, 3.0, 1.0], [1.2, -0.7, 2.5]]
                    ),
                )
            ],
        )
    response = FrozenBidirectionalResponseOperator(x=x, y=y)
    mass = torch.tensor(0.125, dtype=torch.float64)

    terms = response.tangent_column_gram_terms(point_mass=mass)
    hx, hy = _dense(x), _dense(y)

    torch.testing.assert_close(terms.a, torch.diagonal(hx.T @ (mass * hx)))
    torch.testing.assert_close(terms.b, torch.diagonal(hy.T @ (mass * hy)))
    torch.testing.assert_close(terms.c, torch.diagonal(hx.T @ (mass * hy)))
    torch.testing.assert_close(x.diagonal_response(), torch.diagonal(hx))


@pytest.mark.parametrize("variant", TANGENT_PRECONDITIONER_VARIANTS)
def test_tangent_preconditioner_variants_match_reference(variant: str) -> None:
    a = torch.tensor([4.0, 9.0, 1.0], dtype=torch.float64)
    b = torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)
    c = torch.tensor([1.0, -3.0, 2.0], dtype=torch.float64)
    terms = build_tangent_preconditioner_terms(
        gram=TangentColumnGramTerms(a=a, b=b, c=c),
        variant=variant,
        relative_lambda=0.1,
        denominator_relative_eps=1.0e-12,
        cross_axis_relative_eps=1.0e-12,
    )

    separable = a + b
    exact = separable + 2.0 * c
    absolute = separable + 2.0 * c.abs()
    q = c.square() / (separable + 1.0e-12 * separable.mean())
    quadratic = separable + 4.0 * q
    expected = {
        "separable": separable,
        "exact_diagonal": exact,
        "absolute_cross_axis": absolute,
        "normalized_quadratic_cross_axis": quadratic,
    }[variant]
    damping = (0.1 + 1.0e-12) * separable.mean()

    torch.testing.assert_close(terms.selected_base, expected)
    torch.testing.assert_close(terms.denominator, expected + damping)
    torch.testing.assert_close(terms.q, q)
    torch.testing.assert_close(terms.rho, c / torch.sqrt(a * b))
    assert terms.exact_roundoff_clamp_count == 0


def test_separable_preconditioner_preserves_existing_arithmetic() -> None:
    a = torch.tensor([0.125, 3.5, 7.0], dtype=torch.float64)
    b = torch.tensor([0.75, 2.25, 0.5], dtype=torch.float64)
    c = torch.zeros_like(a)
    base = a + b
    expected = base + (0.01 + 1.0e-12) * base.mean()

    terms = build_tangent_preconditioner_terms(
        gram=TangentColumnGramTerms(a=a, b=b, c=c),
        variant="separable",
        relative_lambda=0.01,
        denominator_relative_eps=1.0e-12,
        cross_axis_relative_eps=1.0e-12,
    )

    assert torch.equal(terms.separable_base, base)
    assert torch.equal(terms.denominator, expected)


def test_tangent_preconditioner_rejects_cauchy_violation() -> None:
    with pytest.raises(ValueError, match="Cauchy-Schwarz"):
        build_tangent_preconditioner_terms(
            gram=TangentColumnGramTerms(
                a=torch.ones(2, dtype=torch.float64),
                b=torch.ones(2, dtype=torch.float64),
                c=torch.tensor([2.0, 0.0], dtype=torch.float64),
            ),
            variant="exact_diagonal",
            relative_lambda=0.01,
            denominator_relative_eps=1.0e-12,
            cross_axis_relative_eps=1.0e-12,
        )


def test_context_variant_selection_reuses_operator_and_stored_denominator() -> None:
    response = FrozenBidirectionalResponseOperator(
        x=_operator(
            "x",
            [([0, 1], torch.tensor([[1.0, 0.2], [0.3, 0.9]]))],
        ),
        y=_operator(
            "y",
            [([0, 1], torch.tensor([[0.8, -0.1], [0.4, 1.1]]))],
        ),
    )
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=response,
        point_mass=torch.tensor(0.25, dtype=torch.float64),
        config={
            "preconditioner_variant": "separable",
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
            "cross_axis_relative_eps": 1.0e-12,
        },
    )

    selected = context.with_preconditioner_variant("exact_diagonal")

    assert selected.response_operator is context.response_operator
    assert selected.exact_denominator.data_ptr() == context.exact_denominator.data_ptr()
    assert selected.denominator.data_ptr() == context.exact_denominator.data_ptr()
    assert selected.preconditioner_variant == "exact_diagonal"
