from __future__ import annotations

import pytest
import torch

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_projection import (
    apply_complex_balance_projection,
    reconstruct_complex_projection,
    symmetric_tangent_metric_tensors,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from greenonet.config import BalanceProjectionConfig
from test.complex_fixtures import ConstantGreen, write_geometry_npz


def _response_operator(
    x_matrix: torch.Tensor,
    y_matrix: torch.Tensor,
) -> FrozenBidirectionalResponseOperator:
    point_count = int(x_matrix.shape[0])
    indices = torch.arange(point_count, dtype=torch.long)
    return FrozenBidirectionalResponseOperator(
        x=FrozenAxialResponseOperator(
            axis="x",
            point_count=point_count,
            blocks=(AxialResponseBlock(indices, x_matrix),),
        ),
        y=FrozenAxialResponseOperator(
            axis="y",
            point_count=point_count,
            blocks=(AxialResponseBlock(indices, y_matrix),),
        ),
    )


def _context(
    *,
    eta: float,
    relative_lambda: float = 0.01,
) -> SymmetricTangentGreenResponseContext:
    dtype = torch.float64
    operator = _response_operator(
        torch.tensor(
            [[1.0, 0.2, 0.0], [0.1, 0.8, -0.1], [0.0, 0.3, 1.2]],
            dtype=dtype,
        ),
        torch.tensor(
            [[0.9, -0.1, 0.0], [0.2, 1.1, 0.1], [0.0, -0.2, 0.7]],
            dtype=dtype,
        ),
    )
    return SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=operator,
        point_mass=torch.tensor(0.125, dtype=dtype),
        config={
            "eta": eta,
            "relative_lambda": relative_lambda,
            "denominator_relative_eps": 1.0e-12,
        },
    )


def _raw_response(
    geometry,
    raw_physical: torch.Tensor,
) -> torch.Tensor:
    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    return torch.stack(
        (sigma_x * raw_physical[:, 0], sigma_y * raw_physical[:, 1]),
        dim=1,
    )


def test_tangent_projection_matches_fixed_preconditioned_step_and_balance(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(eta=0.05, relative_lambda=0.1)
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
    )
    raw_response = _raw_response(geometry, raw_physical)
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 0.05,
                "relative_lambda": 0.1,
            },
        ),
        symmetric_tangent_context=context,
    )

    tangent = result.symmetric_tangent_diagnostics
    assert tangent is not None
    raw_difference = raw_physical[:, 0] - raw_physical[:, 1]
    symmetric = torch.stack(
        (0.5 * (rhs + raw_difference), 0.5 * (rhs - raw_difference)),
        dim=1,
    )
    symmetric_solution = context.response_operator.forward_pair(symmetric)
    mismatch = symmetric_solution[:, 0] - symmetric_solution[:, 1]
    gradient = context.response_operator.tangent_gradient(
        mismatch,
        point_mass=context.point_mass,
    )
    delta = -0.05 * gradient / context.denominator.unsqueeze(0)

    torch.testing.assert_close(tangent.symmetric_physical, symmetric)
    torch.testing.assert_close(tangent.gradient, gradient)
    torch.testing.assert_close(tangent.delta, delta)
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(result.difference_update, 2.0 * delta)
    assert result.column_diagonal_context is None
    assert context.statistics()["global_matrix_materialized"] is False
    assert context.statistics()["full_gram_solve"] is False


def test_zero_eta_is_bitwise_physical_symmetric_projection(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 11.0, 12.0]], dtype=torch.float64)
    symmetric = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="physical_symmetric"),
    )
    tangent = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={"eta": 0.0},
        ),
        symmetric_tangent_context=_context(eta=0.0),
    )

    assert torch.equal(tangent.projected_physical, symmetric.projected_physical)
    assert torch.equal(tangent.projected_response, symmetric.projected_response)
    assert torch.equal(tangent.difference_update, symmetric.difference_update)


def test_tangent_projection_preserves_autograd_and_reuses_cached_solution(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(eta=0.01)
    raw_response = torch.tensor(
        [[[1.0, 0.5, -0.25], [0.25, -0.75, 0.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.tensor([[0.5, -0.25, 0.75]], dtype=torch.float64)
    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="symmetric_tangent_green_response"),
        symmetric_tangent_context=context,
    )
    reconstruction = reconstruct_complex_projection(
        projection=result,
        green_model=ConstantGreen(99.0),
        geometry=geometry,
        x_green_branch=torch.empty(0),
        y_green_branch=torch.empty(0),
    )
    diagnostics = result.symmetric_tangent_diagnostics
    assert diagnostics is not None
    torch.testing.assert_close(
        reconstruction.u_phi_valid,
        diagnostics.projected_solution[:, 0],
    )
    torch.testing.assert_close(
        reconstruction.u_psi_valid,
        diagnostics.projected_solution[:, 1],
    )

    loss = reconstruction.u_mean_valid.square().mean()
    loss.backward()
    assert raw_response.grad is not None
    assert torch.all(torch.isfinite(raw_response.grad))
    assert not context.denominator.requires_grad
    metrics = symmetric_tangent_metric_tensors(result)
    assert set(metrics) == {
        "tangent_response_mismatch_pre",
        "tangent_response_mismatch_post",
        "tangent_response_mismatch_ratio",
        "tangent_gradient_rms",
        "tangent_delta_rms",
        "tangent_delta_max_abs",
        "tangent_correction_rel_symmetric_pair",
    }
    assert all(torch.isfinite(value) for value in metrics.values())


def test_tangent_projection_requires_context(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.ones((1, 2, geometry.num_points), dtype=torch.float64)
    rhs = torch.ones((1, geometry.num_points), dtype=torch.float64)

    with pytest.raises(ValueError, match="requires.*context"):
        apply_complex_balance_projection(
            raw_response,
            rhs,
            geometry,
            BalanceProjectionConfig(mode="symmetric_tangent_green_response"),
        )
