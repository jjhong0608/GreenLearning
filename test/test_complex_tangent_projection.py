from __future__ import annotations

import math

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
    SymmetricTangentEtaCapSchedule,
    SymmetricTangentGreenResponseContext,
)
from greenonet.config import BalanceProjectionConfig
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule
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
    eta_strategy: str = "fixed",
    line_search_relative_eps: float = 1.0e-12,
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
            "eta_strategy": eta_strategy,
            "line_search_relative_eps": line_search_relative_eps,
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


def test_closed_loop_eta_matches_analytic_line_search_and_preserves_balance(
    tmp_path,
    monkeypatch,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=100.0,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
    )
    raw_response = _raw_response(geometry, raw_physical)
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)

    def fail_solve(*args, **kwargs):
        raise AssertionError("Closed-loop tangent projection must not solve a matrix.")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)
    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 100.0,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 1.0e-15,
                "relative_lambda": 0.1,
            },
        ),
        symmetric_tangent_context=context,
    )

    tangent = result.symmetric_tangent_diagnostics
    assert tangent is not None
    assert tangent.eta_star is not None
    assert tangent.line_search_numerator is not None
    assert tangent.line_search_denominator is not None
    assert tangent.response_direction is not None
    direction = tangent.gradient / context.denominator.unsqueeze(0)
    directional_response = context.response_operator.forward_pair(
        torch.stack((direction, direction), dim=1)
    )
    response_direction = directional_response[:, 0] + directional_response[:, 1]
    mismatch_energy = context.point_mass * tangent.mismatch_pre.square().sum(dim=1)
    response_energy = context.point_mass * response_direction.square().sum(dim=1)
    numerical_eps = (
        context.line_search_relative_eps
        * torch.maximum(mismatch_energy, response_energy)
        + torch.finfo(torch.float64).tiny
    )
    numerator = (tangent.gradient * direction).sum(dim=1).clamp_min(0.0)
    expected_eta = numerator / (response_energy + numerical_eps)

    torch.testing.assert_close(tangent.eta_star, expected_eta)
    torch.testing.assert_close(tangent.eta_applied, expected_eta)
    torch.testing.assert_close(tangent.response_direction, response_direction)
    torch.testing.assert_close(
        tangent.mismatch_post,
        tangent.mismatch_pre - expected_eta.unsqueeze(1) * response_direction,
    )
    assert torch.linalg.vector_norm(tangent.mismatch_post) < torch.linalg.vector_norm(
        tangent.mismatch_pre
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
        atol=0.0,
        rtol=0.0,
    )


def test_closed_loop_eta_is_scale_invariant_and_batch_independent(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=0.05,
        eta_strategy="closed_loop_exact_line_search",
    )
    raw_physical = torch.tensor(
        [
            [[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]],
            [[-1.0, 0.5, 0.3], [0.2, -0.7, 0.9]],
        ],
        dtype=torch.float64,
    )
    rhs = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.2, 0.4, 1.1]],
        dtype=torch.float64,
    )

    batched = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 0.05,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )
    single = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical[:1]),
        rhs[:1],
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 0.05,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )
    scaled = apply_complex_balance_projection(
        _raw_response(geometry, 7.0 * raw_physical[:1]),
        7.0 * rhs[:1],
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 0.05,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )

    batched_tangent = batched.symmetric_tangent_diagnostics
    single_tangent = single.symmetric_tangent_diagnostics
    scaled_tangent = scaled.symmetric_tangent_diagnostics
    assert batched_tangent is not None
    assert single_tangent is not None
    assert scaled_tangent is not None
    assert batched_tangent.eta_star is not None
    assert single_tangent.eta_star is not None
    assert scaled_tangent.eta_star is not None
    torch.testing.assert_close(batched_tangent.eta_star[:1], single_tangent.eta_star)
    torch.testing.assert_close(
        batched_tangent.eta_applied[:1], single_tangent.eta_applied
    )
    torch.testing.assert_close(
        batched.projected_physical[:1], single.projected_physical
    )
    torch.testing.assert_close(single_tangent.eta_star, scaled_tangent.eta_star)


def test_closed_loop_eta_zero_mismatch_is_finite_and_differentiable(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=10.0,
        eta_strategy="closed_loop_exact_line_search",
    )
    raw_response = torch.zeros(
        (1, 2, geometry.num_points),
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.zeros((1, geometry.num_points), dtype=torch.float64)
    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 10.0,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )

    tangent = result.symmetric_tangent_diagnostics
    assert tangent is not None
    assert tangent.eta_star is not None
    assert torch.equal(tangent.eta_star, torch.zeros_like(tangent.eta_star))
    assert torch.all(torch.isfinite(tangent.eta_star))
    loss = result.projected_physical.square().sum()
    loss.backward()
    assert raw_response.grad is not None
    assert torch.all(torch.isfinite(raw_response.grad))


def test_tangent_eta_cap_schedule_reuses_lr_warmup_and_holds_final_eta():
    lr_schedule = CouplingLearningRateSchedule.from_values(
        enabled=True,
        base_learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        warmup_epochs=4,
        total_epochs=8,
        field_prefix="coupling_training",
    )
    eta_schedule = SymmetricTangentEtaCapSchedule.from_learning_rate_schedule(
        config={
            "eta": 0.02,
            "eta_strategy": "closed_loop_exact_line_search",
        },
        learning_rate_schedule=lr_schedule,
    )

    expected = [
        0.02 * 0.5 * (1.0 - math.cos(math.pi * step / 4.0)) for step in (1, 2, 3, 4)
    ]
    assert [eta_schedule.cap_for_epoch_index(index) for index in range(8)] == (
        pytest.approx(expected + [0.02] * 4)
    )
    assert eta_schedule.kind == "closed_loop_half_cosine_warmup_hold"

    fixed_lr = CouplingLearningRateSchedule.from_values(
        enabled=False,
        base_learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        warmup_epochs=4,
        total_epochs=8,
        field_prefix="coupling_training",
    )
    immediate = SymmetricTangentEtaCapSchedule.from_learning_rate_schedule(
        config={
            "eta": 0.02,
            "eta_strategy": "closed_loop_exact_line_search",
        },
        learning_rate_schedule=fixed_lr,
    )
    assert [immediate.cap_for_epoch_index(index) for index in range(8)] == [0.02] * 8
    assert immediate.kind == "closed_loop_final_cap"
