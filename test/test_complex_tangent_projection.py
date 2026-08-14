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
    post_line_search_stationarity_from_projection,
    reconstruct_complex_projection,
    symmetric_tangent_metric_tensors,
    tangent_auxiliary_losses_from_projection,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentEtaCapSchedule,
    SymmetricTangentGreenResponseContext,
    matrix_free_krylov_k2_step,
    matrix_free_krylov_subspace_step,
    normalized_post_line_search_stationarity_loss,
    tangent_response_trust_loss,
    tangent_source_response_normalization,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexPostLineSearchStationarityConfig,
    ComplexResponseTrustConfig,
)
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
    subspace_dimension: int = 1,
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
            "subspace_dimension": subspace_dimension,
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


def test_omitted_subspace_dimension_is_bitwise_identical_to_explicit_k1(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
    )
    raw_response = _raw_response(geometry, raw_physical)
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    shared = {
        "eta": 0.05,
        "eta_strategy": "closed_loop_exact_line_search",
        "line_search_relative_eps": 1.0e-12,
        "relative_lambda": 0.1,
    }
    omitted = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response=shared,
        ),
        symmetric_tangent_context=_context(
            eta=0.05,
            eta_strategy="closed_loop_exact_line_search",
            relative_lambda=0.1,
        ),
    )
    explicit = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={**shared, "subspace_dimension": 1},
        ),
        symmetric_tangent_context=_context(
            eta=0.05,
            subspace_dimension=1,
            eta_strategy="closed_loop_exact_line_search",
            relative_lambda=0.1,
        ),
    )

    assert torch.equal(omitted.projected_physical, explicit.projected_physical)
    assert torch.equal(omitted.projected_response, explicit.projected_response)
    omitted_tangent = omitted.symmetric_tangent_diagnostics
    explicit_tangent = explicit.symmetric_tangent_diagnostics
    assert omitted_tangent is not None
    assert explicit_tangent is not None
    assert omitted_tangent.subspace_dimension == 1
    assert explicit_tangent.subspace_dimension == 1
    assert torch.equal(omitted_tangent.delta, explicit_tangent.delta)
    assert torch.equal(omitted_tangent.mismatch_post, explicit_tangent.mismatch_post)
    assert torch.equal(omitted_tangent.eta_star, explicit_tangent.eta_star)
    assert torch.equal(omitted_tangent.eta_applied, explicit_tangent.eta_applied)


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


def test_k2_projection_is_balanced_matrix_free_and_improves_k1_cost(
    tmp_path,
    monkeypatch,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=0.015,
        subspace_dimension=2,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)

    def fail_solve(*args, **kwargs):
        raise AssertionError("K=2 tangent projection must not solve a matrix.")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)
    result = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "subspace_dimension": 2,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 1.0e-15,
                "relative_lambda": 0.1,
            },
        ),
        symmetric_tangent_context=context,
    )
    tangent = result.symmetric_tangent_diagnostics
    assert tangent is not None
    assert tangent.subspace_dimension == 2
    assert tangent.eta_applied is None
    assert tangent.eta_cap is None
    assert tangent.coefficient_0 is not None
    assert tangent.coefficient_1 is not None
    assert tangent.second_direction_active is not None
    assert tangent.mismatch_k1 is not None
    assert tangent.cost_k1 is not None
    assert tangent.cost_k2 is not None
    assert tangent.residual_gradient_post is not None
    torch.testing.assert_close(
        context.point_mass * tangent.mismatch_post.square().sum(dim=1),
        tangent.cost_k2,
    )
    assert torch.all(tangent.cost_k2 <= tangent.cost_k1 * (1.0 + 1.0e-10))
    torch.testing.assert_close(
        result.projected_physical.sum(dim=1),
        rhs,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    direct_solution = context.response_operator.forward_pair(result.projected_physical)
    torch.testing.assert_close(tangent.projected_solution, direct_solution)
    metrics = symmetric_tangent_metric_tensors(result)
    assert metrics["tangent_subspace_dimension"].item() == 2
    assert "tangent_eta_cap" not in metrics
    tangent.projected_solution.square().mean().backward()
    assert raw_physical.grad is not None
    assert torch.all(torch.isfinite(raw_physical.grad))


def test_generic_k2_subspace_preserves_existing_k2_tensors_bitwise():
    context = _context(
        eta=0.015,
        subspace_dimension=2,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    mismatch = torch.tensor(
        [[1.2, -0.4, 0.8], [-0.3, 1.5, 0.2]],
        dtype=torch.float64,
    )
    gradient = context.tangent_gradient(mismatch)
    k2 = matrix_free_krylov_k2_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        relative_eps=1.0e-15,
        monotonicity_relative_tol=1.0e-10,
    )
    generic = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=2,
        relative_eps=1.0e-15,
        monotonicity_relative_tol=1.0e-10,
    )

    assert torch.equal(generic.directions[0], k2.direction_0)
    assert torch.equal(generic.directions[1], k2.direction_1)
    assert torch.equal(generic.coefficients[0], k2.coefficient_0)
    assert torch.equal(generic.coefficients[1], k2.coefficient_1)
    assert torch.equal(generic.deltas[0], k2.delta_k1)
    assert torch.equal(generic.deltas[1], k2.delta_k2)
    assert torch.equal(generic.mismatches[0], k2.mismatch_k1)
    assert torch.equal(generic.mismatches[1], k2.mismatch_k2)
    assert torch.equal(generic.costs[0], k2.cost_k1)
    assert torch.equal(generic.costs[1], k2.cost_k2)
    assert torch.equal(
        generic.residual_gradient_post,
        k2.residual_gradient_post,
    )


@pytest.mark.parametrize("subspace_dimension", [3, 4])
def test_k3_k4_projection_is_nested_balanced_and_matrix_free(
    tmp_path,
    monkeypatch,
    subspace_dimension,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=0.015,
        subspace_dimension=subspace_dimension,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)

    def fail_solve(*args, **kwargs):
        raise AssertionError("K=3/K=4 tangent projection must not solve a matrix.")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)
    projection = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "subspace_dimension": subspace_dimension,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 1.0e-15,
                "relative_lambda": 0.1,
            },
        ),
        symmetric_tangent_context=context,
    )
    tangent = projection.symmetric_tangent_diagnostics
    assert tangent is not None
    subspace = tangent.subspace_result
    assert subspace is not None
    assert subspace.subspace_dimension == subspace_dimension
    assert tangent.eta_applied is None
    assert tangent.eta_cap is None
    assert torch.all(subspace.costs[1:] <= subspace.costs[:-1] * (1.0 + 1.0e-10))
    assert torch.all(subspace.response_orthogonality_max[-1] <= 1.0e-10)
    torch.testing.assert_close(
        projection.projected_physical.sum(dim=1),
        rhs,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    direct_solution = context.response_operator.forward_pair(
        projection.projected_physical
    )
    torch.testing.assert_close(tangent.projected_solution, direct_solution)
    metrics = symmetric_tangent_metric_tensors(projection)
    assert metrics["tangent_subspace_dimension"].item() == subspace_dimension
    assert f"tangent_coefficient_{subspace_dimension - 1}_mean" in metrics
    assert f"tangent_response_cost_k{subspace_dimension}_mean" in metrics
    assert "tangent_eta_cap" not in metrics
    tangent.projected_solution.square().mean().backward()
    assert raw_physical.grad is not None
    assert torch.all(torch.isfinite(raw_physical.grad))


@pytest.mark.parametrize("subspace_dimension", [2, 3, 4])
def test_k2_plus_rejects_eta_cap_and_degenerate_directions_fall_back(
    subspace_dimension,
):
    dtype = torch.float64
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=_response_operator(
            torch.ones((1, 1), dtype=dtype),
            torch.zeros((1, 1), dtype=dtype),
        ),
        point_mass=1.0,
        config={
            "subspace_dimension": subspace_dimension,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-12,
            "relative_lambda": 0.0,
            "denominator_relative_eps": 1.0e-12,
        },
    )
    mismatch = torch.tensor([[2.0]], dtype=dtype)
    gradient = context.tangent_gradient(mismatch)
    with pytest.raises(ValueError, match="eta_cap is not applicable"):
        context.tangent_step(mismatch=mismatch, gradient=gradient, eta_cap=0.01)

    step = context.tangent_step(mismatch=mismatch, gradient=gradient)
    assert step.second_direction_active is not None
    assert step.coefficient_1 is not None
    assert step.mismatch_k1 is not None
    assert step.mismatch_k2 is not None
    assert not bool(step.second_direction_active.item())
    assert step.coefficient_1.item() == 0.0
    torch.testing.assert_close(step.mismatch_k2, step.mismatch_k1)
    subspace = step.subspace_result
    assert subspace is not None
    assert not torch.any(subspace.direction_active[1:])
    torch.testing.assert_close(
        subspace.coefficients[1:],
        torch.zeros_like(subspace.coefficients[1:]),
    )
    torch.testing.assert_close(
        subspace.deltas[1:],
        subspace.deltas[0].expand_as(subspace.deltas[1:]),
    )


@pytest.mark.parametrize("subspace_dimension", [2, 3, 4])
def test_k2_plus_stationarity_and_response_trust_use_final_subspace_residual(
    tmp_path,
    subspace_dimension,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=0.015,
        subspace_dimension=subspace_dimension,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    projection = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "subspace_dimension": subspace_dimension,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 1.0e-15,
                "relative_lambda": 0.1,
            },
        ),
        symmetric_tangent_context=context,
    )
    auxiliary = tangent_auxiliary_losses_from_projection(
        projection=projection,
        context=context,
        rhs_phys=rhs,
        stationarity_config={"enabled": True, "weight": 1.0e-4},
        response_trust_config={
            "enabled": True,
            "weight": 1.0e-3,
            "trust_weight": 0.01,
        },
    )
    tangent = projection.symmetric_tangent_diagnostics
    assert tangent is not None
    assert tangent.residual_gradient_post is not None
    assert auxiliary.stationarity is not None
    assert auxiliary.response_trust is not None
    torch.testing.assert_close(
        auxiliary.stationarity.stationarity_residual,
        tangent.residual_gradient_post,
    )
    torch.testing.assert_close(
        auxiliary.response_trust.correction_response,
        tangent.mismatch_post - tangent.mismatch_pre,
    )
    assert (
        auxiliary.stationarity.source_response.data_ptr()
        == auxiliary.response_trust.source_response.data_ptr()
    )
    (auxiliary.stationarity.loss + auxiliary.response_trust.loss).backward()
    assert raw_physical.grad is not None
    assert torch.all(torch.isfinite(raw_physical.grad))


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
        warmup_steps=None,
        total_epochs=8,
        steps_per_epoch=1,
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
    assert [eta_schedule.cap_for_step_index(index) for index in range(8)] == (
        pytest.approx(expected + [0.02] * 4)
    )
    assert eta_schedule.kind == "closed_loop_half_cosine_warmup_hold"

    fixed_lr = CouplingLearningRateSchedule.from_values(
        enabled=False,
        base_learning_rate=1.0e-3,
        min_learning_rate=1.0e-5,
        warmup_epochs=4,
        warmup_steps=None,
        total_epochs=8,
        steps_per_epoch=1,
        field_prefix="coupling_training",
    )
    immediate = SymmetricTangentEtaCapSchedule.from_learning_rate_schedule(
        config={
            "eta": 0.02,
            "eta_strategy": "closed_loop_exact_line_search",
        },
        learning_rate_schedule=fixed_lr,
    )
    assert [immediate.cap_for_step_index(index) for index in range(8)] == [0.02] * 8
    assert immediate.kind == "closed_loop_final_cap"

    with pytest.raises(ValueError, match="subspace_dimension=1"):
        SymmetricTangentEtaCapSchedule.from_learning_rate_schedule(
            config={
                "subspace_dimension": 2,
                "eta_strategy": "closed_loop_exact_line_search",
            },
            learning_rate_schedule=fixed_lr,
        )


def test_normalized_stationarity_is_zero_when_tangent_line_reaches_minimizer(
    monkeypatch,
):
    dtype = torch.float64
    identity = torch.eye(3, dtype=dtype)
    zeros = torch.zeros((3, 3), dtype=dtype)
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=_response_operator(identity, zeros),
        point_mass=1.0,
        config={
            "eta": 10.0,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-15,
            "relative_lambda": 0.0,
            "denominator_relative_eps": 1.0e-12,
        },
    )
    mismatch = torch.tensor([[1.0, -2.0, 0.5]], dtype=dtype)
    gradient = context.tangent_gradient(mismatch)
    step = context.tangent_step(mismatch=mismatch, gradient=gradient)
    assert step.eta_star is not None
    assert step.response_direction is not None

    def fail_solve(*args, **kwargs):
        raise AssertionError("Stationarity loss must not solve a matrix.")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=torch.ones_like(mismatch),
    )
    result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=gradient,
        response_direction=step.response_direction,
        eta_star=step.eta_star,
        source_normalization=source_normalization,
        config=ComplexPostLineSearchStationarityConfig(enabled=True),
    )

    assert result.loss.item() < 1.0e-20
    assert torch.all(torch.isfinite(result.stationarity_residual))
    torch.testing.assert_close(
        result.hessian_direction,
        context.tangent_gradient(step.response_direction),
    )


def test_normalized_stationarity_detects_tangent_line_missing_full_minimizer():
    context = _context(
        eta=10.0,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    mismatch = torch.tensor([[1.0, -0.25, 0.75]], dtype=torch.float64)
    gradient = context.tangent_gradient(mismatch)
    step = context.tangent_step(mismatch=mismatch, gradient=gradient)
    assert step.eta_star is not None
    assert step.response_direction is not None

    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=torch.ones_like(mismatch),
    )
    result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=gradient,
        response_direction=step.response_direction,
        eta_star=step.eta_star,
        source_normalization=source_normalization,
        config={"enabled": True, "eps": 1.0e-12},
    )

    assert result.loss.item() > 0.0
    assert result.loss_per_sample.shape == (1,)
    assert result.stationarity_residual.shape == gradient.shape


def test_normalized_stationarity_uses_uncapped_eta_and_preserves_legacy_ratio():
    context = _context(
        eta=1.0,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
        line_search_relative_eps=1.0e-15,
    )
    mismatch = torch.tensor([[1.0, -0.25, 0.75]], dtype=torch.float64)
    gradient = context.tangent_gradient(mismatch)
    capped = context.tangent_step(
        mismatch=mismatch,
        gradient=gradient,
        eta_cap=1.0e-3,
    )
    uncapped = context.tangent_step(
        mismatch=mismatch,
        gradient=gradient,
        eta_cap=1.0,
    )
    assert capped.eta_star is not None
    assert capped.response_direction is not None
    assert uncapped.eta_star is not None
    assert uncapped.response_direction is not None
    assert not torch.equal(capped.eta_applied, uncapped.eta_applied)

    config = ComplexPostLineSearchStationarityConfig(enabled=True, eps=1.0e-14)
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=torch.ones_like(mismatch),
    )
    capped_result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=gradient,
        response_direction=capped.response_direction,
        eta_star=capped.eta_star,
        source_normalization=source_normalization,
        config=config,
    )
    uncapped_result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=gradient,
        response_direction=uncapped.response_direction,
        eta_star=uncapped.eta_star,
        source_normalization=source_normalization,
        config=config,
    )
    scaled_result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=7.0 * gradient,
        response_direction=7.0 * capped.response_direction,
        eta_star=capped.eta_star,
        source_normalization=source_normalization,
        config=config,
    )
    scaled_source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=7.0 * torch.ones_like(mismatch),
    )
    jointly_scaled_result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=7.0 * gradient,
        response_direction=7.0 * capped.response_direction,
        eta_star=capped.eta_star,
        source_normalization=scaled_source_normalization,
        config=config,
    )

    torch.testing.assert_close(capped_result.loss, uncapped_result.loss)
    torch.testing.assert_close(scaled_result.loss, 49.0 * capped_result.loss)
    torch.testing.assert_close(
        scaled_result.relative_ratio,
        capped_result.relative_ratio,
    )
    torch.testing.assert_close(
        scaled_result.initial_source_ratio,
        49.0 * capped_result.initial_source_ratio,
    )
    torch.testing.assert_close(jointly_scaled_result.loss, capped_result.loss)
    torch.testing.assert_close(
        jointly_scaled_result.relative_ratio,
        capped_result.relative_ratio,
    )


def test_normalized_stationarity_zero_gradient_is_finite_and_differentiable():
    context = _context(
        eta=1.0,
        eta_strategy="closed_loop_exact_line_search",
    )
    gradient = torch.zeros(
        (2, context.num_points),
        dtype=torch.float64,
        requires_grad=True,
    )
    response_direction = torch.zeros_like(gradient)
    eta_star = torch.zeros(2, dtype=torch.float64)
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=torch.zeros_like(gradient),
    )

    result = normalized_post_line_search_stationarity_loss(
        context=context,
        gradient=gradient,
        response_direction=response_direction,
        eta_star=eta_star,
        source_normalization=source_normalization,
        config={"enabled": True, "eps": 1.0e-12},
    )

    assert torch.equal(result.loss_per_sample, torch.zeros(2, dtype=torch.float64))
    assert torch.all(torch.isfinite(result.loss_per_sample))
    result.loss.backward()
    assert gradient.grad is not None
    assert torch.all(torch.isfinite(gradient.grad))


def test_response_trust_matches_source_normalized_capped_response_formula():
    context = _context(
        eta=1.0,
        relative_lambda=0.1,
        eta_strategy="closed_loop_exact_line_search",
    )
    rhs = torch.tensor(
        [[1.0, -0.5, 0.25], [0.25, 0.75, -1.0]],
        dtype=torch.float64,
    )
    mismatch_pre = torch.tensor(
        [[0.8, -0.3, 0.4], [-0.5, 0.2, 0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    mismatch_post = torch.tensor(
        [[0.2, -0.1, 0.3], [-0.1, 0.15, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    trust_weight = 0.025
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=rhs,
    )

    result = tangent_response_trust_loss(
        context=context,
        mismatch_pre=mismatch_pre,
        mismatch_post=mismatch_post,
        source_normalization=source_normalization,
        config=ComplexResponseTrustConfig(
            enabled=True,
            trust_weight=trust_weight,
            eps=1.0e-14,
        ),
    )

    half_rhs = 0.5 * rhs
    expected_source_response = context.response_operator.forward_pair(
        torch.stack((half_rhs, half_rhs), dim=1)
    )
    expected_source_energy = context.point_mass * expected_source_response.square().sum(
        dim=(1, 2)
    )
    expected_post_energy = context.point_mass * mismatch_post.square().sum(dim=1)
    expected_correction = mismatch_post - mismatch_pre
    expected_correction_energy = context.point_mass * expected_correction.square().sum(
        dim=1
    )
    denominator = expected_source_energy + 1.0e-14
    expected_post_ratio = expected_post_energy / denominator
    expected_correction_ratio = expected_correction_energy / denominator
    expected_loss = expected_post_ratio + trust_weight * expected_correction_ratio

    torch.testing.assert_close(result.source_response, expected_source_response)
    torch.testing.assert_close(
        result.source_response_energy_per_sample, expected_source_energy
    )
    torch.testing.assert_close(result.correction_response, expected_correction)
    torch.testing.assert_close(
        result.post_mismatch_ratio_per_sample, expected_post_ratio
    )
    torch.testing.assert_close(
        result.correction_ratio_per_sample, expected_correction_ratio
    )
    torch.testing.assert_close(result.loss_per_sample, expected_loss)
    torch.testing.assert_close(result.loss, expected_loss.mean())
    result.loss.backward()
    assert mismatch_pre.grad is not None
    assert mismatch_post.grad is not None
    assert torch.all(torch.isfinite(mismatch_pre.grad))
    assert torch.all(torch.isfinite(mismatch_post.grad))


def test_response_trust_is_absolute_in_response_scale_and_zero_safe():
    context = _context(
        eta=1.0,
        eta_strategy="closed_loop_exact_line_search",
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    mismatch_pre = torch.tensor([[0.8, -0.3, 0.4]], dtype=torch.float64)
    mismatch_post = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float64)
    config = ComplexResponseTrustConfig(enabled=True, trust_weight=0.01)
    source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=rhs,
    )

    baseline = tangent_response_trust_loss(
        context=context,
        mismatch_pre=mismatch_pre,
        mismatch_post=mismatch_post,
        source_normalization=source_normalization,
        config=config,
    )
    scaled = tangent_response_trust_loss(
        context=context,
        mismatch_pre=7.0 * mismatch_pre,
        mismatch_post=7.0 * mismatch_post,
        source_normalization=source_normalization,
        config=config,
    )
    zero_source_normalization = tangent_source_response_normalization(
        context=context,
        rhs_phys=torch.zeros_like(rhs),
    )
    zero = tangent_response_trust_loss(
        context=context,
        mismatch_pre=torch.zeros_like(mismatch_pre),
        mismatch_post=torch.zeros_like(mismatch_post),
        source_normalization=zero_source_normalization,
        config=config,
    )

    torch.testing.assert_close(scaled.loss, 49.0 * baseline.loss)
    assert torch.equal(zero.loss_per_sample, torch.zeros_like(zero.loss_per_sample))
    assert torch.all(torch.isfinite(zero.loss_per_sample))


def test_stationarity_bridge_adds_one_adjoint_only_when_enabled(tmp_path, monkeypatch):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=1.0,
        eta_strategy="closed_loop_exact_line_search",
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    calls = 0
    original = FrozenBidirectionalResponseOperator.tangent_gradient

    def counted(self, mismatch, *, point_mass):
        nonlocal calls
        calls += 1
        return original(self, mismatch, point_mass=point_mass)

    monkeypatch.setattr(
        FrozenBidirectionalResponseOperator,
        "tangent_gradient",
        counted,
    )
    projection = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 1.0,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )
    assert calls == 1

    disabled = post_line_search_stationarity_from_projection(
        projection=projection,
        context=context,
        config=ComplexPostLineSearchStationarityConfig(enabled=False),
    )
    assert disabled is None
    assert calls == 1

    enabled = post_line_search_stationarity_from_projection(
        projection=projection,
        context=context,
        config=ComplexPostLineSearchStationarityConfig(enabled=True),
    )
    assert enabled is not None
    assert calls == 2


@pytest.mark.parametrize(
    ("stationarity_enabled", "response_trust_enabled", "expected_calls"),
    (
        (True, False, 1),
        (False, True, 1),
        (True, True, 1),
        (False, False, 0),
    ),
)
def test_tangent_auxiliary_losses_share_source_forward_and_stationarity_adjoint(
    tmp_path,
    monkeypatch,
    stationarity_enabled,
    response_trust_enabled,
    expected_calls,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    context = _context(
        eta=1.0,
        eta_strategy="closed_loop_exact_line_search",
    )
    raw_physical = torch.tensor(
        [[[0.2, -0.4, 0.8], [0.7, 0.1, -0.3]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float64)
    projection = apply_complex_balance_projection(
        _raw_response(geometry, raw_physical),
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 1.0,
                "eta_strategy": "closed_loop_exact_line_search",
            },
        ),
        symmetric_tangent_context=context,
    )

    forward_calls = 0
    adjoint_calls = 0
    original_forward = FrozenBidirectionalResponseOperator.forward_pair
    original_adjoint = FrozenBidirectionalResponseOperator.tangent_gradient

    def counted_forward(self, source_pair):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(self, source_pair)

    def counted_adjoint(self, mismatch, *, point_mass):
        nonlocal adjoint_calls
        adjoint_calls += 1
        return original_adjoint(self, mismatch, point_mass=point_mass)

    monkeypatch.setattr(
        FrozenBidirectionalResponseOperator,
        "forward_pair",
        counted_forward,
    )
    monkeypatch.setattr(
        FrozenBidirectionalResponseOperator,
        "tangent_gradient",
        counted_adjoint,
    )

    result = tangent_auxiliary_losses_from_projection(
        projection=projection,
        context=context,
        rhs_phys=rhs,
        stationarity_config={
            "enabled": stationarity_enabled,
            "weight": 1.0e-4,
        },
        response_trust_config={
            "enabled": response_trust_enabled,
            "weight": 1.0e-3,
        },
    )

    auxiliary_enabled = stationarity_enabled or response_trust_enabled
    assert (result.stationarity is not None) is auxiliary_enabled
    assert (result.response_trust is not None) is response_trust_enabled
    assert (result.source_normalization is not None) is auxiliary_enabled
    assert forward_calls == expected_calls
    assert adjoint_calls == expected_calls
    if result.stationarity is not None and result.response_trust is not None:
        assert (
            result.stationarity.source_response.data_ptr()
            == result.response_trust.source_response.data_ptr()
        )
