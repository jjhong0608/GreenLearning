from __future__ import annotations

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
)
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.config import BalanceProjectionConfig
from test.complex_fixtures import write_geometry_npz


def test_physical_symmetric_projection_enforces_balance_and_pull_back(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="physical_symmetric"),
    )

    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    expected_raw_physical = torch.stack(
        (
            raw_response[:, 0] / sigma_x,
            raw_response[:, 1] / sigma_y,
        ),
        dim=1,
    )
    difference = expected_raw_physical[:, 0] - expected_raw_physical[:, 1]
    expected_physical = torch.stack(
        (0.5 * (rhs + difference), 0.5 * (rhs - difference)),
        dim=1,
    )
    expected_response = torch.stack(
        (sigma_x * expected_physical[:, 0], sigma_y * expected_physical[:, 1]),
        dim=1,
    )

    torch.testing.assert_close(result.raw_response, raw_response)
    torch.testing.assert_close(result.raw_physical, expected_raw_physical)
    torch.testing.assert_close(result.projected_physical, expected_physical)
    torch.testing.assert_close(result.projected_response, expected_response)
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
    )
    torch.testing.assert_close(
        result.projected_response[:, 0] / sigma_x
        + result.projected_response[:, 1] / sigma_y,
        rhs,
    )
    torch.testing.assert_close(
        result.response_constraint_residual,
        torch.zeros_like(rhs),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        result.physical_balance_residual,
        torch.zeros_like(rhs),
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(result.raw_difference, difference)
    torch.testing.assert_close(result.projected_difference, difference)


def test_physical_symmetric_projection_preserves_raw_physical_difference(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]], dtype=torch.float64
    )
    rhs = torch.full((1, geometry.num_points), 10.0, dtype=torch.float64)

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="physical_symmetric"),
    )

    sigma_x = geometry.x_lengths_for_valid_points().square()
    sigma_y = geometry.y_lengths_for_valid_points().square()
    raw_physical_difference = (
        raw_response[:, 0] / sigma_x - raw_response[:, 1] / sigma_y
    )
    torch.testing.assert_close(result.projected_difference, raw_physical_difference)


def test_physical_projection_rejects_invalid_shapes_and_retired_modes(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    rhs = torch.ones((1, geometry.num_points), dtype=torch.float64)
    config = BalanceProjectionConfig(mode="physical_symmetric")

    with pytest.raises(ValueError, match="raw_response"):
        apply_complex_balance_projection(
            torch.ones((1, geometry.num_points), dtype=torch.float64),
            rhs,
            geometry,
            config,
        )
    with pytest.raises(ValueError, match="rhs_phys"):
        apply_complex_balance_projection(
            torch.ones((1, 2, geometry.num_points), dtype=torch.float64),
            torch.ones((1, geometry.num_points + 1), dtype=torch.float64),
            geometry,
            config,
        )
    with pytest.raises(ValueError, match="removed.*physical_symmetric"):
        BalanceProjectionConfig.from_raw("response_preconditioned")

    with pytest.raises(ValueError, match="version 6.*physical_symmetric"):
        apply_complex_balance_projection(
            torch.ones((1, 2, geometry.num_points), dtype=torch.float64),
            rhs,
            geometry,
            BalanceProjectionConfig(mode="response_space"),
        )


def test_column_diagonal_projection_uses_opposite_response_cost_and_exact_balance(
    tmp_path,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    raw_physical = torch.tensor(
        [[[1.0, 2.0, 3.0], [0.5, 0.25, 1.0]]],
        dtype=torch.float64,
    )
    raw_response = torch.stack(
        (sigma_x * raw_physical[:, 0], sigma_y * raw_physical[:, 1]),
        dim=1,
    )
    rhs = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float64)
    context = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=torch.tensor([1.0, 9.0, 0.0], dtype=torch.float64),
        gamma_y_squared=torch.tensor([9.0, 1.0, 0.0], dtype=torch.float64),
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
    )

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="column_diagonal_green_response"),
        column_diagonal_context=context,
    )

    residual = rhs - raw_physical[:, 0] - raw_physical[:, 1]
    expected_weight_phi = context.correction_weight_phi.unsqueeze(0)
    torch.testing.assert_close(
        context.correction_weight_phi,
        context.regularized_gamma_y_squared
        / (context.regularized_gamma_x_squared + context.regularized_gamma_y_squared),
    )
    assert context.gain_exponent == 1.0
    expected_difference_update = (2.0 * expected_weight_phi - 1.0) * residual
    torch.testing.assert_close(result.correction_weight_phi, expected_weight_phi)
    torch.testing.assert_close(result.difference_update, expected_difference_update)
    torch.testing.assert_close(
        result.correction_phi + result.correction_psi,
        residual,
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.projected_difference,
        result.raw_difference + expected_difference_update,
    )
    assert result.column_diagonal_context is context
    assert not result.raw_physical.requires_grad


def test_column_diagonal_equal_gain_matches_symmetric_projection(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]], dtype=torch.float64
    )
    rhs = torch.full((1, geometry.num_points), 10.0, dtype=torch.float64)
    gains = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    context = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=gains,
        gamma_y_squared=gains.clone(),
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
    )

    symmetric = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="physical_symmetric"),
    )
    column = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="column_diagonal_green_response"),
        column_diagonal_context=context,
    )

    torch.testing.assert_close(column.projected_physical, symmetric.projected_physical)
    torch.testing.assert_close(column.projected_response, symmetric.projected_response)
    torch.testing.assert_close(column.difference_update, torch.zeros_like(rhs))


def test_omitted_gain_exponent_is_bitwise_identical_to_explicit_one():
    gamma_x_squared = torch.tensor([0.0, 1.0, 9.0], dtype=torch.float64)
    gamma_y_squared = torch.tensor([9.0, 1.0, 0.0], dtype=torch.float64)
    default = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=gamma_x_squared,
        gamma_y_squared=gamma_y_squared,
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
    )
    explicit = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=gamma_x_squared,
        gamma_y_squared=gamma_y_squared,
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
        gain_exponent=1.0,
    )

    assert default.gain_exponent == 1.0
    assert torch.equal(default.correction_weight_phi, explicit.correction_weight_phi)
    assert torch.equal(default.correction_weight_psi, explicit.correction_weight_psi)


def test_zero_gain_exponent_matches_physical_symmetric_projection(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]], dtype=torch.float64
    )
    rhs = torch.tensor([[10.0, 11.0, 12.0]], dtype=torch.float64)
    context = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=torch.tensor([1.0, 100.0, 1.0e-6], dtype=torch.float64),
        gamma_y_squared=torch.tensor([100.0, 1.0, 1.0e6], dtype=torch.float64),
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
        gain_exponent=0.0,
    )

    symmetric = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="physical_symmetric"),
    )
    tempered = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="column_diagonal_green_response",
            column_diagonal_green_response={"gain_exponent": 0.0},
        ),
        column_diagonal_context=context,
    )

    torch.testing.assert_close(
        context.correction_weight_phi,
        torch.full_like(context.correction_weight_phi, 0.5),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tempered.projected_physical,
        symmetric.projected_physical,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tempered.projected_response,
        symmetric.projected_response,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tempered.difference_update,
        symmetric.difference_update,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("gain_exponent", [0.25, 0.5])
def test_tempered_gain_exponent_uses_stable_log_ratio(
    tmp_path,
    gain_exponent,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    gamma_x_squared = torch.tensor([1.0e-300, 9.0, 1.0e300], dtype=torch.float64)
    gamma_y_squared = torch.tensor([1.0e300, 1.0, 1.0e-300], dtype=torch.float64)
    context = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=gamma_x_squared,
        gamma_y_squared=gamma_y_squared,
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
        gain_exponent=gain_exponent,
    )
    expected = torch.sigmoid(
        gain_exponent
        * (
            torch.log(context.regularized_gamma_y_squared)
            - torch.log(context.regularized_gamma_x_squared)
        )
    )

    torch.testing.assert_close(context.correction_weight_phi, expected)
    torch.testing.assert_close(
        context.correction_weight_phi + context.correction_weight_psi,
        torch.ones_like(expected),
        atol=0.0,
        rtol=0.0,
    )
    assert torch.all(torch.isfinite(context.correction_weight_phi))
    assert torch.all((context.correction_weight_phi >= 0.0))
    assert torch.all((context.correction_weight_phi <= 1.0))
    assert context.gain_exponent == gain_exponent
    assert isinstance(context.gain_exponent, float)

    raw_response = torch.ones((1, 2, geometry.num_points), dtype=torch.float64)
    rhs = torch.full((1, geometry.num_points), 3.0, dtype=torch.float64)
    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(
            mode="column_diagonal_green_response",
            column_diagonal_green_response={"gain_exponent": gain_exponent},
        ),
        column_diagonal_context=context,
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
        atol=0.0,
        rtol=0.0,
    )


def test_column_diagonal_projection_requires_context_and_rejects_row_norm(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.ones((1, 2, geometry.num_points), dtype=torch.float64)
    rhs = torch.ones((1, geometry.num_points), dtype=torch.float64)

    with pytest.raises(ValueError, match="requires.*context"):
        apply_complex_balance_projection(
            raw_response,
            rhs,
            geometry,
            BalanceProjectionConfig(mode="column_diagonal_green_response"),
        )
    with pytest.raises(ValueError, match="Row-norm"):
        BalanceProjectionConfig.from_raw("row_norm")
    with pytest.raises(TypeError, match="unknown keys"):
        BalanceProjectionConfig.from_raw(
            {
                "mode": "column_diagonal_green_response",
                "column_diagonal_green_response": {"row_norm": True},
            }
        )


def test_column_diagonal_projection_keeps_finite_gradients(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.full((1, geometry.num_points), 10.0, dtype=torch.float64)
    context = ColumnDiagonalGreenResponseContext.from_gain_squared(
        gamma_x_squared=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        gamma_y_squared=torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        point_mass=0.25,
        gain_squared_eps=1.0e-12,
    )

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="column_diagonal_green_response"),
        column_diagonal_context=context,
    )
    result.projected_response.square().mean().backward()

    assert raw_response.grad is not None
    assert torch.all(torch.isfinite(raw_response.grad))
    assert not context.gamma_x_squared.requires_grad
    with pytest.raises(ValueError, match="finite and positive"):
        BalanceProjectionConfig.from_raw(
            {
                "mode": "column_diagonal_green_response",
                "column_diagonal_green_response": {"gain_squared_eps": 0.0},
            }
        )


@pytest.mark.parametrize("gain_exponent", [0.0, 0.25, 0.5, 1.0])
def test_gain_exponent_accepts_finite_unit_interval(gain_exponent):
    config = BalanceProjectionConfig.from_raw(
        {
            "mode": "column_diagonal_green_response",
            "column_diagonal_green_response": {"gain_exponent": gain_exponent},
        }
    )

    assert config.column_diagonal_green_response.gain_exponent == gain_exponent


@pytest.mark.parametrize("gain_exponent", [True, False, "0.25"])
def test_gain_exponent_rejects_non_numeric_values(gain_exponent):
    with pytest.raises(TypeError, match="gain_exponent must be numeric"):
        BalanceProjectionConfig.from_raw(
            {
                "mode": "column_diagonal_green_response",
                "column_diagonal_green_response": {
                    "gain_exponent": gain_exponent,
                },
            }
        )


@pytest.mark.parametrize(
    "gain_exponent",
    [float("nan"), float("inf"), float("-inf"), -0.1, 1.1],
)
def test_gain_exponent_rejects_values_outside_finite_unit_interval(gain_exponent):
    with pytest.raises(
        ValueError, match=r"gain_exponent must be finite and in \[0, 1\]"
    ):
        BalanceProjectionConfig.from_raw(
            {
                "mode": "column_diagonal_green_response",
                "column_diagonal_green_response": {
                    "gain_exponent": gain_exponent,
                },
            }
        )
