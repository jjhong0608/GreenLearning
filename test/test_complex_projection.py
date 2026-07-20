from __future__ import annotations

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_projection import apply_complex_balance_projection
from greenonet.config import BalanceProjectionConfig
from test.complex_fixtures import write_geometry_npz


def test_response_projection_enforces_physical_balance(tmp_path):
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
        BalanceProjectionConfig(mode="response_space"),
    )

    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    scale = torch.maximum(sigma_x, sigma_y)
    normal_x = sigma_y / scale
    normal_y = sigma_x / scale
    constraint = sigma_x * sigma_y / scale * rhs
    residual = (
        normal_x * raw_response[:, 0] + normal_y * raw_response[:, 1] - constraint
    )
    norm_squared = normal_x.square() + normal_y.square()
    expected_response = torch.stack(
        (
            raw_response[:, 0] - normal_x * residual / norm_squared,
            raw_response[:, 1] - normal_y * residual / norm_squared,
        ),
        dim=1,
    )

    torch.testing.assert_close(result.raw_response, raw_response)
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


def test_response_projection_is_equal_correction_for_equal_lengths(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_response = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]], dtype=torch.float64
    )
    rhs = torch.full((1, geometry.num_points), 10.0, dtype=torch.float64)

    result = apply_complex_balance_projection(
        raw_response,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="response_space"),
    )

    sigma_x = geometry.x_lengths_for_valid_points().square()
    sigma_y = geometry.y_lengths_for_valid_points().square()
    equal = sigma_x == sigma_y
    correction_x = result.raw_response[:, 0] - result.projected_response[:, 0]
    correction_y = result.raw_response[:, 1] - result.projected_response[:, 1]
    torch.testing.assert_close(correction_x[:, equal], correction_y[:, equal])


def test_response_projection_rejects_invalid_shapes_and_retired_modes(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    rhs = torch.ones((1, geometry.num_points), dtype=torch.float64)
    config = BalanceProjectionConfig(mode="response_space")

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
    with pytest.raises(ValueError, match="removed.*response_space"):
        BalanceProjectionConfig.from_raw("response_preconditioned")
