from __future__ import annotations

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
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
