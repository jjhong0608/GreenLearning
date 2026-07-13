from __future__ import annotations

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_projection import (
    apply_complex_balance_projection,
    apply_hard_symmetric_projection,
)
from greenonet.config import BalanceProjectionConfig
from test.complex_fixtures import write_geometry_npz


def test_symmetric_projection_uses_physical_raw_outputs(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_physical = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

    result = apply_hard_symmetric_projection(raw_physical, rhs, geometry)

    torch.testing.assert_close(result.raw_physical, raw_physical)
    torch.testing.assert_close(
        result.balance_residual,
        rhs - raw_physical[:, 0] - raw_physical[:, 1],
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] - result.projected_physical[:, 1],
        raw_physical[:, 0] - raw_physical[:, 1],
    )
    torch.testing.assert_close(
        result.raw_difference, raw_physical[:, 0] - raw_physical[:, 1]
    )
    torch.testing.assert_close(result.projected_difference, result.raw_difference)
    torch.testing.assert_close(
        result.response_baseline_difference,
        torch.zeros_like(rhs),
    )
    torch.testing.assert_close(result.response_gain, torch.ones_like(rhs))


def test_response_preconditioned_projection_matches_length_formula(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_physical = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

    result = apply_complex_balance_projection(
        raw_physical,
        rhs,
        geometry,
        BalanceProjectionConfig(mode="response_preconditioned"),
    )

    sigma_x = geometry.x_lengths_for_valid_points().square().unsqueeze(0)
    sigma_y = geometry.y_lengths_for_valid_points().square().unsqueeze(0)
    denominator = sigma_x + sigma_y
    expected_d0 = (sigma_y - sigma_x) * rhs / denominator
    expected_kappa = 4.0 * sigma_x * sigma_y / denominator.square()
    expected_difference = expected_d0 + expected_kappa * (
        raw_physical[:, 0] - raw_physical[:, 1]
    )
    torch.testing.assert_close(result.response_baseline_difference, expected_d0)
    torch.testing.assert_close(result.response_gain, expected_kappa)
    torch.testing.assert_close(result.projected_difference, expected_difference)
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
    )
    equal_length = sigma_x == sigma_y
    torch.testing.assert_close(
        result.projected_difference[equal_length],
        result.raw_difference[equal_length],
    )


def test_symmetric_projection_rejects_invalid_shapes(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    rhs = torch.ones((1, geometry.num_points), dtype=torch.float64)

    with pytest.raises(ValueError, match="raw_physical"):
        apply_hard_symmetric_projection(
            torch.ones((1, geometry.num_points), dtype=torch.float64),
            rhs,
            geometry,
        )
    with pytest.raises(ValueError, match="rhs_phys"):
        apply_hard_symmetric_projection(
            torch.ones((1, 2, geometry.num_points), dtype=torch.float64),
            torch.ones((1, geometry.num_points + 1), dtype=torch.float64),
            geometry,
        )
