from __future__ import annotations

import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_projection import (
    apply_geometry_weighted_projection,
    apply_hard_symmetric_projection,
)
from test.complex_fixtures import write_geometry_npz


def test_complex_projection_converts_unit_physical_and_back(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_unit = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

    result = apply_hard_symmetric_projection(raw_unit, rhs, geometry)

    torch.testing.assert_close(
        result.raw_physical[:, 0],
        torch.tensor([[4.0, 8.0, 8.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.raw_physical[:, 1],
        torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.projected_physical[:, 0] + result.projected_physical[:, 1],
        rhs,
    )
    torch.testing.assert_close(
        result.projected_unit[:, 0],
        result.projected_physical[:, 0]
        * geometry.x_lengths_for_valid_points().pow(2).unsqueeze(0),
    )
    torch.testing.assert_close(
        result.projected_unit[:, 1],
        result.projected_physical[:, 1]
        * geometry.y_lengths_for_valid_points().pow(2).unsqueeze(0),
    )


def test_geometry_weighted_projection_preserves_balance_and_uses_lengths(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    raw_unit = torch.tensor(
        [[[4.0, 8.0, 2.0], [1.0, 3.0, 5.0]]],
        dtype=torch.float64,
    )
    rhs = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

    symmetric = apply_hard_symmetric_projection(raw_unit, rhs, geometry)
    weighted = apply_geometry_weighted_projection(raw_unit, rhs, geometry)

    torch.testing.assert_close(
        weighted.projected_physical[:, 0] + weighted.projected_physical[:, 1],
        rhs,
    )
    x_scale = geometry.x_lengths_for_valid_points().pow(2)
    y_scale = geometry.y_lengths_for_valid_points().pow(2)
    w_phi = x_scale / (x_scale + y_scale)
    w_psi = y_scale / (x_scale + y_scale)
    beta = 2.0 * w_phi * w_psi
    raw_difference = weighted.raw_physical[:, 0] - weighted.raw_physical[:, 1]
    torch.testing.assert_close(
        weighted.projected_physical[:, 0],
        w_phi.unsqueeze(0) * rhs + beta.unsqueeze(0) * raw_difference,
    )
    torch.testing.assert_close(
        weighted.projected_physical[:, 1],
        w_psi.unsqueeze(0) * rhs - beta.unsqueeze(0) * raw_difference,
    )
    assert not torch.allclose(
        weighted.projected_physical[:, :, 2],
        symmetric.projected_physical[:, :, 2],
    )
