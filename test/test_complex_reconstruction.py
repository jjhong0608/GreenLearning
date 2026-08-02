from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContextBuilder,
    ColumnDiagonalGreenResponseContextCache,
    column_diagonal_gain_squared,
)
from greenonet.complex_reconstruction import reconstruct_from_projected_response
from test.complex_fixtures import (
    ConstantGreen,
    write_coefficients,
    write_geometry_npz,
    write_sample_npz,
)


def test_complex_reconstruction_uses_response_directly_and_zero_endpoints(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    item = dataset[0]
    projected_response = torch.tensor(
        [[[2.0, 4.0, 6.0], [10.0, 20.0, 30.0]]],
        dtype=torch.float64,
    )

    result = reconstruct_from_projected_response(
        green_model=ConstantGreen(1.0),
        geometry=geometry,
        projected_response=projected_response,
        x_green_branch=item.x_green_branch.unsqueeze(0),
        y_green_branch=item.y_green_branch.unsqueeze(0),
    )

    torch.testing.assert_close(result.projected_response, projected_response)
    torch.testing.assert_close(
        result.u_phi_valid,
        torch.tensor([[2.25, 2.25, 3.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.u_psi_valid,
        torch.tensor([[15.0, 10.0, 15.0]], dtype=torch.float64),
    )


def test_column_diagonal_gain_is_source_column_norm_not_row_norm():
    kernel = torch.tensor(
        [[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]],
        dtype=torch.float64,
    )
    weights = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)

    gain = column_diagonal_gain_squared(
        kernel=kernel,
        source_weights=weights,
        segment_length=2.0,
        point_mass=0.25,
    )
    response = kernel * (weights * 4.0).unsqueeze(0)
    expected_column_diagonal = 0.25 * response.square().sum(dim=0)
    row_norm = 0.25 * response.square().sum(dim=1)

    torch.testing.assert_close(gain, expected_column_diagonal)
    assert gain.shape == (3,)
    assert row_norm.shape == (2,)


def test_column_diagonal_gain_has_length_to_fourth_scaling():
    kernel = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    gain_l = column_diagonal_gain_squared(
        kernel=kernel,
        source_weights=weights,
        segment_length=1.0,
        point_mass=0.5,
    )
    gain_2l = column_diagonal_gain_squared(
        kernel=kernel,
        source_weights=weights,
        segment_length=2.0,
        point_mass=0.5,
    )

    torch.testing.assert_close(gain_2l, 16.0 * gain_l)


def test_column_diagonal_context_reuses_reconstruction_nodes_and_cache(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)
    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    item = dataset[0]
    x_branch = item.x_green_branch.unsqueeze(0)
    y_branch = item.y_green_branch.unsqueeze(0)
    builder = ColumnDiagonalGreenResponseContextBuilder(
        {"gain_squared_eps": 1e-12, "gain_exponent": 0.25}
    )

    context = builder.build(
        green_model=ConstantGreen(1.0),
        geometry=geometry,
        x_green_branch=x_branch,
        y_green_branch=y_branch,
    )

    torch.testing.assert_close(
        context.gamma_x_squared,
        torch.tensor([0.0703125, 0.0703125, 0.00390625], dtype=torch.float64),
    )
    torch.testing.assert_close(
        context.gamma_y_squared,
        torch.tensor([0.0703125, 0.0625, 0.0703125], dtype=torch.float64),
    )
    assert not context.gamma_x_squared.requires_grad
    assert not context.gamma_y_squared.requires_grad
    assert context.gain_exponent == 0.25

    cache = ColumnDiagonalGreenResponseContextCache(
        {"gain_squared_eps": 1e-12, "gain_exponent": 0.25}
    )
    first = cache.get_or_build(
        green_model=ConstantGreen(1.0),
        geometry=geometry,
        x_green_branch=x_branch,
        y_green_branch=y_branch,
    )
    second = cache.get_or_build(
        green_model=ConstantGreen(2.0),
        geometry=geometry,
        x_green_branch=x_branch,
        y_green_branch=y_branch,
    )
    assert first is second
    assert cache.build_count == 1
    assert first.gain_exponent == 0.25
