from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import load_complex_geometry
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
