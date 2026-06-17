from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_geometry import load_complex_geometry
from test.complex_fixtures import (
    write_coefficients,
    write_geometry_npz,
    write_sample_npz,
)


def test_complex_dataset_gathers_full_grid_and_preferred_flux(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)

    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        dtype=torch.float64,
    )
    item = dataset[0]

    torch.testing.assert_close(
        item.rhs_valid,
        torch.tensor([7.0, 9.0, 17.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        item.sol_valid,
        torch.tensor([8.0, 10.0, 18.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        item.flux_valid[0],
        torch.tensor([11.0, 13.0, 21.0], dtype=torch.float64),
    )
    assert bool(item.has_flux)
    assert dataset.x_branch.shape == (2, 3, 4)
    assert dataset.y_branch.shape == (3, 3, 4)
    assert dataset.x_green_branch.shape == (2, 4, 4)
    assert dataset.y_green_branch.shape == (3, 4, 4)


def test_complex_dataset_uses_legacy_flux_fallback_and_collates(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir, legacy_flux=True)

    dataset = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)
    batch = complex_coupling_collate_fn([dataset[0]])

    torch.testing.assert_close(
        batch.flux_valid[0, 0],
        torch.tensor([9.0, 11.0, 19.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        batch.flux_valid[0, 1],
        torch.tensor([10.0, 12.0, 20.0], dtype=torch.float64),
    )
    assert batch.geometry.num_x_segments != batch.geometry.num_y_segments
    assert batch.file_stems == ("sample_0000",)


def test_complex_dataset_allows_missing_optional_flux(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir, include_flux=False)

    item = ComplexCouplingDataset(data_dir, geometry, coeffs, branch_input_dim=4)[0]

    assert not bool(item.has_flux)
    torch.testing.assert_close(
        item.flux_valid, torch.zeros((2, 3), dtype=torch.float64)
    )
