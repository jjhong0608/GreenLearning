from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import CouplingCoefficientTermsConfig
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
    assert dataset.x_coefficient_branch.shape == (2, 1, 4)
    assert dataset.y_coefficient_branch.shape == (3, 1, 4)
    assert dataset.x_green_branch.shape == (2, 4, 4)
    assert dataset.y_green_branch.shape == (3, 4, 4)
    assert item.x_source_branch.shape == (2, 4)
    assert item.y_source_branch.shape == (3, 4)
    assert item.x_source_norm.shape == (2,)
    assert item.y_source_norm.shape == (3,)
    torch.testing.assert_close(
        item.x_source_branch[:, 0],
        torch.zeros(2, dtype=torch.float64),
    )
    torch.testing.assert_close(
        item.x_source_branch[:, -1],
        torch.zeros(2, dtype=torch.float64),
    )


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
    assert batch.x_source_branch.shape == (1, 2, 4)
    assert batch.y_source_branch.shape == (1, 3, 4)


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


def test_complex_dataset_builds_source_branch_with_unit_scaling(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)

    item = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        integration_rule="trapezoid",
    )[0]

    expected_x1_unit = torch.tensor(
        [0.0, 17.0 / 6.0, 17.0 / 6.0, 0.0],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        item.x_source_branch[1] * item.x_source_norm[1],
        expected_x1_unit,
    )
    torch.testing.assert_close(
        item.y_source_branch[2] * item.y_source_norm[2],
        torch.zeros(4, dtype=torch.float64),
    )


def test_complex_dataset_respects_coefficient_terms(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data_dir = tmp_path / "data"
    write_sample_npz(data_dir)

    dataset = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=True,
            convection=True,
            reaction=True,
        ),
    )
    assert dataset.x_coefficient_branch.shape == (2, 4, 4)
    assert dataset.y_coefficient_branch.shape == (3, 4, 4)
    torch.testing.assert_close(
        dataset.x_coefficient_branch[:, 0],
        torch.ones((2, 4), dtype=torch.float64),
    )
    torch.testing.assert_close(
        dataset.x_coefficient_branch[:, 1],
        torch.tensor(
            [[4.0, 4.0, 4.0, 4.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        dataset.x_coefficient_branch[:, 2],
        torch.tensor(
            [[5.0, 5.0, 5.0, 5.0], [2.5, 2.5, 2.5, 2.5]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        dataset.x_coefficient_branch[:, 3],
        torch.tensor(
            [[6.0, 6.0, 6.0, 6.0], [1.5, 1.5, 1.5, 1.5]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        dataset.y_coefficient_branch[:, 1],
        torch.full((3, 4), 5.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        dataset.y_coefficient_branch[:, 2],
        torch.full((3, 4), 4.0, dtype=torch.float64),
    )
    torch.testing.assert_close(
        dataset.x_green_branch[:, 2],
        torch.tensor(
            [[4.0, 4.0, 4.0, 4.0], [2.0, 2.0, 2.0, 2.0]],
            dtype=torch.float64,
        ),
    )

    source_only = ComplexCouplingDataset(
        data_dir,
        geometry,
        coeffs,
        branch_input_dim=4,
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=False,
            convection=False,
            reaction=False,
        ),
    )
    assert source_only.x_coefficient_branch.shape == (2, 0, 4)
