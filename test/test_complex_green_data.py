from __future__ import annotations

import numpy as np
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_data import (
    ComplexGreenDataset,
    complex_green_collate_fn,
    generate_complex_green_data,
)
from test.complex_fixtures import write_coefficients, write_geometry_npz


def test_complex_green_data_flattens_unequal_segment_counts(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=2,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        dtype=torch.float64,
    )

    assert geometry.num_x_segments == 2
    assert geometry.num_y_segments == 3
    assert data.num_intervals == 5
    assert data.solution.shape == (2, 5, 5)
    assert data.source.shape == (2, 5, 5)
    torch.testing.assert_close(
        data.unit_grid,
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
    )
    assert data.axis_id.tolist() == [0, 0, 1, 1, 1]
    assert data.segment_id.tolist() == [0, 1, 0, 1, 2]


def test_complex_green_data_maps_physical_coordinates(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=1,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        data.physical_coords[0, :, 0],
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        data.physical_coords[0, :, 1],
        torch.full((5,), 0.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        data.physical_coords[1, :, 0],
        torch.linspace(0.0, 0.5, 5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        data.physical_coords[1, :, 1],
        torch.full((5,), 0.75, dtype=torch.float64),
    )
    torch.testing.assert_close(
        data.physical_coords[2, :, 0],
        torch.full((5,), 0.25, dtype=torch.float64),
    )
    torch.testing.assert_close(
        data.physical_coords[2, :, 1],
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
    )


def test_complex_green_collate_keeps_shared_metadata(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=2,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        dtype=torch.float64,
    )

    dataset = ComplexGreenDataset(data)
    batch = complex_green_collate_fn([dataset[0], dataset[1]])

    assert batch.solution.shape == (2, 5, 5)
    assert batch.source.shape == (2, 5, 5)
    torch.testing.assert_close(batch.a_vals, data.a_vals)
    torch.testing.assert_close(batch.physical_coords, data.physical_coords)


def test_complex_green_data_generates_fine_source_on_same_unit_grid(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=1,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        source_sampling_factor=3,
        dtype=torch.float64,
    )

    assert data.source_fine is not None
    assert data.source_fine_grid is not None
    assert data.source_fine.shape == (1, 5, 13)
    torch.testing.assert_close(
        data.source_fine_grid,
        torch.linspace(0.0, 1.0, 13, dtype=torch.float64),
    )
    torch.testing.assert_close(data.source_fine[..., ::3], data.source)

    batch = complex_green_collate_fn([ComplexGreenDataset(data)[0]]).to("cpu")
    assert batch.source_fine is not None
    assert batch.source_fine_grid is not None
    torch.testing.assert_close(batch.source_fine[..., ::3], batch.source)


def test_complex_green_data_preserves_duplicate_fixed_disconnected_segments(tmp_path):
    geometry_path = write_geometry_npz(
        tmp_path / "geometry.npz",
        x_segment_y=np.array([0.25, 0.25], dtype=np.float64),
    )
    geometry = load_complex_geometry(geometry_path)
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=1,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        dtype=torch.float64,
    )

    assert data.fixed[:2].tolist() == [0.25, 0.25]
    assert data.segment_id[:2].tolist() == [0, 1]
