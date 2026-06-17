from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_data import generate_complex_green_data
from greenonet.complex_green_trainer import ComplexGreenTrainer
from test.complex_fixtures import write_coefficients, write_geometry_npz


def test_complex_green_coefficients_use_unit_interval_scaling(tmp_path):
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

    torch.testing.assert_close(data.a_vals, torch.ones((5, 5), dtype=torch.float64))
    expected_ap = torch.tensor([2.0, 1.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    expected_b = torch.tensor([4.0, 2.0, 5.0, 5.0, 5.0], dtype=torch.float64)
    expected_c = torch.tensor([6.0, 1.5, 6.0, 6.0, 6.0], dtype=torch.float64)
    torch.testing.assert_close(data.ap_vals, expected_ap[:, None].expand(5, 5))
    torch.testing.assert_close(data.b_vals, expected_b[:, None].expand(5, 5))
    torch.testing.assert_close(data.c_vals, expected_c[:, None].expand(5, 5))


def test_complex_green_forward_sampler_outputs_unit_source(tmp_path):
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

    assert data.source.shape == (1, 5, 5)
    assert torch.isfinite(data.source).all()
    assert torch.isfinite(data.solution).all()


def test_complex_green_reconstruction_has_no_segment_length_factor():
    unit_grid = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    kernel = torch.ones((3, 5, 5), dtype=torch.float64)
    source = torch.ones((2, 3, 5), dtype=torch.float64)

    reconstruction = ComplexGreenTrainer._reconstruct_solution(
        kernel=kernel,
        source=source,
        unit_grid=unit_grid,
        integration_rule="trapezoid",
    )

    torch.testing.assert_close(
        reconstruction,
        torch.ones((2, 3, 5), dtype=torch.float64),
    )
