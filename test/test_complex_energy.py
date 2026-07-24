from __future__ import annotations

import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import (
    build_boundary_energy_context,
    canonical_complex_energy_loss,
    physical_boundary_energy_loss,
    physical_edge_energy_loss,
    relative_split_consistency_loss,
)
from greenonet.config import ComplexRelativeSplitConsistencyConfig
from test.complex_fixtures import write_geometry_npz


def test_complex_energy_uses_edges_spacing_area_and_face_average(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    a_valid = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float64)

    loss = physical_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=a_valid,
        geometry=geometry,
    )

    expected_x = 0.5 * 0.5 * 0.5 * (2.0 + 4.0) * ((3.0 - 1.0) / 0.5) ** 2
    expected_y = 0.5 * 0.5 * 0.5 * (2.0 + 6.0) * ((5.0 - 1.0) / 0.5) ** 2
    torch.testing.assert_close(
        loss,
        torch.tensor(expected_x + expected_y, dtype=torch.float64),
    )


def test_boundary_context_covers_every_connected_segment_endpoint(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))

    context = build_boundary_energy_context(geometry)

    assert context.x_anchor_count == 4
    assert context.y_anchor_count == 4
    assert context.total_anchors == 8
    torch.testing.assert_close(
        context.physical_distance,
        torch.tensor(
            [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75],
            dtype=torch.float64,
        ),
    )
    assert torch.all(context.point_indices >= 0)
    assert torch.all(context.physical_distance > 0.0)


def test_boundary_energy_anchors_constant_residual(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    residual = torch.ones((1, geometry.num_points), dtype=torch.float64)
    context = build_boundary_energy_context(geometry)

    boundary = physical_boundary_energy_loss(
        u_phi_valid=residual,
        u_psi_valid=torch.zeros_like(residual),
        a_valid=torch.ones_like(residual),
        context=context,
    )
    energy = canonical_complex_energy_loss(
        u_phi_valid=residual,
        u_psi_valid=torch.zeros_like(residual),
        a_valid=torch.ones_like(residual),
        geometry=geometry,
        boundary_context=context,
    )

    expected = torch.tensor(44.0 / 3.0, dtype=torch.float64)
    torch.testing.assert_close(boundary.total, expected)
    torch.testing.assert_close(energy.bulk, torch.zeros_like(expected))
    torch.testing.assert_close(energy.boundary, expected)
    torch.testing.assert_close(energy.total, expected)


def test_canonical_energy_sums_all_edges_without_transition_partition(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    a_valid = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float64)

    result = canonical_complex_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=a_valid,
        geometry=geometry,
    )

    expected_x = 0.5 * 0.5 * 0.5 * (2.0 + 4.0) * ((3.0 - 1.0) / 0.5) ** 2
    expected_y = 0.5 * 0.5 * 0.5 * (2.0 + 6.0) * ((5.0 - 1.0) / 0.5) ** 2
    torch.testing.assert_close(
        result.bulk,
        torch.tensor(expected_x + expected_y, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.total,
        result.bulk + result.boundary,
    )


def test_canonical_energy_exposes_per_sample_values(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 6.0, 10.0]],
        dtype=torch.float64,
    )
    result = canonical_complex_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=torch.zeros_like(u_phi),
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
    )

    assert result.total_per_sample.shape == (2,)
    assert result.bulk_per_sample.shape == (2,)
    torch.testing.assert_close(
        result.total,
        result.total_per_sample.mean(),
    )
    torch.testing.assert_close(
        result.bulk,
        result.bulk_per_sample.mean(),
    )
    torch.testing.assert_close(
        result.total_per_sample[1],
        4.0 * result.total_per_sample[0],
    )


def test_relative_split_mass_detects_constant_solution_mismatch(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.ones((1, geometry.num_points), dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    rhs = torch.ones_like(u_phi)
    energy = canonical_complex_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
    )

    result = relative_split_consistency_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        rhs_valid=rhs,
        energy=energy,
        geometry=geometry,
        config=ComplexRelativeSplitConsistencyConfig(
            enabled=True,
            weight=2.0,
            mass_weight=3.0,
        ),
    )

    assert result.energy_relative.item() > 0.0
    torch.testing.assert_close(
        result.mass_relative,
        torch.ones((), dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        result.loss,
        2.0 * (result.energy_relative + 3.0 * result.mass_relative),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_relative_split_loss_is_invariant_to_common_source_scale(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    base_phi = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    base_psi = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    base_rhs = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    config = ComplexRelativeSplitConsistencyConfig(enabled=True)

    def compute(scale: float):
        u_phi = scale * base_phi
        u_psi = scale * base_psi
        rhs = scale * base_rhs
        energy = canonical_complex_energy_loss(
            u_phi_valid=u_phi,
            u_psi_valid=u_psi,
            a_valid=torch.ones_like(u_phi),
            geometry=geometry,
        )
        return relative_split_consistency_loss(
            u_phi_valid=u_phi,
            u_psi_valid=u_psi,
            rhs_valid=rhs,
            energy=energy,
            geometry=geometry,
            config=config,
        )

    base = compute(1.0)
    scaled = compute(7.0)

    torch.testing.assert_close(base.loss, scaled.loss)
    torch.testing.assert_close(base.energy_relative, scaled.energy_relative)
    torch.testing.assert_close(base.mass_relative, scaled.mass_relative)


def test_relative_split_normalizes_each_sample_before_batch_reduction(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    base_phi = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    base_psi = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    base_rhs = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    scales = torch.tensor([[1.0], [0.01]], dtype=torch.float64)
    u_phi = scales * base_phi
    u_psi = scales * base_psi
    rhs = scales * base_rhs
    energy = canonical_complex_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
    )

    result = relative_split_consistency_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        rhs_valid=rhs,
        energy=energy,
        geometry=geometry,
        config=ComplexRelativeSplitConsistencyConfig(enabled=True),
    )

    torch.testing.assert_close(
        result.loss_per_sample[0],
        result.loss_per_sample[1],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    torch.testing.assert_close(result.loss, result.loss_per_sample.mean())
