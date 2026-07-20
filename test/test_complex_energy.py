from __future__ import annotations

import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import (
    build_length_jump_partition,
    length_jump_balanced_edge_energy_loss,
    physical_edge_energy_loss,
    relative_split_consistency_loss,
)
from greenonet.config import (
    ComplexLengthJumpBalanceConfig,
    ComplexRelativeSplitConsistencyConfig,
)
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


def test_length_jump_balanced_energy_groups_response_scale_transitions(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    a_valid = torch.tensor([[2.0, 4.0, 6.0]], dtype=torch.float64)
    config = ComplexLengthJumpBalanceConfig(
        enabled=True,
        transition_fraction=0.25,
    )
    partition = build_length_jump_partition(geometry, config)

    result = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=a_valid,
        geometry=geometry,
        config=config,
        partition=partition,
    )

    expected_x = 0.5 * 0.5 * 0.5 * (2.0 + 4.0) * ((3.0 - 1.0) / 0.5) ** 2
    expected_y = 0.5 * 0.5 * 0.5 * (2.0 + 6.0) * ((5.0 - 1.0) / 0.5) ** 2
    assert not bool(partition.x_transition_mask[0])
    assert bool(partition.y_transition_mask[0])
    torch.testing.assert_close(
        partition.y_score,
        torch.tensor([torch.log(torch.tensor(4.0))], dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.unweighted,
        torch.tensor(expected_x + expected_y, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.balanced,
        torch.tensor(1.5 * expected_x + 0.5 * expected_y, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.regular_mean,
        torch.tensor(expected_x, dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.transition_mean,
        torch.tensor(expected_y, dtype=torch.float64),
    )
    assert result.transition_edge_fraction.item() == 0.5


def test_length_jump_balanced_energy_falls_back_without_transition_edges(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float64)
    config = ComplexLengthJumpBalanceConfig(
        enabled=True,
        log_sigma_jump_threshold=10.0,
    )

    result = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=torch.zeros_like(u_phi),
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
        config=config,
    )

    torch.testing.assert_close(result.balanced, result.unweighted)
    assert result.transition_edge_fraction.item() == 0.0


def test_length_jump_balanced_energy_exposes_per_sample_values(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.tensor(
        [[1.0, 3.0, 5.0], [2.0, 6.0, 10.0]],
        dtype=torch.float64,
    )
    result = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=torch.zeros_like(u_phi),
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
        config=ComplexLengthJumpBalanceConfig(enabled=True),
    )

    assert result.unweighted_per_sample.shape == (2,)
    assert result.balanced_per_sample.shape == (2,)
    torch.testing.assert_close(
        result.unweighted,
        result.unweighted_per_sample.mean(),
    )
    torch.testing.assert_close(
        result.balanced,
        result.balanced_per_sample.mean(),
    )
    torch.testing.assert_close(
        result.unweighted_per_sample[1],
        4.0 * result.unweighted_per_sample[0],
    )


def test_relative_split_mass_detects_constant_solution_mismatch(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.ones((1, geometry.num_points), dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    rhs = torch.ones_like(u_phi)
    energy = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
        config=ComplexLengthJumpBalanceConfig(enabled=True),
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

    torch.testing.assert_close(
        result.energy_relative,
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        result.mass_relative,
        torch.ones((), dtype=torch.float64),
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        result.loss,
        torch.tensor(6.0, dtype=torch.float64),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_relative_split_loss_is_invariant_to_common_source_scale(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    base_phi = torch.tensor([[1.0, 2.0, 4.0]], dtype=torch.float64)
    base_psi = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
    base_rhs = torch.tensor([[2.0, 3.0, 5.0]], dtype=torch.float64)
    config = ComplexRelativeSplitConsistencyConfig(enabled=True)
    jump_config = ComplexLengthJumpBalanceConfig(enabled=True)

    def compute(scale: float):
        u_phi = scale * base_phi
        u_psi = scale * base_psi
        rhs = scale * base_rhs
        energy = length_jump_balanced_edge_energy_loss(
            u_phi_valid=u_phi,
            u_psi_valid=u_psi,
            a_valid=torch.ones_like(u_phi),
            geometry=geometry,
            config=jump_config,
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
    energy = length_jump_balanced_edge_energy_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=torch.ones_like(u_phi),
        geometry=geometry,
        config=ComplexLengthJumpBalanceConfig(enabled=True),
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
