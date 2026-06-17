from __future__ import annotations

import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import physical_edge_energy_loss
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
