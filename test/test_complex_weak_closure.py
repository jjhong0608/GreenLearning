from __future__ import annotations

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_weak_closure import (
    assemble_directional_weak_residuals,
    build_directional_weak_context,
    directional_weak_operator_closure_loss,
)
from test.complex_fixtures import write_coefficients, write_geometry_npz


def test_directional_weak_context_uses_full_p1_operator_and_axis_convection(
    tmp_path,
):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    context = build_directional_weak_context(geometry, coeffs)

    expected_x = torch.tensor(
        [[1.125, -0.9375], [-2.9375, 3.125]],
        dtype=torch.float64,
    )
    expected_y = torch.tensor(
        [[0.875, -0.6875], [-3.1875, 3.375]],
        dtype=torch.float64,
    )
    expected_mass = torch.tensor(
        [
            [1.0 / 24.0, 1.0 / 48.0],
            [1.0 / 48.0, 1.0 / 24.0],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(context.x.local_operator[0], expected_x)
    torch.testing.assert_close(context.y.local_operator[0], expected_y)
    torch.testing.assert_close(context.x.local_mass[0], expected_mass)
    torch.testing.assert_close(context.y.local_mass[0], expected_mass)


def test_directional_weak_context_preserves_true_boundary_endpoint_nodes(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    context = build_directional_weak_context(geometry, coeffs)

    assert context.x.element_valid_index[0].tolist() == [-1, 0]
    assert context.x.element_valid_index[2].tolist() == [1, -1]
    torch.testing.assert_close(
        context.x.element_length[:3],
        torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
    )
    assert torch.all(context.x.nodal_mass > 0.0)


def test_zero_trial_and_zero_directional_source_have_zero_weak_residual(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    context = build_directional_weak_context(geometry, coeffs)
    u_pred = torch.zeros((2, geometry.num_points), dtype=torch.float64)
    projected_physical = torch.zeros(
        (2, 2, geometry.num_points),
        dtype=torch.float64,
    )
    rhs = torch.ones_like(u_pred)

    result = directional_weak_operator_closure_loss(
        u_pred_valid=u_pred,
        projected_physical=projected_physical,
        rhs_valid=rhs,
        context=context,
        eps=1.0e-12,
    )

    torch.testing.assert_close(result.loss, torch.zeros_like(result.loss))
    torch.testing.assert_close(result.x_loss, torch.zeros_like(result.x_loss))
    torch.testing.assert_close(result.y_loss, torch.zeros_like(result.y_loss))
    torch.testing.assert_close(
        result.x_residual,
        torch.zeros_like(result.x_residual),
    )
    torch.testing.assert_close(
        result.y_residual,
        torch.zeros_like(result.y_residual),
    )


def test_public_directional_residual_helper_matches_weak_loss_residuals(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    context = build_directional_weak_context(geometry, coeffs)
    u_pred = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)
    projected_physical = torch.tensor(
        [[[0.5, 1.0, 1.5], [1.5, 1.0, 0.5]]],
        dtype=torch.float64,
    )

    residuals = assemble_directional_weak_residuals(
        u_valid=u_pred,
        projected_physical=projected_physical,
        context=context,
    )
    loss_result = directional_weak_operator_closure_loss(
        u_pred_valid=u_pred,
        projected_physical=projected_physical,
        rhs_valid=projected_physical.sum(dim=1),
        context=context,
        eps=1.0e-12,
    )

    torch.testing.assert_close(residuals.x, loss_result.x_residual)
    torch.testing.assert_close(residuals.y, loss_result.y_residual)
    torch.testing.assert_close(
        residuals.full,
        loss_result.x_residual + loss_result.y_residual,
    )


def test_weak_closure_uses_common_u_pred_and_is_source_scale_normalized(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    context = build_directional_weak_context(geometry, coeffs)
    base_u = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    base_source = torch.tensor(
        [[[0.5, 1.0, 1.5], [1.5, 1.0, 0.5]]],
        dtype=torch.float64,
    )
    base_rhs = base_source.sum(dim=1)

    base = directional_weak_operator_closure_loss(
        u_pred_valid=base_u,
        projected_physical=base_source,
        rhs_valid=base_rhs,
        context=context,
        eps=1.0e-12,
    )
    scaled = directional_weak_operator_closure_loss(
        u_pred_valid=7.0 * base_u,
        projected_physical=7.0 * base_source,
        rhs_valid=7.0 * base_rhs,
        context=context,
        eps=1.0e-12,
    )

    assert base.loss.item() > 0.0
    torch.testing.assert_close(base.loss, scaled.loss)
    torch.testing.assert_close(base.x_loss, scaled.x_loss)
    torch.testing.assert_close(base.y_loss, scaled.y_loss)


def test_manufactured_p1_directional_sources_close_nonzero_common_solution(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    context = build_directional_weak_context(geometry, coeffs)
    u_pred = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float64)

    def assembled_matrices(axis_context):
        operator = torch.zeros(
            (geometry.num_points, geometry.num_points),
            dtype=torch.float64,
        )
        mass = torch.zeros_like(operator)
        for element_index, nodes in enumerate(axis_context.element_valid_index):
            for local_row in range(2):
                row = int(nodes[local_row].item())
                if row < 0:
                    continue
                for local_col in range(2):
                    col = int(nodes[local_col].item())
                    if col < 0:
                        continue
                    operator[row, col] += axis_context.local_operator[
                        element_index,
                        local_row,
                        local_col,
                    ]
                    mass[row, col] += axis_context.local_mass[
                        element_index,
                        local_row,
                        local_col,
                    ]
        return operator, mass

    x_operator, x_mass = assembled_matrices(context.x)
    y_operator, y_mass = assembled_matrices(context.y)
    phi = torch.linalg.solve(x_mass, x_operator @ u_pred[0])
    psi = torch.linalg.solve(y_mass, y_operator @ u_pred[0])
    projected_physical = torch.stack((phi, psi), dim=0).unsqueeze(0)

    result = directional_weak_operator_closure_loss(
        u_pred_valid=u_pred,
        projected_physical=projected_physical,
        rhs_valid=phi.add(psi).unsqueeze(0),
        context=context,
        eps=1.0e-12,
    )

    torch.testing.assert_close(
        result.loss,
        torch.zeros((), dtype=torch.float64),
        atol=1.0e-24,
        rtol=0.0,
    )


def test_directional_weak_context_moves_all_tensors(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))

    context = build_directional_weak_context(geometry, coeffs).to("cpu")

    for axis in (context.x, context.y):
        assert axis.element_valid_index.device.type == "cpu"
        assert axis.local_operator.device.type == "cpu"
        assert axis.local_mass.device.type == "cpu"
        assert axis.nodal_mass.device.type == "cpu"
