from __future__ import annotations

from pathlib import Path

import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_gluing import (
    build_complex_gluing_context,
    complex_admissibility_gluing_loss,
)
from greenonet.config import ComplexAdmissibilityGluingConfig
from test.test_make_annular_geometry import (
    AnnularGeometryBuilder,
    AnnularGeometryConfig,
)


def _annulus_geometry(tmp_path: Path):
    path = tmp_path / "annulus.npz"
    AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.125,
            out=path,
        )
    ).write()
    return load_complex_geometry(path)


def test_annulus_context_detects_transition_only_carriers(tmp_path):
    geometry = _annulus_geometry(tmp_path)
    config = ComplexAdmissibilityGluingConfig(enabled=True)

    context = build_complex_gluing_context(geometry, config)

    assert context.x.interface_count == 14
    assert context.y.interface_count == 14
    assert context.x.transition_interface_count > 0
    assert context.y.transition_interface_count > 0
    assert torch.any(context.x.transition_mask)
    assert torch.any(context.y.transition_mask)
    assert context.x.carrier_self_index.numel() > 0
    assert context.y.carrier_self_index.numel() > 0
    assert torch.all(context.x.transition_mask[context.x.carrier_self_index])
    assert torch.all(context.y.transition_mask[context.y.carrier_self_index])
    assert context.x.boundary_coords.shape[0] > 0
    assert context.y.boundary_coords.shape[0] > 0


def test_affine_transverse_profile_has_zero_self_trace_loss(tmp_path):
    geometry = _annulus_geometry(tmp_path)
    config = ComplexAdmissibilityGluingConfig(
        enabled=True,
        transition_carrier_weight=0.0,
    )
    context = build_complex_gluing_context(geometry, config)
    coords = geometry.coords_valid
    affine = (1.0 + 2.0 * coords[:, 0] + 3.0 * coords[:, 1]).unsqueeze(0)

    result = complex_admissibility_gluing_loss(
        u_phi_valid=affine,
        u_psi_valid=affine,
        a_valid=torch.ones_like(affine),
        context=context,
        config=config,
    )

    torch.testing.assert_close(result.self_loss, torch.zeros_like(result.self_loss))
    torch.testing.assert_close(result.x_self_rms, torch.zeros_like(result.x_self_rms))
    torch.testing.assert_close(result.y_self_rms, torch.zeros_like(result.y_self_rms))


def test_slice_jump_produces_positive_self_trace_loss_and_gradient(tmp_path):
    geometry = _annulus_geometry(tmp_path)
    config = ComplexAdmissibilityGluingConfig(
        enabled=True,
        transition_carrier_weight=0.0,
    )
    context = build_complex_gluing_context(geometry, config)
    coords = geometry.coords_valid
    values = (coords[:, 0] + coords[:, 1]).unsqueeze(0)
    values = values + (coords[:, 1] >= 0.5).to(values.dtype).unsqueeze(0)
    u_phi = values.clone().requires_grad_(True)

    result = complex_admissibility_gluing_loss(
        u_phi_valid=u_phi,
        u_psi_valid=torch.zeros_like(u_phi),
        a_valid=torch.ones_like(u_phi),
        context=context,
        config=config,
    )
    result.loss.backward()

    assert result.self_loss.item() > 0.0
    assert u_phi.grad is not None
    assert torch.all(torch.isfinite(u_phi.grad))


def test_transition_carrier_uses_orthogonal_solution_and_boundary_zero(tmp_path):
    geometry = _annulus_geometry(tmp_path)
    config = ComplexAdmissibilityGluingConfig(
        enabled=True,
        self_trace_weight=0.0,
        transition_carrier_weight=1.0,
    )
    context = build_complex_gluing_context(geometry, config)
    u_phi = torch.ones((1, geometry.num_points), dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)

    result = complex_admissibility_gluing_loss(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        a_valid=torch.ones_like(u_phi),
        context=context,
        config=config,
    )

    assert result.carrier_transition.item() > 0.0
    assert result.x_carrier_residual.shape[-1] == (
        2 * context.x.carrier_coords.shape[0] + context.x.boundary_coords.shape[0]
    )
    assert result.y_carrier_residual.shape[-1] == (
        2 * context.y.carrier_coords.shape[0] + context.y.boundary_coords.shape[0]
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("trace_order", 0, "supports only 1"),
        ("carrier_scope", "all", "transition_only"),
        ("transition_fraction", 0.0, "strictly between"),
        ("self_trace_weight", -1.0, "non-negative"),
        ("eps", 0.0, "positive"),
    ],
)
def test_gluing_config_rejects_invalid_values(field, value, error):
    with pytest.raises((TypeError, ValueError), match=error):
        ComplexAdmissibilityGluingConfig(**{field: value})


def test_gluing_config_rejects_unknown_key():
    with pytest.raises(TypeError, match="unknown keys"):
        ComplexAdmissibilityGluingConfig.from_raw({"unknown": 1})
