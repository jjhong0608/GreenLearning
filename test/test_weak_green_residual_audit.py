import torch

from cli.audit_weak_green_residuals import residual_metrics
from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_weak_closure import build_directional_weak_context
from test.complex_fixtures import write_coefficients, write_geometry_npz


def test_weak_diagnostic_scale_invariance_and_signed_sum(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "g.npz"))
    coefficients = load_coefficient_functions(write_coefficients(tmp_path / "c.py"))
    context = build_directional_weak_context(geometry, coefficients)
    fields = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
    pair = torch.ones_like(fields)
    first, raw = residual_metrics(fields, fields.mean(1), pair, pair.sum(1), context)
    second, _ = residual_metrics(
        7 * fields, 7 * fields.mean(1), 7 * pair, 7 * pair.sum(1), context
    )
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    for name in ("phi", "psi", "equal", "weak"):
        torch.testing.assert_close(
            raw[f"{name}_full"], raw[f"{name}_x"] + raw[f"{name}_y"]
        )
    torch.testing.assert_close(first["equal_full"], first["weak_full"])
    torch.testing.assert_close(
        first["own"].square(), 0.5 * (first["phi_x"].square() + first["psi_y"].square())
    )


def test_weak_diagnostic_detects_nonzero_signed_source_defect(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "g.npz"))
    coefficients = load_coefficient_functions(write_coefficients(tmp_path / "c.py"))
    context = build_directional_weak_context(geometry, coefficients)
    fields = torch.zeros((1, 2, 3), dtype=torch.float64)
    pair = torch.ones_like(fields)
    metrics, raw = residual_metrics(fields, fields[:, 0], pair, pair.sum(1), context)
    assert metrics["own"].item() > 0
    assert torch.all(raw["phi_x"] < 0)
    assert torch.all(raw["psi_y"] < 0)
