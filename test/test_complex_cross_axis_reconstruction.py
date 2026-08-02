from __future__ import annotations

from pathlib import Path

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructor,
    LocalWeakResidualReliabilityContext,
)
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_weak_closure import build_directional_weak_context
from greenonet.config import ComplexCrossAxisReconstructionConfig
from test.complex_fixtures import write_coefficients, write_geometry_npz


def _context(tmp_path: Path) -> LocalWeakResidualReliabilityContext:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    geometry = load_complex_geometry(geometry_path)
    coeffs = load_coefficient_functions(
        write_coefficients(tmp_path / "coefficients.py")
    )
    return LocalWeakResidualReliabilityContext.build(
        geometry,
        build_directional_weak_context(geometry, coeffs),
    )


def test_cross_axis_config_defaults_to_disabled_equal_mean() -> None:
    config = ComplexCrossAxisReconstructionConfig()

    assert config.enabled is False
    assert config.mode == "local_weak_residual_reliability"
    assert config.gamma == pytest.approx(0.5)
    assert config.smoothing_steps == 2
    assert config.smoothing_relaxation == pytest.approx(0.5)
    assert config.relative_floor == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"enabled": 1}, "enabled"),
        ({"mode": "geometry_only_compact_c2"}, "mode"),
        ({"mode": "mismatch_detected_seam_c2"}, "mode"),
        ({"gamma": 1.1}, "gamma"),
        ({"smoothing_steps": -1}, "smoothing_steps"),
        ({"smoothing_relaxation": 0.0}, "smoothing_relaxation"),
        ({"relative_floor": -1.0}, "relative_floor"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_cross_axis_config_rejects_invalid_values(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ComplexCrossAxisReconstructionConfig(**kwargs)  # type: ignore[arg-type]


def test_disabled_reconstruction_is_exact_equal_mean_and_builds_no_context(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    reconstructor = ComplexCrossAxisReconstructor(
        ComplexCrossAxisReconstructionConfig(enabled=False)
    )
    u_phi = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, requires_grad=True)
    u_psi = torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    projected = torch.ones((1, 2, 3), dtype=torch.float64)
    geometry = load_complex_geometry(
        write_geometry_npz(tmp_path / "second_geometry.npz")
    )

    result = reconstructor.reconstruct(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        projected_physical=projected,
        geometry=geometry,
        weak_context=context.weak,
    )

    torch.testing.assert_close(result.u_pred_valid, 0.5 * (u_phi + u_psi))
    assert result.mode == "equal_mean"
    assert result.reliability is None
    assert result.u_pred_valid.requires_grad is False
    assert reconstructor.context_build_count == 0


def test_reliability_weights_prefer_lower_indicator_and_form_partition(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    config = ComplexCrossAxisReconstructionConfig(
        enabled=True,
        gamma=1.0,
        smoothing_steps=0,
        relative_floor=0.0,
    )
    phi_indicator = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float64)
    psi_indicator = torch.tensor([[4.0, 1.0, 2.0]], dtype=torch.float64)

    fields = ComplexCrossAxisReconstructor.fields_from_raw_indicators(
        phi_residuals_x=torch.zeros_like(phi_indicator),
        phi_residuals_y=torch.zeros_like(phi_indicator),
        psi_residuals_x=torch.zeros_like(phi_indicator),
        psi_residuals_y=torch.zeros_like(phi_indicator),
        nodal_mass=torch.ones(3, dtype=torch.float64),
        phi_indicator_raw=phi_indicator,
        psi_indicator_raw=psi_indicator,
        context=context,
        config=config,
    )

    assert fields.w_phi[0, 0] > 0.5
    assert fields.w_phi[0, 1] < 0.5
    assert fields.w_phi[0, 2] == pytest.approx(0.5)
    assert fields.w_phi.min() >= 0.0
    assert fields.w_phi.max() <= 1.0
    torch.testing.assert_close(
        fields.w_phi + fields.w_psi, torch.ones_like(fields.w_phi)
    )


def test_locked_graph_smoothing_matches_two_fifty_fifty_steps(tmp_path: Path) -> None:
    context = _context(tmp_path)
    config = ComplexCrossAxisReconstructionConfig(
        enabled=True,
        smoothing_steps=2,
        smoothing_relaxation=0.5,
    )
    values = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float64)

    smoothed = ComplexCrossAxisReconstructor._smooth_indicator(
        values,
        context,
        config,
    )

    # The fixture graph has edges 0-1 and 0-2.
    expected = torch.tensor([[1.0, 1.5, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(smoothed, expected)


def test_gamma_zero_recovers_equal_mean_without_global_solve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(
        write_coefficients(tmp_path / "coefficients.py")
    )
    weak_context = build_directional_weak_context(geometry, coeffs)
    reconstructor = ComplexCrossAxisReconstructor(
        ComplexCrossAxisReconstructionConfig(
            enabled=True,
            gamma=0.0,
            smoothing_steps=0,
        )
    )
    u_phi = torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float64)
    u_psi = torch.tensor([[0.25, -0.25, 0.75]], dtype=torch.float64)
    projected = torch.tensor(
        [[[1.0, 0.5, -0.5], [0.25, 0.75, 1.0]]],
        dtype=torch.float64,
    )

    def fail_solve(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("Cross-axis reliability must not solve a matrix system.")

    monkeypatch.setattr(torch.linalg, "solve", fail_solve)
    result = reconstructor.reconstruct(
        u_phi_valid=u_phi,
        u_psi_valid=u_psi,
        projected_physical=projected,
        geometry=geometry,
        weak_context=weak_context,
    )

    torch.testing.assert_close(result.u_pred_valid, 0.5 * (u_phi + u_psi))
    assert result.reliability is not None
    torch.testing.assert_close(
        result.reliability.w_phi,
        torch.full_like(u_phi, 0.5),
    )
    assert reconstructor.context_build_count == 1
