from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_mismatch_blend_diagnostics import (
    CrossAxisBlendComparisonRequest,
    CrossAxisBlendEstimatorComparison,
    MismatchGradientBlendConfig,
    MismatchSeamC2BlendConfig,
    run_cross_axis_blend_estimator_comparison,
)
from greenonet.complex_smooth_blend_diagnostics import (
    FixedSmoothBlendConfig,
    FixedSmoothBlendDiagnosticRequest,
    FixedSmoothCrossAxisBlendDiagnostic,
    run_fixed_smooth_cross_axis_blend_diagnostic,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingModelConfig,
    ModelConfig,
    TransverseTrunkConfig,
)
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.model import GreenONetModel
from test.complex_fixtures import (
    write_coefficients,
    write_complex_config,
    write_geometry_npz,
    write_sample_npz,
)


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def _load_annular_geometry_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_annular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "smooth_blend_make_annular_geometry",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_annular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_annular_geometry(tmp_path: Path) -> Path:
    module = _load_annular_geometry_module()
    path = tmp_path / "annulus.npz"
    module.AnnularGeometryBuilder(
        module.AnnularGeometryConfig(
            inner_radius=0.4,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        )
    ).write()
    return path


def _write_diagnostic_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coefficient_path = write_coefficients(tmp_path / "coefficients.py")
    test_path = tmp_path / "test"
    write_sample_npz(test_path)

    coupling_config = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        axis_1d_trunk=Axis1DTrunkConfig(
            enabled=True,
            num_frequencies=2,
            max_frequency=2.0,
            transverse_trunk=TransverseTrunkConfig(
                enabled=True,
                fusion="product",
                length_context=True,
            ),
        ),
    )
    green_config = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_checkpoint = tmp_path / "coupling.safetensors"
    green_checkpoint = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_config).state_dict(),
        coupling_checkpoint,
    )
    save_model_with_config(
        GreenONetModel(green_config),
        green_config,
        green_checkpoint,
    )
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=test_path,
        coefficient_path=coefficient_path,
    )
    payload = json.loads(config_path.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_path.write_text(json.dumps(payload))
    return (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
    )


def test_fixed_smooth_blend_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="alpha"):
        FixedSmoothBlendConfig(alpha=0.0)
    with pytest.raises(ValueError, match="smoothing_steps"):
        FixedSmoothBlendConfig(smoothing_steps=-1)
    with pytest.raises(ValueError, match="smoothing_relaxation"):
        FixedSmoothBlendConfig(smoothing_relaxation=1.1)
    with pytest.raises(ValueError, match="transition_dilation_steps"):
        FixedSmoothBlendConfig(transition_dilation_steps=-1)
    with pytest.raises(ValueError, match="weight_construction"):
        FixedSmoothBlendConfig(weight_construction="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ramp_gamma"):
        FixedSmoothBlendConfig(ramp_gamma=1.1)
    with pytest.raises(ValueError, match="ramp_width"):
        FixedSmoothBlendConfig(ramp_width=0.0)


def test_mismatch_gradient_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="gamma"):
        MismatchGradientBlendConfig(gamma=1.1)
    with pytest.raises(ValueError, match="smoothing_steps"):
        MismatchGradientBlendConfig(smoothing_steps=-1)
    with pytest.raises(ValueError, match="smoothing_relaxation"):
        MismatchGradientBlendConfig(smoothing_relaxation=0.0)
    with pytest.raises(ValueError, match="activation_lower"):
        MismatchGradientBlendConfig(activation_lower=-0.1)
    with pytest.raises(ValueError, match="activation_upper"):
        MismatchGradientBlendConfig(
            activation_lower=0.2,
            activation_upper=0.2,
        )
    with pytest.raises(ValueError, match="scale_eps"):
        MismatchGradientBlendConfig(scale_eps=0.0)


def test_mismatch_seam_c2_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="gamma"):
        MismatchSeamC2BlendConfig(gamma=1.1)
    with pytest.raises(ValueError, match="ramp_width"):
        MismatchSeamC2BlendConfig(ramp_width=0.0)
    with pytest.raises(ValueError, match="max_seams_per_axis"):
        MismatchSeamC2BlendConfig(max_seams_per_axis=0)
    with pytest.raises(ValueError, match="peak_relative_threshold"):
        MismatchSeamC2BlendConfig(peak_relative_threshold=0.0)
    with pytest.raises(ValueError, match="profile_smoothing_steps"):
        MismatchSeamC2BlendConfig(profile_smoothing_steps=-1)
    with pytest.raises(ValueError, match="minimum_separation"):
        MismatchSeamC2BlendConfig(minimum_separation=0.0)
    with pytest.raises(ValueError, match="scale_eps"):
        MismatchSeamC2BlendConfig(scale_eps=0.0)


def test_mismatch_gradient_sensor_detects_jump_orientation(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    config = MismatchGradientBlendConfig(
        gamma=1.0,
        smoothing_steps=0,
        activation_lower=0.0,
        activation_upper=0.1,
    )
    zeros = torch.zeros((1, geometry.num_points), dtype=torch.float64)
    vertical = CrossAxisBlendEstimatorComparison.build_mismatch_gradient_fields(
        geometry,
        torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64),
        zeros,
        config,
    )
    horizontal = CrossAxisBlendEstimatorComparison.build_mismatch_gradient_fields(
        geometry,
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
        zeros,
        config,
    )
    constant = CrossAxisBlendEstimatorComparison.build_mismatch_gradient_fields(
        geometry,
        torch.ones_like(zeros),
        zeros,
        config,
    )

    assert torch.all(vertical.theta[0, :2] > 0.0)
    assert torch.all(vertical.w_phi[0, :2] > 0.5)
    assert torch.all(horizontal.theta[0, [0, 2]] < 0.0)
    assert torch.all(horizontal.w_psi[0, [0, 2]] > 0.5)
    torch.testing.assert_close(
        constant.theta,
        torch.zeros_like(constant.theta),
    )
    torch.testing.assert_close(
        constant.w_phi,
        torch.full_like(constant.w_phi, 0.5),
    )
    torch.testing.assert_close(
        vertical.w_phi + vertical.w_psi,
        torch.ones_like(vertical.w_phi),
    )


def test_mismatch_seam_c2_detects_location_then_builds_smooth_profile(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(_write_annular_geometry(tmp_path))
    x = geometry.coords_valid[:, 0]
    mismatch = (x.abs() < 0.4).to(torch.float64).unsqueeze(0)
    zeros = torch.zeros_like(mismatch)
    direct = CrossAxisBlendEstimatorComparison.build_mismatch_gradient_fields(
        geometry,
        mismatch,
        zeros,
        MismatchGradientBlendConfig(
            gamma=0.5,
            smoothing_steps=0,
            activation_lower=0.0,
            activation_upper=0.1,
        ),
    )
    seam = CrossAxisBlendEstimatorComparison.build_mismatch_seam_c2_fields(
        geometry,
        direct,
        MismatchSeamC2BlendConfig(
            gamma=0.5,
            ramp_width=0.5,
            max_seams_per_axis=2,
            peak_relative_threshold=0.2,
            profile_smoothing_steps=0,
            minimum_separation=0.5,
        ),
    )

    assert seam.x_seam_counts.tolist() == [2]
    assert seam.y_seam_counts.tolist() == [0]
    torch.testing.assert_close(
        torch.sort(seam.x_seam_coordinates[0]).values,
        torch.tensor([-0.375, 0.375], dtype=torch.float64),
    )
    assert torch.all(torch.isnan(seam.y_seam_coordinates))
    torch.testing.assert_close(
        seam.w_phi + seam.w_psi,
        torch.ones_like(seam.w_phi),
    )
    assert torch.all(seam.w_phi[seam.support_mask] >= 0.5)
    assert torch.all(seam.w_phi[~seam.support_mask] == 0.5)


def test_mismatch_seam_c2_constant_mismatch_falls_back_to_equal_mean(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    u_phi = torch.ones((1, geometry.num_points), dtype=torch.float64)
    u_psi = torch.zeros_like(u_phi)
    direct = CrossAxisBlendEstimatorComparison.build_mismatch_gradient_fields(
        geometry,
        u_phi,
        u_psi,
        MismatchGradientBlendConfig(),
    )
    seam = CrossAxisBlendEstimatorComparison.build_mismatch_seam_c2_fields(
        geometry,
        direct,
        MismatchSeamC2BlendConfig(),
    )

    assert seam.x_seam_counts.tolist() == [0]
    assert seam.y_seam_counts.tolist() == [0]
    torch.testing.assert_close(
        seam.w_phi,
        torch.full_like(seam.w_phi, 0.5),
    )
    torch.testing.assert_close(
        seam.w_psi,
        torch.full_like(seam.w_psi, 0.5),
    )


def test_compact_c2_ramp_has_exact_endpoint_properties() -> None:
    distance = torch.tensor(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        dtype=torch.float64,
    )
    bump = FixedSmoothCrossAxisBlendDiagnostic._compact_c2_bump(
        distance,
        width=1.0,
    )

    torch.testing.assert_close(
        bump,
        torch.tensor(
            [1.0, 0.896484375, 0.5, 0.103515625, 0.0, 0.0],
            dtype=torch.float64,
        ),
    )
    assert torch.all(bump[:-1] >= bump[1:])


def test_compact_c2_ramp_uses_topology_transition_distance(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(_write_annular_geometry(tmp_path))
    config = FixedSmoothBlendConfig(
        weight_construction="compact_c2_ramp",
        ramp_gamma=0.5,
        ramp_width=0.25,
        transition_dilation_steps=0,
    )
    fields = FixedSmoothCrossAxisBlendDiagnostic.build_fixed_blend_fields(
        geometry,
        config,
    )

    torch.testing.assert_close(
        fields.phi_transition_coordinates,
        torch.tensor([-0.375, 0.375], dtype=torch.float64),
    )
    torch.testing.assert_close(
        fields.psi_transition_coordinates,
        torch.tensor([-0.375, 0.375], dtype=torch.float64),
    )
    torch.testing.assert_close(
        fields.w_phi + fields.w_psi,
        torch.ones_like(fields.w_phi),
    )
    assert fields.w_phi.min().item() >= 0.25 - 1.0e-12
    assert fields.w_phi.max().item() <= 0.75 + 1.0e-12
    outside = ~fields.ramp_support_mask
    assert torch.any(outside)
    torch.testing.assert_close(
        fields.w_phi[outside],
        torch.full_like(fields.w_phi[outside], 0.5),
    )
    torch.testing.assert_close(
        fields.w_psi[outside],
        torch.full_like(fields.w_psi[outside], 0.5),
    )

    phi_only = (fields.influence_phi > 0.0) & (fields.influence_psi == 0.0)
    psi_only = (fields.influence_psi > 0.0) & (fields.influence_phi == 0.0)
    assert torch.any(phi_only)
    assert torch.any(psi_only)
    assert torch.all(fields.w_phi[phi_only] < 0.5)
    assert torch.all(fields.w_phi[psi_only] > 0.5)

    zero_gamma = FixedSmoothCrossAxisBlendDiagnostic.build_fixed_blend_fields(
        geometry,
        FixedSmoothBlendConfig(
            weight_construction="compact_c2_ramp",
            ramp_gamma=0.0,
            ramp_width=0.25,
            transition_dilation_steps=0,
        ),
    )
    torch.testing.assert_close(
        zero_gamma.w_phi,
        torch.full_like(zero_gamma.w_phi, 0.5),
    )
    torch.testing.assert_close(
        zero_gamma.w_psi,
        torch.full_like(zero_gamma.w_psi, 0.5),
    )


def test_paired_bootstrap_summary_is_deterministic_and_paired() -> None:
    rows = [
        {
            "baseline_rel_sol": 2.0,
            "blend_rel_sol": 1.0,
            "baseline_transition_error_rms": 4.0,
            "blend_transition_error_rms": 2.0,
            "baseline_transition_trace_error_jump_rms": 6.0,
            "blend_transition_trace_error_jump_rms": 3.0,
        },
        {
            "baseline_rel_sol": 4.0,
            "blend_rel_sol": 2.0,
            "baseline_transition_error_rms": 8.0,
            "blend_transition_error_rms": 4.0,
            "baseline_transition_trace_error_jump_rms": 12.0,
            "blend_transition_trace_error_jump_rms": 6.0,
        },
    ]
    first = FixedSmoothCrossAxisBlendDiagnostic._paired_bootstrap_summary(
        rows,  # type: ignore[arg-type]
        draws=1000,
        seed=7,
    )
    second = FixedSmoothCrossAxisBlendDiagnostic._paired_bootstrap_summary(
        rows,  # type: ignore[arg-type]
        draws=1000,
        seed=7,
    )

    assert first == second
    for metric in first["metrics"].values():
        assert metric["observed_relative_change"] == pytest.approx(-0.5)
        assert metric["relative_change_ci95"] == pytest.approx([-0.5, -0.5])
        assert metric["bootstrap_probability_improvement"] == pytest.approx(1.0)


def test_geometry_only_weights_downweight_chart_with_transverse_length_jump(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    fields = FixedSmoothCrossAxisBlendDiagnostic.build_fixed_blend_fields(
        geometry,
        FixedSmoothBlendConfig(
            alpha=1.0 / math.log(2.0),
            smoothing_steps=0,
            transition_dilation_steps=0,
        ),
    )

    assert fields.j_phi_raw[0].item() > 0.0
    assert fields.j_phi_raw[2].item() > 0.0
    torch.testing.assert_close(fields.j_psi_raw, torch.zeros_like(fields.j_psi_raw))
    assert fields.w_phi[0].item() < 0.5
    assert fields.w_phi[2].item() < 0.5
    assert fields.w_psi[0].item() > 0.5
    torch.testing.assert_close(
        fields.w_phi + fields.w_psi,
        torch.ones_like(fields.w_phi),
    )
    assert fields.transition_point_mask.tolist() == [True, False, True]


def test_fixed_blend_is_partition_of_unity_and_sample_independent(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    config = FixedSmoothBlendConfig(
        smoothing_steps=1,
        transition_dilation_steps=0,
    )
    first = FixedSmoothCrossAxisBlendDiagnostic.build_fixed_blend_fields(
        geometry,
        config,
    )
    second = FixedSmoothCrossAxisBlendDiagnostic.build_fixed_blend_fields(
        geometry,
        config,
    )

    torch.testing.assert_close(first.w_phi, second.w_phi)
    torch.testing.assert_close(first.w_psi, second.w_psi)
    assert torch.all(first.w_phi >= 0.0)
    assert torch.all(first.w_phi <= 1.0)
    torch.testing.assert_close(
        first.w_phi + first.w_psi,
        torch.ones_like(first.w_phi),
    )


def test_checkpoint_backed_diagnostic_writes_expected_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
    ) = _write_diagnostic_fixture(tmp_path)
    outdir = tmp_path / "diagnostic"

    summary = run_fixed_smooth_cross_axis_blend_diagnostic(
        FixedSmoothBlendDiagnosticRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            selected_samples=(0,),
            batch_size=1,
            blend=FixedSmoothBlendConfig(
                smoothing_steps=0,
                transition_dilation_steps=0,
            ),
        )
    )

    assert summary["diagnostic"] == ("fixed_smooth_cross_axis_reconstruction_blend")
    assert summary["status"] == "post_hoc_diagnostic_only"
    assert summary["production_code_changed"] is False
    assert summary["training_or_checkpoint_changed"] is False
    assert summary["weight_construction"]["sample_independent"] is True
    assert summary["weight_construction"]["uses_sol"] is False
    assert summary["metric_role"] == "evaluation_only_full_reference_test"
    assert summary["paired_bootstrap"]["draws"] == 100_000
    assert summary["num_samples"] == 1
    assert summary["selected_samples"] == [0]
    assert (outdir / "summary.json").is_file()
    assert (outdir / "diagnosis_report.md").is_file()
    assert (outdir / "metrics" / "per_sample_blend_comparison.csv").is_file()
    raw_path = outdir / "data" / "selected_fixed_smooth_blend_arrays.npz"
    assert raw_path.is_file()
    with np.load(raw_path) as raw:
        assert {
            "distance_phi",
            "distance_psi",
            "influence_phi",
            "influence_psi",
            "theta",
            "w_phi",
            "w_psi",
            "u_baseline",
            "u_blend",
            "baseline_error",
            "blend_error",
        }.issubset(raw.files)
        np.testing.assert_allclose(raw["w_phi"] + raw["w_psi"], 1.0)
    assert (outdir / "figures" / "geometry" / "fixed_blend_fields.json").is_file()
    assert (
        outdir / "figures" / "selected" / "sample_0000_blend_comparison.json"
    ).is_file()


def test_checkpoint_backed_four_estimator_comparison_writes_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
    ) = _write_diagnostic_fixture(tmp_path)
    outdir = tmp_path / "comparison"

    summary = run_cross_axis_blend_estimator_comparison(
        CrossAxisBlendComparisonRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            selected_samples=(0,),
            batch_size=1,
            blend=FixedSmoothBlendConfig(
                smoothing_steps=0,
                transition_dilation_steps=0,
            ),
            mismatch=MismatchGradientBlendConfig(
                smoothing_steps=0,
                activation_lower=0.0,
                activation_upper=0.1,
            ),
            seam_c2=MismatchSeamC2BlendConfig(
                ramp_width=0.5,
                profile_smoothing_steps=0,
            ),
            seam_sweep=True,
            seam_sweep_gammas=(0.5,),
            seam_sweep_width_steps=(2.0,),
            seam_sweep_peak_thresholds=(0.25,),
        )
    )

    assert summary["diagnostic"] == (
        "cross_axis_reconstruction_blend_estimator_comparison"
    )
    assert summary["status"] == "post_hoc_diagnostic_only"
    assert summary["production_code_changed"] is False
    assert summary["estimators"]["mismatch_gradient"]["sample_dependent"] is True
    assert summary["estimators"]["mismatch_gradient"]["uses_sol"] is False
    assert summary["estimators"]["mismatch_gradient"]["uses_line_lengths"] is False
    assert (
        summary["estimators"]["mismatch_detected_seam_c2"][
            "detector_and_weight_profile_separated"
        ]
        is True
    )
    assert (
        summary["estimators"]["mismatch_detected_seam_c2"]["uses_line_lengths"] is False
    )
    assert set(summary["aggregate_metrics"]) == {
        "geometry_only",
        "mismatch_gradient",
        "mismatch_detected_seam_c2",
    }
    assert summary["geometry_vs_mismatch"]["baseline"] == "geometry_only"
    assert summary["geometry_vs_mismatch"]["candidate"] == "mismatch_gradient"
    assert set(summary["geometry_vs_mismatch"]["paired_bootstrap"]["metrics"]) == {
        "rel_sol",
        "transition_error_rms",
        "transition_trace_error_jump_rms",
    }
    assert summary["seam_c2_vs_mismatch"]["candidate"] == ("mismatch_detected_seam_c2")
    assert summary["seam_c2_parameter_sweep"]["row_count"] == 1
    assert (outdir / "metrics" / "per_sample_estimator_comparison.csv").is_file()
    assert (outdir / "metrics" / "seam_c2_parameter_sweep.csv").is_file()
    raw_path = outdir / "data" / "selected_cross_axis_blend_comparison_arrays.npz"
    assert raw_path.is_file()
    with np.load(raw_path) as raw:
        assert {
            "u_equal_mean",
            "u_geometry_blend",
            "u_mismatch_blend",
            "mismatch",
            "mismatch_j_x",
            "mismatch_j_y",
            "mismatch_activation",
            "mismatch_theta",
            "mismatch_w_phi",
            "mismatch_w_psi",
            "u_seam_c2_blend",
            "seam_c2_x_edge_profile",
            "seam_c2_y_edge_profile",
            "seam_c2_x_seam_coordinates",
            "seam_c2_y_seam_coordinates",
            "seam_c2_influence_x",
            "seam_c2_influence_y",
            "seam_c2_theta",
            "seam_c2_w_phi",
            "seam_c2_w_psi",
        }.issubset(raw.files)
        np.testing.assert_allclose(
            raw["mismatch_w_phi"] + raw["mismatch_w_psi"],
            1.0,
        )
        np.testing.assert_allclose(
            raw["seam_c2_w_phi"] + raw["seam_c2_w_psi"],
            1.0,
        )
    assert (outdir / "figures" / "aggregate" / "four_estimator_rel_sol.json").is_file()
    assert (
        outdir / "figures" / "selected" / "sample_0000_four_estimator_comparison.json"
    ).is_file()
