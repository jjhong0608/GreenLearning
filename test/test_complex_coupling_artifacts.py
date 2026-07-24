from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_artifacts import (
    ComplexCouplingArtifactExporter,
    export_complex_coupling_artifacts,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexPreProjectionFusionConfig,
    CouplingCoefficientTermsConfig,
    CouplingModelConfig,
    ModelConfig,
    TransverseTrunkConfig,
)
from greenonet.coupling_artifacts import CouplingArtifactRequest
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


def _marker_for_field(outdir: Path, field: str) -> dict:
    figure = json.loads(
        (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).read_text()
    )
    return figure["data"][0]["marker"]


def _coefficient_figure(outdir: Path, field: str) -> dict:
    return json.loads(
        (outdir / "figures" / "coefficients" / f"{field}.json").read_text()
    )


def _write_zero_coefficients(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return torch.ones_like(x)",
                "def apx_fun(x, y): return torch.zeros_like(x)",
                "def apy_fun(x, y): return torch.zeros_like(x)",
                "def bx_fun(x, y): return torch.zeros_like(x)",
                "def by_fun(x, y): return torch.zeros_like(x)",
                "def c_fun(x, y): return torch.zeros_like(x)",
            ]
        )
    )
    return path


def test_complex_artifact_export_writes_outputs_without_cross_fields(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        pre_projection_fusion=ComplexPreProjectionFusionConfig(
            enabled=True,
            nonlinear_hidden_dim=8,
            nonlinear_depth=1,
            gate_initial_value=0.05,
        ),
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=True,
            convection=True,
            reaction=True,
        ),
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(), coupling_path
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coeff_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["coefficient_terms"] = {
        "diffusion": True,
        "convection": True,
        "reaction": True,
    }
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_payload["coupling_model"]["pre_projection_fusion"] = {
        "enabled": True,
        "nonlinear_hidden_dim": 8,
        "nonlinear_depth": 1,
        "gate_initial_value": 0.05,
        "eps": 1e-12,
    }
    config_payload["coupling_training"]["relative_split_consistency"] = {
        "enabled": True,
        "weight": 2.0,
        "mass_weight": 3.0,
        "eps": 1e-12,
    }
    config_payload["coupling_training"]["weak_operator_closure"] = {
        "enabled": True,
        "weight": 4.0,
        "eps": 1e-12,
    }
    config_payload["coupling_training"]["optimizer"] = {
        "name": "soap",
        "betas": [0.95, 0.95],
        "profile_step_time": True,
        "soap": {
            "precondition_frequency": 10,
            "max_precondition_dim": 64,
        },
    }
    config_payload["coupling_training"]["best_physics_checkpoint"] = {"enabled": True}
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            device="cpu",
            theme="plotly_white",
            coefficient_vector_max_points=2,
        )
    )

    assert summary["geometry_mode"] == "complex"
    assert summary["selected_samples"] == [0]
    assert "cross" not in json.dumps(summary)
    assert (outdir / "summary.json").exists()
    assert (outdir / "metrics" / "per_sample_metrics.csv").exists()
    assert (outdir / "data" / "selected_raw_arrays.npz").exists()
    expected_figure_fields = {
        "rhs",
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "u_split_mismatch",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
        "weak_residual_x",
        "weak_residual_y",
        "split_mass_relative_contribution",
        "base_physical_difference",
        "fused_physical_difference",
        "linear_difference_correction",
        "nonlinear_difference_correction",
        "blended_difference_correction",
    }
    assert set(summary["figure_fields"]) == expected_figure_fields
    assert summary["error_convention"] == "signed_difference"
    assert summary["solution_prediction"] == "u_pred=0.5*(u_phi+u_psi)"
    assert summary["raw_output_space"] == "reference_response"
    assert summary["output_contract_version"] == 6
    assert summary["optimizer"]["name"] == "soap"
    assert summary["optimizer"]["betas"] == (0.95, 0.95)
    assert summary["optimizer"]["soap"]["precondition_frequency"] == 10
    assert summary["optimizer"]["upstream_commit"] == (
        "a1e553530fde97d0e6b307d7c82ac6d38b072340"
    )
    assert summary["balance_projection"]["enabled"] is True
    assert summary["balance_projection"]["mode"] == "physical_symmetric"
    assert summary["balance_projection"]["space"] == "physical_source"
    assert summary["balance_projection"]["uses_reference_targets"] is False
    assert "p=P_raw/Lx^2" in summary["balance_projection"]["formula"]
    assert summary["pre_projection_fusion"]["enabled"] is True
    assert summary["pre_projection_fusion"]["space"] == ("physical_directional_source")
    assert summary["pre_projection_fusion"]["correction_mode"] == (
        "antisymmetric_difference"
    )
    assert summary["pre_projection_fusion"]["common_mode_preserved"] is True
    assert summary["pre_projection_fusion"]["gate_value"] == pytest.approx(0.05)
    assert summary["pre_projection_fusion"]["uses_reference_targets"] is False
    assert summary["reconstruction_response_input"] == {
        "phi": "projected Phi is used directly",
        "psi": "projected Psi is used directly",
        "additional_length_scaling": False,
    }
    assert summary["reference_targets_used_for_training"] is False
    assert summary["canonical_boundary_energy"] == {
        "enabled": True,
        "definition": "endpoint_p1_edge",
        "formula": "a_i * r_i^2 * h_perp / d_endpoint",
        "coefficient_evaluation": "one_sided_nearest_valid_point",
        "endpoint_value": 0.0,
        "anchor_count": 8,
        "x_anchor_count": 4,
        "y_anchor_count": 4,
        "covers_all_connected_segment_endpoints": True,
        "uses_reference_targets": False,
    }
    assert summary["canonical_energy"] == {
        "enabled": True,
        "domain": "all_valid_same_segment_edges",
        "bulk_formula": (
            "sum_edges arithmetic_mean(a)*(delta(u_phi-u_psi)/h_axis)^2*hx*hy"
        ),
        "boundary_included": True,
        "transition_partition": False,
        "checkpoint_metric": "loss_energy_consistency",
        "uses_reference_targets": False,
    }
    assert "length_jump_balance" not in summary
    assert summary["relative_split_consistency"] == {
        "enabled": True,
        "weight": 2.0,
        "mass_weight": 3.0,
        "eps": 1e-12,
        "source_normalization": "physical_rhs_l2_squared",
        "domain_length_scale": "max_global_extent",
        "uses_reference_targets": False,
    }
    assert summary["weak_operator_closure"] == {
        "enabled": True,
        "weight": 4.0,
        "eps": 1e-12,
        "trial_solution": "u_pred=0.5*(u_phi+u_psi)",
        "test_space": "directional_segment_p1_nodal",
        "coefficient_evaluation": "direct_at_physical_element_midpoints",
        "reaction_split": "c/2_per_direction",
        "uses_reference_targets": False,
    }
    assert summary["checkpoint_selection"] == {
        "best_energy": True,
        "best_physics": True,
        "reference_metric_used": False,
    }
    assert (
        summary["non_error_color_range_policy"] == "shared_reference_prediction_groups"
    )
    assert summary["non_error_color_range_groups"]["solution"] == [
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
    ]
    assert summary["optional_flux_targets_exported"] is True
    assert summary["coefficient_figure_fields"] == [
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
        "convection_vector",
    ]
    assert summary["coefficient_figure_count"] == 6
    assert summary["figure_count"] == len(expected_figure_fields) + 6
    assert summary["coefficient_field_space"] == "physical"
    assert summary["coefficient_evaluation"] == "direct_at_coords_valid"
    assert summary["coefficient_raw_archive"] == "data/coefficient_fields.npz"
    assert summary["coefficient_vector"]["max_points"] == 2
    assert 0 < summary["coefficient_vector"]["selected_points"] <= 2
    assert summary["coefficient_vector"]["background_points"] == 3
    assert summary["coefficient_field_statistics"]["a"] == {
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
        "physical_nonzero": True,
        "constant": True,
        "branch_enabled": True,
        "figure_exported": True,
    }
    assert summary["coefficient_field_statistics"]["bx"]["min"] == 4.0
    assert summary["coefficient_field_statistics"]["by"]["max"] == 5.0
    assert summary["coefficient_field_statistics"]["c"]["mean"] == 6.0
    assert summary["coefficient_branch_channel_order"] == [
        "a",
        "b_primary",
        "b_transverse",
        "c",
    ]
    assert summary["coefficient_branch_convection"] == "primary_transverse"
    assert (
        summary["coefficient_branch_transverse_convection_scaling"]
        == "primary_segment_length"
    )
    assert summary["transverse_trunk"] == {
        "enabled": True,
        "fusion": "product",
        "length_context": True,
        "features": [
            "t_perpendicular",
            "log(L_perpendicular/L_ref)",
            "log(L_parallel/L_perpendicular)",
            "kappa",
        ],
    }
    for field in expected_figure_fields:
        assert (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).exists()
    for field in summary["coefficient_figure_fields"]:
        assert (outdir / "figures" / "coefficients" / f"{field}.json").exists()

    with (outdir / "metrics" / "per_sample_metrics.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert rows
    assert all("cross" not in key for key in rows[0])

    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    assert any(key.endswith("_raw_response_phi") for key in raw.files)
    assert any(key.endswith("_raw_response_psi") for key in raw.files)
    assert any(key.endswith("_raw_physical_phi") for key in raw.files)
    assert any(key.endswith("_raw_physical_psi") for key in raw.files)
    assert any(key.endswith("_projected_response_phi") for key in raw.files)
    assert any(key.endswith("_projected_response_psi") for key in raw.files)
    assert any(key.endswith("_x_length_squared") for key in raw.files)
    assert any(key.endswith("_y_length_squared") for key in raw.files)
    assert any(key.endswith("_raw_difference") for key in raw.files)
    assert any(key.endswith("_projected_difference") for key in raw.files)
    assert any(key.endswith("_raw_response_constraint_residual") for key in raw.files)
    assert any(key.endswith("_response_constraint_residual") for key in raw.files)
    assert any(key.endswith("_base_raw_response_phi") for key in raw.files)
    assert any(key.endswith("_base_raw_response_psi") for key in raw.files)
    assert any(key.endswith("_base_physical_p") for key in raw.files)
    assert any(key.endswith("_base_physical_q") for key in raw.files)
    assert any(key.endswith("_base_physical_difference") for key in raw.files)
    assert any(key.endswith("_fused_physical_p") for key in raw.files)
    assert any(key.endswith("_fused_physical_q") for key in raw.files)
    assert any(key.endswith("_fused_physical_difference") for key in raw.files)
    assert any(key.endswith("_linear_difference_correction") for key in raw.files)
    assert any(key.endswith("_nonlinear_difference_correction") for key in raw.files)
    assert any(key.endswith("_blended_difference_correction") for key in raw.files)
    assert any(key.endswith("_fusion_source_scale") for key in raw.files)
    assert any(key.endswith("_fusion_gate") for key in raw.files)
    assert not any(key.endswith("_x_length_jump_score") for key in raw.files)
    assert not any(key.endswith("_y_length_jump_score") for key in raw.files)
    assert not any(key.endswith("_x_transition_edge_mask") for key in raw.files)
    assert not any(key.endswith("_y_transition_edge_mask") for key in raw.files)
    assert any(key.endswith("_x_transverse_length_context") for key in raw.files)
    assert any(key.endswith("_y_transverse_length_context") for key in raw.files)
    assert any(key.endswith("_weak_residual_x") for key in raw.files)
    assert any(key.endswith("_weak_residual_y") for key in raw.files)
    assert any(key.endswith("_weak_nodal_mass_x") for key in raw.files)
    assert any(key.endswith("_weak_nodal_mass_y") for key in raw.files)
    assert any(key.endswith("_split_mass_relative_contribution") for key in raw.files)
    assert any(key.endswith("_boundary_endpoint_coords") for key in raw.files)
    assert any(key.endswith("_boundary_split_residual") for key in raw.files)
    assert any(key.endswith("_boundary_physical_distance") for key in raw.files)
    for suffix in (
        "_u_pred",
        "_u_pred_error",
        "_u_phi_error",
        "_u_psi_error",
        "_u_split_mismatch",
        "_target_phi",
        "_target_psi",
        "_phi_error",
        "_psi_error",
    ):
        assert any(key.endswith(suffix) for key in raw.files)
    assert all("cross" not in key for key in raw.files)

    coefficient_raw = np.load(outdir / "data" / "coefficient_fields.npz")
    assert set(coefficient_raw.files) == {
        "coords_valid",
        "a",
        "bx",
        "by",
        "b_magnitude",
        "c",
        "quiver_indices",
    }
    np.testing.assert_allclose(coefficient_raw["a"], 1.0)
    np.testing.assert_allclose(coefficient_raw["bx"], 4.0)
    np.testing.assert_allclose(coefficient_raw["by"], 5.0)
    np.testing.assert_allclose(coefficient_raw["b_magnitude"], np.sqrt(41.0))
    np.testing.assert_allclose(coefficient_raw["c"], 6.0)
    assert coefficient_raw["quiver_indices"].size <= 2

    bx_marker = _coefficient_figure(outdir, "convection_bx")["data"][0]["marker"]
    assert bx_marker["cmin"] == -4.0
    assert bx_marker["cmax"] == 4.0
    vector_figure = _coefficient_figure(outdir, "convection_vector")
    assert vector_figure["data"][0]["type"] == "scattergl"
    assert vector_figure["data"][0]["showlegend"] is False
    arrow_trace = vector_figure["data"][-1]
    arrow_dx = arrow_trace["x"][1] - arrow_trace["x"][0]
    arrow_dy = arrow_trace["y"][1] - arrow_trace["y"][0]
    assert arrow_dx / arrow_dy == pytest.approx(4.0 / 5.0)

    error_figure = json.loads(
        (
            outdir
            / "figures"
            / "u_pred_error"
            / "sample_0000_sample_0000_u_pred_error.json"
        ).read_text()
    )
    marker = error_figure["data"][0]["marker"]
    assert marker["colorscale"]
    assert marker["cmin"] == -marker["cmax"]

    solution_ranges = {
        (
            _marker_for_field(outdir, field)["cmin"],
            _marker_for_field(outdir, field)["cmax"],
        )
        for field in ("sol", "u_pred", "u_phi", "u_psi")
    }
    assert len(solution_ranges) == 1
    phi_range = (
        _marker_for_field(outdir, "target_phi")["cmin"],
        _marker_for_field(outdir, "target_phi")["cmax"],
    )
    assert phi_range == (
        _marker_for_field(outdir, "phi")["cmin"],
        _marker_for_field(outdir, "phi")["cmax"],
    )
    psi_range = (
        _marker_for_field(outdir, "target_psi")["cmin"],
        _marker_for_field(outdir, "target_psi")["cmax"],
    )
    assert psi_range == (
        _marker_for_field(outdir, "psi")["cmin"],
        _marker_for_field(outdir, "psi")["cmax"],
    )


def test_complex_coefficient_artifacts_distinguish_physical_and_branch_activity(
    tmp_path: Path,
) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coefficient_path = _write_zero_coefficients(tmp_path / "coefficients.py")
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
        coefficient_vector_max_points=2,
    )
    exporter = ComplexCouplingArtifactExporter(request)
    geometry = load_complex_geometry(geometry_path)
    coefficients = load_coefficient_functions(coefficient_path)
    fields = exporter._evaluate_coefficient_fields(geometry, coefficients)
    terms = CouplingCoefficientTermsConfig(
        diffusion=False,
        convection=False,
        reaction=False,
    )

    figure_fields = exporter._coefficient_figure_fields(fields, terms)
    statistics = exporter._coefficient_field_statistics(fields, terms, figure_fields)

    assert figure_fields == ("diffusion_a",)
    assert statistics["a"]["physical_nonzero"] is True
    assert statistics["a"]["branch_enabled"] is False
    assert statistics["a"]["figure_exported"] is True
    assert statistics["b_magnitude"]["physical_nonzero"] is False
    assert statistics["b_magnitude"]["figure_exported"] is False
    assert statistics["c"]["physical_nonzero"] is False
    assert statistics["c"]["figure_exported"] is False


def test_complex_coefficient_artifacts_export_enabled_zero_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coefficient_path = _write_zero_coefficients(tmp_path / "coefficients.py")
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
        save_generated_data=False,
        coefficient_vector_max_points=1,
    )
    exporter = ComplexCouplingArtifactExporter(request)
    fields = exporter._evaluate_coefficient_fields(
        load_complex_geometry(geometry_path),
        load_coefficient_functions(coefficient_path),
    )
    terms = CouplingCoefficientTermsConfig(
        diffusion=False,
        convection=True,
        reaction=True,
    )

    paths, figure_fields = exporter._write_coefficient_figures(
        fields,
        terms,
        "plotly_white",
    )
    exporter._write_coefficient_npz(fields)

    assert len(paths) == 6
    assert figure_fields == (
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
        "convection_vector",
    )
    vector_figure = _coefficient_figure(request.outdir, "convection_vector")
    annotations = vector_figure["layout"]["annotations"]
    assert any("Zero convection field" in item["text"] for item in annotations)
    assert not (request.outdir / "data" / "coefficient_fields.npz").exists()
