from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from greenonet.complex_coupling_artifacts import export_complex_coupling_artifacts
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingCoefficientTermsConfig,
    CouplingModelConfig,
    ModelConfig,
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


def test_complex_artifact_export_writes_outputs_without_cross_fields(
    tmp_path, monkeypatch
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
        balance_projection=BalanceProjectionConfig(mode="symmetric"),
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=True,
            convection=True,
            reaction=True,
        ),
        axis_1d_trunk=Axis1DTrunkConfig(
            num_frequencies=2,
            max_frequency=2.0,
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
        "mode": "symmetric",
    }
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
    }
    assert set(summary["figure_fields"]) == expected_figure_fields
    assert summary["error_convention"] == "signed_difference"
    assert summary["solution_prediction"] == "u_pred=0.5*(u_phi+u_psi)"
    assert summary["raw_output_space"] == "physical"
    assert summary["output_contract_version"] == 2
    assert summary["balance_projection"] == {
        "enabled": True,
        "mode": "symmetric",
        "space": "physical",
        "residual_split": "equal_half",
    }
    assert summary["post_projection_unit_conversion"] == {
        "phi": "Phi_unit=Lx^2*phi_physical",
        "psi": "Psi_unit=Ly^2*psi_physical",
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
        "enabled": False,
        "fusion": "product",
        "coordinate": {
            "x_path": "y_local_t",
            "y_path": "x_local_t",
        },
    }
    for field in expected_figure_fields:
        assert (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).exists()

    with (outdir / "metrics" / "per_sample_metrics.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert rows
    assert all("cross" not in key for key in rows[0])

    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    assert any(key.endswith("_raw_physical_phi") for key in raw.files)
    assert any(key.endswith("_raw_physical_psi") for key in raw.files)
    assert any(key.endswith("_projected_unit_phi") for key in raw.files)
    assert any(key.endswith("_projected_unit_psi") for key in raw.files)
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
