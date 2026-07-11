from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from cli.compare_complex_projection_posthoc import CompareComplexProjectionPosthocCLI
from greenonet.config import ModelConfig
from greenonet.io import save_model_with_config
from greenonet.model import GreenONetModel
from test.complex_fixtures import (
    write_coefficients,
    write_complex_config,
    write_geometry_npz,
)


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def _write_selected_raw_archive(
    artifact_root: Path,
    geometry_path: Path,
    *,
    include_raw_unit_psi: bool = True,
) -> None:
    artifact_data = artifact_root / "data"
    artifact_data.mkdir(parents=True, exist_ok=True)
    with np.load(geometry_path, allow_pickle=False) as raw:
        coords = np.array(raw["coords_valid"])
    prefix = "sample_0000_fixture"
    payload = {
        f"{prefix}_coords_valid": coords,
        f"{prefix}_rhs": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        f"{prefix}_sol": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        f"{prefix}_raw_unit_phi": np.array([4.0, 8.0, 2.0], dtype=np.float64),
    }
    if include_raw_unit_psi:
        payload[f"{prefix}_raw_unit_psi"] = np.array(
            [1.0, 3.0, 5.0],
            dtype=np.float64,
        )
    np.savez(artifact_data / "selected_raw_arrays.npz", **payload)


def _write_green_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    config = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    save_model_with_config(GreenONetModel(config), config, path)
    return path


def test_compare_complex_projection_posthoc_writes_comparison_outputs(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(
        tmp_path / "geometry.npz",
        inner_radius=np.array(0.2, dtype=np.float64),
    )
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=tmp_path / "unused_test_data",
        coefficient_path=coeff_path,
    )
    green_path = _write_green_checkpoint(tmp_path / "green.safetensors")
    artifact_root = tmp_path / "artifacts"
    _write_selected_raw_archive(artifact_root, geometry_path)
    outdir = tmp_path / "projection_compare"

    summary = CompareComplexProjectionPosthocCLI().run(
        [
            "--artifact-root",
            str(artifact_root),
            "--geometry",
            str(geometry_path),
            "--green-checkpoint",
            str(green_path),
            "--config",
            str(config_path),
            "--outdir",
            str(outdir),
            "--device",
            "cpu",
            "--theme",
            "plotly_white",
        ]
    )

    assert summary["projection_modes"] == ["symmetric", "geometry_weighted"]
    assert summary["geometry_weighted_rule"] == "direct_length_squared"
    assert summary["transition_coordinate"] == 0.25
    assert (outdir / "summary.json").exists()
    assert (outdir / "per_sample_projection_comparison.csv").exists()
    assert (outdir / "zone_projection_comparison.csv").exists()
    assert (outdir / "data" / "selected_projection_raw_arrays.npz").exists()
    assert (
        outdir
        / "figures"
        / "symmetric"
        / "u_pred_error"
        / "sample_0000_fixture_symmetric_u_pred_error.json"
    ).exists()
    assert (
        outdir
        / "figures"
        / "weighted_minus_symmetric"
        / "u_pred"
        / "sample_0000_fixture_weighted_minus_symmetric_u_pred.json"
    ).exists()

    with (outdir / "per_sample_projection_comparison.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 1
    assert rows[0]["sample_key"] == "sample_0000_fixture"
    assert float(rows[0]["symmetric_balance_max_abs"]) < 1.0e-12
    assert float(rows[0]["geometry_weighted_balance_max_abs"]) < 1.0e-12

    with (outdir / "zone_projection_comparison.csv").open() as fp:
        zone_rows = list(csv.DictReader(fp))
    assert {row["zone"] for row in zone_rows} >= {
        "global",
        "horizontal_abs_y_0.25",
        "vertical_abs_x_0.25",
    }

    raw = np.load(outdir / "data" / "selected_projection_raw_arrays.npz")
    assert "sample_0000_fixture_symmetric_u_pred" in raw.files
    assert "sample_0000_fixture_geometry_weighted_u_pred" in raw.files
    assert "sample_0000_fixture_weighted_minus_symmetric_u_pred" in raw.files
    assert not np.allclose(
        raw["sample_0000_fixture_symmetric_phi"],
        raw["sample_0000_fixture_geometry_weighted_phi"],
    )


def test_compare_complex_projection_posthoc_rejects_missing_raw_fields(tmp_path):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=tmp_path / "unused_test_data",
        coefficient_path=coeff_path,
    )
    green_path = _write_green_checkpoint(tmp_path / "green.safetensors")
    artifact_root = tmp_path / "artifacts"
    _write_selected_raw_archive(
        artifact_root,
        geometry_path,
        include_raw_unit_psi=False,
    )

    with pytest.raises(KeyError, match="raw_unit_psi"):
        CompareComplexProjectionPosthocCLI().run(
            [
                "--artifact-root",
                str(artifact_root),
                "--geometry",
                str(geometry_path),
                "--green-checkpoint",
                str(green_path),
                "--config",
                str(config_path),
                "--outdir",
                str(tmp_path / "projection_compare"),
                "--device",
                "cpu",
            ]
        )
