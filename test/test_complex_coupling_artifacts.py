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
    assert (outdir / "figures" / "phi" / "sample_0000_sample_0000_phi.json").exists()

    with (outdir / "metrics" / "per_sample_metrics.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert rows
    assert all("cross" not in key for key in rows[0])

    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    assert any(key.endswith("_raw_unit_phi") for key in raw.files)
    assert all("cross" not in key for key in raw.files)
