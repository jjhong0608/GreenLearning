from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from greenonet.config import (
    BalanceProjectionConfig,
    CouplingModelConfig,
    ModelConfig,
)
from greenonet.coupling_artifacts import (
    CouplingArtifactExporter,
    CouplingArtifactRequest,
    export_coupling_artifacts,
)
from greenonet.coupling_model import CouplingNet
from greenonet.io import save_model_with_config
from greenonet.model import GreenONetModel


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def _write_npz(data_dir: Path, name: str = "sample.npz") -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 1.0, 257)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    sol = xx * (1.0 - xx) * yy * (1.0 - yy)
    rhs = 1.0 + xx - 0.5 * yy
    uxx = -2.0 * yy[1:-1, 1:-1] * (1.0 - yy[1:-1, 1:-1])
    uyy = -2.0 * xx[1:-1, 1:-1] * (1.0 - xx[1:-1, 1:-1])
    path = data_dir / name
    np.savez(path, sol=sol, rhs=rhs, uxx=uxx, uyy=uyy)
    return path


def _write_coefficients(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return torch.ones_like(x)",
                "def apx_fun(x, y): return torch.zeros_like(x)",
                "def apy_fun(x, y): return torch.zeros_like(x)",
                "def bx_fun(x, y): return torch.full_like(x, 10.0)",
                "def by_fun(x, y): return torch.full_like(x, 20.0)",
                "def c_fun(x, y): return torch.zeros_like(x)",
            ]
        )
    )


def _green_config() -> ModelConfig:
    return ModelConfig(
        input_dim=2,
        hidden_dim=4,
        depth=1,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        use_green=True,
        branch_input_dim=5,
        use_fourier=False,
        dtype=torch.float64,
    )


def _coupling_config() -> CouplingModelConfig:
    return CouplingModelConfig(
        branch_input_dim=5,
        trunk_input_dim=2,
        hidden_dim=4,
        depth=1,
        activation="tanh",
        use_bias=True,
        dropout=0.0,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(enabled=False),
    )


def _write_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    green_cfg = _green_config()
    coupling_cfg = _coupling_config()
    green_path = tmp_path / "green.safetensors"
    coupling_path = tmp_path / "coupling.safetensors"
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    save_model_with_config(CouplingNet(coupling_cfg), coupling_cfg, coupling_path)
    return coupling_path, green_path


def _write_config(
    path: Path,
    *,
    test_path: Path,
    coefficient_path: Path,
    device: str = "cpu",
) -> None:
    payload = {
        "dataset": {
            "step_size": 0.25,
            "n_points_per_line": 5,
            "samples_per_line": 1,
            "validation_samples_per_line": 1,
            "scale_length": 0.1,
            "validation_scale_length": 0.1,
            "deterministic": True,
            "sampler_mode": "forward",
            "validation_sampler_mode": "forward",
            "use_operator_learning": True,
            "test_path": str(test_path),
            "coefficient_functions_path": str(coefficient_path),
            "dtype": "float64",
        },
        "model": {
            "input_dim": 2,
            "hidden_dim": 4,
            "depth": 1,
            "activation": "tanh",
            "use_bias": True,
            "dropout": 0.0,
            "use_green": True,
            "branch_input_dim": 5,
            "use_fourier": False,
            "dtype": "float64",
        },
        "training": {
            "learning_rate": 1e-3,
            "epochs": 1,
            "batch_size": 1,
            "log_interval": 1,
            "device": "cpu",
            "integration_rule": "trapezoid",
            "compile": {"enabled": False},
        },
        "coupling_model": {
            "branch_input_dim": 5,
            "trunk_input_dim": 2,
            "hidden_dim": 4,
            "depth": 1,
            "activation": "tanh",
            "use_bias": True,
            "dropout": 0.0,
            "dtype": "float64",
            "balance_projection": {"enabled": False, "mode": "symmetric"},
        },
        "coupling_training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 1,
            "log_interval": 1,
            "device": device,
            "integration_rule": "trapezoid",
            "losses": {
                "l2_consistency": {"enabled": True, "weight": 1.0},
                "energy_consistency": {"enabled": True, "weight": 1.0},
                "cross_consistency": {"enabled": True, "weight": 1.0},
                "balance_loss": {"enabled": False, "weight": 1.0},
                "symmetric_boundary_loss": {"enabled": False, "weight": 1.0},
            },
            "compile": {"enabled": False},
        },
    }
    path.write_text(json.dumps(payload))


def test_coupling_artifact_helpers_keep_signed_differences(tmp_path: Path) -> None:
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
    )
    exporter = CouplingArtifactExporter(request)

    sol_components = torch.zeros((2, 3, 3), dtype=torch.float64)
    pred_sol_x = torch.full((3, 3), -1.0, dtype=torch.float64)
    pred_sol_y = torch.zeros((3, 3), dtype=torch.float64)
    solution_grids = exporter._solution_grids(sol_components, pred_sol_x, pred_sol_y)

    assert torch.all(solution_grids["u_error"] < 0.0)
    assert torch.all(solution_grids["u_pred_x_error"] == -1.0)
    assert torch.all(solution_grids["u_pred_x_minus_u_pred_y"] == -1.0)

    flux_components = torch.zeros((2, 3, 3), dtype=torch.float64)
    pred_flux_components = torch.zeros((2, 3, 3), dtype=torch.float64)
    pred_flux_components[0] = -2.0
    pred_flux_components[1] = 0.5
    source_grid = torch.full((3, 3), 1.0, dtype=torch.float64)
    flux_grids = exporter._flux_grids(
        flux_components,
        pred_flux_components,
        source_grid,
    )

    assert torch.all(flux_grids["phi_error"] == -2.0)
    assert torch.all(flux_grids["psi_error"] == 0.5)
    assert torch.allclose(
        flux_grids["phi_plus_psi"],
        torch.full((3, 3), -1.5, dtype=torch.float64),
    )
    assert torch.allclose(
        flux_grids["balance_residual"],
        torch.full((3, 3), 2.5, dtype=torch.float64),
    )


def test_export_coupling_artifacts_smoke(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    data_dir = tmp_path / "test_data"
    _write_npz(data_dir)
    coefficient_path = tmp_path / "coefficients.py"
    _write_coefficients(coefficient_path)
    coupling_checkpoint, green_checkpoint = _write_checkpoints(tmp_path)
    config_path = tmp_path / "config.json"
    outdir = tmp_path / "coupling_artifacts"
    _write_config(
        config_path,
        test_path=data_dir,
        coefficient_path=coefficient_path,
        device="cuda",
    )

    summary = export_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            device="cpu",
            selected_samples=(0,),
            theme="plotly_white",
        )
    )

    assert summary["device"] == "cpu"
    assert summary["selected_samples"] == [0]
    assert (outdir / "summary.json").exists()
    assert (outdir / "metrics" / "per_sample_metrics.csv").exists()
    assert (outdir / "metrics" / "aggregate_metrics.csv").exists()
    assert (outdir / "metrics" / "balance_residual_metrics.csv").exists()
    assert (outdir / "metrics" / "boundary_residual_metrics.csv").exists()
    assert (outdir / "data" / "selected_samples.npz").exists()
    assert (outdir / "data" / "selected_predictions.npz").exists()
    assert (outdir / "data" / "selected_diagnostics.npz").exists()

    figure_base = outdir / "figures" / "solution" / "sample_0000_sample_u_error"
    assert figure_base.with_suffix(".html").exists()
    assert figure_base.with_suffix(".json").exists()
    assert (
        outdir
        / "figures"
        / "balance"
        / "sample_0000_sample_balance_residual.json"
    ).exists()
    assert not any("null" in path.name for path in (outdir / "figures").rglob("*"))
    assert not any("closure" in path.name for path in (outdir / "figures").rglob("*"))

    selected = np.load(outdir / "data" / "selected_samples.npz")
    assert np.nanmax(selected["bx"]) == 10.0
    assert np.nanmax(selected["by"]) == 20.0
    saved_summary = json.loads((outdir / "summary.json").read_text())
    assert saved_summary["source_grid_policy"] == ["npz_rhs_resampled_to_model_grid"]
    assert saved_summary["coefficients"] == str(coefficient_path)
