from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.config import ModelConfig
from greenonet.green_artifacts import (
    GreenArtifactRequest,
    GreenArtifactExporter,
    export_green_artifacts,
    load_green_artifact_configs,
)
from greenonet.io import save_model_with_config
from greenonet.model import GreenONetModel


def _write_config(
    path: Path,
    coefficient_path: Path | None = None,
) -> None:
    dataset: dict[str, object] = {
        "step_size": 0.5,
        "n_points_per_line": 5,
        "samples_per_line": 1,
        "validation_samples_per_line": 1,
        "scale_length": 0.1,
        "validation_scale_length": 0.1,
        "deterministic": True,
        "sampler_mode": "forward",
        "validation_sampler_mode": "forward",
        "use_operator_learning": True,
        "dtype": "float64",
    }
    if coefficient_path is not None:
        dataset["coefficient_functions_path"] = str(coefficient_path)
    payload = {
        "dataset": dataset,
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
            "batch_size": 2,
            "log_interval": 1,
            "compute_validation_rel_sol": True,
            "device": "cpu",
            "integration_rule": "trapezoid",
            "compile": {"enabled": False},
        },
        "pipeline": {
            "run_green": True,
            "run_coupling": False,
            "green_pretrained_path": None,
            "coupling_pretrained_path": None,
        },
    }
    path.write_text(json.dumps(payload))


def _write_checkpoint(path: Path) -> None:
    cfg = ModelConfig(
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
    torch.manual_seed(0)
    save_model_with_config(GreenONetModel(cfg), cfg, path)


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def test_export_green_artifacts_smoke_diffusion_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    checkpoint_path = tmp_path / "green.safetensors"
    config_path = tmp_path / "config.json"
    outdir = tmp_path / "artifacts"
    _write_checkpoint(checkpoint_path)
    _write_config(config_path)

    summary = export_green_artifacts(
        GreenArtifactRequest(
            checkpoint=checkpoint_path,
            config=config_path,
            outdir=outdir,
            eval_seed=7,
            line_indices=(0,),
            xi_fractions=(0.5,),
            theme="plotly_white",
        )
    )

    assert summary["rel_green_valid"] is True
    assert (outdir / "summary.json").exists()
    assert (outdir / "metrics" / "per_line_metrics.csv").exists()
    assert (outdir / "metrics" / "sample_metrics.csv").exists()
    assert (outdir / "metrics" / "boundary_diagnostics.csv").exists()
    assert (outdir / "metrics" / "green_slice_metrics.csv").exists()
    assert (
        outdir
        / "figures"
        / "green_heatmaps"
        / "axis0_line000_green_heatmap_pred.html"
    ).exists()
    assert (
        outdir
        / "figures"
        / "green_heatmaps"
        / "axis0_line000_green_heatmap_pred.json"
    ).exists()
    assert (
        outdir
        / "figures"
        / "green_slices"
        / "axis0_line000_xi002_green_slice.html"
    ).exists()
    assert (
        outdir
        / "figures"
        / "green_slices"
        / "axis0_line000_xi002_green_slice.json"
    ).exists()
    assert (outdir / "data" / "generated_eval_data.npz").exists()
    assert (outdir / "data" / "selected_green_kernels.npz").exists()
    assert (outdir / "data" / "selected_reconstructions.npz").exists()

    saved_summary = json.loads((outdir / "summary.json").read_text())
    assert saved_summary["selected_xi"] == [
        {"index": 2, "value": 0.5, "label": "0.5"}
    ]
    assert saved_summary["eval_seed"] == 7
    assert saved_summary["eval_sampling"]["samples_per_line"] == 1


def test_export_green_artifacts_seed_reproduces_generated_eval_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    checkpoint_path = tmp_path / "green.safetensors"
    config_path = tmp_path / "config.json"
    _write_checkpoint(checkpoint_path)
    _write_config(config_path)

    for name in ("run_a", "run_b"):
        export_green_artifacts(
            GreenArtifactRequest(
                checkpoint=checkpoint_path,
                config=config_path,
                outdir=tmp_path / name,
                eval_seed=11,
                line_indices=(0,),
                xi_fractions=(0.5,),
            )
        )

    data_a = np.load(tmp_path / "run_a" / "data" / "generated_eval_data.npz")
    data_b = np.load(tmp_path / "run_b" / "data" / "generated_eval_data.npz")
    np.testing.assert_allclose(data_a["source"], data_b["source"])
    np.testing.assert_allclose(data_a["solution"], data_b["solution"])


def test_export_green_artifacts_marks_rel_green_invalid_for_reaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    coefficient_path = tmp_path / "reaction_coefficients.py"
    coefficient_path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return torch.ones_like(x)",
                "def apx_fun(x, y): return torch.zeros_like(x)",
                "def apy_fun(x, y): return torch.zeros_like(x)",
                "def bx_fun(x, y): return torch.zeros_like(x)",
                "def by_fun(x, y): return torch.zeros_like(x)",
                "def c_fun(x, y): return torch.ones_like(x)",
            ]
        )
    )
    checkpoint_path = tmp_path / "green.safetensors"
    config_path = tmp_path / "config.json"
    outdir = tmp_path / "artifacts"
    _write_checkpoint(checkpoint_path)
    _write_config(config_path, coefficient_path=coefficient_path)

    summary = export_green_artifacts(
        GreenArtifactRequest(
            checkpoint=checkpoint_path,
            config=config_path,
            outdir=outdir,
            eval_seed=7,
            line_indices=(0,),
            xi_fractions=(0.5,),
        )
    )

    assert summary["rel_green_valid"] is False
    assert summary["rel_green"] is None
    assert "nonzero" in str(summary["rel_green_skip_reason"])


def test_exporter_reconstruction_metric_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    checkpoint_path = tmp_path / "green.safetensors"
    config_path = tmp_path / "config.json"
    _write_checkpoint(checkpoint_path)
    _write_config(config_path)

    request = GreenArtifactRequest(
        checkpoint=checkpoint_path,
        config=config_path,
        outdir=tmp_path / "artifacts",
        eval_seed=5,
        line_indices=(0,),
        xi_fractions=(0.5,),
    )
    exporter = GreenArtifactExporter(request)
    dataset_cfg, _model_cfg, training_cfg, _raw = load_green_artifact_configs(
        config_path
    )
    sampling_cfg = exporter._resolve_sampling_config(dataset_cfg)
    dataset = exporter._generate_dataset(
        dataset_cfg=dataset_cfg,
        training_cfg=training_cfg,
        sampling_cfg=sampling_cfg,
        coeffs=load_coefficient_functions(None),
    )
    kernel = torch.zeros(
        (2, dataset.coords.shape[1], dataset.coords.shape[2], dataset.coords.shape[2]),
        dtype=torch.float64,
    )
    trunk_grid = exporter._build_trunk_grid(
        m_points=dataset.coords.shape[2],
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    reconstruction = exporter._reconstruct_solution(
        kernel=kernel,
        source=dataset.sources,
        trunk_grid=trunk_grid,
        integration_rule=training_cfg.integration_rule,
    )
    rel = exporter._relative_solution_error_by_line(
        reconstruction=reconstruction,
        solution=dataset.solutions,
        trunk_grid=trunk_grid,
        integration_rule=training_cfg.integration_rule,
    )

    assert rel.shape == (1, 2, 1)
