from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_data import (
    ComplexGreenDataset,
    generate_complex_green_data,
)
from greenonet.complex_green_trainer import ComplexGreenTrainer
from greenonet.config import CompileConfig, ModelConfig, TrainingConfig
from greenonet.model import GreenONetModel
from test.complex_fixtures import write_geometry_npz


class CountingGreenONetModel(GreenONetModel):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.forward_pairs_calls = 0

    def forward_pairs(
        self,
        trunk_coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> torch.Tensor:
        self.forward_pairs_calls += 1
        return super().forward_pairs(trunk_coords, a_vals, ap_vals, b_vals, c_vals)

    def forward(
        self,
        trunk_grid: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> torch.Tensor:
        raise AssertionError(
            "ComplexGreenTrainer must call forward_pairs, not forward."
        )


def _write_reaction_free_coefficients(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return 1.0 + 0.0 * x + 0.0 * y",
                "def apx_fun(x, y): return 0.0 * x",
                "def apy_fun(x, y): return 0.0 * y",
                "def bx_fun(x, y): return 0.0 * x",
                "def by_fun(x, y): return 0.0 * y",
                "def c_fun(x, y): return 0.0 * x",
            ]
        )
    )
    return path


def test_complex_green_trainer_one_epoch_outputs_safe_metrics(tmp_path):
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(
        _write_reaction_free_coefficients(tmp_path / "coeffs.py")
    )
    data = generate_complex_green_data(
        geometry,
        coeffs,
        branch_input_dim=5,
        samples_per_interval=1,
        sampler_mode="forward",
        scale_length=0.1,
        deterministic=True,
        integration_rule="trapezoid",
        dtype=torch.float64,
    )
    dataset = ComplexGreenDataset(data)
    model_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        activation="tanh",
        use_green=False,
        branch_input_dim=5,
        dtype=torch.float64,
    )
    model = CountingGreenONetModel(model_cfg)
    trainer = ComplexGreenTrainer(
        model=model,
        config=TrainingConfig(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            log_interval=1,
            device="cpu",
            integration_rule="trapezoid",
            compile=CompileConfig(enabled=False),
            lbfgs_max_iter=0,
        ),
        work_dir=tmp_path / "work",
        model_cfg=model_cfg,
    )

    trainer.train(dataset)

    assert model.forward_pairs_calls > 0
    assert (tmp_path / "work" / "model.safetensors").exists()
    assert (tmp_path / "work" / "loss_curve.html").exists()
    assert (tmp_path / "work" / "rel_sol_curve.html").exists()
    assert (tmp_path / "work" / "rel_green_curve.html").exists()
    assert (tmp_path / "work" / "green_heatmap.html").exists()
    assert (tmp_path / "work" / "per_interval_metrics.csv").exists()
    assert (tmp_path / "work" / "per_interval_metrics_summary.json").exists()

    with (tmp_path / "work" / "per_interval_metrics.csv").open() as fp:
        fieldnames = set(next(csv.DictReader(fp)).keys())
    assert "rel_sol" in fieldnames
    assert "rel_green" in fieldnames
    assert not any("cross" in key for key in fieldnames)

    summary = json.loads(
        (tmp_path / "work" / "per_interval_metrics_summary.json").read_text()
    )
    assert summary["num_intervals"] == 5
    assert summary["rel_green_valid"] is True
    assert not any("cross" in key for key in summary)
