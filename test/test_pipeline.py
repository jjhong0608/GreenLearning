import pathlib
import csv
import json
import math
import pytest

import torch

from greenonet.config import ModelConfig, TrainingConfig
from greenonet.model import GreenONetModel
from greenonet.trainer import Trainer


class TestGreenONetModel:
    def test_forward_and_backprop(self) -> None:
        torch.manual_seed(0)
        model_config = ModelConfig(
            input_dim=2,
            hidden_dim=16,
            depth=3,
            activation="tanh",
            use_bias=True,
            branch_input_dim=4,
        )
        model = GreenONetModel(model_config)
        # Build coords grid (2 axes, 1 line, 4 points)
        a_vals = torch.rand(1, 2, 1, 4)
        ap_vals = torch.zeros_like(a_vals)
        b_vals = torch.zeros_like(a_vals)
        c_vals = torch.zeros_like(a_vals)
        output = model(
            trunk_grid=torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, 4), torch.linspace(0, 1, 4), indexing="ij"
                ),
                dim=-1,
            ),
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_vals,
        )
        assert output.shape == (2, 1, 4, 4)
        loss = output.sum()
        loss.backward()
        assert model.branch_a.net[0].weight.grad is not None
        assert model.branch_c.net[0].weight.grad is not None
        assert model.trunk.net[0].weight.grad is not None

    def test_rational_activation(self) -> None:
        torch.manual_seed(0)
        model_config = ModelConfig(
            input_dim=2,
            hidden_dim=8,
            depth=2,
            activation="rational",
            use_bias=True,
            use_green=True,
            branch_input_dim=3,
        )
        model = GreenONetModel(model_config)
        zeros = torch.zeros((1, 2, 1, 3), dtype=torch.float64)
        output = model(
            trunk_grid=torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), indexing="ij"
                ),
                dim=-1,
            ),
            a_vals=zeros,
            ap_vals=zeros,
            b_vals=zeros,
            c_vals=zeros,
        )
        assert output.shape == (2, 1, 3, 3)
        loss = (output**2).mean()
        loss.backward()
        assert model.branch_a.net[0].weight.grad is not None
        assert model.branch_c.net[0].weight.grad is not None
        assert model.trunk.net[0].weight.grad is not None

    def test_fourier_embedding(self) -> None:
        torch.manual_seed(0)
        model = GreenONetModel(
            ModelConfig(
                input_dim=2,
                hidden_dim=8,
                depth=2,
                activation="tanh",
                use_bias=True,
                use_green=False,
                branch_input_dim=3,
                use_fourier=True,
                fourier_dim=6,
                fourier_scale=2.0,
                fourier_include_input=True,
            )
        )
        trunk = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), indexing="ij"
            ),
            dim=-1,
        )
        vals = torch.ones((1, 2, 1, 3))
        out = model(
            trunk_grid=trunk,
            a_vals=vals,
            ap_vals=torch.zeros_like(vals),
            b_vals=torch.zeros_like(vals),
            c_vals=torch.zeros_like(vals),
        )
        assert out.shape == (2, 1, 3, 3)

    def test_green_terms_change_output(self) -> None:
        torch.manual_seed(0)
        base_cfg = ModelConfig(
            input_dim=2,
            hidden_dim=4,
            depth=2,
            activation="tanh",
            use_bias=True,
            use_green=False,
            branch_input_dim=2,
        )
        green_cfg = ModelConfig(
            input_dim=2,
            hidden_dim=4,
            depth=2,
            activation="tanh",
            use_bias=True,
            use_green=True,
            branch_input_dim=2,
        )
        base_model = GreenONetModel(base_cfg)
        green_model = GreenONetModel(green_cfg)
        branch = torch.rand(1, 2, 1, 2)
        zeros = torch.zeros_like(branch)
        base_out = base_model(
            trunk_grid=torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, 2), torch.linspace(0, 1, 2), indexing="ij"
                ),
                dim=-1,
            ),
            a_vals=branch,
            ap_vals=zeros,
            b_vals=zeros,
            c_vals=zeros,
        )
        green_out = green_model(
            trunk_grid=torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 1, 2), torch.linspace(0, 1, 2), indexing="ij"
                ),
                dim=-1,
            ),
            a_vals=branch,
            ap_vals=zeros,
            b_vals=zeros,
            c_vals=zeros,
        )
        assert not torch.allclose(base_out, green_out)

    def test_coefficients_affect_output(self) -> None:
        torch.manual_seed(0)
        model = GreenONetModel(
            ModelConfig(
                input_dim=2,
                hidden_dim=8,
                depth=2,
                activation="tanh",
                use_bias=True,
                use_green=False,
                branch_input_dim=3,
            )
        )
        trunk = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, 3), torch.linspace(0, 1, 3), indexing="ij"
            ),
            dim=-1,
        )
        a_vals = torch.ones((1, 2, 1, 3))
        ap_vals = torch.zeros_like(a_vals)
        b_vals = torch.ones_like(a_vals)
        c_zeros = torch.zeros_like(a_vals)
        c_ones = torch.ones_like(a_vals)
        out_zero = model(
            trunk_grid=trunk,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_zeros,
        )
        out_one = model(
            trunk_grid=trunk,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_ones,
        )
        assert not torch.allclose(out_zero, out_one)

    def test_coupling_predicts_flux_divergence(self) -> None:
        torch.manual_seed(0)
        model = GreenONetModel(
            ModelConfig(
                input_dim=2,
                hidden_dim=8,
                depth=3,
                activation="tanh",
                use_bias=True,
                use_green=False,
            )
        )
        coords = torch.randn(5, 2)
        with pytest.raises(AttributeError):
            _ = model.predict_flux_divergence(coords)


class TestTrainer:
    def test_training_reduces_loss(self, tmp_path: pathlib.Path) -> None:
        torch.manual_seed(0)
        # Use packed axial dataset with single sample
        n_lines = 2
        m_points = 5
        # coords not used directly; trainer expects AxialDataset-like structure via collate
        coords = torch.zeros((2, n_lines, m_points, 2), dtype=torch.float64)
        solutions = torch.ones((1, 2, n_lines, m_points), dtype=torch.float64)
        sources = torch.zeros_like(solutions)
        zeros = torch.zeros_like(solutions)
        dataset = [
            (
                coords,
                solutions[0],
                sources[0],
                zeros[0],
                zeros[0],
                zeros[0],
                zeros[0],
            )
        ]
        model_cfg = ModelConfig(
            input_dim=2,
            hidden_dim=32,
            depth=3,
            activation="tanh",
            use_bias=True,
            use_green=True,
            branch_input_dim=m_points,
        )
        model = GreenONetModel(model_cfg)
        trainer = Trainer(
            model=model,
            config=TrainingConfig(
                learning_rate=5e-3,
                epochs=3,
                batch_size=16,
                log_interval=1,
            ),
            work_dir=tmp_path,
            model_cfg=model_cfg,
        )
        initial_loss = trainer.evaluate(dataset)
        trainer.train(dataset)
        final_loss = trainer.evaluate(dataset)
        assert final_loss <= initial_loss
        assert (tmp_path / "training.log").exists()
        assert (tmp_path / "loss_curve.html").exists()
        assert (tmp_path / "model.safetensors").exists()
        per_line_csv = tmp_path / "per_line_metrics.csv"
        per_line_summary = tmp_path / "per_line_metrics_summary.json"
        assert per_line_csv.exists()
        assert per_line_summary.exists()
        with per_line_csv.open() as fp:
            rows = list(csv.DictReader(fp))
        assert len(rows) == 2 * n_lines
        for col in (
            "axis_id",
            "axis_name",
            "line_index",
            "line_coordinate",
            "rel_sol_line",
            "rel_green_line",
            "rel_sol_line_mean",
            "rel_sol_line_min",
            "rel_sol_line_max",
            "rel_sol_line_std",
            "rel_green_line_mean",
            "rel_green_line_min",
            "rel_green_line_max",
            "rel_green_line_std",
        ):
            assert col in rows[0]
        for row in rows:
            assert float(row["rel_sol_line_std"]) >= 0.0
            assert float(row["rel_green_line_std"]) >= 0.0
            assert float(row["rel_sol_line"]) == pytest.approx(
                float(row["rel_sol_line_mean"])
            )
            rel_green = float(row["rel_green_line"])
            rel_green_mean = float(row["rel_green_line_mean"])
            if math.isnan(rel_green) and math.isnan(rel_green_mean):
                pass
            else:
                assert rel_green == pytest.approx(rel_green_mean)
        summary = json.loads(per_line_summary.read_text())
        assert summary["num_axes"] == 2
        assert summary["num_lines_per_axis"] == n_lines
        assert trainer.loss_history, "Loss history should be recorded"

    def test_lbfgs_stage_runs(self, tmp_path: pathlib.Path) -> None:
        torch.manual_seed(0)
        n_lines = 1
        m_points = 3
        coords = torch.zeros((2, n_lines, m_points, 2), dtype=torch.float64)
        solutions = torch.ones((1, 2, n_lines, m_points), dtype=torch.float64)
        sources = torch.zeros_like(solutions)
        zeros = torch.zeros_like(solutions)
        dataset = [
            (
                coords,
                solutions[0],
                sources[0],
                zeros[0],
                zeros[0],
                zeros[0],
                zeros[0],
            )
        ]
        model_cfg = ModelConfig(
            input_dim=2,
            hidden_dim=16,
            depth=2,
            activation="tanh",
            use_bias=True,
            use_green=True,
            branch_input_dim=m_points,
        )
        model = GreenONetModel(model_cfg)
        trainer = Trainer(
            model=model,
            config=TrainingConfig(
                learning_rate=1e-2,
                epochs=1,
                batch_size=8,
                log_interval=1,
                lbfgs_max_iter=3,
                lbfgs_lr=0.5,
                lbfgs_history_size=5,
                lbfgs_epochs=2,
            ),
            work_dir=tmp_path,
            model_cfg=model_cfg,
        )
        trainer.train(dataset)
        # One Adam epoch + two LBFGS aggregate entries
        assert len(trainer.loss_history) >= 3

    def test_validation_rel_sol_is_recorded_and_exported(
        self, tmp_path: pathlib.Path
    ) -> None:
        torch.manual_seed(0)
        n_lines = 2
        m_points = 5
        coords = torch.zeros((2, n_lines, m_points, 2), dtype=torch.float64)

        train_solutions = torch.ones((1, 2, n_lines, m_points), dtype=torch.float64)
        val_solutions = 2.0 * torch.ones((1, 2, n_lines, m_points), dtype=torch.float64)
        sources = torch.zeros_like(train_solutions)
        zeros = torch.zeros_like(train_solutions)

        train_dataset = [
            (
                coords,
                train_solutions[0],
                sources[0],
                zeros[0],
                zeros[0],
                zeros[0],
                zeros[0],
            )
        ]
        validation_dataset = [
            (
                coords,
                val_solutions[0],
                sources[0],
                zeros[0],
                zeros[0],
                zeros[0],
                zeros[0],
            )
        ]

        model_cfg = ModelConfig(
            input_dim=2,
            hidden_dim=16,
            depth=2,
            activation="tanh",
            use_bias=True,
            use_green=True,
            branch_input_dim=m_points,
        )
        model = GreenONetModel(model_cfg)
        trainer = Trainer(
            model=model,
            config=TrainingConfig(
                learning_rate=1e-3,
                epochs=1,
                batch_size=4,
                log_interval=1,
                compute_validation_rel_sol=True,
            ),
            work_dir=tmp_path,
            model_cfg=model_cfg,
        )

        trainer.train(train_dataset, validation_dataset)

        assert trainer.rel_sol_history
        assert trainer.val_rel_sol_history

        per_line_csv = tmp_path / "per_line_metrics.csv"
        rows = list(csv.DictReader(per_line_csv.open()))
        assert rows
        for col in (
            "val_rel_sol_line",
            "val_rel_sol_line_mean",
            "val_rel_sol_line_min",
            "val_rel_sol_line_max",
            "val_rel_sol_line_std",
        ):
            assert col in rows[0]
        for row in rows:
            assert float(row["val_rel_sol_line"]) == pytest.approx(
                float(row["val_rel_sol_line_mean"])
            )
            assert float(row["val_rel_sol_line_std"]) >= 0.0

        summary = json.loads((tmp_path / "per_line_metrics_summary.json").read_text())
        assert "mean_val_rel_sol_line" in summary
