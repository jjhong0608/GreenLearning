import json
import sys
from types import SimpleNamespace

from greenonet.model import GreenONetModel
from cli.train import TrainCLI


class TestTrainCLIConfigCopy:
    def _write_config(self, path):
        payload = {
            "dataset": {
                "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
            },
            "model": {},
            "training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        path.write_text(json.dumps(payload))
        return payload

    def test_copies_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        payload = self._write_config(config_path)
        work_dir = tmp_path / "work"

        def _fake_run_green_o_net(*_args, **kwargs):
            return SimpleNamespace(model=GreenONetModel(kwargs["model_cfg"]))

        monkeypatch.setattr("cli.train.run_green_o_net", _fake_run_green_o_net)
        monkeypatch.setattr(
            sys,
            "argv",
            ["train.py", "--config", str(config_path), "--work-dir", str(work_dir)],
        )

        TrainCLI().run()

        copied = work_dir / "config_used.json"
        assert copied.exists()
        assert json.loads(copied.read_text()) == payload


class TestTrainCLIDatasetConfig:
    def test_ignores_domain_block(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {
                "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
                "step_size": 0.25,
            },
            "model": {},
            "training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        dataset_cfg, *_rest = TrainCLI()._build_configs(config_path)

        assert dataset_cfg.step_size == 0.25
        assert not hasattr(dataset_cfg, "domain")

    def test_parses_green_validation_dataset_controls(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {
                "step_size": 0.25,
                "samples_per_line": 3,
                "validation_samples_per_line": 2,
                "scale_length": [0.05, 0.25],
                "validation_scale_length": [0.10, 0.20],
                "validation_sampler_mode": "backward",
            },
            "model": {},
            "training": {
                "compute_validation_rel_sol": True,
            },
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        dataset_cfg, _model_cfg, training_cfg, *_rest = TrainCLI()._build_configs(
            config_path
        )

        assert dataset_cfg.samples_per_line == 3
        assert dataset_cfg.validation_samples_per_line == 2
        assert dataset_cfg.scale_length == (0.05, 0.25)
        assert dataset_cfg.validation_scale_length == (0.10, 0.20)
        assert dataset_cfg.validation_sampler_mode == "backward"
        assert training_cfg.compute_validation_rel_sol is True

    def test_parses_integration_rules(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {
                "integration_rule": "trapezoid",
                "compile": {
                    "enabled": True,
                },
            },
            "coupling_model": {},
            "coupling_training": {
                "integration_rule": "trapezoid",
                "flux_consistency_enabled": True,
                "lambda_flux_consistency": 0.25,
                "lambda_raw_balance": 0.5,
                "learning_rate": 5e-4,
                "epochs": 11,
                "use_lr_schedule": True,
                "warmup_epochs": 3,
                "min_lr": 1e-6,
                "periodic_checkpoint": {
                    "enabled": True,
                    "every_epochs": 4,
                },
                "best_rel_sol_checkpoint": {
                    "enabled": True
                },
                "compile": {
                    "enabled": True,
                },
            },
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            _pipeline_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert training_cfg.integration_rule == "trapezoid"
        assert training_cfg.compile.enabled is True
        assert coupling_training_cfg.integration_rule == "trapezoid"
        assert coupling_training_cfg.flux_consistency_enabled is True
        assert coupling_training_cfg.lambda_flux_consistency == 0.25
        assert coupling_training_cfg.lambda_raw_balance == 0.5
        assert coupling_training_cfg.learning_rate == 5e-4
        assert coupling_training_cfg.epochs == 11
        assert coupling_training_cfg.use_lr_schedule is True
        assert coupling_training_cfg.warmup_epochs == 3
        assert coupling_training_cfg.min_lr == 1e-6
        assert coupling_training_cfg.periodic_checkpoint.enabled is True
        assert coupling_training_cfg.periodic_checkpoint.every_epochs == 4
        assert coupling_training_cfg.best_rel_sol_checkpoint.enabled is True
        assert coupling_training_cfg.compile.enabled is True
        assert not hasattr(coupling_model_cfg, "use_fourier")
        assert not hasattr(coupling_model_cfg, "fourier_dim")
        assert not hasattr(coupling_model_cfg, "fourier_scale")
        assert not hasattr(coupling_model_cfg, "fourier_include_input")
