import math
import json
import sys
from types import SimpleNamespace

import pytest

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
            "coupling_model": {
                "hidden_dim": 128,
                "activation": "rational",
                "fusion": {
                    "type": "gated_attention",
                    "gated_attention": {
                        "activation": "gelu",
                        "use_bias": False,
                        "dropout": 0.1
                    }
                },
                "trunk_fourier": {
                    "enabled": True,
                    "frequencies": [math.pi, 2.0 * math.pi],
                },
                "line_encoder": {
                    "type": "cnn1d",
                    "in_channels": 3,
                    "include_position": True,
                    "include_boundary_distance": True,
                    "conv_channels": [32, 64, 64],
                    "kernel_size": 5,
                    "dilations": [1, 2, 4],
                    "pooling": "meanmax",
                    "activation": "rational",
                    "mlp_head": {
                        "depth": 2,
                        "hidden_dim": 32,
                        "activation": "rational",
                        "use_bias": True,
                        "dropout": 0.0
                    }
                }
            },
            "coupling_training": {
                "integration_rule": "trapezoid",
                "losses": {
                    "l2_consistency": {
                        "enabled": True,
                        "weight": 1.5,
                        "weight_mode": "manual",
                    },
                    "flux_consistency": {
                        "enabled": True,
                        "weight": 0.25,
                        "weight_mode": "auto_operator",
                    },
                    "cross_consistency": {
                        "enabled": False,
                        "weight": 2.0,
                        "weight_mode": "manual",
                    },
                },
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
        assert coupling_training_cfg.losses.l2_consistency.enabled is True
        assert coupling_training_cfg.losses.l2_consistency.weight == 1.5
        assert coupling_training_cfg.losses.l2_consistency.weight_mode == "manual"
        assert coupling_training_cfg.losses.flux_consistency.enabled is True
        assert coupling_training_cfg.losses.flux_consistency.weight == 0.25
        assert (
            coupling_training_cfg.losses.flux_consistency.weight_mode
            == "auto_operator"
        )
        assert coupling_training_cfg.losses.cross_consistency.enabled is False
        assert coupling_training_cfg.losses.cross_consistency.weight == 2.0
        assert coupling_training_cfg.losses.cross_consistency.weight_mode == "manual"
        assert coupling_training_cfg.learning_rate == 5e-4
        assert coupling_training_cfg.epochs == 11
        assert coupling_training_cfg.use_lr_schedule is True
        assert coupling_training_cfg.warmup_epochs == 3
        assert coupling_training_cfg.min_lr == 1e-6
        assert coupling_training_cfg.periodic_checkpoint.enabled is True
        assert coupling_training_cfg.periodic_checkpoint.every_epochs == 4
        assert coupling_training_cfg.best_rel_sol_checkpoint.enabled is True
        assert coupling_training_cfg.compile.enabled is True
        assert coupling_model_cfg.hidden_dim == 128
        assert coupling_model_cfg.activation == "rational"
        assert coupling_model_cfg.fusion.type == "gated_attention"
        assert coupling_model_cfg.fusion.gated_attention.activation == "gelu"
        assert coupling_model_cfg.fusion.gated_attention.use_bias is False
        assert coupling_model_cfg.fusion.gated_attention.dropout == 0.1
        assert coupling_model_cfg.line_encoder.type == "cnn1d"
        assert coupling_model_cfg.line_encoder.in_channels == 3
        assert coupling_model_cfg.line_encoder.conv_channels == [32, 64, 64]
        assert coupling_model_cfg.line_encoder.kernel_size == 5
        assert coupling_model_cfg.line_encoder.dilations == [1, 2, 4]
        assert coupling_model_cfg.line_encoder.pooling == "meanmax"
        assert coupling_model_cfg.line_encoder.activation == "rational"
        assert coupling_model_cfg.line_encoder.mlp_head.depth == 2
        assert coupling_model_cfg.line_encoder.mlp_head.hidden_dim == 32
        assert coupling_model_cfg.trunk_fourier.enabled is True
        assert coupling_model_cfg.trunk_fourier.frequencies == [
            math.pi,
            2.0 * math.pi,
        ]
        assert not hasattr(coupling_model_cfg, "use_fourier")
        assert not hasattr(coupling_model_cfg, "fourier_dim")
        assert not hasattr(coupling_model_cfg, "fourier_scale")
        assert not hasattr(coupling_model_cfg, "fourier_include_input")

    def test_rejects_deprecated_flat_coupling_loss_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {},
            "coupling_training": {
                "lambda_consistency": 1.0,
                "lambda_flux_consistency": 0.25,
            },
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="deprecated flat coupling loss"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_invalid_coupling_loss_weight_mode(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {},
            "coupling_training": {
                "losses": {
                    "flux_consistency": {
                        "enabled": True,
                        "weight": 1.0,
                        "weight_mode": "bad_mode",
                    }
                }
            },
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="weight_mode must be one of"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_object_coupling_line_encoder_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "line_encoder": "cnn1d",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.line_encoder must be an object"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_object_coupling_trunk_fourier_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "trunk_fourier": True,
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.trunk_fourier must be an object"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_object_coupling_fusion_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "fusion": "gated_attention",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.fusion must be an object"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_invalid_coupling_fusion_type(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "fusion": {
                    "type": "bad_fusion",
                }
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.fusion.type must be one of"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_auto_operator_mode_for_l2_loss(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {},
            "coupling_training": {
                "losses": {
                    "l2_consistency": {
                        "enabled": True,
                        "weight": 1.0,
                        "weight_mode": "auto_operator",
                    }
                }
            },
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="l2_consistency.weight_mode must be 'manual'"):
            TrainCLI()._build_configs(config_path)
