import json
import sys
from types import SimpleNamespace

import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    ComplexCouplingSourceConfig,
    ComplexPreProjectionFusionConfig,
    ComplexReferenceDiagnosticsConfig,
    CouplingBranchFusionConfig,
    CouplingModelConfig,
    CouplingTrainingConfig,
    DatasetConfig,
    IndexedGpSourceConfig,
    PipelineConfig,
    TrainingConfig,
    validate_complex_coupling_source_config,
)
from greenonet.model import GreenONetModel
from cli.eval_coupling import EvalCouplingCLI
from cli.train import TrainCLI
from test.complex_fixtures import write_coefficients, write_geometry_npz


class TestTrainCLIConfigCopy:
    def _write_config(self, path):
        payload = {
            "dataset": {
                "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0}
            },
            "model": {},
            "training": {},
            "terminal": {"width": 250},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        path.write_text(json.dumps(payload))
        return payload

    def test_copies_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        self._write_config(config_path)
        work_dir = tmp_path / "work"
        captured = {}

        def _fake_run_green_o_net(*_args, **kwargs):
            captured["terminal_width"] = kwargs["terminal_width"]
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
        used = json.loads(copied.read_text())
        assert used["training"]["optimizer"]["name"] == "adamw"
        assert used["training"]["optimizer"]["betas"] == [0.9, 0.999]
        assert used["green_optimizer_provenance"]["implementation"] == (
            "torch.optim.AdamW"
        )
        assert used["green_learning_rate_schedule"]["kind"] == "fixed"
        assert captured["terminal_width"] == 250

    def test_green_config_copy_materializes_soap_provenance(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {},
                    "training": {
                        "learning_rate": 0.002,
                        "epochs": 3,
                        "use_lr_schedule": True,
                        "warmup_epochs": 1,
                        "min_lr": 1e-5,
                        "optimizer": {
                            "name": "soap",
                            "soap": {"precondition_frequency": 7},
                        },
                    },
                    "pipeline": {"run_green": True, "run_coupling": False},
                }
            )
        )
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        training = TrainingConfig(
            learning_rate=0.002,
            epochs=3,
            use_lr_schedule=True,
            warmup_epochs=1,
            min_lr=1e-5,
            optimizer={
                "name": "soap",
                "soap": {"precondition_frequency": 7},
            },
        )

        TrainCLI._write_config_used(
            config_path=config_path,
            work_dir=work_dir,
            dataset_cfg=DatasetConfig(),
            training_cfg=training,
            coupling_training_cfg=CouplingTrainingConfig(),
            pipeline_cfg=PipelineConfig(run_green=True, run_coupling=False),
        )

        used = json.loads((work_dir / "config_used.json").read_text())
        assert used["training"]["optimizer"]["name"] == "soap"
        assert used["training"]["optimizer"]["soap"]["precondition_frequency"] == 7
        assert used["green_optimizer_provenance"]["upstream_commit"] == (
            "a1e553530fde97d0e6b307d7c82ac6d38b072340"
        )
        assert used["green_learning_rate_schedule"]["enabled"] is True

    def test_training_config_rejects_removed_adam_optimizer(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {},
                    "training": {"optimizer": {"name": "adam"}},
                    "pipeline": {"run_green": True, "run_coupling": False},
                }
            )
        )

        with pytest.raises(ValueError, match="Adam has been removed"):
            TrainCLI()._build_configs(config_path)

    def test_complex_config_copy_materializes_optimizer_provenance(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset": {"geometry_mode": "complex"},
                    "coupling_training": {
                        "learning_rate": 0.002,
                        "optimizer": {
                            "name": "soap",
                            "soap": {"precondition_frequency": 7},
                        },
                    },
                    "pipeline": {"run_green": False, "run_coupling": True},
                }
            )
        )
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        training = CouplingTrainingConfig(
            learning_rate=0.002,
            optimizer={
                "name": "soap",
                "soap": {"precondition_frequency": 7},
            },
        )

        TrainCLI._write_config_used(
            config_path=config_path,
            work_dir=work_dir,
            dataset_cfg=DatasetConfig(geometry_mode="complex"),
            coupling_training_cfg=training,
            pipeline_cfg=PipelineConfig(run_green=False, run_coupling=True),
        )

        used = json.loads((work_dir / "config_used.json").read_text())
        assert used["coupling_training"]["optimizer"]["name"] == "soap"
        assert used["coupling_training"]["optimizer"]["betas"] == [0.9, 0.999]
        assert (
            used["coupling_training"]["optimizer"]["soap"]["precondition_frequency"]
            == 7
        )
        assert used["optimizer_provenance"]["upstream_commit"] == (
            "a1e553530fde97d0e6b307d7c82ac6d38b072340"
        )
        assert used["dataset"]["coupling_source"]["mode"] == "npz"
        assert used["dataset"]["reference_diagnostics"] == {
            "training": True,
            "validation": True,
        }
        assert used["complex_source_provenance"]["fixed_across_epochs"] is True

    def test_uses_custom_coefficient_functions_for_green_training(
        self, tmp_path, monkeypatch
    ):
        config_path = tmp_path / "config.json"
        coeff_path = tmp_path / "custom_coefficients.py"
        coeff_path.write_text(
            "\n".join(
                [
                    "import torch",
                    "def a_fun(x, y): return x + y + 10.0",
                    "def apx_fun(x, y): return torch.ones_like(x) * 20.0",
                    "def apy_fun(x, y): return torch.ones_like(y) * 30.0",
                    "def bx_fun(x, y): return torch.ones_like(x) * 40.0",
                    "def by_fun(x, y): return torch.ones_like(y) * 60.0",
                    "def c_fun(x, y): return torch.ones_like(y) * 50.0",
                ]
            )
        )
        payload = self._write_config(config_path)
        payload["dataset"]["coefficient_functions_path"] = str(coeff_path)
        config_path.write_text(json.dumps(payload))
        work_dir = tmp_path / "work"
        captured = {}

        def _fake_run_green_o_net(*_args, **kwargs):
            x = torch.tensor([0.25], dtype=torch.float64)
            y = torch.tensor([0.5], dtype=torch.float64)
            captured["a"] = kwargs["a_fun"](x, y)
            captured["apx"] = kwargs["apx_fun"](x, y)
            captured["apy"] = kwargs["apy_fun"](x, y)
            captured["bx"] = kwargs["bx_fun"](x, y)
            captured["by"] = kwargs["by_fun"](x, y)
            captured["c"] = kwargs["c_fun"](x, y)
            return SimpleNamespace(model=GreenONetModel(kwargs["model_cfg"]))

        monkeypatch.setattr("cli.train.run_green_o_net", _fake_run_green_o_net)
        monkeypatch.setattr(
            sys,
            "argv",
            ["train.py", "--config", str(config_path), "--work-dir", str(work_dir)],
        )

        TrainCLI().run()

        torch.testing.assert_close(
            captured["a"], torch.tensor([10.75], dtype=torch.float64)
        )
        torch.testing.assert_close(
            captured["apx"], torch.tensor([20.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            captured["apy"], torch.tensor([30.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            captured["bx"], torch.tensor([40.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            captured["by"], torch.tensor([60.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            captured["c"], torch.tensor([50.0], dtype=torch.float64)
        )

    def test_convection_from_coords_uses_axis_specific_functions(self):
        coords = torch.tensor(
            [
                [[[0.0, 0.25], [1.0, 0.25]]],
                [[[0.5, 0.0], [0.5, 1.0]]],
            ],
            dtype=torch.float64,
        )

        def bx_fun(x, y):
            return 10.0 + x + 0.0 * y

        def by_fun(x, y):
            return 20.0 + y + 0.0 * x

        b_vals = TrainCLI._convection_from_coords(coords, bx_fun, by_fun)

        torch.testing.assert_close(
            b_vals,
            torch.tensor(
                [
                    [[10.0, 11.0]],
                    [[20.0, 21.0]],
                ],
                dtype=torch.float64,
            ),
        )


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
        assert dataset_cfg.coefficient_functions_path is None
        assert not hasattr(dataset_cfg, "domain")

    def test_existing_dataset_config_defaults_to_reference_npz(self):
        config = DatasetConfig.from_raw({"geometry_mode": "complex"})

        assert config.coupling_source == ComplexCouplingSourceConfig(mode="npz")
        assert config.reference_diagnostics == ComplexReferenceDiagnosticsConfig(
            training=True,
            validation=True,
        )

    def test_parses_indexed_gp_source_config(self):
        config = DatasetConfig.from_raw(
            {
                "geometry_mode": "complex",
                "geometry_path": "geometry.npz",
                "test_path": "test",
                "coupling_source": {
                    "mode": "indexed_gp",
                    "indexed_gp": {
                        "num_train": 12,
                        "num_valid": 3,
                        "seed": 4,
                        "lengthscale": 0.3,
                        "amplitude": 1.5,
                        "mean": -0.2,
                    },
                },
                "reference_diagnostics": {
                    "training": False,
                    "validation": False,
                },
            }
        )

        assert config.coupling_source.indexed_gp == IndexedGpSourceConfig(
            num_train=12,
            num_valid=3,
            seed=4,
            lengthscale=0.3,
            amplitude=1.5,
            mean=-0.2,
        )
        assert config.geometry_path is not None
        assert config.test_path is not None

    @pytest.mark.parametrize(
        ("raw", "message"),
        [
            (
                {"coupling_source": {"mode": "unsupported"}},
                "coupling_source.mode",
            ),
            (
                {
                    "coupling_source": {
                        "mode": "indexed_gp",
                        "indexed_gp": {
                            "num_train": 1,
                            "num_valid": 0,
                            "seed": -1,
                        },
                    }
                },
                "indexed_gp.seed",
            ),
            (
                {"reference_diagnostics": {"training": 1}},
                "reference_diagnostics.training",
            ),
            (
                {"coupling_source": {"mode": "npz", "unknown": True}},
                "unknown keys",
            ),
        ],
    )
    def test_rejects_invalid_source_config(self, raw, message):
        with pytest.raises((TypeError, ValueError), match=message):
            DatasetConfig.from_raw({"geometry_mode": "complex", **raw})

    def test_rejects_indexed_gp_paths_and_reference_diagnostics(self, tmp_path):
        source = ComplexCouplingSourceConfig(
            mode="indexed_gp",
            indexed_gp=IndexedGpSourceConfig(num_train=2, num_valid=1),
        )
        diagnostics = ComplexReferenceDiagnosticsConfig(
            training=False,
            validation=False,
        )
        training = CouplingTrainingConfig()

        with pytest.raises(ValueError, match="training_path"):
            validate_complex_coupling_source_config(
                DatasetConfig(
                    geometry_mode="complex",
                    training_path=tmp_path / "train",
                    coupling_source=source,
                    reference_diagnostics=diagnostics,
                ),
                training,
            )

    def test_rejects_indexed_gp_with_reference_diagnostics(self):
        source = ComplexCouplingSourceConfig(
            mode="indexed_gp",
            indexed_gp=IndexedGpSourceConfig(num_train=2, num_valid=1),
        )

        with pytest.raises(ValueError, match="reference_diagnostics"):
            validate_complex_coupling_source_config(
                DatasetConfig(
                    geometry_mode="complex",
                    coupling_source=source,
                ),
                CouplingTrainingConfig(),
            )

    def test_rejects_complex_source_options_for_unit_square(self):
        with pytest.raises(ValueError, match="complex geometry"):
            validate_complex_coupling_source_config(
                DatasetConfig(
                    geometry_mode="unit_square",
                    reference_diagnostics=ComplexReferenceDiagnosticsConfig(
                        training=False,
                        validation=False,
                    ),
                ),
                CouplingTrainingConfig(),
            )

    def test_requires_validation_source_for_best_checkpoint(self, tmp_path):
        with pytest.raises(ValueError, match="validation source"):
            validate_complex_coupling_source_config(
                DatasetConfig(
                    geometry_mode="complex",
                    training_path=tmp_path / "train",
                ),
                CouplingTrainingConfig(
                    best_energy_checkpoint={"enabled": True},
                ),
            )

    def test_builds_indexed_gp_train_and_validation_datasets(self, tmp_path):
        geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
        geometry = load_complex_geometry(geometry_path)
        coeffs = load_coefficient_functions(
            write_coefficients(tmp_path / "coefficients.py")
        )
        dataset_config = DatasetConfig(
            geometry_mode="complex",
            geometry_path=geometry_path,
            coupling_source=ComplexCouplingSourceConfig(
                mode="indexed_gp",
                indexed_gp=IndexedGpSourceConfig(
                    num_train=3,
                    num_valid=1,
                    seed=8,
                ),
            ),
            reference_diagnostics=ComplexReferenceDiagnosticsConfig(
                training=False,
                validation=False,
            ),
        )
        model_config = CouplingModelConfig(branch_input_dim=4)
        training_config = CouplingTrainingConfig(integration_rule="trapezoid")

        train = TrainCLI._build_complex_source_dataset(
            split="train",
            dataset_cfg=dataset_config,
            coupling_model_cfg=model_config,
            coupling_training_cfg=training_config,
            geometry=geometry,
            coeffs=coeffs,
        )
        validation = TrainCLI._build_complex_source_dataset(
            split="valid",
            dataset_cfg=dataset_config,
            coupling_model_cfg=model_config,
            coupling_training_cfg=training_config,
            geometry=geometry,
            coeffs=coeffs,
        )

        assert train is not None
        assert validation is not None
        assert len(train) == 3
        assert len(validation) == 1
        fixed_rhs = train[2].rhs_valid.clone()
        _ = train[0]
        torch.testing.assert_close(
            train[2].rhs_valid,
            fixed_rhs,
            rtol=0.0,
            atol=0.0,
        )
        assert not bool(train[0].has_solution)
        assert not bool(validation[0].has_solution)

    def test_parses_optional_complex_pre_projection_fusion(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"geometry_mode": "complex"},
            "model": {},
            "training": {},
            "coupling_model": {
                "pre_projection_fusion": {
                    "enabled": True,
                    "nonlinear_hidden_dim": 12,
                    "nonlinear_depth": 2,
                    "gate_initial_value": 0.1,
                    "eps": 1e-10,
                }
            },
            "coupling_training": {},
            "pipeline": {"run_green": False, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset,
            _model,
            _training,
            coupling_model,
            _coupling_training,
            _pipeline,
            _terminal,
        ) = TrainCLI()._build_configs(config_path)

        assert coupling_model.pre_projection_fusion == (
            ComplexPreProjectionFusionConfig(
                enabled=True,
                nonlinear_hidden_dim=12,
                nonlinear_depth=2,
                gate_initial_value=0.1,
                eps=1e-10,
            )
        )

    def test_parses_green_validation_dataset_controls(self, tmp_path):
        config_path = tmp_path / "config.json"
        coefficient_functions_path = tmp_path / "coefficients.py"
        payload = {
            "dataset": {
                "step_size": 0.25,
                "samples_per_line": 3,
                "validation_samples_per_line": 2,
                "scale_length": [0.05, 0.25],
                "validation_scale_length": [0.10, 0.20],
                "validation_sampler_mode": "backward",
                "coefficient_functions_path": str(coefficient_functions_path),
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
        assert dataset_cfg.coefficient_functions_path == coefficient_functions_path
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
                "losses": {
                    "l2_consistency": {
                        "enabled": True,
                        "weight": 1.5,
                    },
                    "energy_consistency": {
                        "enabled": True,
                        "weight": 0.25,
                    },
                    "cross_consistency": {
                        "enabled": False,
                        "weight": 2.0,
                    },
                    "balance_loss": {
                        "enabled": False,
                        "weight": 4.0,
                    },
                    "symmetric_boundary_loss": {
                        "enabled": False,
                        "weight": 5.0,
                    },
                },
                "learning_rate": 5e-4,
                "source_stencil_lift_learning_rate": 2.5e-5,
                "weight_decay": 1.0e-2,
                "source_stencil_lift_weight_decay": 2.0e-3,
                "gradient_clip_max_norm": 0.75,
                "epochs": 11,
                "use_lr_schedule": True,
                "warmup_epochs": 3,
                "min_lr": 1e-6,
                "periodic_checkpoint": {
                    "enabled": True,
                    "every_epochs": 4,
                },
                "best_rel_sol_checkpoint": {"enabled": True},
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
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert training_cfg.integration_rule == "trapezoid"
        assert training_cfg.compile.enabled is True
        assert coupling_training_cfg.integration_rule == "trapezoid"
        assert coupling_training_cfg.losses.l2_consistency.enabled is True
        assert coupling_training_cfg.losses.l2_consistency.weight == 1.5
        assert coupling_training_cfg.losses.energy_consistency.enabled is True
        assert coupling_training_cfg.losses.energy_consistency.weight == 0.25
        assert coupling_training_cfg.losses.cross_consistency.enabled is False
        assert coupling_training_cfg.losses.cross_consistency.weight == 2.0
        assert coupling_training_cfg.losses.balance_loss.enabled is False
        assert coupling_training_cfg.losses.balance_loss.weight == 4.0
        assert coupling_training_cfg.losses.symmetric_boundary_loss.enabled is False
        assert coupling_training_cfg.losses.symmetric_boundary_loss.weight == 5.0
        assert coupling_training_cfg.learning_rate == 5e-4
        assert coupling_training_cfg.source_stencil_lift_learning_rate == 2.5e-5
        assert coupling_training_cfg.weight_decay == 1.0e-2
        assert coupling_training_cfg.source_stencil_lift_weight_decay == 2.0e-3
        assert coupling_training_cfg.gradient_clip_max_norm == 0.75
        assert coupling_training_cfg.epochs == 11
        assert coupling_training_cfg.use_lr_schedule is True
        assert coupling_training_cfg.warmup_epochs == 3
        assert coupling_training_cfg.min_lr == 1e-6
        assert coupling_training_cfg.periodic_checkpoint.enabled is True
        assert coupling_training_cfg.periodic_checkpoint.every_epochs == 4
        assert coupling_training_cfg.best_rel_sol_checkpoint.enabled is True
        assert coupling_training_cfg.compile.enabled is True
        assert coupling_model_cfg.balance_projection.enabled is True
        assert coupling_model_cfg.balance_projection.mode == "symmetric"
        assert coupling_model_cfg.balance_projection.mask == "quadratic"
        assert coupling_model_cfg.smooth_mask_normalize is True
        assert coupling_model_cfg.smooth_mask_eps == 1e-12
        assert coupling_model_cfg.smooth_mask_power == 1.0
        assert coupling_model_cfg.smooth_mask_diff_power == 1.0
        assert coupling_model_cfg.smooth_mask_diff_power_trainable is False
        assert coupling_model_cfg.smooth_mask_diff_power_min == 0.25
        assert coupling_model_cfg.smooth_mask_diff_power_max == 2.0
        assert coupling_model_cfg.coefficient_terms.diffusion is True
        assert coupling_model_cfg.coefficient_terms.convection is False
        assert coupling_model_cfg.coefficient_terms.reaction is False
        assert coupling_model_cfg.branch_fusion.mode == "product"
        assert coupling_model_cfg.source_stencil_lift.enabled is False
        assert coupling_model_cfg.green_response_feature.enabled is False
        assert coupling_model_cfg.trunk_positional_encoding.enabled is False
        assert coupling_model_cfg.trunk_positional_encoding.mode == "fourier"
        assert coupling_model_cfg.trunk_positional_encoding.num_frequencies == 4
        assert coupling_model_cfg.trunk_positional_encoding.max_frequency == 8.0
        assert coupling_model_cfg.trunk_positional_encoding.include_input is True
        assert not hasattr(coupling_model_cfg, "use_fourier")
        assert not hasattr(coupling_model_cfg, "fourier_dim")
        assert not hasattr(coupling_model_cfg, "fourier_scale")
        assert not hasattr(coupling_model_cfg, "fourier_include_input")

    def test_parses_balance_projection_object_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "balance_projection": {
                    "enabled": True,
                    "mode": "symmetric",
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert coupling_model_cfg.balance_projection.enabled is True
        assert coupling_model_cfg.balance_projection.mode == "symmetric"
        assert coupling_training_cfg.losses.balance_loss.enabled is False

    def test_parses_branch_fusion_object_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "branch_fusion": {
                    "mode": "product_fuser",
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert coupling_model_cfg.branch_fusion.mode == "product_fuser"

    def test_parses_source_stencil_lift_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "balance_projection": "smooth_mask",
                "smooth_mask_normalize": False,
                "smooth_mask_eps": 1e-9,
                "smooth_mask_power": 0.5,
                "smooth_mask_diff_power": 0.75,
                "smooth_mask_diff_power_trainable": True,
                "smooth_mask_diff_power_min": 0.25,
                "smooth_mask_diff_power_max": 2.0,
                "source_stencil_lift": {
                    "enabled": True,
                    "encoder_type": "linear",
                    "coefficient_normalization": "tanh",
                    "coefficient_tanh_beta": 1.7,
                    "hidden_dim": 48,
                    "depth": 2,
                    "activation": "gelu",
                    "use_bias": False,
                    "dropout": 0.0,
                    "use_g_normalization": True,
                    "eps": 1e-12,
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        source_lift = coupling_model_cfg.source_stencil_lift
        assert coupling_model_cfg.balance_projection.enabled is True
        assert coupling_model_cfg.balance_projection.mode == "smooth_mask"
        assert coupling_model_cfg.balance_projection.mask == "quadratic"
        assert coupling_model_cfg.smooth_mask_normalize is False
        assert coupling_model_cfg.smooth_mask_eps == 1e-9
        assert coupling_model_cfg.smooth_mask_power == 0.5
        assert coupling_model_cfg.smooth_mask_diff_power == 0.75
        assert coupling_model_cfg.smooth_mask_diff_power_trainable is True
        assert coupling_model_cfg.smooth_mask_diff_power_min == 0.25
        assert coupling_model_cfg.smooth_mask_diff_power_max == 2.0
        assert source_lift.enabled is True
        assert source_lift.encoder_type == "linear"
        assert source_lift.coefficient_normalization == "tanh"
        assert source_lift.coefficient_tanh_beta == 1.7
        assert source_lift.hidden_dim == 48
        assert source_lift.depth == 2
        assert source_lift.activation == "gelu"
        assert source_lift.use_bias is False
        assert source_lift.dropout == 0.0
        assert source_lift.use_g_normalization is True
        assert source_lift.eps == 1e-12

    def test_parses_green_response_feature_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "green_response_feature": {
                    "enabled": True,
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert coupling_model_cfg.green_response_feature.enabled is True

    def test_parses_coefficient_terms_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "coefficient_terms": {
                    "diffusion": False,
                    "convection": True,
                    "reaction": True,
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        assert coupling_model_cfg.coefficient_terms.diffusion is False
        assert coupling_model_cfg.coefficient_terms.convection is True
        assert coupling_model_cfg.coefficient_terms.reaction is True

    def test_rejects_non_object_coefficient_terms_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "coefficient_terms": "diffusion",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.coefficient_terms"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_object_green_response_feature_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "green_response_feature": "enabled",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.green_response_feature"):
            TrainCLI()._build_configs(config_path)

    def test_parses_trunk_positional_encoding_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "trunk_positional_encoding": {
                    "enabled": True,
                    "mode": "boundary_algebraic",
                    "num_frequencies": 5,
                    "max_frequency": 16.0,
                    "include_input": False,
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        positional = coupling_model_cfg.trunk_positional_encoding
        assert positional.enabled is True
        assert positional.mode == "boundary_algebraic"
        assert positional.num_frequencies == 5
        assert positional.max_frequency == 16.0
        assert positional.include_input is False

    def test_rejects_non_object_trunk_positional_encoding_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "trunk_positional_encoding": "enabled",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.trunk_positional_encoding"):
            TrainCLI()._build_configs(config_path)

    def test_parses_axis_1d_trunk_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "axis_1d_trunk": {
                    "enabled": True,
                    "boundary_aware_modes": 5,
                    "num_frequencies": 6,
                    "max_frequency": 12.0,
                    "transverse_trunk": {
                        "enabled": True,
                        "fusion": "product_fuser",
                        "length_context": True,
                    },
                },
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        (
            _dataset_cfg,
            _model_cfg,
            _training_cfg,
            coupling_model_cfg,
            _coupling_training_cfg,
            _pipeline_cfg,
            _terminal_cfg,
        ) = TrainCLI()._build_configs(config_path)

        axis_1d_trunk = coupling_model_cfg.axis_1d_trunk
        assert axis_1d_trunk.enabled is True
        assert axis_1d_trunk.boundary_aware_modes == 5
        assert axis_1d_trunk.num_frequencies == 6
        assert axis_1d_trunk.max_frequency == 12.0
        assert axis_1d_trunk.transverse_trunk.enabled is True
        assert axis_1d_trunk.transverse_trunk.fusion == "product_fuser"
        assert axis_1d_trunk.transverse_trunk.length_context is True

    def test_rejects_non_object_axis_1d_trunk_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "axis_1d_trunk": "enabled",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.axis_1d_trunk"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_object_source_stencil_lift_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {
                "source_stencil_lift": "enabled",
            },
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.source_stencil_lift"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_removed_coupler_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {"coupler": {"enabled": True}},
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="coupling_model.coupler has been removed"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_removed_hybrid_detach_config(self):
        with pytest.raises(
            TypeError, match="coupling_training.hybrid_detach has been removed"
        ):
            TrainCLI._build_coupling_training_config({"hybrid_detach": True})

    def test_rejects_removed_stage2_config(self):
        with pytest.raises(
            TypeError, match="coupling_training.stage2 has been removed"
        ):
            TrainCLI._build_coupling_training_config({"stage2": True})

    def test_source_stencil_lift_learning_rate_defaults_to_none(self):
        cfg = TrainCLI._build_coupling_training_config({})

        assert cfg.source_stencil_lift_learning_rate is None
        assert cfg.weight_decay == 0.0
        assert cfg.source_stencil_lift_weight_decay is None
        assert cfg.gradient_clip_max_norm == 1.0

    def test_coupling_gradient_clip_max_norm_can_be_disabled(self):
        cfg = TrainCLI._build_coupling_training_config({"gradient_clip_max_norm": None})

        assert cfg.gradient_clip_max_norm is None

    def test_eval_cli_parses_source_stencil_lift_learning_rate(self):
        cfg = EvalCouplingCLI._build_coupling_training_config(
            {
                "learning_rate": 1.0e-3,
                "source_stencil_lift_learning_rate": 5.0e-5,
                "weight_decay": 1.0e-2,
                "source_stencil_lift_weight_decay": 2.0e-3,
                "gradient_clip_max_norm": 0.5,
            }
        )

        assert cfg.learning_rate == 1.0e-3
        assert cfg.source_stencil_lift_learning_rate == 5.0e-5
        assert cfg.weight_decay == 1.0e-2
        assert cfg.source_stencil_lift_weight_decay == 2.0e-3
        assert cfg.gradient_clip_max_norm == 0.5

    def test_eval_cli_rejects_removed_hybrid_detach_config(self):
        with pytest.raises(
            TypeError, match="coupling_training.hybrid_detach has been removed"
        ):
            EvalCouplingCLI._build_coupling_training_config({"hybrid_detach": True})

    def test_eval_cli_rejects_removed_stage2_config(self):
        with pytest.raises(
            TypeError, match="coupling_training.stage2 has been removed"
        ):
            EvalCouplingCLI._build_coupling_training_config({"stage2": True})

    def test_source_stencil_lift_defaults(self):
        cfg = TrainCLI._build_source_stencil_lift_config(None, "coupling_model")

        assert cfg.enabled is False
        assert cfg.encoder_type == "mlp"
        assert cfg.coefficient_normalization == "rms"
        assert cfg.coefficient_tanh_beta == 1.0
        assert cfg.hidden_dim == 32
        assert cfg.depth == 2
        assert cfg.activation == "gelu"
        assert cfg.use_bias is True
        assert cfg.dropout == 0.0
        assert cfg.use_g_normalization is True
        assert cfg.eps == 1e-12

    def test_green_response_feature_defaults(self):
        cfg = TrainCLI._build_green_response_feature_config(None, "coupling_model")

        assert cfg.enabled is False

    def test_coefficient_terms_defaults(self):
        cfg = TrainCLI._build_coefficient_terms_config(None, "coupling_model")

        assert cfg.diffusion is True
        assert cfg.convection is False
        assert cfg.reaction is False

    def test_branch_fusion_defaults(self):
        cfg = TrainCLI._build_branch_fusion_config(None, "coupling_model")

        assert cfg.mode == "product"

    def test_eval_cli_parses_branch_fusion_config(self):
        cfg = EvalCouplingCLI._build_branch_fusion_config(
            {"mode": "product_fuser"},
            "coupling_model",
        )

        assert cfg == CouplingBranchFusionConfig(mode="product_fuser")

    def test_eval_cli_rejects_non_object_branch_fusion_config(self):
        with pytest.raises(TypeError, match="coupling_model.branch_fusion"):
            EvalCouplingCLI._build_branch_fusion_config(
                "product_fuser",
                "coupling_model",
            )

    def test_eval_cli_rejects_invalid_branch_fusion_mode(self):
        with pytest.raises(ValueError, match="coupling_model.branch_fusion.mode"):
            EvalCouplingCLI._build_branch_fusion_config(
                {"mode": "concat_fuser"},
                "coupling_model",
            )

    def test_eval_cli_parses_coefficient_terms_config(self):
        cfg = EvalCouplingCLI._build_coefficient_terms_config(
            {
                "diffusion": False,
                "convection": True,
                "reaction": True,
            },
            "coupling_model",
        )

        assert cfg.diffusion is False
        assert cfg.convection is True
        assert cfg.reaction is True

    def test_eval_cli_rejects_non_object_coefficient_terms_config(self):
        with pytest.raises(TypeError, match="coupling_model.coefficient_terms"):
            EvalCouplingCLI._build_coefficient_terms_config(
                "diffusion",
                "coupling_model",
            )

    def test_eval_cli_parses_green_response_feature_config(self):
        cfg = EvalCouplingCLI._build_green_response_feature_config(
            {"enabled": True},
            "coupling_model",
        )

        assert cfg.enabled is True

    def test_eval_cli_rejects_non_object_green_response_feature_config(self):
        with pytest.raises(TypeError, match="coupling_model.green_response_feature"):
            EvalCouplingCLI._build_green_response_feature_config(
                "enabled",
                "coupling_model",
            )

    def test_trunk_positional_encoding_defaults(self):
        cfg = TrainCLI._build_trunk_positional_encoding_config(None, "coupling_model")

        assert cfg.enabled is False
        assert cfg.mode == "fourier"
        assert cfg.num_frequencies == 4
        assert cfg.max_frequency == 8.0
        assert cfg.include_input is True

    def test_axis_1d_trunk_defaults(self):
        cfg = TrainCLI._build_axis_1d_trunk_config(None, "coupling_model")

        assert cfg.enabled is False
        assert cfg.boundary_aware_modes == 4
        assert cfg.num_frequencies == 4
        assert cfg.max_frequency == 8.0

    def test_eval_cli_parses_axis_1d_trunk_config(self):
        cfg = EvalCouplingCLI._build_axis_1d_trunk_config(
            {
                "enabled": True,
                "boundary_aware_modes": 6,
                "num_frequencies": 7,
                "max_frequency": 14.0,
                "transverse_trunk": {
                    "enabled": True,
                    "fusion": "product",
                    "length_context": True,
                },
            },
            "coupling_model",
        )

        assert cfg.enabled is True
        assert cfg.boundary_aware_modes == 6
        assert cfg.num_frequencies == 7
        assert cfg.max_frequency == 14.0
        assert cfg.transverse_trunk.enabled is True
        assert cfg.transverse_trunk.fusion == "product"
        assert cfg.transverse_trunk.length_context is True

    def test_eval_cli_rejects_non_object_axis_1d_trunk_config(self):
        with pytest.raises(TypeError, match="coupling_model.axis_1d_trunk"):
            EvalCouplingCLI._build_axis_1d_trunk_config(
                "enabled",
                "coupling_model",
            )


class TestCanonicalEnergyTrainingConfig:
    def test_parses_best_energy_checkpoint_without_transition_config(self):
        cfg = TrainCLI._build_coupling_training_config(
            {
                "best_energy_checkpoint": {"enabled": True},
                "best_rel_sol_checkpoint": {"enabled": False},
            }
        )

        assert not hasattr(cfg, "length_jump_balance")
        assert cfg.best_energy_checkpoint.enabled is True
        assert cfg.best_rel_sol_checkpoint.enabled is False

    def test_train_cli_rejects_retired_length_jump_config(self):
        with pytest.raises(TypeError, match="full-domain canonical"):
            TrainCLI._build_coupling_training_config(
                {"length_jump_balance": {"enabled": True}}
            )

    def test_eval_cli_rejects_retired_length_jump_config(self):
        with pytest.raises(TypeError, match="full-domain canonical"):
            EvalCouplingCLI._build_coupling_training_config(
                {"length_jump_balance": {"enabled": True}}
            )

    def test_eval_cli_parses_trunk_positional_encoding_config(self):
        cfg = EvalCouplingCLI._build_trunk_positional_encoding_config(
            {
                "enabled": True,
                "mode": "boundary_algebraic",
                "num_frequencies": 6,
                "max_frequency": 32.0,
                "include_input": False,
            },
            "coupling_model",
        )

        assert cfg.enabled is True
        assert cfg.mode == "boundary_algebraic"
        assert cfg.num_frequencies == 6
        assert cfg.max_frequency == 32.0
        assert cfg.include_input is False

    def test_eval_cli_rejects_non_object_trunk_positional_encoding_config(self):
        with pytest.raises(TypeError, match="coupling_model.trunk_positional_encoding"):
            EvalCouplingCLI._build_trunk_positional_encoding_config(
                "enabled",
                "coupling_model",
            )


class TestCouplingOptimizerConfig:
    def test_train_and_eval_parse_soap_optimizer_config(self):
        raw = {
            "optimizer": {
                "name": "soap",
                "betas": [0.95, 0.95],
                "eps": 1e-8,
                "profile_step_time": True,
                "soap": {
                    "shampoo_beta": -1.0,
                    "precondition_frequency": 10,
                    "max_precondition_dim": 1024,
                    "merge_dims": False,
                    "precondition_1d": False,
                    "normalize_grads": False,
                    "correct_bias": True,
                },
            }
        }

        for builder in (
            TrainCLI._build_coupling_training_config,
            EvalCouplingCLI._build_coupling_training_config,
        ):
            config = builder(raw)
            assert config.optimizer.name == "soap"
            assert config.optimizer.betas == (0.95, 0.95)
            assert config.optimizer.profile_step_time is True
            assert config.optimizer.soap.precondition_frequency == 10
            assert config.optimizer.soap.max_precondition_dim == 1024

    def test_optimizer_block_is_optional_and_defaults_to_adamw(self):
        config = TrainCLI._build_coupling_training_config({})

        assert config.optimizer.name == "adamw"
        assert config.optimizer.betas == (0.9, 0.999)
        assert config.optimizer.eps == 1e-8
        assert config.optimizer.profile_step_time is False

    @pytest.mark.parametrize(
        "raw",
        (
            {"optimizer": {"unknown": 1}},
            {"optimizer": {"soap": {"unknown": 1}}},
        ),
    )
    def test_optimizer_unknown_keys_fail_fast(self, raw):
        with pytest.raises(TypeError, match="unknown keys"):
            EvalCouplingCLI._build_coupling_training_config(raw)


class TestComplexPhysicsLossTrainingConfig:
    def test_train_and_eval_parse_complex_physics_loss_configs(self):
        raw = {
            "relative_split_consistency": {
                "enabled": True,
                "weight": 2.0,
                "mass_weight": 3.0,
                "eps": 1e-10,
            },
            "weak_operator_closure": {
                "enabled": True,
                "weight": 4.0,
                "eps": 1e-9,
            },
            "best_physics_checkpoint": {"enabled": True},
        }

        for builder in (
            TrainCLI._build_coupling_training_config,
            EvalCouplingCLI._build_coupling_training_config,
        ):
            config = builder(raw)
            assert config.relative_split_consistency.enabled is True
            assert config.relative_split_consistency.weight == 2.0
            assert config.relative_split_consistency.mass_weight == 3.0
            assert config.relative_split_consistency.eps == 1e-10
            assert config.weak_operator_closure.enabled is True
            assert config.weak_operator_closure.weight == 4.0
            assert config.weak_operator_closure.eps == 1e-9
            assert config.best_physics_checkpoint.enabled is True

    @pytest.mark.parametrize(
        ("section", "field", "value"),
        (
            ("relative_split_consistency", "weight", -1.0),
            ("relative_split_consistency", "mass_weight", -1.0),
            ("relative_split_consistency", "eps", 0.0),
            ("weak_operator_closure", "weight", -1.0),
            ("weak_operator_closure", "eps", 0.0),
        ),
    )
    def test_rejects_invalid_complex_physics_loss_values(
        self,
        section,
        field,
        value,
    ):
        with pytest.raises(ValueError, match=rf"{section}\.{field}"):
            TrainCLI._build_coupling_training_config(
                {section: {"enabled": True, field: value}}
            )

    @pytest.mark.parametrize(
        "section",
        (
            "relative_split_consistency",
            "weak_operator_closure",
            "best_physics_checkpoint",
        ),
    )
    def test_rejects_unknown_complex_physics_config_keys(self, section):
        with pytest.raises(TypeError, match=rf"{section} has unknown keys"):
            EvalCouplingCLI._build_coupling_training_config(
                {section: {"enabled": True, "unknown": 1}}
            )

    def test_rejects_retired_admissibility_gluing_config(self):
        with pytest.raises(TypeError, match="admissibility_gluing"):
            TrainCLI._build_coupling_training_config(
                {"admissibility_gluing": {"enabled": True}}
            )

    def test_eval_cli_parses_balance_projection_object_config(self):
        cfg = EvalCouplingCLI._build_balance_projection_config(
            {
                "enabled": True,
                "mode": "symmetric",
            },
            "coupling_model",
        )

        assert cfg.enabled is True
        assert cfg.mode == "symmetric"

    def test_eval_cli_parses_source_stencil_lift_config(self):
        config_kwargs = {
            "branch_input_dim": 5,
            "balance_projection": "smooth_mask",
            "smooth_mask_normalize": False,
            "smooth_mask_eps": 1e-9,
            "smooth_mask_power": 0.5,
            "smooth_mask_diff_power": 0.75,
            "smooth_mask_diff_power_trainable": True,
            "smooth_mask_diff_power_min": 0.25,
            "smooth_mask_diff_power_max": 2.0,
            "source_stencil_lift": EvalCouplingCLI._build_source_stencil_lift_config(
                {
                    "enabled": True,
                    "encoder_type": "linear",
                    "coefficient_normalization": "tanh",
                    "coefficient_tanh_beta": 2.0,
                    "hidden_dim": 16,
                },
                "coupling_model",
            ),
        }
        model_cfg = CouplingModelConfig(
            **config_kwargs,
        )

        assert model_cfg.balance_projection.enabled is True
        assert model_cfg.balance_projection.mode == "smooth_mask"
        assert model_cfg.balance_projection.mask == "quadratic"
        assert model_cfg.smooth_mask_normalize is False
        assert model_cfg.smooth_mask_eps == 1e-9
        assert model_cfg.smooth_mask_power == 0.5
        assert model_cfg.smooth_mask_diff_power == 0.75
        assert model_cfg.smooth_mask_diff_power_trainable is True
        assert model_cfg.smooth_mask_diff_power_min == 0.25
        assert model_cfg.smooth_mask_diff_power_max == 2.0
        assert model_cfg.source_stencil_lift.enabled is True
        assert model_cfg.source_stencil_lift.encoder_type == "linear"
        assert model_cfg.source_stencil_lift.coefficient_normalization == "tanh"
        assert model_cfg.source_stencil_lift.coefficient_tanh_beta == 2.0
        assert model_cfg.source_stencil_lift.hidden_dim == 16

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


class TestTerminalConfig:
    def test_missing_terminal_defaults_to_auto_width(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "coupling_model": {},
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        *_configs, terminal_cfg = TrainCLI()._build_configs(config_path)

        assert terminal_cfg.width is None

    def test_parses_terminal_width(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "terminal": {"width": 250},
            "coupling_model": {},
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        *_configs, terminal_cfg = TrainCLI()._build_configs(config_path)

        assert terminal_cfg.width == 250

    def test_rejects_non_object_terminal_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "terminal": 250,
            "coupling_model": {},
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(TypeError, match="terminal must be an object"):
            TrainCLI()._build_configs(config_path)

    def test_rejects_non_positive_terminal_width(self, tmp_path):
        config_path = tmp_path / "config.json"
        payload = {
            "dataset": {"step_size": 0.25},
            "model": {},
            "training": {},
            "terminal": {"width": 0},
            "coupling_model": {},
            "coupling_training": {},
            "pipeline": {"run_green": True, "run_coupling": False},
        }
        config_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="terminal.width"):
            TrainCLI()._build_configs(config_path)

    def test_eval_cli_parses_terminal_width(self):
        cfg = EvalCouplingCLI._build_terminal_config({"width": 180})

        assert cfg.width == 180

    def test_eval_cli_rejects_non_object_terminal_config(self):
        with pytest.raises(TypeError, match="terminal must be an object"):
            EvalCouplingCLI._build_terminal_config("wide")

    def test_eval_cli_rejects_non_positive_terminal_width(self):
        with pytest.raises(ValueError, match="terminal.width"):
            EvalCouplingCLI._build_terminal_config({"width": -1})
