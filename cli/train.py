from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from greenonet.config import (
    CompileConfig,
    CouplerConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingLossesConfig,
    CouplingLossTermConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
    DatasetConfig,
    ModelConfig,
    PipelineConfig,
    TrainingConfig,
)
from greenonet.compile_utils import maybe_compile_model, model_state_dict_for_save
from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_model import CouplingNet
from greenonet.coupling_trainer import CouplingTrainer
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.runner import run_green_o_net


class TrainCLI:
    """Command-line entrypoint for model training."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Train GreenONet and CouplingNet.")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/default.json",
            help="Path to JSON configuration file.",
        )
        parser.add_argument(
            "--work-dir",
            type=str,
            default="checkpoints/run",
            help="Directory to store logs and artifacts.",
        )
        self.parser = parser

    @staticmethod
    def _parse_scale_length(
        dataset_kwargs: dict[str, object], key: str
    ) -> dict[str, object]:
        scale_len = dataset_kwargs.get(key)
        if isinstance(scale_len, list) and len(scale_len) == 2:
            dataset_kwargs[key] = (float(scale_len[0]), float(scale_len[1]))
        return dataset_kwargs

    def _build_configs(
        self, config_path: Path
    ) -> tuple[
        DatasetConfig,
        ModelConfig,
        TrainingConfig,
        CouplingModelConfig,
        CouplingTrainingConfig,
        PipelineConfig,
    ]:
        with config_path.open() as fp:
            raw = json.load(fp)
        dataset_kwargs = dict(raw["dataset"])
        dataset_kwargs.pop("domain", None)
        dtype_name = dataset_kwargs.pop("dtype", "float64")
        dataset_kwargs["dtype"] = getattr(torch, dtype_name)
        dataset_kwargs = self._parse_scale_length(dataset_kwargs, "scale_length")
        dataset_kwargs = self._parse_scale_length(
            dataset_kwargs, "validation_scale_length"
        )
        if (
            "training_path" in dataset_kwargs
            and dataset_kwargs["training_path"] is not None
        ):
            dataset_kwargs["training_path"] = Path(dataset_kwargs["training_path"])
        if (
            "validation_path" in dataset_kwargs
            and dataset_kwargs["validation_path"] is not None
        ):
            dataset_kwargs["validation_path"] = Path(dataset_kwargs["validation_path"])
        if "test_path" in dataset_kwargs and dataset_kwargs["test_path"] is not None:
            dataset_kwargs["test_path"] = Path(dataset_kwargs["test_path"])
        dataset_cfg = DatasetConfig(**dataset_kwargs)

        # model_kwargs = dict(raw["model"])
        # model_dtype = model_kwargs.pop("dtype", "float64")
        # model_kwargs["dtype"] = getattr(torch, model_dtype)
        # model_cfg = ModelConfig(**model_kwargs)
        # training_cfg = TrainingConfig(**raw["training"])
        model_kwargs = dict(raw.get("model", {}))
        model_dtype = model_kwargs.pop("dtype", "float64")
        model_kwargs["dtype"] = getattr(torch, model_dtype)
        model_cfg = ModelConfig(**model_kwargs)
        training_cfg = self._build_training_config(raw.get("training", {}))

        coupling_model_kwargs = dict(raw.get("coupling_model", {}))
        coupler_raw = coupling_model_kwargs.pop("coupler", None)
        coupler_cfg = self._build_coupler_config(coupler_raw, "coupling_model")
        cm_dtype = coupling_model_kwargs.pop("dtype", "float64")
        coupling_model_kwargs["dtype"] = getattr(torch, cm_dtype)
        coupling_model_cfg = CouplingModelConfig(
            coupler=coupler_cfg,
            **coupling_model_kwargs,
        )

        coupling_training_cfg = self._build_coupling_training_config(
            raw.get("coupling_training", {})
        )
        pipeline_cfg = PipelineConfig(**raw.get("pipeline", {}))
        return (
            dataset_cfg,
            model_cfg,
            training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            pipeline_cfg,
        )

    @staticmethod
    def _build_compile_config(
        raw_compile: object | None, section_name: str
    ) -> CompileConfig:
        if raw_compile is None:
            return CompileConfig()
        if isinstance(raw_compile, dict):
            return CompileConfig(**raw_compile)
        raise TypeError(f"{section_name}.compile must be an object.")

    @staticmethod
    def _build_coupler_config(
        raw_coupler: object | None,
        section_name: str,
    ) -> CouplerConfig:
        if raw_coupler is None:
            return CouplerConfig()
        if not isinstance(raw_coupler, dict):
            raise TypeError(f"{section_name}.coupler must be an object.")
        return CouplerConfig(**dict(raw_coupler))

    @classmethod
    def _build_training_config(
        cls,
        raw_training: dict[str, object],
    ) -> TrainingConfig:
        training_kwargs = dict(raw_training)
        compile_raw = training_kwargs.pop("compile", None)
        compile_cfg = cls._build_compile_config(compile_raw, "training")
        return TrainingConfig(compile=compile_cfg, **training_kwargs)

    @staticmethod
    def _build_coupling_training_config(
        raw_training: dict[str, object],
    ) -> CouplingTrainingConfig:
        coupling_training_kwargs = dict(raw_training)
        deprecated_loss_keys = {
            "lambda_consistency",
            "flux_consistency_enabled",
            "lambda_flux_consistency",
            "energy_consistency_enabled",
            "lambda_energy_consistency",
        }
        found_deprecated = sorted(
            key for key in deprecated_loss_keys if key in coupling_training_kwargs
        )
        if found_deprecated:
            raise TypeError(
                "deprecated flat coupling loss config is not supported; use "
                "coupling_training.losses.* instead "
                f"({', '.join(found_deprecated)})."
            )
        losses_raw = coupling_training_kwargs.pop("losses", None)
        compile_raw = coupling_training_kwargs.pop("compile", None)
        periodic_raw = coupling_training_kwargs.pop("periodic_checkpoint", None)
        best_rel_sol_raw = coupling_training_kwargs.pop("best_rel_sol_checkpoint", None)
        losses_cfg = TrainCLI._build_coupling_losses_config(
            losses_raw, "coupling_training"
        )
        compile_cfg = TrainCLI._build_compile_config(compile_raw, "coupling_training")
        if periodic_raw is None:
            periodic_cfg = CouplingPeriodicCheckpointConfig()
        elif isinstance(periodic_raw, dict):
            periodic_cfg = CouplingPeriodicCheckpointConfig(**periodic_raw)
        else:
            raise TypeError("coupling_training.periodic_checkpoint must be an object.")
        if best_rel_sol_raw is None:
            best_rel_sol_cfg = CouplingBestRelSolCheckpointConfig()
        elif isinstance(best_rel_sol_raw, dict):
            best_rel_sol_cfg = CouplingBestRelSolCheckpointConfig(**best_rel_sol_raw)
        else:
            raise TypeError(
                "coupling_training.best_rel_sol_checkpoint must be an object."
            )
        return CouplingTrainingConfig(
            losses=losses_cfg,
            compile=compile_cfg,
            periodic_checkpoint=periodic_cfg,
            best_rel_sol_checkpoint=best_rel_sol_cfg,
            **coupling_training_kwargs,
        )

    @staticmethod
    def _build_coupling_losses_config(
        raw_losses: object | None, section_name: str
    ) -> CouplingLossesConfig:
        if raw_losses is None:
            return CouplingLossesConfig()
        if not isinstance(raw_losses, dict):
            raise TypeError(f"{section_name}.losses must be an object.")

        loss_kwargs = dict(raw_losses)
        parsed: dict[str, CouplingLossTermConfig] = {}
        for key in ("l2_consistency", "energy_consistency", "cross_consistency"):
            raw_term = loss_kwargs.pop(key, None)
            if raw_term is None:
                parsed[key] = CouplingLossTermConfig()
            elif isinstance(raw_term, dict):
                parsed[key] = CouplingLossTermConfig(**raw_term)
            else:
                raise TypeError(f"{section_name}.losses.{key} must be an object.")
        if loss_kwargs:
            unknown = ", ".join(sorted(loss_kwargs))
            raise TypeError(f"{section_name}.losses has unknown keys: {unknown}.")
        return CouplingLossesConfig(**parsed)

    @staticmethod
    def a_fun(x: Tensor, y: Tensor) -> Tensor:
        # return (
        #     1
        #     + 0.35 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)
        #     + 0.25 * torch.sin(10 * torch.pi * x) * torch.sin(8 * torch.pi * y)
        # )
        # h = 0.0625
        # x = x - h
        # y = y - h
        return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
        # return 1.0 + 0.5 * torch.sin(2 * torch.pi * (x - y))
        # return 1.0 + 0.5 * torch.exp(x + y) / math.exp(2.0)
        # return 1.0 + 0.5 * torch.sin(4 * torch.pi * x) * torch.sin(4 * torch.pi * y)

    @staticmethod
    def apx_fun(x: Tensor, y: Tensor) -> Tensor:
        # pi = torch.pi
        # return 0.7 * pi * torch.cos(2 * pi * x) * torch.sin(
        #     2 * pi * y
        # ) + 2.5 * pi * torch.cos(10 * pi * x) * torch.sin(8 * pi * y)
        # h = 0.0625
        # x = x - h
        # y = y - h
        return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
        # return torch.pi * torch.cos(2 * torch.pi * (x - y))
        # return 0.5 * torch.exp(x + y) / math.exp(2.0)
        # return (
        #     2.0 * torch.pi * torch.cos(4 * torch.pi * x) * torch.sin(4 * torch.pi * y)
        # )

    @staticmethod
    def apy_fun(x: Tensor, y: Tensor) -> Tensor:
        # pi = torch.pi
        # return 0.7 * pi * torch.sin(2 * pi * x) * torch.cos(
        #     2 * pi * y
        # ) + 2.0 * pi * torch.sin(10 * pi * x) * torch.cos(8 * pi * y)
        # h = 0.0625
        # x = x - h
        # y = y - h
        return 2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * y)
        # return -torch.pi * torch.cos(2 * torch.pi * (x - y))
        # return 0.5 * torch.exp(x + y) / math.exp(2.0)
        # return (
        #     2.0 * torch.pi * torch.sin(4 * torch.pi * x) * torch.cos(4 * torch.pi * y)
        # )

    @staticmethod
    def b_fun(x: Tensor, y: Tensor) -> Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def c_fun(x: Tensor, y: Tensor) -> Tensor:
        # return 5 * (1 + 0.3 * torch.cos(6 * torch.pi * x) * torch.cos(6 * torch.pi * y))
        # h = 0.0625
        # x = x - h
        # y = y - h
        # return 2.5 * (1 + 0.4 * torch.cos(torch.pi * x) * torch.sin(torch.pi * y))
        # return 2.5 * (1 + 0.4 * torch.sin(torch.pi * x) * torch.cos(torch.pi * y))
        return torch.zeros_like(x)

    @staticmethod
    def _coeff_from_coords(
        coords: torch.Tensor, fn: Callable[[Tensor, Tensor], Tensor]
    ) -> torch.Tensor:
        x = coords[..., 0]
        y = coords[..., 1]
        return fn(x, y)

    @staticmethod
    def _compute_green_kernel(
        model: GreenONetModel,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor | None = None,
        b_vals: torch.Tensor | None = None,
        c_vals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Green kernel (2,n,m,m) using provided coefficients per axial line."""
        m_points = coords.shape[2]
        device = next(model.parameters()).device
        trunk_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, m_points, device=device),
                torch.linspace(0.0, 1.0, m_points, device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        if a_vals.dim() == 3:
            a_vals = a_vals.unsqueeze(0)
        if ap_vals is None:
            ap_vals = torch.zeros_like(a_vals)
        if b_vals is None:
            b_vals = torch.zeros_like(a_vals)
        if c_vals is None:
            c_vals = torch.zeros_like(a_vals)
        if ap_vals.dim() == 3:
            ap_vals = ap_vals.unsqueeze(0)
        if b_vals.dim() == 3:
            b_vals = b_vals.unsqueeze(0)
        if c_vals.dim() == 3:
            c_vals = c_vals.unsqueeze(0)
        with torch.no_grad():
            kernel = model(
                trunk_grid=trunk_grid,
                a_vals=a_vals.to(device),
                ap_vals=ap_vals.to(device),
                b_vals=b_vals.to(device),
                c_vals=c_vals.to(device),
            )  # (2,n,m,m)
        # from greenonet.greens import ExactGreenFunction
        # kernel_class = ExactGreenFunction(torch.linspace(0, 1, m_points, device=device), a=a_vals.to(device))
        # kernel0 = kernel_class().squeeze(0)
        return kernel.cpu()

    def run(self) -> None:
        args = self.parser.parse_args()
        config_path = Path(args.config)
        (
            dataset_cfg,
            model_cfg,
            training_cfg,
            coupling_model_cfg,
            coupling_training_cfg,
            pipeline_cfg,
        ) = self._build_configs(config_path)

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, work_dir / "config_used.json")

        green_model = GreenONetModel(model_cfg)
        green_kernel: torch.Tensor | None = None

        if pipeline_cfg.run_green:
            green_trainer = run_green_o_net(
                a_fun=self.a_fun,
                apx_fun=self.apx_fun,
                apy_fun=self.apy_fun,
                b_fun=self.b_fun,
                c_fun=self.c_fun,
                activation=model_cfg.activation,
                work_dir=work_dir,
                ndata=dataset_cfg.samples_per_line,
                validation_ndata=dataset_cfg.validation_samples_per_line,
                seed=training_cfg.epochs,
                scale_length=dataset_cfg.scale_length,
                validation_scale_length=dataset_cfg.validation_scale_length,
                use_operator_learning=dataset_cfg.use_operator_learning,
                deterministic=dataset_cfg.deterministic,
                sampler_mode=dataset_cfg.sampler_mode,
                validation_sampler_mode=dataset_cfg.validation_sampler_mode,
                n_points_per_line=model_cfg.branch_input_dim,
                step_size=dataset_cfg.step_size,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
            )
            green_model.load_state_dict(model_state_dict_for_save(green_trainer.model))
        else:
            if pipeline_cfg.green_pretrained_path is None:
                raise ValueError(
                    "green_pretrained_path must be set when not training GreenONet"
                )
            try:
                green_model, model_cfg = load_model_with_config(
                    pipeline_cfg.green_pretrained_path
                )
            except Exception:
                load_state_dict_auto(green_model, pipeline_cfg.green_pretrained_path)

        if pipeline_cfg.run_coupling:
            train_dir = dataset_cfg.training_path or Path("2D_data_variable")
            val_dir = dataset_cfg.validation_path
            coupling_train_dataset = CouplingDataset(
                data_dir=train_dir,
                step_size=dataset_cfg.step_size,
                n_points_per_line=dataset_cfg.n_points_per_line,
                dtype=dataset_cfg.dtype,
                integration_rule=coupling_training_cfg.integration_rule,
                a_fun=self.a_fun,
                b_fun=self.b_fun,
                c_fun=self.c_fun,
                ap_fun_x=self.apx_fun,
                ap_fun_y=self.apy_fun,
            )
            coupling_val_dataset = None
            if val_dir is not None:
                coupling_val_dataset = CouplingDataset(
                    data_dir=val_dir,
                    step_size=dataset_cfg.step_size,
                    n_points_per_line=dataset_cfg.n_points_per_line,
                    dtype=dataset_cfg.dtype,
                    integration_rule=coupling_training_cfg.integration_rule,
                    a_fun=self.a_fun,
                    b_fun=self.b_fun,
                    c_fun=self.c_fun,
                    ap_fun_x=self.apx_fun,
                    ap_fun_y=self.apy_fun,
                )
            test_dir = dataset_cfg.test_path
            coupling_test_dataset = None
            if test_dir is not None:
                coupling_test_dataset = CouplingDataset(
                    data_dir=test_dir,
                    step_size=dataset_cfg.step_size,
                    n_points_per_line=dataset_cfg.n_points_per_line,
                    dtype=dataset_cfg.dtype,
                    integration_rule=coupling_training_cfg.integration_rule,
                    a_fun=self.a_fun,
                    b_fun=self.b_fun,
                    c_fun=self.c_fun,
                    ap_fun_x=self.apx_fun,
                    ap_fun_y=self.apy_fun,
                )
            if green_kernel is None:
                (
                    sample_coords,
                    _,
                    _,
                    _,
                    _,
                    _,
                    sample_kappa,
                    _,
                    _,
                    sample_ap,
                ) = coupling_train_dataset[0]
                device = torch.device(coupling_training_cfg.device)
                sample_b = self._coeff_from_coords(sample_coords, self.b_fun)
                sample_c = self._coeff_from_coords(sample_coords, self.c_fun)
                green_model = maybe_compile_model(
                    green_model.to(device),
                    training_cfg.compile,
                    model_name="GreenONetModel",
                )
                green_kernel = self._compute_green_kernel(
                    green_model,
                    sample_coords.to(device),
                    a_vals=sample_kappa.to(device),
                    ap_vals=sample_ap.to(device),
                    b_vals=sample_b.to(device),
                    c_vals=sample_c.to(device),
                )
            coupling_model = CouplingNet(coupling_model_cfg)
            if pipeline_cfg.coupling_pretrained_path is not None:
                try:
                    coupling_model, coupling_model_cfg = load_model_with_config(
                        pipeline_cfg.coupling_pretrained_path
                    )
                except Exception:
                    load_state_dict_auto(
                        coupling_model, pipeline_cfg.coupling_pretrained_path
                    )
            coupling_trainer = CouplingTrainer(
                model=coupling_model,
                config=coupling_training_cfg,
                work_dir=work_dir,
                green_kernel=green_kernel,
                model_cfg=coupling_model_cfg,
            )
            coupling_trainer.train(coupling_train_dataset, coupling_val_dataset)
            if coupling_test_dataset is not None:
                evaluator = CouplingEvaluator(
                    model=coupling_model,
                    green_kernel=green_kernel,
                    device=torch.device(coupling_training_cfg.device),
                    work_dir=work_dir,
                    integration_rule=coupling_training_cfg.integration_rule,
                )
                evaluator.evaluate(
                    coupling_test_dataset,
                    dataset_name="test",
                    batch_size=coupling_training_cfg.batch_size,
                )


if __name__ == "__main__":
    TrainCLI().run()
