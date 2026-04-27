from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from greenonet.config import (
    CompileConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingLossesConfig,
    CouplingLossTermConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from greenonet.compile_utils import maybe_compile_model
from greenonet.coupling_data import CouplingDataset
from greenonet.coupling_evaluator import CouplingEvaluator
from greenonet.coupling_model import CouplingNet
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel


class EvalCouplingCLI:
    """CLI for evaluating CouplingNet on test datasets."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Evaluate CouplingNet on test data."
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to JSON configuration file.",
        )
        parser.add_argument(
            "--work-dir",
            type=str,
            default="checkpoints/eval",
            help="Directory to store evaluation outputs.",
        )
        parser.add_argument(
            "--coupling-checkpoint",
            type=str,
            required=True,
            help="Path to CouplingNet checkpoint (.safetensors or .pth).",
        )
        parser.add_argument(
            "--green-checkpoint",
            type=str,
            required=True,
            help="Path to GreenONet checkpoint for computing kernels.",
        )
        self.parser = parser

    def _build_dataset_config(self, raw_dataset: dict) -> DatasetConfig:
        dataset_kwargs = dict(raw_dataset)
        dataset_kwargs.pop("domain", None)
        dtype_name = dataset_kwargs.pop("dtype", "float64")
        dataset_kwargs["dtype"] = getattr(torch, dtype_name)
        scale_len = dataset_kwargs.get("scale_length")
        if isinstance(scale_len, list) and len(scale_len) == 2:
            dataset_kwargs["scale_length"] = (float(scale_len[0]), float(scale_len[1]))
        for key in ("training_path", "validation_path", "test_path"):
            if key in dataset_kwargs and dataset_kwargs[key] is not None:
                dataset_kwargs[key] = Path(dataset_kwargs[key])
        return DatasetConfig(**dataset_kwargs)

    @staticmethod
    def _build_compile_config(
        raw_compile: object | None, section_name: str
    ) -> CompileConfig:
        if raw_compile is None:
            return CompileConfig()
        if isinstance(raw_compile, dict):
            return CompileConfig(**raw_compile)
        raise TypeError(f"{section_name}.compile must be an object.")

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
        losses_cfg = EvalCouplingCLI._build_coupling_losses_config(
            losses_raw, "coupling_training"
        )
        compile_cfg = EvalCouplingCLI._build_compile_config(compile_raw, "coupling_training")
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
        device = next(model.parameters()).device
        m_points = coords.shape[2]
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
        from greenonet.greens import ExactGreenFunction

        kernel_class = ExactGreenFunction(
            torch.linspace(0, 1, m_points, device=device), a=a_vals.to(device)
        )
        kernel = kernel_class().squeeze(0)
        return kernel.cpu()

    def run(self) -> None:
        args = self.parser.parse_args()
        with Path(args.config).open() as fp:
            raw = json.load(fp)

        dataset_cfg = self._build_dataset_config(raw["dataset"])
        coupling_model_kwargs = dict(raw.get("coupling_model", {}))
        cm_dtype = coupling_model_kwargs.pop("dtype", "float64")
        coupling_model_kwargs["dtype"] = getattr(torch, cm_dtype)
        coupling_model_cfg = CouplingModelConfig(**coupling_model_kwargs)
        training_cfg = self._build_training_config(raw.get("training", {}))
        coupling_training_cfg = self._build_coupling_training_config(
            raw.get("coupling_training", {})
        )

        if dataset_cfg.test_path is None:
            raise ValueError("test_path must be set in dataset config for evaluation.")

        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        def a_fun(x: Tensor, y: Tensor) -> Tensor:
            # h = 0.0625
            # x = x - h
            # y = y - h
            # return 1.0 + 0.5 * torch.exp(x + y) / math.exp(2.0)
            return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
            return 1.0 + 0.5 * torch.sin(4 * torch.pi * x) * torch.sin(4 * torch.pi * y)
            return 1.0 + 0.5 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
            return x**2 + y**2 + 1

        def apx_fun(x: Tensor, y: Tensor) -> Tensor:
            # h = 0.0625
            # x = x - h
            # y = y - h
            return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)
            return 0.5 * torch.exp(x + y) / math.exp(2.0)
            return (
                2.0
                * torch.pi
                * torch.cos(4 * torch.pi * x)
                * torch.sin(4 * torch.pi * y)
            )
            return 0.5 * torch.pi * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
            return 2 * x

        def apy_fun(x: Tensor, y: Tensor) -> Tensor:
            # h = 0.0625
            # x = x - h
            # y = y - h
            return (
                2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * y)
            )
            return 0.5 * torch.exp(x + y) / math.exp(2.0)
            return (
                2.0
                * torch.pi
                * torch.sin(4 * torch.pi * x)
                * torch.cos(4 * torch.pi * y)
            )
            return 0.5 * torch.pi * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
            return 2 * y

        def b_fun(x: Tensor, y: Tensor) -> Tensor:
            return torch.zeros_like(x)

        def c_fun(x: Tensor, y: Tensor) -> Tensor:
            # h = 0.0625
            # x = x - h
            # y = y - h
            return torch.zeros_like(x)
            # return 2.5 * (1 + 0.4 * torch.cos(torch.pi * x) * torch.sin(torch.pi * y))

        test_dataset = CouplingDataset(
            data_dir=dataset_cfg.test_path,
            step_size=dataset_cfg.step_size,
            n_points_per_line=dataset_cfg.n_points_per_line,
            dtype=dataset_cfg.dtype,
            integration_rule=coupling_training_cfg.integration_rule,
            a_fun=a_fun,
            b_fun=b_fun,
            c_fun=c_fun,
            ap_fun_x=apx_fun,
            ap_fun_y=apy_fun,
        )

        coupling_model = CouplingNet(coupling_model_cfg)
        try:
            coupling_model, coupling_model_cfg = load_model_with_config(
                Path(args.coupling_checkpoint)
            )
        except Exception:
            load_state_dict_auto(coupling_model, Path(args.coupling_checkpoint))
        coupling_model = maybe_compile_model(
            coupling_model.to(torch.device(coupling_training_cfg.device)),
            coupling_training_cfg.compile,
            model_name="CouplingNet",
        )

        # Build green kernel from green checkpoint
        model_kwargs = dict(raw.get("model", {}))
        model_dtype = model_kwargs.pop("dtype", "float64")
        model_kwargs["dtype"] = getattr(torch, model_dtype)
        green_model = GreenONetModel(ModelConfig(**model_kwargs))
        try:
            green_model, model_cfg = load_model_with_config(Path(args.green_checkpoint))
        except Exception:
            load_state_dict_auto(green_model, Path(args.green_checkpoint))
        green_model = maybe_compile_model(
            green_model.to(torch.device(coupling_training_cfg.device)),
            training_cfg.compile,
            model_name="GreenONetModel",
        )

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
        ) = test_dataset[0]
        sample_b = self._coeff_from_coords(sample_coords, b_fun)
        sample_c = self._coeff_from_coords(sample_coords, c_fun)
        green_kernel = self._compute_green_kernel(
            green_model,
            sample_coords.to(torch.device(coupling_training_cfg.device)),
            a_vals=sample_kappa.to(torch.device(coupling_training_cfg.device)),
            ap_vals=sample_ap.to(torch.device(coupling_training_cfg.device)),
            b_vals=sample_b.to(torch.device(coupling_training_cfg.device)),
            c_vals=sample_c.to(torch.device(coupling_training_cfg.device)),
        )

        evaluator = CouplingEvaluator(
            model=coupling_model,
            green_kernel=green_kernel,
            device=torch.device(coupling_training_cfg.device),
            work_dir=work_dir,
            integration_rule=coupling_training_cfg.integration_rule,
        )
        evaluator.evaluate(
            test_dataset,
            dataset_name="test",
            batch_size=coupling_training_cfg.batch_size,
        )


if __name__ == "__main__":
    EvalCouplingCLI().run()
