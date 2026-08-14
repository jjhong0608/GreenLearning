from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, List, cast

import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim, nn

from greenonet.data import axial_collate_fn
from greenonet.config import ModelConfig, TrainingConfig
from greenonet.logging_mixin import LoggingMixin
from greenonet.numerics import IntegrationRule, integrate
from greenonet.visualizer import GreenVisualizer, LossVisualizer
from greenonet.greens import (
    exact_green_kernel_from_coefficients,
    select_green_reference_policy,
)
from greenonet.compile_utils import maybe_compile_model, model_state_dict_for_save
from greenonet.green_lr_scheduler import GreenLearningRateSchedule
from greenonet.green_optimizer import GreenOptimizerFactory, GreenTrainingRecorder
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.optimizer_support import OptimizerStepProfiler
from greenonet.reproducibility import TrainingSeedContext, seed_dataloader_worker
from greenonet.training_step_schedule import StepValidationSchedule


AxialBatch = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class Trainer(LoggingMixin):
    """Training loop for the GreenONet model."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        work_dir: Path | str,
        model_cfg: ModelConfig | None = None,
        terminal_width: int | None = None,
        seed_context: TrainingSeedContext | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.model_cfg = model_cfg
        self.seed_context = seed_context
        if self.seed_context is None and config.seed is not None:
            self.seed_context = TrainingSeedContext(
                stage="green",
                base_seed=config.seed,
                deterministic_algorithms=config.deterministic_algorithms,
                device=config.device,
            )
        if self.seed_context is not None and self.seed_context.stage != "green":
            raise ValueError("Trainer requires a green seed context.")
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="Trainer",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model = maybe_compile_model(
            self.model,
            self.config.compile,
            logger=self.logger,
            model_name="GreenONetModel",
        )
        self._train_loader_generator: torch.Generator | None
        self._lbfgs_loader_generator: torch.Generator | None
        if self.seed_context is not None:
            self.seed_context.configure_process()
            self.seed_context.apply("runtime")
            self.seed_context.log(self.logger)
            self._train_loader_generator = self.seed_context.make_generator(
                "loader_train"
            )
            self._lbfgs_loader_generator = self.seed_context.make_generator(
                "loader_lbfgs"
            )
        else:
            self._train_loader_generator = None
            self._lbfgs_loader_generator = None
        self.loss_history: List[float] = []
        self.rel_sol_history: List[float] = []
        self.val_rel_sol_history: List[float] = []
        self.rel_green_history: List[float] = []
        self.optimizer_factory = GreenOptimizerFactory(config)
        self.optimizer_provenance = self.optimizer_factory.provenance()
        self.training_recorder = GreenTrainingRecorder(
            work_dir=self.work_dir,
            logger=self.logger,
            provenance=self.optimizer_provenance,
        )

    @staticmethod
    def _green_reconstruction_loss(
        prediction: torch.Tensor,
        source: torch.Tensor,
        solution: torch.Tensor,
        trunk_grid: torch.Tensor,
        integration_rule: IntegrationRule = "simpson",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate Green's function against the source and compare with the exact solution.

        prediction: (B, 2, n_lines, m, m) or (2, n_lines, m, m)
        source, solution: (B, 2, n_lines, m)
        trunk_grid: (m, m, 2)
        """
        if prediction.dim() == 4:
            # Expand shared Green kernel across batch
            prediction = prediction.unsqueeze(0).expand(source.shape[0], -1, -1, -1, -1)

        xi = trunk_grid[0, :, 1]
        x = trunk_grid[:, 0, 0]

        source_exp = source.unsqueeze(-2)
        rhs = torch.where(
            source_exp.abs() > 0,
            source_exp * prediction,
            torch.zeros_like(prediction),
        )  # (B,2,n,m,m)
        integ = integrate(
            rhs, x=xi, dim=-1, rule=integration_rule
        )  # integrate over xi -> (B,2,n,m)
        residual = solution - integ
        residual_energy = integrate(
            residual.pow(2), x=x, dim=-1, rule=integration_rule
        )  # (B,2,n)
        solution_energy = integrate(solution.pow(2), x=x, dim=-1, rule=integration_rule)
        # relative = torch.sqrt(residual_energy / (solution_energy + 1e-12)).mean()
        relative = torch.sqrt(residual_energy / solution_energy).mean()
        loss = residual_energy.mean()
        return loss, relative

    @staticmethod
    def _green_kernel_error(
        prediction: torch.Tensor,
        coords: torch.Tensor,
        a_val: torch.Tensor,
        ap_val: torch.Tensor,
        b_val: torch.Tensor,
        c_val: torch.Tensor,
        integration_rule: IntegrationRule = "simpson",
    ) -> torch.Tensor:
        """
        Compute relative L2 error between predicted and exact Green's kernel.

        prediction: (B,2,n,m,m) or (2,n,m,m)
        coords: (2,n,m,2)
        coefficient tensors: (B,2,n,m)
        """
        if prediction.dim() == 4:
            prediction = prediction.unsqueeze(0)
        if a_val.dim() == 3:
            a_val = a_val.unsqueeze(0)
        if b_val.dim() == 3:
            b_val = b_val.unsqueeze(0)
        if c_val.dim() == 3:
            c_val = c_val.unsqueeze(0)
        # Align dimensions
        b_pred, axes_pred, n_lines_pred, _m_points, _ = prediction.shape
        b_coeff = min(a_val.shape[0], b_val.shape[0], c_val.shape[0])
        axes = min(axes_pred, a_val.shape[1], b_val.shape[1], c_val.shape[1])
        n_lines = min(
            n_lines_pred,
            a_val.shape[2],
            b_val.shape[2],
            c_val.shape[2],
        )
        batch = min(b_pred, b_coeff)
        prediction = prediction[:batch, :axes, :n_lines]
        a_val = a_val[:batch, :axes, :n_lines]
        b_val = b_val[:batch, :axes, :n_lines]
        c_val = c_val[:batch, :axes, :n_lines]
        x_axis = coords[0, 0, :, 0].to(prediction.device)  # (m,)

        policy = select_green_reference_policy(b_val, c_val)
        if not policy.valid:
            return torch.full((), float("nan"), device=prediction.device)
        assert policy.reference is not None
        exact_kernel = exact_green_kernel_from_coefficients(
            coords=coords,
            a_vals=a_val,
            b_vals=b_val,
            reference=policy.reference,
        )

        # num = (prediction - exact_kernel).pow(2).sum()
        # den = exact_kernel.pow(2).sum().clamp_min(1e-12)
        num = (prediction - exact_kernel).pow(2)
        den = exact_kernel.pow(2)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule).clamp_min(1e-12)

        return torch.sqrt(num / den).mean()

    @staticmethod
    def _green_reconstruction_rel_by_line(
        prediction: torch.Tensor,
        source: torch.Tensor,
        solution: torch.Tensor,
        trunk_grid: torch.Tensor,
        integration_rule: IntegrationRule = "simpson",
    ) -> torch.Tensor:
        """
        Relative reconstruction error per sample/axis/line.

        Returns:
            rel_line: (B, 2, n_lines)
        """
        if prediction.dim() == 4:
            prediction = prediction.unsqueeze(0).expand(source.shape[0], -1, -1, -1, -1)

        xi = trunk_grid[0, :, 1]
        x = trunk_grid[:, 0, 0]
        source_exp = source.unsqueeze(-2)
        rhs = torch.where(
            source_exp.abs() > 0,
            source_exp * prediction,
            torch.zeros_like(prediction),
        )
        integ = integrate(rhs, x=xi, dim=-1, rule=integration_rule)
        residual = solution - integ
        residual_energy = integrate(residual.pow(2), x=x, dim=-1, rule=integration_rule)
        solution_energy = integrate(
            solution.pow(2), x=x, dim=-1, rule=integration_rule
        ).clamp_min(1e-12)
        return torch.sqrt(residual_energy / solution_energy)

    @staticmethod
    def _green_kernel_rel_by_line(
        prediction: torch.Tensor,
        coords: torch.Tensor,
        a_val: torch.Tensor,
        ap_val: torch.Tensor,
        b_val: torch.Tensor,
        c_val: torch.Tensor,
        integration_rule: IntegrationRule = "simpson",
    ) -> torch.Tensor:
        """
        Relative Green kernel error per sample/axis/line.

        Returns:
            rel_line: (B, 2, n_lines)
        """
        if prediction.dim() == 4:
            prediction = prediction.unsqueeze(0)
        if a_val.dim() == 3:
            a_val = a_val.unsqueeze(0)
        if b_val.dim() == 3:
            b_val = b_val.unsqueeze(0)
        if c_val.dim() == 3:
            c_val = c_val.unsqueeze(0)

        b_pred, axes_pred, n_lines_pred, _, _ = prediction.shape
        b_coeff = min(a_val.shape[0], b_val.shape[0], c_val.shape[0])
        axes = min(axes_pred, a_val.shape[1], b_val.shape[1], c_val.shape[1])
        n_lines = min(
            n_lines_pred,
            a_val.shape[2],
            b_val.shape[2],
            c_val.shape[2],
        )
        batch = min(b_pred, b_coeff)
        prediction = prediction[:batch, :axes, :n_lines]
        a_val = a_val[:batch, :axes, :n_lines]
        b_val = b_val[:batch, :axes, :n_lines]
        c_val = c_val[:batch, :axes, :n_lines]
        x_axis = coords[0, 0, :, 0].to(prediction.device)

        policy = select_green_reference_policy(b_val, c_val)
        if not policy.valid:
            return torch.full(
                (batch, axes, n_lines),
                float("nan"),
                device=prediction.device,
            )
        assert policy.reference is not None
        exact_kernel = exact_green_kernel_from_coefficients(
            coords=coords,
            a_vals=a_val,
            b_vals=b_val,
            reference=policy.reference,
        )

        num = (prediction - exact_kernel).pow(2)
        den = exact_kernel.pow(2)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule).clamp_min(1e-12)
        return torch.sqrt(num / den)

    def _aggregate_rel_sol_by_line(
        self, dataset: Dataset[AxialBatch]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eval_loader = self._make_eval_loader(dataset)
        rel_sol_sum: torch.Tensor | None = None
        rel_sol_sum_sq: torch.Tensor | None = None
        rel_sol_min: torch.Tensor | None = None
        rel_sol_max: torch.Tensor | None = None
        rel_sol_count = 0

        self.model.eval()
        with torch.no_grad():
            for (
                coords,
                solution,
                source,
                a_val,
                ap_val,
                b_val,
                c_val,
            ) in eval_loader:
                coords = coords.to(self.device)
                solution = solution.to(self.device)
                source = source.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)

                trunk_grid = self._build_trunk_grid(coords.shape[2])
                prediction = self.model(
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                )
                rel_sol_line = self._green_reconstruction_rel_by_line(
                    prediction=prediction,
                    source=source,
                    solution=solution,
                    trunk_grid=trunk_grid,
                    integration_rule=self.config.integration_rule,
                )
                rel_sol_cpu = rel_sol_line.detach().cpu().to(torch.float64)
                rel_sol_batch_sum = rel_sol_cpu.sum(dim=0)
                rel_sol_batch_sum_sq = rel_sol_cpu.pow(2).sum(dim=0)
                rel_sol_batch_min = rel_sol_cpu.min(dim=0).values
                rel_sol_batch_max = rel_sol_cpu.max(dim=0).values

                if rel_sol_sum is None:
                    rel_sol_sum = rel_sol_batch_sum
                    rel_sol_sum_sq = rel_sol_batch_sum_sq
                    rel_sol_min = rel_sol_batch_min
                    rel_sol_max = rel_sol_batch_max
                else:
                    assert rel_sol_sum_sq is not None
                    assert rel_sol_min is not None
                    assert rel_sol_max is not None
                    rel_sol_sum = rel_sol_sum + rel_sol_batch_sum
                    rel_sol_sum_sq = rel_sol_sum_sq + rel_sol_batch_sum_sq
                    rel_sol_min = torch.minimum(rel_sol_min, rel_sol_batch_min)
                    rel_sol_max = torch.maximum(rel_sol_max, rel_sol_batch_max)
                rel_sol_count += int(rel_sol_line.shape[0])

        if (
            rel_sol_sum is None
            or rel_sol_sum_sq is None
            or rel_sol_min is None
            or rel_sol_max is None
        ):
            raise ValueError(
                "Per-line reconstruction aggregation skipped: empty dataset."
            )

        mean = rel_sol_sum / max(rel_sol_count, 1)
        if rel_sol_count > 1:
            var = (rel_sol_sum_sq - rel_sol_sum.pow(2) / rel_sol_count) / (
                rel_sol_count - 1
            )
            std = torch.sqrt(var.clamp_min(0.0))
        else:
            std = torch.zeros_like(mean)
        return mean, rel_sol_min, rel_sol_max, std

    def _save_per_line_metrics(
        self,
        dataset: Dataset[AxialBatch],
        validation_dataset: Dataset[AxialBatch] | None = None,
    ) -> None:
        """
        Aggregate per-line reconstruction/Green errors across the full dataset
        and save CSV + summary JSON.
        """
        eval_loader = self._make_eval_loader(dataset)
        rel_sol_sum: torch.Tensor | None = None
        rel_sol_sum_sq: torch.Tensor | None = None
        rel_sol_min: torch.Tensor | None = None
        rel_sol_max: torch.Tensor | None = None
        rel_green_sum: torch.Tensor | None = None
        rel_green_sum_sq: torch.Tensor | None = None
        rel_green_min: torch.Tensor | None = None
        rel_green_max: torch.Tensor | None = None
        rel_sol_count = 0
        rel_green_count = 0
        line_coords_x: torch.Tensor | None = None
        line_coords_y: torch.Tensor | None = None

        self.model.eval()
        with torch.no_grad():
            for (
                coords,
                solution,
                source,
                a_val,
                ap_val,
                b_val,
                c_val,
            ) in eval_loader:
                coords = coords.to(self.device)
                solution = solution.to(self.device)
                source = source.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)

                if line_coords_x is None or line_coords_y is None:
                    line_coords_x = coords[0, :, 0, 1].detach().cpu()
                    line_coords_y = coords[1, :, 0, 0].detach().cpu()

                trunk_grid = self._build_trunk_grid(coords.shape[2])
                prediction = self.model(
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                )
                rel_sol_line = self._green_reconstruction_rel_by_line(
                    prediction=prediction,
                    source=source,
                    solution=solution,
                    trunk_grid=trunk_grid,
                    integration_rule=self.config.integration_rule,
                )  # (B,2,n)
                rel_green_line = self._green_kernel_rel_by_line(
                    prediction=prediction,
                    coords=coords,
                    a_val=a_val,
                    ap_val=ap_val,
                    b_val=b_val,
                    c_val=c_val,
                    integration_rule=self.config.integration_rule,
                )  # (B_or_1,2,n)

                rel_sol_cpu = rel_sol_line.detach().cpu().to(torch.float64)
                rel_green_cpu = rel_green_line.detach().cpu().to(torch.float64)

                rel_sol_batch_sum = rel_sol_cpu.sum(dim=0)
                rel_sol_batch_sum_sq = rel_sol_cpu.pow(2).sum(dim=0)
                rel_sol_batch_min = rel_sol_cpu.min(dim=0).values
                rel_sol_batch_max = rel_sol_cpu.max(dim=0).values

                rel_green_batch_sum = rel_green_cpu.sum(dim=0)
                rel_green_batch_sum_sq = rel_green_cpu.pow(2).sum(dim=0)
                rel_green_batch_min = rel_green_cpu.min(dim=0).values
                rel_green_batch_max = rel_green_cpu.max(dim=0).values

                if rel_sol_sum is None:
                    rel_sol_sum = rel_sol_batch_sum
                    rel_sol_sum_sq = rel_sol_batch_sum_sq
                    rel_sol_min = rel_sol_batch_min
                    rel_sol_max = rel_sol_batch_max
                else:
                    assert rel_sol_sum_sq is not None
                    assert rel_sol_min is not None
                    assert rel_sol_max is not None
                    rel_sol_sum = rel_sol_sum + rel_sol_batch_sum
                    rel_sol_sum_sq = rel_sol_sum_sq + rel_sol_batch_sum_sq
                    rel_sol_min = torch.minimum(rel_sol_min, rel_sol_batch_min)
                    rel_sol_max = torch.maximum(rel_sol_max, rel_sol_batch_max)
                if rel_green_sum is None:
                    rel_green_sum = rel_green_batch_sum
                    rel_green_sum_sq = rel_green_batch_sum_sq
                    rel_green_min = rel_green_batch_min
                    rel_green_max = rel_green_batch_max
                else:
                    assert rel_green_sum_sq is not None
                    assert rel_green_min is not None
                    assert rel_green_max is not None
                    rel_green_sum = rel_green_sum + rel_green_batch_sum
                    rel_green_sum_sq = rel_green_sum_sq + rel_green_batch_sum_sq
                    rel_green_min = torch.minimum(rel_green_min, rel_green_batch_min)
                    rel_green_max = torch.maximum(rel_green_max, rel_green_batch_max)
                rel_sol_count += int(rel_sol_line.shape[0])
                rel_green_count += int(rel_green_line.shape[0])

        if (
            rel_sol_sum is None
            or rel_sol_sum_sq is None
            or rel_sol_min is None
            or rel_sol_max is None
            or rel_green_sum is None
            or rel_green_sum_sq is None
            or rel_green_min is None
            or rel_green_max is None
            or line_coords_x is None
            or line_coords_y is None
        ):
            self.logger.warning("Per-line metric aggregation skipped: empty dataset.")
            return

        def _sample_stats(
            metric_sum: torch.Tensor,
            metric_sum_sq: torch.Tensor,
            metric_min: torch.Tensor,
            metric_max: torch.Tensor,
            metric_count: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            denom = max(metric_count, 1)
            mean = metric_sum / denom
            if metric_count > 1:
                var = (metric_sum_sq - metric_sum.pow(2) / metric_count) / (
                    metric_count - 1
                )
                std = torch.sqrt(var.clamp_min(0.0))
            else:
                std = torch.zeros_like(mean)
            return mean, metric_min, metric_max, std

        rel_sol_mean, rel_sol_min, rel_sol_max, rel_sol_std = _sample_stats(
            metric_sum=rel_sol_sum,
            metric_sum_sq=rel_sol_sum_sq,
            metric_min=rel_sol_min,
            metric_max=rel_sol_max,
            metric_count=rel_sol_count,
        )
        rel_green_mean, rel_green_min, rel_green_max, rel_green_std = _sample_stats(
            metric_sum=rel_green_sum,
            metric_sum_sq=rel_green_sum_sq,
            metric_min=rel_green_min,
            metric_max=rel_green_max,
            metric_count=rel_green_count,
        )
        n_axes, n_lines = rel_sol_mean.shape

        val_rel_sol_mean: torch.Tensor | None = None
        val_rel_sol_min: torch.Tensor | None = None
        val_rel_sol_max: torch.Tensor | None = None
        val_rel_sol_std: torch.Tensor | None = None
        if validation_dataset is not None:
            (
                val_rel_sol_mean,
                val_rel_sol_min,
                val_rel_sol_max,
                val_rel_sol_std,
            ) = self._aggregate_rel_sol_by_line(validation_dataset)

        csv_path = self.work_dir / "per_line_metrics.csv"
        with csv_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
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
                    "val_rel_sol_line",
                    "val_rel_sol_line_mean",
                    "val_rel_sol_line_min",
                    "val_rel_sol_line_max",
                    "val_rel_sol_line_std",
                ],
            )
            writer.writeheader()
            for axis in range(n_axes):
                axis_name = "x" if axis == 0 else "y"
                axis_coords = line_coords_x if axis == 0 else line_coords_y
                for line_idx in range(n_lines):
                    writer.writerow(
                        {
                            "axis_id": axis,
                            "axis_name": axis_name,
                            "line_index": line_idx,
                            "line_coordinate": float(axis_coords[line_idx].item()),
                            "rel_sol_line": float(rel_sol_mean[axis, line_idx].item()),
                            "rel_sol_line_mean": float(
                                rel_sol_mean[axis, line_idx].item()
                            ),
                            "rel_sol_line_min": float(
                                rel_sol_min[axis, line_idx].item()
                            ),
                            "rel_sol_line_max": float(
                                rel_sol_max[axis, line_idx].item()
                            ),
                            "rel_sol_line_std": float(
                                rel_sol_std[axis, line_idx].item()
                            ),
                            "rel_green_line": float(
                                rel_green_mean[axis, line_idx].item()
                            ),
                            "rel_green_line_mean": float(
                                rel_green_mean[axis, line_idx].item()
                            ),
                            "rel_green_line_min": float(
                                rel_green_min[axis, line_idx].item()
                            ),
                            "rel_green_line_max": float(
                                rel_green_max[axis, line_idx].item()
                            ),
                            "rel_green_line_std": float(
                                rel_green_std[axis, line_idx].item()
                            ),
                            "val_rel_sol_line": (
                                ""
                                if val_rel_sol_mean is None
                                else float(val_rel_sol_mean[axis, line_idx].item())
                            ),
                            "val_rel_sol_line_mean": (
                                ""
                                if val_rel_sol_mean is None
                                else float(val_rel_sol_mean[axis, line_idx].item())
                            ),
                            "val_rel_sol_line_min": (
                                ""
                                if val_rel_sol_min is None
                                else float(val_rel_sol_min[axis, line_idx].item())
                            ),
                            "val_rel_sol_line_max": (
                                ""
                                if val_rel_sol_max is None
                                else float(val_rel_sol_max[axis, line_idx].item())
                            ),
                            "val_rel_sol_line_std": (
                                ""
                                if val_rel_sol_std is None
                                else float(val_rel_sol_std[axis, line_idx].item())
                            ),
                        }
                    )

        flat_sol = rel_sol_mean.reshape(-1)
        flat_green = rel_green_mean.reshape(-1)
        sol_argmax = int(torch.argmax(flat_sol).item())
        green_argmax = int(torch.argmax(flat_green).item())
        summary = {
            "num_axes": int(n_axes),
            "num_lines_per_axis": int(n_lines),
            "mean_rel_sol_line": float(flat_sol.mean().item()),
            "mean_rel_green_line": float(flat_green.mean().item()),
            "max_rel_sol_line": float(flat_sol[sol_argmax].item()),
            "max_rel_sol_axis_id": int(sol_argmax // n_lines),
            "max_rel_sol_line_index": int(sol_argmax % n_lines),
            "max_rel_green_line": float(flat_green[green_argmax].item()),
            "max_rel_green_axis_id": int(green_argmax // n_lines),
            "max_rel_green_line_index": int(green_argmax % n_lines),
        }
        if val_rel_sol_mean is not None:
            flat_val_sol = val_rel_sol_mean.reshape(-1)
            val_sol_argmax = int(torch.argmax(flat_val_sol).item())
            summary.update(
                {
                    "mean_val_rel_sol_line": float(flat_val_sol.mean().item()),
                    "max_val_rel_sol_line": float(flat_val_sol[val_sol_argmax].item()),
                    "max_val_rel_sol_axis_id": int(val_sol_argmax // n_lines),
                    "max_val_rel_sol_line_index": int(val_sol_argmax % n_lines),
                }
            )
        summary_path = self.work_dir / "per_line_metrics_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self.logger.info(
            "Saved per-line metrics: %s and %s (mean rel_sol=%.4e, mean rel_green=%.4e)",
            csv_path,
            summary_path,
            summary["mean_rel_sol_line"],
            summary["mean_rel_green_line"],
        )

    def _make_loader(
        self,
        dataset: Dataset[AxialBatch],
        *,
        generator: torch.Generator | None = None,
    ) -> DataLoader[AxialBatch]:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=axial_collate_fn,
            pin_memory=True,
            generator=generator,
            worker_init_fn=seed_dataloader_worker if generator is not None else None,
        )

    def _make_eval_loader(self, dataset: Dataset[AxialBatch]) -> DataLoader[AxialBatch]:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=axial_collate_fn,
            pin_memory=True,
        )

    def _build_trunk_grid(self, m_points: int) -> torch.Tensor:
        return torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, m_points, device=self.device),
                torch.linspace(0.0, 1.0, m_points, device=self.device),
                indexing="ij",
            ),
            dim=-1,
        )

    def _dataset_rel_sol(self, dataset: Dataset[AxialBatch]) -> float:
        loader = self._make_eval_loader(dataset)
        total = 0.0
        count = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for (
                coords,
                solution,
                source,
                a_val,
                ap_val,
                b_val,
                c_val,
            ) in loader:
                coords = coords.to(self.device)
                solution = solution.to(self.device)
                source = source.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)
                trunk_grid = self._build_trunk_grid(coords.shape[2])
                prediction = self.model(
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                )
                rel_line = self._green_reconstruction_rel_by_line(
                    prediction=prediction,
                    source=source,
                    solution=solution,
                    trunk_grid=trunk_grid,
                    integration_rule=self.config.integration_rule,
                )
                total += float(rel_line.sum().item())
                count += int(rel_line.numel())
        if was_training:
            self.model.train()
        return total / max(count, 1)

    def train(
        self,
        dataset: Dataset[AxialBatch],
        validation_dataset: Dataset[AxialBatch] | None = None,
    ) -> None:
        self.model.train()
        loader = self._make_loader(
            dataset,
            generator=self._train_loader_generator,
        )
        if self.config.compute_validation_rel_sol and validation_dataset is None:
            raise ValueError(
                "validation_dataset must be provided when compute_validation_rel_sol=True."
            )
        steps_per_epoch = len(loader)
        if steps_per_epoch < 1:
            raise ValueError("GreenNet training loader must not be empty.")
        schedule_config = GreenLearningRateSchedule.from_config(
            self.config,
            steps_per_epoch=steps_per_epoch,
        )
        validation_schedule = (
            StepValidationSchedule.for_validation(
                validation_every_steps=self.config.validation_every_steps,
                total_optimizer_steps=schedule_config.total_optimizer_steps,
                field_prefix="training",
            )
            if self.config.compute_validation_rel_sol
            else None
        )
        optimizer = self.optimizer_factory.build(self.model.parameters())
        optimizer_profiler = OptimizerStepProfiler(
            optimizer=optimizer,
            enabled=self.optimizer_provenance.profile_step_time,
            device=self.device,
        )
        scheduler = schedule_config.build(optimizer)
        self.training_recorder.log_startup(schedule_config, validation_schedule)
        self.training_recorder.write_provenance(
            schedule_config,
            validation_schedule,
        )
        global_step = 0
        validation_index = 0

        for epoch in range(1, self.config.epochs + 1):
            epoch_losses: List[float] = []
            last_batch = None
            first_learning_rate: float | None = None
            last_learning_rate: float | None = None
            optimizer_profiler.begin_epoch()
            for step_in_epoch, (
                coords,
                solution,
                source,
                a_val,
                ap_val,
                b_val,
                c_val,
            ) in enumerate(loader, start=1):
                # coords: (2, n, m, 2) shared; fields: (B, 2, n, m)
                learning_rate = float(optimizer.param_groups[0]["lr"])
                if first_learning_rate is None:
                    first_learning_rate = learning_rate
                last_learning_rate = learning_rate
                coords = coords.to(self.device)
                solution = solution.to(self.device)
                source = source.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)

                trunk_grid = self._build_trunk_grid(coords.shape[2])

                optimizer.zero_grad()
                prediction = self.model(
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                )

                loss, _ = self._green_reconstruction_loss(
                    prediction=prediction,
                    source=source,
                    solution=solution,
                    trunk_grid=trunk_grid,
                    integration_rule=self.config.integration_rule,
                )
                cast(Any, loss).backward()
                optimizer_profiler.step()
                global_step += 1
                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(loss.detach().item())
                last_batch = (
                    coords,
                    solution,
                    source,
                    a_val,
                    ap_val,
                    b_val,
                    c_val,
                    trunk_grid,
                )
                if validation_schedule is not None and validation_schedule.is_due(
                    global_step
                ):
                    assert validation_dataset is not None
                    validation_index += 1
                    val_rel_sol = self._dataset_rel_sol(validation_dataset)
                    self.val_rel_sol_history.append(val_rel_sol)
                    self.training_recorder.record(
                        phase=self.optimizer_provenance.name,
                        epoch=epoch,
                        split="val",
                        global_step=global_step,
                        step_in_epoch=step_in_epoch,
                        validation_index=validation_index,
                        learning_rate=learning_rate,
                        loss=float("nan"),
                        rel_sol=val_rel_sol,
                    )
                    self.logger.info(
                        "Green validation epoch=%d global_step=%d "
                        "step_in_epoch=%d validation_index=%d "
                        "learning_rate=%.6e rel_sol=%.6e",
                        epoch,
                        global_step,
                        step_in_epoch,
                        validation_index,
                        learning_rate,
                        val_rel_sol,
                    )

            mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            self.loss_history.append(mean_loss)
            optimizer_metrics = optimizer_profiler.finish_epoch()
            epoch_rel_sol: float | None = None
            epoch_rel_green: float | None = None
            if first_learning_rate is None or last_learning_rate is None:
                raise ValueError("GreenNet training loader must not be empty.")

            # Logging metrics at interval using the last processed batch
            if epoch % self.config.log_interval == 0 and last_batch is not None:
                coords, solution, source, a_val, ap_val, b_val, c_val, trunk_grid = (
                    last_batch
                )
                with torch.no_grad():
                    pred_eval = self.model(
                        trunk_grid=trunk_grid,
                        a_vals=a_val,
                        ap_vals=ap_val,
                        b_vals=b_val,
                        c_vals=c_val,
                    )
                    rel_green = self._green_kernel_error(
                        prediction=pred_eval,
                        coords=coords,
                        a_val=a_val,
                        ap_val=ap_val,
                        b_val=b_val,
                        c_val=c_val,
                        integration_rule=self.config.integration_rule,
                    )
                    if self.config.compute_validation_rel_sol:
                        train_rel_sol = self._dataset_rel_sol(dataset)
                        self.rel_sol_history.append(train_rel_sol)
                        epoch_rel_sol = train_rel_sol
                    else:
                        _, rel_sol = self._green_reconstruction_loss(
                            prediction=pred_eval,
                            source=source,
                            solution=solution,
                            trunk_grid=trunk_grid,
                            integration_rule=self.config.integration_rule,
                        )
                        epoch_rel_sol = float(rel_sol.detach().item())
                        self.rel_sol_history.append(epoch_rel_sol)
                    epoch_rel_green = float(rel_green.detach().item())
                    self.rel_green_history.append(epoch_rel_green)
                if self.config.compute_validation_rel_sol:
                    self.logger.info(
                        "Epoch %s train global_step=%d: learning_rate=%.6e "
                        "learning_rate_first=%.6e learning_rate_last=%.6e | "
                        "loss=%.4e | train_rel_sol=%.4e | rel_green=%.4e%s",
                        epoch,
                        global_step,
                        last_learning_rate,
                        first_learning_rate,
                        last_learning_rate,
                        mean_loss,
                        self.rel_sol_history[-1],
                        self.rel_green_history[-1],
                        self._optimizer_metrics_log_suffix(optimizer_metrics),
                    )
                else:
                    self.logger.info(
                        "Epoch %s train global_step=%d: learning_rate=%.6e "
                        "learning_rate_first=%.6e learning_rate_last=%.6e | "
                        "loss=%.4e | rel_sol=%.4e | rel_green=%.4e%s",
                        epoch,
                        global_step,
                        last_learning_rate,
                        first_learning_rate,
                        last_learning_rate,
                        mean_loss,
                        self.rel_sol_history[-1],
                        self.rel_green_history[-1],
                        self._optimizer_metrics_log_suffix(optimizer_metrics),
                    )
            elif epoch % self.config.log_interval == 0:
                self.logger.info(
                    "Epoch %s train global_step=%d: learning_rate=%.6e "
                    "learning_rate_first=%.6e learning_rate_last=%.6e | "
                    "loss=%.4e%s",
                    epoch,
                    global_step,
                    last_learning_rate,
                    first_learning_rate,
                    last_learning_rate,
                    mean_loss,
                    self._optimizer_metrics_log_suffix(optimizer_metrics),
                )
            self.training_recorder.record(
                phase=self.optimizer_provenance.name,
                epoch=epoch,
                split="train",
                global_step=global_step,
                step_in_epoch=steps_per_epoch,
                learning_rate=last_learning_rate,
                learning_rate_first=first_learning_rate,
                learning_rate_last=last_learning_rate,
                loss=mean_loss,
                rel_sol=epoch_rel_sol,
                rel_green=epoch_rel_green,
                telemetry=optimizer_metrics,
            )

        self._save_model_checkpoint("model_pre_lbfgs.safetensors")

        # Optional LBFGS fine-tuning stage
        if self.config.lbfgs_max_iter > 0 and self.config.lbfgs_epochs > 0:
            self.logger.info(
                "Starting LBFGS fine-tuning (epochs=%s, max_iter=%s, lr=%s)",
                self.config.lbfgs_epochs,
                self.config.lbfgs_max_iter,
                self.config.lbfgs_lr,
            )
            lbfgs_optimizer = optim.LBFGS(
                self.model.parameters(),
                lr=self.config.lbfgs_lr,
                max_iter=self.config.lbfgs_max_iter,
                max_eval=1250 * 1000 // 15000,
                history_size=self.config.lbfgs_history_size,
                tolerance_grad=self.config.lbfgs_tolerance_grad,
                tolerance_change=0,
                line_search_fn="strong_wolfe",
            )

            for lbfgs_epoch in range(1, self.config.lbfgs_epochs + 1):
                lbfgs_losses: List[float] = []
                lbfgs_loader = self._make_loader(
                    dataset,
                    generator=self._lbfgs_loader_generator,
                )
                for (
                    coords,
                    solution,
                    source,
                    a_val,
                    ap_val,
                    b_val,
                    c_val,
                ) in lbfgs_loader:
                    coords = coords.to(self.device)
                    solution = solution.to(self.device)
                    source = source.to(self.device)
                    a_val = a_val.to(self.device)
                    ap_val = ap_val.to(self.device)
                    b_val = b_val.to(self.device)
                    c_val = c_val.to(self.device)

                    trunk_grid = self._build_trunk_grid(coords.shape[2])

                    def closure() -> torch.Tensor:
                        lbfgs_optimizer.zero_grad()
                        prediction = self.model(
                            trunk_grid=trunk_grid,
                            a_vals=a_val,
                            ap_vals=ap_val,
                            b_vals=b_val,
                            c_vals=c_val,
                        )
                        loss, _ = self._green_reconstruction_loss(
                            prediction=prediction,
                            source=source,
                            solution=solution,
                            trunk_grid=trunk_grid,
                            integration_rule=self.config.integration_rule,
                        )
                        cast(Any, loss).backward()
                        return loss

                    loss = cast(Any, lbfgs_optimizer).step(closure)
                    lbfgs_losses.append(loss.item())

                if lbfgs_losses:
                    last_lbfgs = float(lbfgs_losses[-1])
                    self.loss_history.append(last_lbfgs)
                    # evaluate rel errors on last batch
                    with torch.no_grad():
                        pred_eval = self.model(
                            trunk_grid=trunk_grid,
                            a_vals=a_val,
                            ap_vals=ap_val,
                            b_vals=b_val,
                            c_vals=c_val,
                        )
                        rel_green = self._green_kernel_error(
                            prediction=pred_eval,
                            coords=coords,
                            a_val=a_val,
                            ap_val=ap_val,
                            b_val=b_val,
                            c_val=c_val,
                            integration_rule=self.config.integration_rule,
                        )
                        if self.config.compute_validation_rel_sol:
                            train_rel_sol = self._dataset_rel_sol(dataset)
                            assert validation_dataset is not None
                            val_rel_sol = self._dataset_rel_sol(validation_dataset)
                            self.rel_sol_history.append(train_rel_sol)
                            self.val_rel_sol_history.append(val_rel_sol)
                        else:
                            _, rel_sol = self._green_reconstruction_loss(
                                prediction=pred_eval,
                                source=source,
                                solution=solution,
                                trunk_grid=trunk_grid,
                                integration_rule=self.config.integration_rule,
                            )
                            self.rel_sol_history.append(float(rel_sol.detach().item()))
                        self.rel_green_history.append(float(rel_green.detach().item()))
                    if self.config.compute_validation_rel_sol:
                        self.logger.info(
                            "LBFGS epoch %s last loss: %.4e | train_rel_sol=%.4e | val_rel_sol=%.4e | rel_green=%.4e",
                            lbfgs_epoch,
                            last_lbfgs,
                            self.rel_sol_history[-1],
                            self.val_rel_sol_history[-1],
                            self.rel_green_history[-1],
                        )
                    else:
                        self.logger.info(
                            "LBFGS epoch %s last loss: %.4e | rel_sol=%.4e | rel_green=%.4e",
                            lbfgs_epoch,
                            last_lbfgs,
                            self.rel_sol_history[-1],
                            self.rel_green_history[-1],
                        )
                    self.training_recorder.record(
                        phase="lbfgs",
                        epoch=lbfgs_epoch,
                        learning_rate=float(lbfgs_optimizer.param_groups[0]["lr"]),
                        loss=last_lbfgs,
                        rel_sol=self.rel_sol_history[-1],
                        val_rel_sol=(
                            self.val_rel_sol_history[-1]
                            if self.config.compute_validation_rel_sol
                            else None
                        ),
                        rel_green=self.rel_green_history[-1],
                    )

        if self.loss_history:
            LossVisualizer.save_loss_curve(
                losses=self.loss_history,
                output_path=self.work_dir / "loss_curve.html",
                logger=self.logger,
            )
            # Save relative error curves
            if self.rel_sol_history:
                LossVisualizer.save_loss_curve(
                    losses=self.rel_sol_history,
                    output_path=self.work_dir / "rel_sol_curve.html",
                    logger=self.logger,
                )
            if self.val_rel_sol_history:
                LossVisualizer.save_loss_curve(
                    losses=self.val_rel_sol_history,
                    output_path=self.work_dir / "val_rel_sol_curve.html",
                    logger=self.logger,
                )
            if self.rel_green_history:
                LossVisualizer.save_loss_curve(
                    losses=self.rel_green_history,
                    output_path=self.work_dir / "rel_green_curve.html",
                    logger=self.logger,
                )
            # Plot a Green heatmap using the last batch from the last epoch
            try:
                sample = next(iter(loader))
                coords, solution, source, a_val, ap_val, b_val, c_val = sample
                coords = coords.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)
                trunk_grid = self._build_trunk_grid(coords.shape[2])
                GreenVisualizer.save_green_heatmap(
                    model=self.model,
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                    output_path=self.work_dir / "green_heatmap.html",
                    logger=self.logger,
                )
            except StopIteration:
                self.logger.warning("No data available to plot Green heatmap.")
        self._save_per_line_metrics(dataset, validation_dataset)
        self.training_recorder.write_csv()
        self._save_model_checkpoint("model.safetensors")

    @staticmethod
    def _optimizer_metrics_log_suffix(metrics: dict[str, float]) -> str:
        if not metrics:
            return ""
        return " | " + " | ".join(
            f"{key}={value:.6e}" for key, value in metrics.items()
        )

    def _save_model_checkpoint(self, filename: str) -> None:
        model_path = self.work_dir / filename
        if self.model_cfg is not None:
            save_model_with_config(self.model, self.model_cfg, model_path, self.logger)
        else:
            save_state_dict_safetensors(
                model_state_dict_for_save(self.model), model_path, self.logger
            )

    def evaluate(self, dataset: Dataset[AxialBatch]) -> float:
        self.model.eval()
        loader = self._make_eval_loader(dataset)
        losses: List[float] = []
        with torch.no_grad():
            for (
                coords,
                solution,
                source,
                a_val,
                ap_val,
                b_val,
                c_val,
            ) in loader:
                coords = coords.to(self.device)
                solution = solution.to(self.device)
                source = source.to(self.device)
                a_val = a_val.to(self.device)
                ap_val = ap_val.to(self.device)
                b_val = b_val.to(self.device)
                c_val = c_val.to(self.device)
                trunk_grid = self._build_trunk_grid(coords.shape[2])
                prediction = self.model(
                    trunk_grid=trunk_grid,
                    a_vals=a_val,
                    ap_vals=ap_val,
                    b_vals=b_val,
                    c_vals=c_val,
                )
                loss, _ = self._green_reconstruction_loss(
                    prediction=prediction,
                    source=source,
                    solution=solution,
                    trunk_grid=trunk_grid,
                    integration_rule=self.config.integration_rule,
                )
                losses.append(loss.item())
        return float(sum(losses) / max(len(losses), 1))
