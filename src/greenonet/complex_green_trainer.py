from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, cast

import plotly.graph_objects as go
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

from greenonet.compile_utils import (
    maybe_compile_model,
    model_state_dict_for_save,
    unwrap_compiled_model,
)
from greenonet.complex_green_data import (
    ComplexGreenBatch,
    ComplexGreenDataset,
    ComplexGreenItem,
    complex_green_collate_fn,
)
from greenonet.config import ModelConfig, TrainingConfig
from greenonet.greens import (
    exact_green_kernel_from_unit_coefficients,
    select_green_reference_policy,
)
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.logging_mixin import LoggingMixin
from greenonet.numerics import IntegrationRule, integrate
from greenonet.plotly_io import save_plotly_figure
from greenonet.visualizer import LossVisualizer


@dataclass(frozen=True)
class IntervalMetricStats:
    mean: Tensor
    min: Tensor
    max: Tensor
    std: Tensor


class ComplexGreenTrainer(LoggingMixin):
    """Training loop for flat complex-geometry GreenNet intervals."""

    ZERO_TOL = 1.0e-12

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        work_dir: Path | str,
        model_cfg: ModelConfig | None = None,
        terminal_width: int | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.model_cfg = model_cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexGreenTrainer",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model = maybe_compile_model(
            self.model,
            self.config.compile,
            logger=self.logger,
            model_name="ComplexGreenONetModel",
        )
        self.loss_history: List[float] = []
        self.rel_sol_history: List[float] = []
        self.val_rel_sol_history: List[float] = []
        self.rel_green_history: List[float] = []

    @staticmethod
    def _build_trunk_grid(unit_grid: Tensor) -> Tensor:
        return torch.stack(
            torch.meshgrid(unit_grid, unit_grid, indexing="ij"),
            dim=-1,
        )

    def _forward_pairs(
        self,
        trunk_grid: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
    ) -> Tensor:
        pair_forward = getattr(self.model, "forward_pairs", None)
        if not callable(pair_forward):
            original = unwrap_compiled_model(self.model)
            pair_forward = getattr(original, "forward_pairs", None)
        if not callable(pair_forward):
            raise TypeError("Complex GreenNet training requires model.forward_pairs().")
        return cast(Tensor, pair_forward(trunk_grid, a_vals, ap_vals, b_vals, c_vals))

    @staticmethod
    def _reconstruct_solution(
        kernel: Tensor,
        source: Tensor,
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        rhs = source.unsqueeze(-2) * kernel.unsqueeze(0)
        return integrate(rhs, x=unit_grid, dim=-1, rule=integration_rule)

    @classmethod
    def _green_reconstruction_loss(
        cls,
        *,
        kernel: Tensor,
        source: Tensor,
        solution: Tensor,
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> tuple[Tensor, Tensor]:
        reconstruction = cls._reconstruct_solution(
            kernel=kernel,
            source=source,
            unit_grid=unit_grid,
            integration_rule=integration_rule,
        )
        residual = solution - reconstruction
        residual_energy = integrate(
            residual.pow(2),
            x=unit_grid,
            dim=-1,
            rule=integration_rule,
        )
        solution_energy = integrate(
            solution.pow(2),
            x=unit_grid,
            dim=-1,
            rule=integration_rule,
        ).clamp_min(1.0e-12)
        rel_sol = torch.sqrt(residual_energy / solution_energy).mean()
        return residual_energy.mean(), rel_sol

    @classmethod
    def _green_reconstruction_rel_by_interval(
        cls,
        *,
        kernel: Tensor,
        source: Tensor,
        solution: Tensor,
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        reconstruction = cls._reconstruct_solution(
            kernel=kernel,
            source=source,
            unit_grid=unit_grid,
            integration_rule=integration_rule,
        )
        residual = solution - reconstruction
        residual_energy = integrate(
            residual.pow(2),
            x=unit_grid,
            dim=-1,
            rule=integration_rule,
        )
        solution_energy = integrate(
            solution.pow(2),
            x=unit_grid,
            dim=-1,
            rule=integration_rule,
        ).clamp_min(1.0e-12)
        return torch.sqrt(residual_energy / solution_energy)

    @staticmethod
    def _relative_green_error_by_interval(
        *,
        prediction: Tensor,
        exact_kernel: Tensor,
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        num = (prediction - exact_kernel).pow(2)
        den = exact_kernel.pow(2)
        num = integrate(num, x=unit_grid, dim=-1, rule=integration_rule)
        den = integrate(den, x=unit_grid, dim=-1, rule=integration_rule)
        num = integrate(num, x=unit_grid, dim=-1, rule=integration_rule)
        den = integrate(den, x=unit_grid, dim=-1, rule=integration_rule).clamp_min(
            1.0e-12
        )
        return torch.sqrt(num / den)

    def _green_kernel_rel_by_interval(self, batch: ComplexGreenBatch) -> Tensor:
        policy = select_green_reference_policy(
            batch.b_vals,
            batch.c_vals,
            zero_tol=self.ZERO_TOL,
        )
        if not policy.valid:
            return torch.full(
                (batch.a_vals.shape[0],),
                float("nan"),
                dtype=batch.a_vals.dtype,
                device=batch.a_vals.device,
            )
        assert policy.reference is not None
        trunk_grid = self._build_trunk_grid(batch.unit_grid)
        prediction = self._forward_pairs(
            trunk_grid,
            batch.a_vals,
            batch.ap_vals,
            batch.b_vals,
            batch.c_vals,
        )
        exact_kernel = exact_green_kernel_from_unit_coefficients(
            batch.unit_grid,
            batch.a_vals,
            batch.b_vals,
            policy.reference,
        )
        return self._relative_green_error_by_interval(
            prediction=prediction,
            exact_kernel=exact_kernel,
            unit_grid=batch.unit_grid,
            integration_rule=self.config.integration_rule,
        )

    def _make_loader(
        self,
        dataset: Dataset[ComplexGreenItem],
        *,
        shuffle: bool,
    ) -> DataLoader[ComplexGreenBatch]:
        return cast(
            DataLoader[ComplexGreenBatch],
            DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=complex_green_collate_fn,
                pin_memory=self.device.type == "cuda",
            ),
        )

    def _dataset_rel_sol(self, dataset: Dataset[ComplexGreenItem]) -> float:
        loader = self._make_loader(dataset, shuffle=False)
        total = 0.0
        count = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                trunk_grid = self._build_trunk_grid(batch.unit_grid)
                prediction = self._forward_pairs(
                    trunk_grid,
                    batch.a_vals,
                    batch.ap_vals,
                    batch.b_vals,
                    batch.c_vals,
                )
                rel_line = self._green_reconstruction_rel_by_interval(
                    kernel=prediction,
                    source=batch.source,
                    solution=batch.solution,
                    unit_grid=batch.unit_grid,
                    integration_rule=self.config.integration_rule,
                )
                total += float(rel_line.sum().item())
                count += int(rel_line.numel())
        if was_training:
            self.model.train()
        return total / max(count, 1)

    @staticmethod
    def _metric_stats(values: Tensor) -> IntervalMetricStats:
        values = values.detach().cpu().to(torch.float64)
        mean = values.mean(dim=0)
        min_val = values.min(dim=0).values
        max_val = values.max(dim=0).values
        std = (
            values.std(dim=0, unbiased=True)
            if values.shape[0] > 1
            else torch.zeros_like(mean)
        )
        return IntervalMetricStats(mean=mean, min=min_val, max=max_val, std=std)

    @staticmethod
    def _finite_mean(values: Tensor) -> float | None:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return None
        return float(finite.mean().item())

    @staticmethod
    def _finite_max(values: Tensor) -> float | None:
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return None
        return float(finite.max().item())

    def _aggregate_rel_sol_by_interval(
        self,
        dataset: Dataset[ComplexGreenItem],
    ) -> IntervalMetricStats:
        loader = self._make_loader(dataset, shuffle=False)
        rel_values: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                trunk_grid = self._build_trunk_grid(batch.unit_grid)
                prediction = self._forward_pairs(
                    trunk_grid,
                    batch.a_vals,
                    batch.ap_vals,
                    batch.b_vals,
                    batch.c_vals,
                )
                rel_values.append(
                    self._green_reconstruction_rel_by_interval(
                        kernel=prediction,
                        source=batch.source,
                        solution=batch.solution,
                        unit_grid=batch.unit_grid,
                        integration_rule=self.config.integration_rule,
                    )
                    .detach()
                    .cpu()
                )
        if not rel_values:
            raise ValueError("Cannot aggregate metrics over an empty dataset.")
        return self._metric_stats(torch.cat(rel_values, dim=0))

    def _save_interval_metrics(
        self,
        dataset: ComplexGreenDataset,
        validation_dataset: ComplexGreenDataset | None = None,
    ) -> None:
        train_stats = self._aggregate_rel_sol_by_interval(dataset)
        validation_stats = (
            None
            if validation_dataset is None
            else self._aggregate_rel_sol_by_interval(validation_dataset)
        )
        first_batch = complex_green_collate_fn([dataset[0]]).to(self.device)
        with torch.no_grad():
            rel_green = self._green_kernel_rel_by_interval(first_batch).detach().cpu()

        csv_path = self.work_dir / "per_interval_metrics.csv"
        with csv_path.open("w", newline="") as fp:
            fieldnames = [
                "interval_index",
                "axis_id",
                "axis",
                "segment_index",
                "left",
                "right",
                "fixed",
                "length",
                "rel_sol",
                "rel_sol_mean",
                "rel_sol_min",
                "rel_sol_max",
                "rel_sol_std",
                "rel_green",
                "val_rel_sol",
                "val_rel_sol_mean",
                "val_rel_sol_min",
                "val_rel_sol_max",
                "val_rel_sol_std",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            data = dataset.data
            for interval_idx in range(data.num_intervals):
                axis_id = int(data.axis_id[interval_idx].item())
                row: dict[str, object] = {
                    "interval_index": interval_idx,
                    "axis_id": axis_id,
                    "axis": "x" if axis_id == 0 else "y",
                    "segment_index": int(data.segment_id[interval_idx].item()),
                    "left": float(data.left[interval_idx].item()),
                    "right": float(data.right[interval_idx].item()),
                    "fixed": float(data.fixed[interval_idx].item()),
                    "length": float(data.length[interval_idx].item()),
                    "rel_sol": float(train_stats.mean[interval_idx].item()),
                    "rel_sol_mean": float(train_stats.mean[interval_idx].item()),
                    "rel_sol_min": float(train_stats.min[interval_idx].item()),
                    "rel_sol_max": float(train_stats.max[interval_idx].item()),
                    "rel_sol_std": float(train_stats.std[interval_idx].item()),
                    "rel_green": (
                        ""
                        if not torch.isfinite(rel_green[interval_idx])
                        else float(rel_green[interval_idx].item())
                    ),
                    "val_rel_sol": "",
                    "val_rel_sol_mean": "",
                    "val_rel_sol_min": "",
                    "val_rel_sol_max": "",
                    "val_rel_sol_std": "",
                }
                if validation_stats is not None:
                    row.update(
                        {
                            "val_rel_sol": float(
                                validation_stats.mean[interval_idx].item()
                            ),
                            "val_rel_sol_mean": float(
                                validation_stats.mean[interval_idx].item()
                            ),
                            "val_rel_sol_min": float(
                                validation_stats.min[interval_idx].item()
                            ),
                            "val_rel_sol_max": float(
                                validation_stats.max[interval_idx].item()
                            ),
                            "val_rel_sol_std": float(
                                validation_stats.std[interval_idx].item()
                            ),
                        }
                    )
                writer.writerow(row)

        summary = {
            "num_intervals": dataset.data.num_intervals,
            "num_x_segments": int((dataset.data.axis_id == 0).sum().item()),
            "num_y_segments": int((dataset.data.axis_id == 1).sum().item()),
            "mean_rel_sol_interval": float(train_stats.mean.mean().item()),
            "max_rel_sol_interval": float(train_stats.max.max().item()),
            "rel_green_valid": bool(torch.isfinite(rel_green).all().item()),
            "mean_rel_green_interval": self._finite_mean(rel_green),
            "max_rel_green_interval": self._finite_max(rel_green),
        }
        if validation_stats is not None:
            summary.update(
                {
                    "mean_val_rel_sol_interval": float(
                        validation_stats.mean.mean().item()
                    ),
                    "max_val_rel_sol_interval": float(
                        validation_stats.max.max().item()
                    ),
                }
            )
        summary_path = self.work_dir / "per_interval_metrics_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        self.logger.info(
            "Saved per-interval metrics: %s and %s",
            csv_path,
            summary_path,
        )

    def _save_green_heatmap(self, dataset: ComplexGreenDataset) -> None:
        batch = complex_green_collate_fn([dataset[0]]).to(self.device)
        with torch.no_grad():
            trunk_grid = self._build_trunk_grid(batch.unit_grid)
            kernel = self._forward_pairs(
                trunk_grid,
                batch.a_vals,
                batch.ap_vals,
                batch.b_vals,
                batch.c_vals,
            )
        unit_np = batch.unit_grid.detach().cpu().numpy()
        fig = go.Figure(
            data=go.Heatmap(
                z=kernel[0].detach().cpu().numpy(),
                x=unit_np,
                y=unit_np,
                colorscale="Viridis",
            )
        )
        fig.update_layout(
            title="Complex GreenNet kernel interval=0",
            xaxis_title="eta",
            yaxis_title="t",
            template="plotly_white",
            xaxis=dict(constrain="domain"),
            yaxis=dict(scaleanchor="x", scaleratio=1, constrain="domain"),
        )
        save_plotly_figure(fig, self.work_dir / "green_heatmap", self.logger)

    def train(
        self,
        dataset: ComplexGreenDataset,
        validation_dataset: ComplexGreenDataset | None = None,
    ) -> None:
        if self.config.compute_validation_rel_sol and validation_dataset is None:
            raise ValueError(
                "validation_dataset must be provided when compute_validation_rel_sol=True."
            )
        self.model.train()
        loader = self._make_loader(dataset, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        for epoch in range(1, self.config.epochs + 1):
            epoch_losses: List[float] = []
            last_batch: ComplexGreenBatch | None = None
            for batch in loader:
                batch = batch.to(self.device)
                trunk_grid = self._build_trunk_grid(batch.unit_grid)
                optimizer.zero_grad()
                prediction = self._forward_pairs(
                    trunk_grid,
                    batch.a_vals,
                    batch.ap_vals,
                    batch.b_vals,
                    batch.c_vals,
                )
                loss, _rel_sol = self._green_reconstruction_loss(
                    kernel=prediction,
                    source=batch.source,
                    solution=batch.solution,
                    unit_grid=batch.unit_grid,
                    integration_rule=self.config.integration_rule,
                )
                cast(Any, loss).backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
                last_batch = batch

            mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
            self.loss_history.append(mean_loss)

            if epoch % self.config.log_interval == 0 and last_batch is not None:
                with torch.no_grad():
                    trunk_grid = self._build_trunk_grid(last_batch.unit_grid)
                    pred_eval = self._forward_pairs(
                        trunk_grid,
                        last_batch.a_vals,
                        last_batch.ap_vals,
                        last_batch.b_vals,
                        last_batch.c_vals,
                    )
                    if self.config.compute_validation_rel_sol:
                        train_rel_sol = self._dataset_rel_sol(dataset)
                        assert validation_dataset is not None
                        val_rel_sol = self._dataset_rel_sol(validation_dataset)
                        self.rel_sol_history.append(train_rel_sol)
                        self.val_rel_sol_history.append(val_rel_sol)
                    else:
                        _loss, rel_sol = self._green_reconstruction_loss(
                            kernel=pred_eval,
                            source=last_batch.source,
                            solution=last_batch.solution,
                            unit_grid=last_batch.unit_grid,
                            integration_rule=self.config.integration_rule,
                        )
                        self.rel_sol_history.append(float(rel_sol.detach().item()))

                    rel_green_line = self._green_kernel_rel_by_interval(last_batch)
                    rel_green_mean = self._finite_mean(rel_green_line.detach().cpu())
                    if rel_green_mean is not None:
                        self.rel_green_history.append(rel_green_mean)

                if self.config.compute_validation_rel_sol:
                    rel_green_text = (
                        "nan" if rel_green_mean is None else f"{rel_green_mean:.4e}"
                    )
                    self.logger.info(
                        "Epoch %s: loss=%.4e | train_rel_sol=%.4e | val_rel_sol=%.4e | rel_green=%s",
                        epoch,
                        mean_loss,
                        self.rel_sol_history[-1],
                        self.val_rel_sol_history[-1],
                        rel_green_text,
                    )
                else:
                    rel_green_text = (
                        "nan" if rel_green_mean is None else f"{rel_green_mean:.4e}"
                    )
                    self.logger.info(
                        "Epoch %s: loss=%.4e | rel_sol=%.4e | rel_green=%s",
                        epoch,
                        mean_loss,
                        self.rel_sol_history[-1],
                        rel_green_text,
                    )
            elif epoch % self.config.log_interval == 0:
                self.logger.info("Epoch %s: loss=%.4e", epoch, mean_loss)

        if self.config.lbfgs_max_iter > 0 and self.config.lbfgs_epochs > 0:
            self._run_lbfgs(dataset, validation_dataset)

        self._save_outputs(dataset, validation_dataset)

    def _run_lbfgs(
        self,
        dataset: ComplexGreenDataset,
        validation_dataset: ComplexGreenDataset | None,
    ) -> None:
        self.logger.info(
            "Starting LBFGS fine-tuning (epochs=%s, max_iter=%s, lr=%s)",
            self.config.lbfgs_epochs,
            self.config.lbfgs_max_iter,
            self.config.lbfgs_lr,
        )
        optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=self.config.lbfgs_lr,
            max_iter=self.config.lbfgs_max_iter,
            history_size=self.config.lbfgs_history_size,
            tolerance_grad=self.config.lbfgs_tolerance_grad,
            tolerance_change=0,
            line_search_fn="strong_wolfe",
        )
        for lbfgs_epoch in range(1, self.config.lbfgs_epochs + 1):
            losses: list[float] = []
            for batch in self._make_loader(dataset, shuffle=True):
                batch = batch.to(self.device)
                trunk_grid = self._build_trunk_grid(batch.unit_grid)

                def closure() -> Tensor:
                    optimizer.zero_grad()
                    prediction = self._forward_pairs(
                        trunk_grid,
                        batch.a_vals,
                        batch.ap_vals,
                        batch.b_vals,
                        batch.c_vals,
                    )
                    loss, _rel_sol = self._green_reconstruction_loss(
                        kernel=prediction,
                        source=batch.source,
                        solution=batch.solution,
                        unit_grid=batch.unit_grid,
                        integration_rule=self.config.integration_rule,
                    )
                    cast(Any, loss).backward()
                    return loss

                loss = cast(Any, optimizer).step(closure)
                losses.append(float(loss.item()))
            if losses:
                last_loss = losses[-1]
                self.loss_history.append(last_loss)
                train_rel_sol = self._dataset_rel_sol(dataset)
                self.rel_sol_history.append(train_rel_sol)
                if self.config.compute_validation_rel_sol:
                    assert validation_dataset is not None
                    self.val_rel_sol_history.append(
                        self._dataset_rel_sol(validation_dataset)
                    )
                self.logger.info(
                    "LBFGS epoch %s last loss: %.4e | train_rel_sol=%.4e",
                    lbfgs_epoch,
                    last_loss,
                    train_rel_sol,
                )

    def _save_outputs(
        self,
        dataset: ComplexGreenDataset,
        validation_dataset: ComplexGreenDataset | None,
    ) -> None:
        if self.loss_history:
            LossVisualizer.save_loss_curve(
                losses=self.loss_history,
                output_path=self.work_dir / "loss_curve.html",
                logger=self.logger,
            )
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
        self._save_green_heatmap(dataset)
        self._save_interval_metrics(dataset, validation_dataset)

        model_path = self.work_dir / "model.safetensors"
        if self.model_cfg is not None:
            save_model_with_config(self.model, self.model_cfg, model_path, self.logger)
        else:
            save_state_dict_safetensors(
                model_state_dict_for_save(self.model),
                model_path,
                self.logger,
            )

    def evaluate(self, dataset: ComplexGreenDataset) -> float:
        self.model.eval()
        loader = self._make_loader(dataset, shuffle=False)
        losses: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                trunk_grid = self._build_trunk_grid(batch.unit_grid)
                prediction = self._forward_pairs(
                    trunk_grid,
                    batch.a_vals,
                    batch.ap_vals,
                    batch.b_vals,
                    batch.c_vals,
                )
                loss, _rel_sol = self._green_reconstruction_loss(
                    kernel=prediction,
                    source=batch.source,
                    solution=batch.solution,
                    unit_grid=batch.unit_grid,
                    integration_rule=self.config.integration_rule,
                )
                losses.append(float(loss.item()))
        return float(sum(losses) / max(len(losses), 1))
