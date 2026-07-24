from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import optim
from torch.utils.data import DataLoader

from greenonet.compile_utils import (
    maybe_compile_model,
    model_state_dict_for_save,
    unwrap_compiled_model,
)
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_coupling_objective import (
    ComplexCouplingObjectiveResult,
    compute_complex_coupling_objective,
)
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    build_boundary_energy_context,
    relative_l2_valid,
)
from greenonet.complex_projection import (
    ComplexProjectionResult,
    apply_complex_balance_projection,
)
from greenonet.complex_reconstruction import (
    ComplexReconstructionResult,
    reconstruct_from_projected_response,
)
from greenonet.coupling_optimizer import (
    ComplexCouplingOptimizerFactory,
    OptimizerStepProfiler,
)
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexPreProjectionFusionConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingBestEnergyCheckpointConfig,
    CouplingBestPhysicsCheckpointConfig,
    CouplingTrainingConfig,
)
from greenonet.io import save_state_dict_safetensors
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexForwardResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult
    objective: ComplexCouplingObjectiveResult


class ComplexCouplingTrainer(LoggingMixin):
    """Trainer for complex-geometry CouplingNet with no cross diagnostics."""

    _METRIC_KEYS: tuple[str, ...] = (
        "loss",
        "loss_energy_consistency",
        "loss_energy_bulk",
        "loss_energy_boundary",
        "loss_energy_boundary_x",
        "loss_energy_boundary_y",
        "loss_split_relative",
        "loss_split_energy_relative",
        "loss_split_mass_relative",
        "loss_weak_operator_closure",
        "loss_weak_operator_x",
        "loss_weak_operator_y",
        "rel_sol",
        "rel_flux",
        "pre_projection_fusion_gate",
        "learning_rate",
        "optimizer_step_time_mean_ms",
        "optimizer_step_time_p95_ms",
        "optimizer_step_time_max_ms",
        "optimizer_step_count",
        "optimizer_basis_refresh_count",
        "optimizer_peak_memory_mib",
    )

    def __init__(
        self,
        *,
        model: ComplexCouplingNet,
        config: CouplingTrainingConfig,
        work_dir: Path | str,
        green_model: torch.nn.Module,
        terminal_width: int | None = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.balance_projection = BalanceProjectionConfig.from_raw(
            model.config.balance_projection
        )
        self.pre_projection_fusion_config = ComplexPreProjectionFusionConfig.from_raw(
            model.config.pre_projection_fusion
        )
        self.config = config
        self.relative_split_config = ComplexRelativeSplitConsistencyConfig.from_raw(
            config.relative_split_consistency
        )
        self.weak_closure_config = ComplexWeakOperatorClosureConfig.from_raw(
            config.weak_operator_closure
        )
        self.best_energy_checkpoint = CouplingBestEnergyCheckpointConfig.from_raw(
            config.best_energy_checkpoint
        )
        self.best_physics_checkpoint = CouplingBestPhysicsCheckpointConfig.from_raw(
            config.best_physics_checkpoint
        )
        self.optimizer_factory = ComplexCouplingOptimizerFactory(config)
        self.optimizer_provenance = self.optimizer_factory.provenance()
        if config.best_rel_sol_checkpoint.enabled:
            raise ValueError(
                "Complex mode does not allow best_rel_sol_checkpoint because "
                "reference sol must not select a training checkpoint. Use "
                "best_energy_checkpoint instead."
            )
        self.green_model = green_model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexCouplingTrainer",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )
        self._log_pre_projection_fusion(model)
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model = maybe_compile_model(
            self.model,
            self.config.compile,
            logger=self.logger,
            model_name="ComplexCouplingNet",
        )
        self.green_model.to(self.device)
        self.green_model.eval()
        for parameter in self.green_model.parameters():
            parameter.requires_grad_(False)
        self.loss_history: list[float] = []
        self.metric_rows: list[dict[str, float | int | str]] = []
        self._boundary_context: ComplexBoundaryEnergyContext | None = None

    def train(
        self,
        train_dataset: ComplexCouplingDataset,
        validation_dataset: ComplexCouplingDataset | None = None,
    ) -> None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=complex_coupling_collate_fn,
        )
        validation_loader = (
            None
            if validation_dataset is None
            else DataLoader(
                validation_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=complex_coupling_collate_fn,
            )
        )
        schedule_config = CouplingLearningRateSchedule.from_config(
            self.config,
            total_epochs=self.config.epochs,
        )
        optimizer = self.optimizer_factory.build(self.model.parameters())
        optimizer_profiler = OptimizerStepProfiler(
            optimizer=optimizer,
            enabled=self.optimizer_provenance.profile_step_time,
            device=self.device,
        )
        scheduler = schedule_config.build(optimizer)
        self._log_optimizer()
        self._write_optimizer_provenance()
        self._log_learning_rate_schedule(schedule_config)
        best_val_energy: float | None = None
        best_val_physics: float | None = None
        for epoch in range(1, self.config.epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_metrics = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                optimizer_profiler=optimizer_profiler,
            )
            fusion_gate = self._current_pre_projection_fusion_gate()
            if fusion_gate is not None:
                train_metrics["pre_projection_fusion_gate"] = fusion_gate
            train_metrics["learning_rate"] = learning_rate
            self.loss_history.append(float(train_metrics["loss"]))
            self._record_metrics(epoch, "train", train_metrics)
            if epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, "train", train_metrics)
            if validation_loader is not None:
                val_metrics = self._evaluate_loader(validation_loader)
                if fusion_gate is not None:
                    val_metrics["pre_projection_fusion_gate"] = fusion_gate
                val_metrics["learning_rate"] = learning_rate
                self._record_metrics(epoch, "val", val_metrics)
                if epoch % self.config.log_interval == 0:
                    self._log_epoch(epoch, "val", val_metrics)
                if self.best_energy_checkpoint.enabled:
                    validation_energy = float(val_metrics["loss_energy_consistency"])
                    if best_val_energy is None or validation_energy < best_val_energy:
                        best_val_energy = validation_energy
                        self._save_checkpoint(
                            "complex_coupling_model_best_energy.safetensors"
                        )
                if self.best_physics_checkpoint.enabled:
                    validation_physics = float(val_metrics["loss"])
                    if (
                        best_val_physics is None
                        or validation_physics < best_val_physics
                    ):
                        best_val_physics = validation_physics
                        self._save_checkpoint(
                            "complex_coupling_model_best_physics.safetensors"
                        )
            if (
                self.config.periodic_checkpoint.enabled
                and self.config.periodic_checkpoint.every_epochs > 0
                and epoch % self.config.periodic_checkpoint.every_epochs == 0
            ):
                self._save_checkpoint(
                    f"complex_coupling_model_epoch_{epoch:04d}.safetensors"
                )
            if scheduler is not None:
                scheduler.step()

        self._save_checkpoint("complex_coupling_model.safetensors")
        self._save_checkpoint("coupling_model.safetensors")
        self._write_metric_csv()

    def _run_epoch(
        self,
        loader: DataLoader[Any],
        *,
        optimizer: optim.Optimizer,
        optimizer_profiler: OptimizerStepProfiler,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        samples = 0
        optimizer_profiler.begin_epoch()
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            result = self._forward_batch(batch)
            result.loss.backward()  # type: ignore[no-untyped-call]
            if self.config.gradient_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_max_norm,
                )
            optimizer_profiler.step()
            batch_size = int(batch.rhs_valid.shape[0])
            self._accumulate(totals, result.metrics, batch_size=batch_size)
            samples += batch_size
        metrics = self._average(totals, samples)
        metrics.update(optimizer_profiler.finish_epoch())
        return metrics

    def _evaluate_loader(
        self,
        loader: DataLoader[Any],
    ) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        samples = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                result = self._forward_batch(batch)
                batch_size = int(batch.rhs_valid.shape[0])
                self._accumulate(totals, result.metrics, batch_size=batch_size)
                samples += batch_size
        if was_training:
            self.model.train()
        return self._average(totals, samples)

    def _forward_batch(self, batch: ComplexCouplingBatch) -> ComplexForwardResult:
        raw_response = self.model(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
            rhs_phys=batch.rhs_valid,
        )
        projection = apply_complex_balance_projection(
            raw_response=raw_response,
            rhs_phys=batch.rhs_valid,
            geometry=batch.geometry,
            config=self.balance_projection,
        )
        reconstruction = reconstruct_from_projected_response(
            green_model=self.green_model,
            geometry=batch.geometry,
            projected_response=projection.projected_response,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        objective = compute_complex_coupling_objective(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            rhs_valid=batch.rhs_valid,
            projected_physical=projection.projected_physical,
            a_valid=batch.a_valid,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
            relative_split_config=self.relative_split_config,
            weak_closure_config=self.weak_closure_config,
            boundary_context=self._boundary_energy_context(batch),
        )
        loss = objective.loss
        metrics = {
            key: value.detach() for key, value in objective.metric_tensors().items()
        }
        if torch.any(batch.has_solution):
            selected_solution = batch.has_solution
            metrics["rel_sol"] = relative_l2_valid(
                reconstruction.u_mean_valid[selected_solution],
                batch.sol_valid[selected_solution],
            ).detach()
        if torch.any(batch.has_flux):
            selected = batch.has_flux
            metrics["rel_flux"] = relative_l2_valid(
                projection.projected_physical[selected],
                batch.flux_valid[selected],
            ).detach()
        return ComplexForwardResult(
            loss=loss,
            metrics=metrics,
            projection=projection,
            reconstruction=reconstruction,
            objective=objective,
        )

    def _boundary_energy_context(
        self,
        batch: ComplexCouplingBatch,
    ) -> ComplexBoundaryEnergyContext:
        if self._boundary_context is None:
            self._boundary_context = build_boundary_energy_context(batch.geometry)
            self.logger.info(
                "canonical boundary energy anchors=%d x_anchors=%d y_anchors=%d",
                self._boundary_context.total_anchors,
                self._boundary_context.x_anchor_count,
                self._boundary_context.y_anchor_count,
            )
        return self._boundary_context

    @staticmethod
    def _accumulate(
        totals: dict[str, float],
        metrics: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + batch_size * float(value.item())

    @staticmethod
    def _average(totals: dict[str, float], samples: int) -> dict[str, float]:
        if samples == 0:
            raise ValueError("Cannot average zero complex coupling samples.")
        return {key: value / samples for key, value in totals.items()}

    def _record_metrics(
        self,
        epoch: int,
        split: str,
        metrics: dict[str, float],
    ) -> None:
        row: dict[str, float | int | str] = {"epoch": epoch, "split": split}
        for key in self._METRIC_KEYS:
            if key in metrics:
                row[key] = metrics[key]
        self.metric_rows.append(row)

    def _log_epoch(
        self,
        epoch: int,
        split: str,
        metrics: dict[str, float],
    ) -> None:
        parts = [
            f"{key}={metrics[key]:.6e}" for key in self._METRIC_KEYS if key in metrics
        ]
        self.logger.info("epoch %04d %s %s", epoch, split, " ".join(parts))

    def _log_learning_rate_schedule(
        self,
        schedule: CouplingLearningRateSchedule,
    ) -> None:
        self.logger.info(
            "learning-rate schedule enabled=%s kind=%s base_lr=%.6e "
            "min_lr=%.6e configured_warmup_epochs=%d "
            "effective_warmup_epochs=%d total_epochs=%d",
            schedule.enabled,
            schedule.kind,
            schedule.base_learning_rate,
            schedule.min_learning_rate,
            schedule.configured_warmup_epochs,
            schedule.effective_warmup_epochs,
            schedule.total_epochs,
        )

    def _log_optimizer(self) -> None:
        provenance = self.optimizer_provenance
        self.logger.info(
            "optimizer name=%s implementation=%s base_lr=%.6e "
            "weight_decay=%.6e betas=%s eps=%.6e profile_step_time=%s "
            "checkpoint_policy=%s",
            provenance.name,
            provenance.implementation,
            provenance.learning_rate,
            provenance.weight_decay,
            provenance.betas,
            provenance.eps,
            provenance.profile_step_time,
            provenance.checkpoint_policy,
        )
        if provenance.soap is not None:
            self.logger.info(
                "SOAP upstream_commit=%s settings=%s "
                "frequency_unit=optimizer_step "
                "first_step_initializes_preconditioner=true",
                provenance.upstream_commit,
                provenance.soap,
            )

    def _write_optimizer_provenance(self) -> None:
        path = self.work_dir / "optimizer_provenance.json"
        path.write_text(
            json.dumps(self.optimizer_provenance.as_dict(), indent=2) + "\n"
        )

    def _log_pre_projection_fusion(self, model: ComplexCouplingNet) -> None:
        gate_value: float | None = None
        if model.pre_projection_fusion is not None:
            gate_value = float(
                torch.sigmoid(model.pre_projection_fusion.gate_logit.detach()).item()
            )
        self.logger.info(
            "pre-projection fusion enabled=%s space=physical_source "
            "correction=antisymmetric_difference hidden_dim=%d depth=%d "
            "initial_gate=%.6f current_gate=%s",
            self.pre_projection_fusion_config.enabled,
            self.pre_projection_fusion_config.nonlinear_hidden_dim,
            self.pre_projection_fusion_config.nonlinear_depth,
            self.pre_projection_fusion_config.gate_initial_value,
            "disabled" if gate_value is None else f"{gate_value:.6f}",
        )

    def _current_pre_projection_fusion_gate(self) -> float | None:
        module = unwrap_compiled_model(self.model)
        if not isinstance(module, ComplexCouplingNet):
            raise TypeError("Compiled complex trainer model has an invalid origin.")
        fusion = module.pre_projection_fusion
        if fusion is None:
            return None
        return float(torch.sigmoid(fusion.gate_logit.detach()).cpu().item())

    def _save_checkpoint(self, filename: str) -> None:
        save_state_dict_safetensors(
            model_state_dict_for_save(self.model),
            self.work_dir / filename,
            logger=self.logger,
        )

    def _write_metric_csv(self) -> None:
        if not self.metric_rows:
            return
        path = self.work_dir / "complex_training_metrics.csv"
        fieldnames = list(self.metric_rows[0].keys())
        for row in self.metric_rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metric_rows)


def complex_metric_keys_are_safe(keys: Iterable[str]) -> bool:
    return all("cross" not in key for key in keys)
