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
)
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
    ComplexCrossAxisReconstructor,
)
from greenonet.complex_coupling_objective import (
    ComplexCouplingObjectiveResult,
    compute_complex_coupling_objective,
)
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    build_boundary_energy_context,
    relative_l2_valid,
)
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
    ColumnDiagonalGreenResponseContextCache,
)
from greenonet.complex_projection import (
    ComplexProjectionResult,
    apply_complex_balance_projection,
    reconstruct_complex_projection,
    symmetric_tangent_metric_tensors,
)
from greenonet.complex_pre_projection_fusion import (
    FINAL_LAYER_INITIALIZATION,
    FUSION_ARCHITECTURE,
    pre_projection_fusion_formula,
)
from greenonet.complex_reconstruction import (
    ComplexReconstructionResult,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentEtaCapSchedule,
    SymmetricTangentGreenResponseContext,
    SymmetricTangentGreenResponseContextCache,
)
from greenonet.coupling_optimizer import (
    ComplexCouplingOptimizerFactory,
    OptimizerStepProfiler,
)
from greenonet.coupling_lr_scheduler import CouplingLearningRateSchedule
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCanonicalEnergyConfig,
    ComplexCrossAxisReconstructionConfig,
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
    cross_axis_reconstruction: ComplexCrossAxisReconstructionResult | None
    objective: ComplexCouplingObjectiveResult


class ComplexCouplingTrainer(LoggingMixin):
    """Trainer for complex-geometry CouplingNet with no cross diagnostics."""

    _METRIC_KEYS: tuple[str, ...] = (
        "loss",
        "boundary_weight",
        "loss_energy_optimized",
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
        "tangent_response_mismatch_pre",
        "tangent_response_mismatch_post",
        "tangent_response_mismatch_ratio",
        "tangent_gradient_rms",
        "tangent_delta_rms",
        "tangent_delta_max_abs",
        "tangent_correction_rel_symmetric_pair",
        "tangent_eta_cap",
        "tangent_eta_star_mean",
        "tangent_eta_applied_mean",
        "tangent_eta_cap_fraction",
        "tangent_line_search_numerator_mean",
        "tangent_line_search_denominator_mean",
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
        self.cross_axis_reconstruction_config = (
            ComplexCrossAxisReconstructionConfig.from_raw(
                model.config.cross_axis_reconstruction
            )
        )
        self.cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            self.cross_axis_reconstruction_config
        )
        self.config = config
        self.canonical_energy_config = ComplexCanonicalEnergyConfig.from_raw(
            config.canonical_energy
        )
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
        self.logger.info(
            "canonical energy boundary_weight=%.6e "
            "boundary_in_optimization=%s boundary_diagnostic_always=true "
            "optimized_formula=bulk+boundary_weight*boundary "
            "canonical_formula=bulk+boundary",
            self.canonical_energy_config.boundary_weight,
            self.canonical_energy_config.boundary_weight > 0.0,
        )
        self.logger.info(
            "final reconstruction enabled=%s mode=%s gamma=%.6f "
            "smoothing_steps=%d smoothing_relaxation=%.6f relative_floor=%.6f "
            "affects_training_objective=false",
            self.cross_axis_reconstruction_config.enabled,
            self.cross_axis_reconstruction_config.mode,
            self.cross_axis_reconstruction_config.gamma,
            self.cross_axis_reconstruction_config.smoothing_steps,
            self.cross_axis_reconstruction_config.smoothing_relaxation,
            self.cross_axis_reconstruction_config.relative_floor,
        )
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
        self._green_response_context_cache = ColumnDiagonalGreenResponseContextCache(
            self.balance_projection.column_diagonal_green_response
        )
        self._tangent_context_cache = SymmetricTangentGreenResponseContextCache(
            self.balance_projection.symmetric_tangent_green_response
        )

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
        tangent_eta_schedule = self._build_tangent_eta_schedule(schedule_config)
        self._log_optimizer()
        self._write_optimizer_provenance()
        self._log_learning_rate_schedule(schedule_config)
        if tangent_eta_schedule is not None:
            self._log_tangent_eta_schedule(tangent_eta_schedule)
        best_val_energy: float | None = None
        best_val_physics: float | None = None
        for epoch in range(1, self.config.epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            tangent_eta_cap = (
                None
                if tangent_eta_schedule is None
                else tangent_eta_schedule.cap_for_epoch_index(epoch - 1)
            )
            train_metrics = self._run_epoch(
                train_loader,
                optimizer=optimizer,
                optimizer_profiler=optimizer_profiler,
                tangent_eta_cap=tangent_eta_cap,
            )
            train_metrics["learning_rate"] = learning_rate
            self.loss_history.append(float(train_metrics["loss"]))
            self._record_metrics(epoch, "train", train_metrics)
            if epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, "train", train_metrics)
            if validation_loader is not None:
                val_metrics = self._evaluate_loader(validation_loader)
                val_metrics["learning_rate"] = learning_rate
                self._record_metrics(epoch, "val", val_metrics)
                if epoch % self.config.log_interval == 0:
                    self._log_epoch(epoch, "val", val_metrics)
                if self.best_energy_checkpoint.enabled:
                    validation_energy = float(val_metrics["loss_energy_optimized"])
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
        tangent_eta_cap: float | None,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        samples = 0
        optimizer_profiler.begin_epoch()
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            result = self._forward_batch(
                batch,
                symmetric_tangent_eta_cap=tangent_eta_cap,
            )
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

    def _forward_batch(
        self,
        batch: ComplexCouplingBatch,
        *,
        symmetric_tangent_eta_cap: float | None = None,
    ) -> ComplexForwardResult:
        projection_context = self._projection_context(batch)
        tangent_context = self._tangent_projection_context(batch)
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
            column_diagonal_context=projection_context,
            symmetric_tangent_context=tangent_context,
            symmetric_tangent_eta_cap=symmetric_tangent_eta_cap,
        )
        reconstruction = reconstruct_complex_projection(
            projection=projection,
            green_model=self.green_model,
            geometry=batch.geometry,
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
            canonical_energy_config=self.canonical_energy_config,
            relative_split_config=self.relative_split_config,
            weak_closure_config=self.weak_closure_config,
            boundary_context=self._boundary_energy_context(batch),
        )
        loss = objective.loss
        metrics = {
            key: value.detach() for key, value in objective.metric_tensors().items()
        }
        metrics.update(
            {
                key: value.detach()
                for key, value in symmetric_tangent_metric_tensors(projection).items()
            }
        )
        cross_axis_reconstruction: ComplexCrossAxisReconstructionResult | None = None
        if torch.any(batch.has_solution):
            selected_solution = batch.has_solution
            cross_axis_reconstruction = self.cross_axis_reconstructor.reconstruct(
                u_phi_valid=reconstruction.u_phi_valid,
                u_psi_valid=reconstruction.u_psi_valid,
                projected_physical=projection.projected_physical,
                geometry=batch.geometry,
                weak_context=batch.weak_context,
            )
            metrics["rel_sol"] = relative_l2_valid(
                cross_axis_reconstruction.u_pred_valid[selected_solution],
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
            cross_axis_reconstruction=cross_axis_reconstruction,
            objective=objective,
        )

    def _projection_context(
        self,
        batch: ComplexCouplingBatch,
    ) -> ColumnDiagonalGreenResponseContext | None:
        if self.balance_projection.mode != "column_diagonal_green_response":
            return None
        build_count_before = self._green_response_context_cache.build_count
        context = self._green_response_context_cache.get_or_build(
            green_model=self.green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        if self._green_response_context_cache.build_count != build_count_before:
            stats = context.statistics()
            self.logger.info(
                "column-diagonal Green-response context build_seconds=%.6f "
                "gain_exponent=%.6f "
                "gain_x_squared=[%.6e, %.6e] gain_y_squared=[%.6e, %.6e] "
                "weight_phi=[%.6e, %.6e] x_floored=%d y_floored=%d "
                "row_norm_used=false full_gram_solve=false",
                self._green_response_context_cache.build_seconds,
                context.gain_exponent,
                stats["gamma_x_squared_min"],
                stats["gamma_x_squared_max"],
                stats["gamma_y_squared_min"],
                stats["gamma_y_squared_max"],
                stats["weight_phi_min"],
                stats["weight_phi_max"],
                stats["x_floored_point_count"],
                stats["y_floored_point_count"],
            )
        return context

    def _tangent_projection_context(
        self,
        batch: ComplexCouplingBatch,
    ) -> SymmetricTangentGreenResponseContext | None:
        if self.balance_projection.mode != "symmetric_tangent_green_response":
            return None
        build_count_before = self._tangent_context_cache.build_count
        context = self._tangent_context_cache.get_or_build(
            green_model=self.green_model,
            geometry=batch.geometry,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        if self._tangent_context_cache.build_count != build_count_before:
            stats = context.statistics()
            self.logger.info(
                "symmetric-tangent Green-response context build_seconds=%.6f "
                "eta=%.6e eta_strategy=%s line_search_relative_eps=%.6e "
                "relative_lambda=%.6e denominator_relative_eps=%.6e "
                "gain_scale=%.6e denominator=[%.6e, %.6e] "
                "x_blocks=%d y_blocks=%d row_norm_used=false "
                "global_matrix_materialized=false full_gram_solve=false",
                self._tangent_context_cache.build_seconds,
                context.eta,
                context.eta_strategy,
                context.line_search_relative_eps,
                context.relative_lambda,
                context.denominator_relative_eps,
                stats["gain_scale"],
                stats["denominator_min"],
                stats["denominator_max"],
                stats["x_segment_block_count"],
                stats["y_segment_block_count"],
            )
        return context

    def _build_tangent_eta_schedule(
        self,
        learning_rate_schedule: CouplingLearningRateSchedule,
    ) -> SymmetricTangentEtaCapSchedule | None:
        if self.balance_projection.mode != "symmetric_tangent_green_response":
            return None
        return SymmetricTangentEtaCapSchedule.from_learning_rate_schedule(
            config=self.balance_projection.symmetric_tangent_green_response,
            learning_rate_schedule=learning_rate_schedule,
        )

    @property
    def column_diagonal_green_response_context(
        self,
    ) -> ColumnDiagonalGreenResponseContext | None:
        return self._green_response_context_cache.context

    @property
    def column_diagonal_green_response_context_build_count(self) -> int:
        return self._green_response_context_cache.build_count

    @property
    def symmetric_tangent_green_response_context(
        self,
    ) -> SymmetricTangentGreenResponseContext | None:
        return self._tangent_context_cache.context

    @property
    def symmetric_tangent_green_response_context_build_count(self) -> int:
        return self._tangent_context_cache.build_count

    @property
    def symmetric_tangent_green_response_context_build_seconds(self) -> float:
        return self._tangent_context_cache.build_seconds

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

    def _log_tangent_eta_schedule(
        self,
        schedule: SymmetricTangentEtaCapSchedule,
    ) -> None:
        self.logger.info(
            "tangent-eta schedule strategy=%s kind=%s final_eta=%.6e "
            "shared_with_lr_warmup=%s configured_warmup_epochs=%d "
            "effective_warmup_epochs=%d total_epochs=%d "
            "training_cap=scheduled validation_cap=final",
            schedule.eta_strategy,
            schedule.kind,
            schedule.final_eta,
            schedule.enabled,
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
        config = self.pre_projection_fusion_config
        self.logger.info(
            "pre-projection fusion enabled=%s space=physical_source "
            "architecture=%s mode=%s input_dim=2 "
            "hidden_dim=%d depth=%d activation=%s use_bias=%s "
            "identity_skip=%s final_initialization=%s "
            "final_layer_init_scale=%.6g explicit_geometry_features=false "
            "formula=%s",
            config.enabled,
            FUSION_ARCHITECTURE,
            config.mode,
            config.hidden_dim,
            config.depth,
            model.config.activation,
            model.config.use_bias,
            str(config.mode == "residual").lower(),
            FINAL_LAYER_INITIALIZATION,
            config.final_layer_init_scale,
            pre_projection_fusion_formula(config.mode),
        )

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
