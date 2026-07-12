from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import optim
from torch.utils.data import DataLoader

from greenonet.compile_utils import maybe_compile_model, model_state_dict_for_save
from greenonet.complex_coupling_data import (
    ComplexCouplingBatch,
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_losses import physical_edge_energy_loss, relative_l2_valid
from greenonet.complex_projection import (
    ComplexProjectionResult,
    apply_hard_symmetric_projection,
)
from greenonet.complex_reconstruction import (
    ComplexReconstructionResult,
    reconstruct_from_projected_unit,
)
from greenonet.config import CouplingTrainingConfig
from greenonet.io import save_state_dict_safetensors
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexForwardResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult


class ComplexCouplingTrainer(LoggingMixin):
    """Trainer for complex-geometry CouplingNet with no cross diagnostics."""

    _METRIC_KEYS: tuple[str, ...] = (
        "loss",
        "loss_energy_consistency",
        "rel_sol",
        "rel_flux",
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
        self.config = config
        self.green_model = green_model
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexCouplingTrainer",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
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
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_val_rel_sol: float | None = None
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._run_epoch(train_loader, optimizer=optimizer)
            self.loss_history.append(float(train_metrics["loss"]))
            self._record_metrics(epoch, "train", train_metrics)
            if epoch % self.config.log_interval == 0:
                self._log_epoch(epoch, "train", train_metrics)
            if validation_loader is not None:
                val_metrics = self._evaluate_loader(validation_loader)
                self._record_metrics(epoch, "val", val_metrics)
                if epoch % self.config.log_interval == 0:
                    self._log_epoch(epoch, "val", val_metrics)
                if (
                    self.config.best_rel_sol_checkpoint.enabled
                    and "rel_sol" in val_metrics
                ):
                    rel_sol = float(val_metrics["rel_sol"])
                    if best_val_rel_sol is None or rel_sol < best_val_rel_sol:
                        best_val_rel_sol = rel_sol
                        self._save_checkpoint(
                            "complex_coupling_model_best_rel_sol.safetensors"
                        )
            if (
                self.config.periodic_checkpoint.enabled
                and self.config.periodic_checkpoint.every_epochs > 0
                and epoch % self.config.periodic_checkpoint.every_epochs == 0
            ):
                self._save_checkpoint(
                    f"complex_coupling_model_epoch_{epoch:04d}.safetensors"
                )

        self._save_checkpoint("complex_coupling_model.safetensors")
        self._save_checkpoint("coupling_model.safetensors")
        self._write_metric_csv()

    def _run_epoch(
        self,
        loader: DataLoader[Any],
        *,
        optimizer: optim.Optimizer,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        batches = 0
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
            optimizer.step()
            self._accumulate(totals, result.metrics)
            batches += 1
        return self._average(totals, batches)

    def _evaluate_loader(
        self,
        loader: DataLoader[Any],
    ) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        totals: dict[str, float] = {}
        batches = 0
        with torch.no_grad():
            for batch in loader:
                result = self._forward_batch(batch.to(self.device))
                self._accumulate(totals, result.metrics)
                batches += 1
        if was_training:
            self.model.train()
        return self._average(totals, batches)

    def _forward_batch(self, batch: ComplexCouplingBatch) -> ComplexForwardResult:
        raw_physical = self.model(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
        )
        projection = apply_hard_symmetric_projection(
            raw_physical=raw_physical,
            rhs_phys=batch.rhs_valid,
            geometry=batch.geometry,
        )
        reconstruction = reconstruct_from_projected_unit(
            green_model=self.green_model,
            geometry=batch.geometry,
            projected_unit=projection.projected_unit,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        loss_energy = physical_edge_energy_loss(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            a_valid=batch.a_valid,
            geometry=batch.geometry,
        )
        loss = loss_energy
        metrics = {
            "loss": loss.detach(),
            "loss_energy_consistency": loss_energy.detach(),
            "rel_sol": relative_l2_valid(
                reconstruction.u_mean_valid,
                batch.sol_valid,
            ).detach(),
        }
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
        )

    @staticmethod
    def _accumulate(
        totals: dict[str, float],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())

    @staticmethod
    def _average(totals: dict[str, float], batches: int) -> dict[str, float]:
        if batches == 0:
            raise ValueError("Cannot average zero complex coupling batches.")
        return {key: value / batches for key, value in totals.items()}

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
