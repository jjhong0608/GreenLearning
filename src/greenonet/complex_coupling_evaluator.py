from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

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
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexPredictionBatch:
    batch: ComplexCouplingBatch
    raw_unit: torch.Tensor
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult
    metrics: dict[str, torch.Tensor]


class ComplexCouplingEvaluator(LoggingMixin):
    """Evaluate complex-geometry CouplingNet without cross diagnostics."""

    def __init__(
        self,
        *,
        model: ComplexCouplingNet,
        green_model: torch.nn.Module,
        device: torch.device,
        work_dir: Path | str,
        terminal_width: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.green_model = green_model.to(device)
        self.green_model.eval()
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexCouplingEvaluator",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )

    def evaluate(
        self,
        dataset: ComplexCouplingDataset,
        *,
        dataset_name: str,
        batch_size: int,
    ) -> dict[str, float]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        rows: list[dict[str, float | int | str]] = []
        totals: dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for batch in loader:
                prediction = self.predict_batch(batch.to(self.device))
                for sample_offset, sample_index in enumerate(
                    prediction.batch.sample_indices.cpu().tolist()
                ):
                    row = self._sample_metric_row(prediction, sample_offset)
                    row["sample_id"] = int(sample_index)
                    row["file_stem"] = prediction.batch.file_stems[sample_offset]
                    rows.append(row)
                for key, value in prediction.metrics.items():
                    totals[key] = totals.get(key, 0.0) + float(value.item())
                count += 1
        summary = {key: value / max(count, 1) for key, value in totals.items()}
        self._write_outputs(dataset_name, rows, summary)
        return summary

    def predict_batch(self, batch: ComplexCouplingBatch) -> ComplexPredictionBatch:
        raw_unit = self.model(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_unit_norm=batch.x_source_unit_norm,
            y_source_unit_norm=batch.y_source_unit_norm,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
        )
        projection = apply_hard_symmetric_projection(
            raw_unit=raw_unit,
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
        metrics = {
            "loss": loss_energy.detach(),
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
        return ComplexPredictionBatch(
            batch=batch,
            raw_unit=raw_unit,
            projection=projection,
            reconstruction=reconstruction,
            metrics=metrics,
        )

    @staticmethod
    def _sample_metric_row(
        prediction: ComplexPredictionBatch,
        sample_offset: int,
    ) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "loss": float(
                physical_edge_energy_loss(
                    u_phi_valid=prediction.reconstruction.u_phi_valid[
                        sample_offset : sample_offset + 1
                    ],
                    u_psi_valid=prediction.reconstruction.u_psi_valid[
                        sample_offset : sample_offset + 1
                    ],
                    a_valid=prediction.batch.a_valid[sample_offset : sample_offset + 1],
                    geometry=prediction.batch.geometry,
                ).item()
            ),
            "loss_energy_consistency": float(
                physical_edge_energy_loss(
                    u_phi_valid=prediction.reconstruction.u_phi_valid[
                        sample_offset : sample_offset + 1
                    ],
                    u_psi_valid=prediction.reconstruction.u_psi_valid[
                        sample_offset : sample_offset + 1
                    ],
                    a_valid=prediction.batch.a_valid[sample_offset : sample_offset + 1],
                    geometry=prediction.batch.geometry,
                ).item()
            ),
            "rel_sol": float(
                relative_l2_valid(
                    prediction.reconstruction.u_mean_valid[
                        sample_offset : sample_offset + 1
                    ],
                    prediction.batch.sol_valid[sample_offset : sample_offset + 1],
                ).item()
            ),
        }
        if bool(prediction.batch.has_flux[sample_offset].item()):
            row["rel_flux"] = float(
                relative_l2_valid(
                    prediction.projection.projected_physical[
                        sample_offset : sample_offset + 1
                    ],
                    prediction.batch.flux_valid[sample_offset : sample_offset + 1],
                ).item()
            )
        return row

    def _write_outputs(
        self,
        dataset_name: str,
        rows: list[dict[str, float | int | str]],
        summary: dict[str, float],
    ) -> None:
        metrics_dir = self.work_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / f"{dataset_name}_metrics.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        csv_path = metrics_dir / f"{dataset_name}_per_sample_metrics.csv"
        if rows:
            fieldnames = list(rows[0].keys())
            for row in rows[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            with csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
