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
from greenonet.complex_coupling_objective import (
    ComplexCouplingObjectiveResult,
    compute_complex_coupling_objective,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_gluing import (
    ComplexGluingContext,
    build_complex_gluing_context,
)
from greenonet.complex_losses import (
    ComplexEnergyLossResult,
    ComplexLengthJumpPartition,
    build_length_jump_partition,
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
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexAdmissibilityGluingConfig,
    ComplexLengthJumpBalanceConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingTrainingConfig,
)
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexPredictionBatch:
    batch: ComplexCouplingBatch
    raw_response: torch.Tensor
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult
    energy: ComplexEnergyLossResult
    objective: ComplexCouplingObjectiveResult
    metrics: dict[str, torch.Tensor]


class ComplexCouplingEvaluator(LoggingMixin):
    """Evaluate complex-geometry CouplingNet without cross diagnostics."""

    def __init__(
        self,
        *,
        model: ComplexCouplingNet,
        green_model: torch.nn.Module,
        config: CouplingTrainingConfig,
        device: torch.device,
        work_dir: Path | str,
        terminal_width: int | None = None,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.balance_projection = BalanceProjectionConfig.from_raw(
            model.config.balance_projection
        )
        self.length_jump_config = ComplexLengthJumpBalanceConfig.from_raw(
            config.length_jump_balance
        )
        self.relative_split_config = ComplexRelativeSplitConsistencyConfig.from_raw(
            config.relative_split_consistency
        )
        self.weak_closure_config = ComplexWeakOperatorClosureConfig.from_raw(
            config.weak_operator_closure
        )
        self.gluing_config = ComplexAdmissibilityGluingConfig.from_raw(
            config.admissibility_gluing
        )
        if not self.length_jump_config.enabled:
            raise ValueError(
                "ComplexCouplingEvaluator output-contract version 6 requires "
                "coupling_training.length_jump_balance.enabled=true."
            )
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
        self._length_jump_partition: ComplexLengthJumpPartition | None = None
        self._gluing_context: ComplexGluingContext | None = None

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
        sample_count = 0
        with torch.no_grad():
            for batch in loader:
                prediction = self.predict_batch(batch.to(self.device))
                batch_size = int(prediction.batch.rhs_valid.shape[0])
                for sample_offset, sample_index in enumerate(
                    prediction.batch.sample_indices.cpu().tolist()
                ):
                    row = self._sample_metric_row(prediction, sample_offset)
                    row["sample_id"] = int(sample_index)
                    row["file_stem"] = prediction.batch.file_stems[sample_offset]
                    rows.append(row)
                for key, value in prediction.metrics.items():
                    totals[key] = totals.get(key, 0.0) + batch_size * float(
                        value.item()
                    )
                sample_count += batch_size
        summary = {key: value / max(sample_count, 1) for key, value in totals.items()}
        self._write_outputs(dataset_name, rows, summary)
        return summary

    def predict_batch(self, batch: ComplexCouplingBatch) -> ComplexPredictionBatch:
        raw_response = self.model(
            geometry=batch.geometry,
            x_source_branch=batch.x_source_branch,
            y_source_branch=batch.y_source_branch,
            x_source_amplitude=batch.x_source_amplitude,
            y_source_amplitude=batch.y_source_amplitude,
            x_coefficient_branch=batch.x_coefficient_branch,
            y_coefficient_branch=batch.y_coefficient_branch,
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
            length_jump_config=self.length_jump_config,
            relative_split_config=self.relative_split_config,
            weak_closure_config=self.weak_closure_config,
            gluing_config=self.gluing_config,
            gluing_context=self.trace_gluing_context(batch.geometry),
            partition=self.energy_partition(batch.geometry),
        )
        metrics = {
            key: value.detach() for key, value in objective.metric_tensors().items()
        }
        metrics["rel_sol"] = relative_l2_valid(
            reconstruction.u_mean_valid,
            batch.sol_valid,
        ).detach()
        if torch.any(batch.has_flux):
            selected = batch.has_flux
            metrics["rel_flux"] = relative_l2_valid(
                projection.projected_physical[selected],
                batch.flux_valid[selected],
            ).detach()
        return ComplexPredictionBatch(
            batch=batch,
            raw_response=raw_response,
            projection=projection,
            reconstruction=reconstruction,
            energy=objective.energy,
            objective=objective,
            metrics=metrics,
        )

    def _sample_metric_row(
        self,
        prediction: ComplexPredictionBatch,
        sample_offset: int,
    ) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            key: float(value.item())
            for key, value in prediction.objective.sample_metric_tensors(
                sample_offset
            ).items()
        }
        row["rel_sol"] = float(
            relative_l2_valid(
                prediction.reconstruction.u_mean_valid[
                    sample_offset : sample_offset + 1
                ],
                prediction.batch.sol_valid[sample_offset : sample_offset + 1],
            ).item()
        )
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

    def energy_partition(
        self,
        geometry: ComplexGeometryMetadata,
    ) -> ComplexLengthJumpPartition:
        if self._length_jump_partition is None:
            self._length_jump_partition = build_length_jump_partition(
                geometry,
                self.length_jump_config,
            )
        return self._length_jump_partition

    def trace_gluing_context(
        self,
        geometry: ComplexGeometryMetadata,
    ) -> ComplexGluingContext | None:
        if not self.gluing_config.enabled:
            return None
        if self._gluing_context is None:
            self._gluing_context = build_complex_gluing_context(
                geometry,
                self.gluing_config,
            )
        return self._gluing_context

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
