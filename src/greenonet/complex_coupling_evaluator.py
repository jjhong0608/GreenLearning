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
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
    ComplexCrossAxisReconstructor,
)
from greenonet.complex_pre_projection_fusion import (
    ComplexPreProjectionFusionResult,
)
from greenonet.complex_coupling_objective import (
    ComplexCouplingObjectiveResult,
    compute_complex_coupling_objective,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
    ColumnDiagonalGreenResponseContextCache,
)
from greenonet.complex_losses import (
    ComplexBoundaryEnergyContext,
    ComplexEnergyLossResult,
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
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexRelativeSplitConsistencyConfig,
    ComplexWeakOperatorClosureConfig,
    CouplingTrainingConfig,
)
from greenonet.logging_mixin import LoggingMixin


@dataclass(frozen=True)
class ComplexPredictionBatch:
    batch: ComplexCouplingBatch
    raw_response: torch.Tensor
    pre_projection_fusion: ComplexPreProjectionFusionResult | None
    projection: ComplexProjectionResult
    reconstruction: ComplexReconstructionResult
    cross_axis_reconstruction: ComplexCrossAxisReconstructionResult
    energy: ComplexEnergyLossResult
    objective: ComplexCouplingObjectiveResult
    metrics: dict[str, torch.Tensor]


class ComplexCouplingEvaluator(LoggingMixin):
    """Evaluate complex CouplingNet with the configured final reconstruction."""

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
        self.cross_axis_reconstruction_config = (
            ComplexCrossAxisReconstructionConfig.from_raw(
                model.config.cross_axis_reconstruction
            )
        )
        self.cross_axis_reconstructor = ComplexCrossAxisReconstructor(
            self.cross_axis_reconstruction_config
        )
        self.relative_split_config = ComplexRelativeSplitConsistencyConfig.from_raw(
            config.relative_split_consistency
        )
        self.weak_closure_config = ComplexWeakOperatorClosureConfig.from_raw(
            config.weak_operator_closure
        )
        self.green_model = green_model.to(device)
        self.green_model.eval()
        for parameter in self.green_model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            logger_name="ComplexCouplingEvaluator",
            work_dir=self.work_dir,
            terminal_width=terminal_width,
        )
        self._boundary_context: ComplexBoundaryEnergyContext | None = None
        self._green_response_context_cache = ColumnDiagonalGreenResponseContextCache(
            self.balance_projection.column_diagonal_green_response
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
        projection_context = self._projection_context(batch)
        raw_response, fusion = self.model.forward_with_fusion_diagnostics(
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
        )
        reconstruction = reconstruct_from_projected_response(
            green_model=self.green_model,
            geometry=batch.geometry,
            projected_response=projection.projected_response,
            x_green_branch=batch.x_green_branch,
            y_green_branch=batch.y_green_branch,
        )
        cross_axis_reconstruction = self.cross_axis_reconstructor.reconstruct(
            u_phi_valid=reconstruction.u_phi_valid,
            u_psi_valid=reconstruction.u_psi_valid,
            projected_physical=projection.projected_physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
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
            boundary_context=self.boundary_energy_context(batch.geometry),
        )
        metrics = {
            key: value.detach() for key, value in objective.metric_tensors().items()
        }
        if torch.any(batch.has_solution):
            selected_solution = batch.has_solution
            metrics["rel_sol"] = relative_l2_valid(
                cross_axis_reconstruction.u_pred_valid[selected_solution],
                batch.sol_valid[selected_solution],
            ).detach()
            if self.cross_axis_reconstruction_config.enabled:
                metrics["rel_sol_equal_mean"] = relative_l2_valid(
                    cross_axis_reconstruction.u_equal_mean_valid[selected_solution],
                    batch.sol_valid[selected_solution],
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
            pre_projection_fusion=fusion,
            projection=projection,
            reconstruction=reconstruction,
            cross_axis_reconstruction=cross_axis_reconstruction,
            energy=objective.energy,
            objective=objective,
            metrics=metrics,
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

    @property
    def column_diagonal_green_response_context(
        self,
    ) -> ColumnDiagonalGreenResponseContext | None:
        return self._green_response_context_cache.context

    @property
    def column_diagonal_green_response_context_build_count(self) -> int:
        return self._green_response_context_cache.build_count

    @property
    def column_diagonal_green_response_context_build_seconds(self) -> float:
        return self._green_response_context_cache.build_seconds

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
        if bool(prediction.batch.has_solution[sample_offset].item()):
            row["rel_sol"] = float(
                relative_l2_valid(
                    prediction.cross_axis_reconstruction.u_pred_valid[
                        sample_offset : sample_offset + 1
                    ],
                    prediction.batch.sol_valid[sample_offset : sample_offset + 1],
                ).item()
            )
            if self.cross_axis_reconstruction_config.enabled:
                row["rel_sol_equal_mean"] = float(
                    relative_l2_valid(
                        prediction.cross_axis_reconstruction.u_equal_mean_valid[
                            sample_offset : sample_offset + 1
                        ],
                        prediction.batch.sol_valid[sample_offset : sample_offset + 1],
                    ).item()
                )
        reliability = prediction.cross_axis_reconstruction.reliability
        if reliability is not None:
            weight = reliability.w_phi[sample_offset]
            row["weak_weight_phi_mean"] = float(weight.mean().item())
            row["weak_weight_phi_min"] = float(weight.min().item())
            row["weak_weight_phi_max"] = float(weight.max().item())
            row["weak_support_fraction"] = float(
                reliability.support_mask[sample_offset]
                .to(dtype=torch.float64)
                .mean()
                .item()
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

    def boundary_energy_context(
        self,
        geometry: ComplexGeometryMetadata,
    ) -> ComplexBoundaryEnergyContext:
        if self._boundary_context is None:
            self._boundary_context = build_boundary_energy_context(geometry)
        return self._boundary_context

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
