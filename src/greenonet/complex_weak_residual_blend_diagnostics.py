from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from greenonet.coefficients import CoefficientFunctions, load_coefficient_functions
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructor,
    LocalWeakResidualReliabilityContext,
    LocalWeakResidualReliabilityFields as WeakResidualReliabilityFields,
)
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_mismatch_blend_diagnostics import (
    CrossAxisBlendComparisonRequest,
    CrossAxisBlendEstimatorComparison,
    MismatchGradientBlendConfig,
    MismatchSeamC2BlendConfig,
)
from greenonet.complex_smooth_blend_diagnostics import (
    FixedSmoothBlendConfig,
    FixedSmoothBlendEvaluation,
)
from greenonet.complex_weak_closure import build_directional_weak_context
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    CouplingArtifactConfigs,
    load_coupling_artifact_configs,
)
from greenonet.config import ComplexCrossAxisReconstructionConfig
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class WeakResidualReliabilityBlendConfig:
    """Prediction-only local weak-residual reliability blend parameters."""

    gamma: float = 0.5
    smoothing_steps: int = 2
    smoothing_relaxation: float = 0.5
    relative_floor: float = 0.1
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if (
            isinstance(self.smoothing_steps, bool)
            or not isinstance(self.smoothing_steps, int)
            or self.smoothing_steps < 0
        ):
            raise ValueError("smoothing_steps must be a non-negative integer.")
        if (
            not math.isfinite(self.smoothing_relaxation)
            or not 0.0 < self.smoothing_relaxation <= 1.0
        ):
            raise ValueError("smoothing_relaxation must be in (0, 1].")
        if not math.isfinite(self.relative_floor) or self.relative_floor < 0.0:
            raise ValueError("relative_floor must be finite and non-negative.")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be finite and positive.")


@dataclass(frozen=True)
class WeakResidualBlendComparisonRequest(CrossAxisBlendComparisonRequest):
    """Frozen-checkpoint request for the general reliability comparison."""

    blend: FixedSmoothBlendConfig = FixedSmoothBlendConfig(
        weight_construction="compact_c2_ramp",
        ramp_gamma=0.5,
    )
    mismatch: MismatchGradientBlendConfig = MismatchGradientBlendConfig()
    seam_c2: MismatchSeamC2BlendConfig = MismatchSeamC2BlendConfig(gamma=0.3)
    weak_residual: WeakResidualReliabilityBlendConfig = (
        WeakResidualReliabilityBlendConfig()
    )
    weak_sweep: bool = False
    weak_sweep_gammas: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    weak_sweep_relative_floors: tuple[float, ...] = (0.01, 0.1, 1.0)
    weak_sweep_smoothing_steps: tuple[int, ...] = (0, 2, 4)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.blend.weight_construction != "compact_c2_ramp":
            raise ValueError("The four-way comparison requires geometry compact C2.")
        if not self.weak_sweep_gammas or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.weak_sweep_gammas
        ):
            raise ValueError("weak_sweep_gammas must contain finite values in [0, 1].")
        if not self.weak_sweep_relative_floors or any(
            not math.isfinite(value) or value < 0.0
            for value in self.weak_sweep_relative_floors
        ):
            raise ValueError(
                "weak_sweep_relative_floors must contain non-negative values."
            )
        if not self.weak_sweep_smoothing_steps or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.weak_sweep_smoothing_steps
        ):
            raise ValueError(
                "weak_sweep_smoothing_steps must contain non-negative integers."
            )


@dataclass(frozen=True)
class WeakResidualBlendEvaluation(FixedSmoothBlendEvaluation):
    """Directional reconstructions plus reference-free projection inputs."""

    rhs: torch.Tensor
    projected_physical: torch.Tensor


class WeakResidualReliabilityMixin:
    """Build a local reliability blend from candidate full-PDE weak defects."""

    @staticmethod
    def _validate_candidate_inputs(
        geometry: ComplexGeometryMetadata,
        evaluation: WeakResidualBlendEvaluation,
    ) -> None:
        expected = (len(evaluation.sample_ids), geometry.num_points)
        if evaluation.u_phi.shape != expected or evaluation.u_psi.shape != expected:
            raise ValueError("Directional solutions must have shape (B, P).")
        if evaluation.rhs.shape != expected:
            raise ValueError("rhs must have shape (B, P).")
        if evaluation.projected_physical.shape != (
            expected[0],
            2,
            expected[1],
        ):
            raise ValueError("projected_physical must have shape (B, 2, P).")

    @staticmethod
    def _production_config(
        config: WeakResidualReliabilityBlendConfig,
    ) -> ComplexCrossAxisReconstructionConfig:
        return ComplexCrossAxisReconstructionConfig(
            enabled=True,
            gamma=config.gamma,
            smoothing_steps=config.smoothing_steps,
            smoothing_relaxation=config.smoothing_relaxation,
            relative_floor=config.relative_floor,
            eps=config.eps,
        )

    @classmethod
    def _weights_from_raw_indicators(
        cls,
        geometry: ComplexGeometryMetadata,
        phi_indicator_raw: torch.Tensor,
        psi_indicator_raw: torch.Tensor,
        config: WeakResidualReliabilityBlendConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            _w_psi,
            support_mask,
        ) = ComplexCrossAxisReconstructor.weights_from_geometry(
            phi_indicator_raw,
            psi_indicator_raw,
            geometry=geometry,
            config=cls._production_config(config),
        )
        return (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            support_mask,
        )

    @classmethod
    def build_weak_residual_reliability_fields(
        cls,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
        evaluation: WeakResidualBlendEvaluation,
        config: WeakResidualReliabilityBlendConfig,
    ) -> WeakResidualReliabilityFields:
        cls._validate_candidate_inputs(geometry, evaluation)
        weak_context = build_directional_weak_context(geometry, coeffs)
        context = LocalWeakResidualReliabilityContext.build(
            geometry,
            weak_context,
        )
        return ComplexCrossAxisReconstructor.build_reliability_fields(
            u_phi_valid=evaluation.u_phi,
            u_psi_valid=evaluation.u_psi,
            projected_physical=evaluation.projected_physical,
            context=context,
            config=cls._production_config(config),
        )

    @classmethod
    def reweight_weak_residual_fields(
        cls,
        geometry: ComplexGeometryMetadata,
        fields: WeakResidualReliabilityFields,
        config: WeakResidualReliabilityBlendConfig,
    ) -> WeakResidualReliabilityFields:
        (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            _w_psi,
            support_mask,
        ) = ComplexCrossAxisReconstructor.weights_from_geometry(
            fields.phi_indicator_raw,
            fields.psi_indicator_raw,
            geometry=geometry,
            config=cls._production_config(config),
        )
        return replace(
            fields,
            phi_indicator=phi_indicator,
            psi_indicator=psi_indicator,
            sample_floor=sample_floor,
            theta=theta,
            w_phi=w_phi,
            w_psi=1.0 - w_phi,
            support_mask=support_mask,
        )


class WeakResidualBlendComparison(
    WeakResidualReliabilityMixin,
    CrossAxisBlendEstimatorComparison,
):
    """Compare four post-hoc estimators on one frozen complex checkpoint."""

    request: WeakResidualBlendComparisonRequest

    def __init__(
        self,
        request: WeakResidualBlendComparisonRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(request, logger=logger)
        self.weak_fields: WeakResidualReliabilityFields

    def run(self) -> dict[str, Any]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError("Weak-residual comparison requires complex geometry.")
        geometry_path = self.request.geometry or configs.dataset.geometry_path
        test_path = self.request.test_path or configs.dataset.test_path
        coefficient_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        if geometry_path is None or test_path is None or coefficient_path is None:
            raise ValueError("Geometry, test data, and coefficients are required.")

        device = torch.device(self.request.device or configs.coupling_training.device)
        self.geometry = load_complex_geometry(
            geometry_path,
            dtype=configs.dataset.dtype,
        )
        self.blend_fields = self.build_fixed_blend_fields(
            self.geometry,
            self.request.blend,
        )
        coeffs = load_coefficient_functions(coefficient_path)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            coeffs,
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        if len(dataset) == 0:
            raise ValueError("The test dataset is empty.")

        geometry_evaluation = self._evaluate_dataset_with_projection(
            dataset,
            configs,
            device,
        )
        mismatch_fields = self.build_mismatch_gradient_fields(
            self.geometry,
            geometry_evaluation.u_phi,
            geometry_evaluation.u_psi,
            self.request.mismatch,
        )
        self.seam_c2_fields = self.build_mismatch_seam_c2_fields(
            self.geometry,
            mismatch_fields,
            self.request.seam_c2,
        )
        seam_evaluation = self._evaluation_with_sample_weights(
            geometry_evaluation,
            self.seam_c2_fields.w_phi,
            self.seam_c2_fields.w_psi,
        )
        self.weak_fields = self.build_weak_residual_reliability_fields(
            self.geometry,
            coeffs,
            geometry_evaluation,
            self.request.weak_residual,
        )
        weak_evaluation = self._evaluation_with_sample_weights(
            geometry_evaluation,
            self.weak_fields.w_phi,
            self.weak_fields.w_psi,
        )

        rows = self._four_way_rows(
            geometry_evaluation,
            seam_evaluation,
            weak_evaluation,
        )
        aggregate = self._four_way_aggregate(rows)
        selected, roles = self._select_samples_from_rows(rows)
        metrics_dir = self.request.outdir / "metrics"
        self._write_csv(
            metrics_dir / "per_sample_weak_residual_blend_comparison.csv",
            rows,
        )
        sweep_rows = (
            self._run_weak_sweep(geometry_evaluation) if self.request.weak_sweep else []
        )
        if sweep_rows:
            self._write_csv(
                metrics_dir / "weak_residual_parameter_sweep.csv",
                sweep_rows,
            )
        if self.request.save_generated_data:
            self._write_selected_arrays(
                geometry_evaluation,
                seam_evaluation,
                weak_evaluation,
                selected,
            )
        figure_paths = [
            self._write_four_way_metric_figure(rows),
            *self._write_weak_selected_figures(
                geometry_evaluation,
                seam_evaluation,
                weak_evaluation,
                selected,
            ),
        ]
        summary = self._build_weak_summary(
            configs=configs,
            dataset=dataset,
            geometry_path=Path(geometry_path),
            test_path=Path(test_path),
            coefficient_path=Path(coefficient_path),
            aggregate=aggregate,
            rows=rows,
            sweep_rows=sweep_rows,
            selected=selected,
            roles=roles,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self._write_report(summary)
        if self.logger is not None:
            self.logger.info(
                "Weak-residual comparison complete: equal=%.6f geometry=%.6f "
                "seam=%.6f weak=%.6f",
                aggregate["equal_mean"]["rel_sol_mean"],
                aggregate["geometry_c2"]["rel_sol_mean"],
                aggregate["mismatch_seam_c2"]["rel_sol_mean"],
                aggregate["weak_residual_reliability"]["rel_sol_mean"],
            )
        return summary

    def _evaluate_dataset_with_projection(
        self,
        dataset: ComplexCouplingDataset,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> WeakResidualBlendEvaluation:
        loader_request = CouplingArtifactRequest(
            config=self.request.config,
            coupling_checkpoint=self.request.coupling_checkpoint,
            green_checkpoint=self.request.green_checkpoint,
            outdir=self.request.outdir,
            coefficients=self.request.coefficients,
            device=str(device),
            theme=self.request.theme,
        )
        model_loader = ComplexCouplingArtifactExporter(
            loader_request,
            logger=self.logger,
        )
        coupling_model = model_loader._load_complex_model(configs, device)
        green_model = model_loader._load_green_model(configs, device)
        evaluator = ComplexCouplingEvaluator(
            model=coupling_model,
            green_model=green_model,
            config=configs.coupling_training,
            device=device,
            work_dir=self.request.outdir / "_evaluator",
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self.request.batch_size, len(dataset)),
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        sample_ids: list[torch.Tensor] = []
        file_stems: list[str] = []
        sol: list[torch.Tensor] = []
        rhs: list[torch.Tensor] = []
        projected_physical: list[torch.Tensor] = []
        u_phi: list[torch.Tensor] = []
        u_psi: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                prediction = evaluator.predict_batch(batch.to(device))
                if not bool(torch.all(prediction.batch.has_solution).item()):
                    raise ValueError(
                        "All comparison samples must contain evaluation-only sol."
                    )
                sample_ids.append(prediction.batch.sample_indices.detach().cpu())
                file_stems.extend(prediction.batch.file_stems)
                sol.append(prediction.batch.sol_valid.detach().cpu())
                rhs.append(prediction.batch.rhs_valid.detach().cpu())
                projected_physical.append(
                    prediction.projection.projected_physical.detach().cpu()
                )
                u_phi.append(prediction.reconstruction.u_phi_valid.detach().cpu())
                u_psi.append(prediction.reconstruction.u_psi_valid.detach().cpu())
        sample_ids_tensor = torch.cat(sample_ids, dim=0)
        sol_tensor = torch.cat(sol, dim=0)
        rhs_tensor = torch.cat(rhs, dim=0)
        projected_tensor = torch.cat(projected_physical, dim=0)
        u_phi_tensor = torch.cat(u_phi, dim=0)
        u_psi_tensor = torch.cat(u_psi, dim=0)
        baseline = 0.5 * (u_phi_tensor + u_psi_tensor)
        geometry_blend = (
            self.blend_fields.w_phi.detach().cpu().unsqueeze(0) * u_phi_tensor
            + self.blend_fields.w_psi.detach().cpu().unsqueeze(0) * u_psi_tensor
        )
        return WeakResidualBlendEvaluation(
            sample_ids=sample_ids_tensor,
            file_stems=tuple(file_stems),
            sol=sol_tensor,
            u_phi=u_phi_tensor,
            u_psi=u_psi_tensor,
            baseline=baseline,
            blend=geometry_blend,
            rhs=rhs_tensor,
            projected_physical=projected_tensor,
        )

    @staticmethod
    def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
        denominator = torch.linalg.vector_norm(target).clamp_min(1.0e-12)
        return float(
            (torch.linalg.vector_norm(prediction - target) / denominator).item()
        )

    @staticmethod
    def _rms(values: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(values.square())).item())

    def _transition_metric(
        self,
        error: torch.Tensor,
    ) -> tuple[float, float]:
        point_mask = self.blend_fields.transition_point_mask
        transition_edges, _ = self._transition_edges(self.blend_fields)
        zone_rms = math.nan
        trace_rms = math.nan
        if torch.any(point_mask):
            zone_rms = self._rms(error[point_mask])
        if transition_edges.numel() > 0:
            trace_rms = self._rms(
                error[transition_edges[:, 1]] - error[transition_edges[:, 0]]
            )
        return zone_rms, trace_rms

    def _four_way_rows(
        self,
        geometry: WeakResidualBlendEvaluation,
        seam: FixedSmoothBlendEvaluation,
        weak: FixedSmoothBlendEvaluation,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for offset, sample_id in enumerate(geometry.sample_ids.tolist()):
            sol = geometry.sol[offset]
            predictions = {
                "equal_mean": geometry.baseline[offset],
                "geometry_c2": geometry.blend[offset],
                "mismatch_seam_c2": seam.blend[offset],
                "weak_residual_reliability": weak.blend[offset],
            }
            row: dict[str, float | int | str] = {
                "sample_id": int(sample_id),
                "file_stem": geometry.file_stems[offset],
                "u_phi_rel_sol": self._relative_l2(geometry.u_phi[offset], sol),
                "u_psi_rel_sol": self._relative_l2(geometry.u_psi[offset], sol),
                "weak_weight_phi_mean": float(
                    self.weak_fields.w_phi[offset].mean().item()
                ),
                "weak_weight_phi_min": float(
                    self.weak_fields.w_phi[offset].min().item()
                ),
                "weak_weight_phi_max": float(
                    self.weak_fields.w_phi[offset].max().item()
                ),
                "weak_support_fraction": float(
                    self.weak_fields.support_mask[offset]
                    .to(dtype=torch.float64)
                    .mean()
                    .item()
                ),
                "weak_phi_indicator_mean": float(
                    self.weak_fields.phi_indicator[offset].mean().item()
                ),
                "weak_psi_indicator_mean": float(
                    self.weak_fields.psi_indicator[offset].mean().item()
                ),
            }
            for name, prediction in predictions.items():
                error = prediction - sol
                transition_rms, trace_rms = self._transition_metric(error)
                row[f"{name}_rel_sol"] = self._relative_l2(prediction, sol)
                row[f"{name}_error_rms"] = self._rms(error)
                row[f"{name}_transition_error_rms"] = transition_rms
                row[f"{name}_transition_trace_error_jump_rms"] = trace_rms
            rows.append(row)
        return rows

    @staticmethod
    def _finite_mean(rows: Sequence[dict[str, float | int | str]], key: str) -> float:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return math.nan if finite.size == 0 else float(finite.mean())

    def _four_way_aggregate(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> dict[str, dict[str, float | int]]:
        names = (
            "equal_mean",
            "geometry_c2",
            "mismatch_seam_c2",
            "weak_residual_reliability",
        )
        baseline = self._finite_mean(rows, "equal_mean_rel_sol")
        result: dict[str, dict[str, float | int]] = {}
        for name in names:
            rel_sol = self._finite_mean(rows, f"{name}_rel_sol")
            result[name] = {
                "sample_count": len(rows),
                "rel_sol_mean": rel_sol,
                "rel_sol_relative_change_vs_equal": (
                    (rel_sol - baseline) / max(baseline, 1.0e-12)
                ),
                "rel_sol_win_count_vs_equal": sum(
                    float(row[f"{name}_rel_sol"]) < float(row["equal_mean_rel_sol"])
                    for row in rows
                ),
                "error_rms_mean": self._finite_mean(rows, f"{name}_error_rms"),
                "transition_error_rms_mean": self._finite_mean(
                    rows,
                    f"{name}_transition_error_rms",
                ),
                "transition_trace_error_jump_rms_mean": self._finite_mean(
                    rows,
                    f"{name}_transition_trace_error_jump_rms",
                ),
            }
        return result

    def _select_samples_from_rows(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> tuple[tuple[int, ...], dict[str, int]]:
        if self.request.selected_samples is not None:
            available = {int(row["sample_id"]) for row in rows}
            selected = tuple(dict.fromkeys(self.request.selected_samples))
            missing = sorted(set(selected) - available)
            if missing:
                raise IndexError(f"Selected sample indices are unavailable: {missing}.")
            return selected, {}
        sorted_rows = sorted(rows, key=lambda row: float(row["equal_mean_rel_sol"]))
        positions = {
            "min": 0,
            "q25": round(0.25 * (len(sorted_rows) - 1)),
            "q50": round(0.50 * (len(sorted_rows) - 1)),
            "q75": round(0.75 * (len(sorted_rows) - 1)),
            "max": len(sorted_rows) - 1,
        }
        roles = {
            role: int(sorted_rows[position]["sample_id"])
            for role, position in positions.items()
        }
        return tuple(dict.fromkeys(roles.values())), roles

    def _run_weak_sweep(
        self,
        evaluation: WeakResidualBlendEvaluation,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for gamma in self.request.weak_sweep_gammas:
            for relative_floor in self.request.weak_sweep_relative_floors:
                for smoothing_steps in self.request.weak_sweep_smoothing_steps:
                    config = replace(
                        self.request.weak_residual,
                        gamma=gamma,
                        relative_floor=relative_floor,
                        smoothing_steps=smoothing_steps,
                    )
                    fields = self.reweight_weak_residual_fields(
                        self.geometry,
                        self.weak_fields,
                        config,
                    )
                    candidate = self._evaluation_with_sample_weights(
                        evaluation,
                        fields.w_phi,
                        fields.w_psi,
                    )
                    rel_values = [
                        self._relative_l2(candidate.blend[index], evaluation.sol[index])
                        for index in range(len(evaluation.sample_ids))
                    ]
                    baseline_values = [
                        self._relative_l2(
                            evaluation.baseline[index],
                            evaluation.sol[index],
                        )
                        for index in range(len(evaluation.sample_ids))
                    ]
                    mean_rel = float(np.mean(rel_values))
                    baseline_mean = float(np.mean(baseline_values))
                    rows.append(
                        {
                            "gamma": gamma,
                            "relative_floor": relative_floor,
                            "smoothing_steps": smoothing_steps,
                            "rel_sol_mean": mean_rel,
                            "rel_sol_relative_change_vs_equal": (
                                (mean_rel - baseline_mean) / max(baseline_mean, 1.0e-12)
                            ),
                            "rel_sol_win_count_vs_equal": sum(
                                candidate_value < baseline_value
                                for candidate_value, baseline_value in zip(
                                    rel_values,
                                    baseline_values,
                                    strict=True,
                                )
                            ),
                            "weight_phi_min": float(fields.w_phi.min().item()),
                            "weight_phi_max": float(fields.w_phi.max().item()),
                            "weight_neighbor_jump_max": self._sample_weight_jump_max(
                                fields.w_phi
                            ),
                        }
                    )
        rows.sort(key=lambda row: float(row["rel_sol_mean"]))
        return rows

    def _sample_weight_jump_max(self, weights: torch.Tensor) -> float:
        edges = torch.cat((self.geometry.x_edges, self.geometry.y_edges), dim=0)
        if edges.numel() == 0:
            return 0.0
        jump = (weights[:, edges[:, 1]] - weights[:, edges[:, 0]]).abs()
        return float(jump.max().item())

    def _write_selected_arrays(
        self,
        geometry: WeakResidualBlendEvaluation,
        seam: FixedSmoothBlendEvaluation,
        weak: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> None:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(geometry.sample_ids.tolist())
        }
        offsets = [sample_to_offset[sample_id] for sample_id in selected]
        payload = {
            "selected_sample_ids": np.asarray(selected, dtype=np.int64),
            "selected_file_stems": np.asarray(
                [geometry.file_stems[offset] for offset in offsets]
            ),
            "coords_valid": self._numpy(self.geometry.coords_valid),
            "sol": self._numpy(geometry.sol[offsets]),
            "rhs": self._numpy(geometry.rhs[offsets]),
            "projected_physical": self._numpy(geometry.projected_physical[offsets]),
            "u_phi": self._numpy(geometry.u_phi[offsets]),
            "u_psi": self._numpy(geometry.u_psi[offsets]),
            "u_equal_mean": self._numpy(geometry.baseline[offsets]),
            "u_geometry_c2": self._numpy(geometry.blend[offsets]),
            "u_mismatch_seam_c2": self._numpy(seam.blend[offsets]),
            "u_weak_residual_reliability": self._numpy(weak.blend[offsets]),
            "weak_phi_x_residual": self._numpy(
                self.weak_fields.phi_x_residual[offsets]
            ),
            "weak_phi_y_residual": self._numpy(
                self.weak_fields.phi_y_residual[offsets]
            ),
            "weak_phi_full_residual": self._numpy(
                self.weak_fields.phi_full_residual[offsets]
            ),
            "weak_psi_x_residual": self._numpy(
                self.weak_fields.psi_x_residual[offsets]
            ),
            "weak_psi_y_residual": self._numpy(
                self.weak_fields.psi_y_residual[offsets]
            ),
            "weak_psi_full_residual": self._numpy(
                self.weak_fields.psi_full_residual[offsets]
            ),
            "weak_nodal_mass": self._numpy(self.weak_fields.nodal_mass),
            "weak_phi_indicator_raw": self._numpy(
                self.weak_fields.phi_indicator_raw[offsets]
            ),
            "weak_psi_indicator_raw": self._numpy(
                self.weak_fields.psi_indicator_raw[offsets]
            ),
            "weak_phi_indicator": self._numpy(self.weak_fields.phi_indicator[offsets]),
            "weak_psi_indicator": self._numpy(self.weak_fields.psi_indicator[offsets]),
            "weak_sample_floor": self._numpy(self.weak_fields.sample_floor[offsets]),
            "weak_theta": self._numpy(self.weak_fields.theta[offsets]),
            "weak_w_phi": self._numpy(self.weak_fields.w_phi[offsets]),
            "weak_w_psi": self._numpy(self.weak_fields.w_psi[offsets]),
        }
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "selected_weak_residual_blend_arrays.npz",
            **payload,  # type: ignore[arg-type]
        )

    def _write_four_way_metric_figure(
        self,
        rows: list[dict[str, float | int | str]],
    ) -> str:
        baseline = np.asarray(
            [float(row["equal_mean_rel_sol"]) * 100.0 for row in rows]
        )
        figure = go.Figure()
        series = (
            ("Geometry-only C2", "geometry_c2_rel_sol", "#2A6FBB", "circle"),
            (
                "Mismatch-detected seam C2",
                "mismatch_seam_c2_rel_sol",
                "#D1495B",
                "diamond",
            ),
            (
                "Local weak-residual reliability",
                "weak_residual_reliability_rel_sol",
                "#238B45",
                "square",
            ),
        )
        for name, key, color, symbol in series:
            candidate = np.asarray([float(row[key]) * 100.0 for row in rows])
            figure.add_trace(
                go.Scatter(
                    x=baseline,
                    y=candidate,
                    mode="markers",
                    name=name,
                    marker={"color": color, "symbol": symbol, "size": 8},
                    customdata=np.asarray([int(row["sample_id"]) for row in rows]),
                    hovertemplate=(
                        "sample=%{customdata}<br>equal=%{x:.4f}%"
                        "<br>candidate=%{y:.4f}%<extra></extra>"
                    ),
                )
            )
        limit_min = float(
            min(
                baseline.min(),
                *(
                    np.asarray([float(row[key]) * 100.0 for row in rows]).min()
                    for _, key, _, _ in series
                ),
            )
        )
        limit_max = float(
            max(
                baseline.max(),
                *(
                    np.asarray([float(row[key]) * 100.0 for row in rows]).max()
                    for _, key, _, _ in series
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[limit_min, limit_max],
                y=[limit_min, limit_max],
                mode="lines",
                name="No change",
                line={"color": "#555555", "dash": "dash"},
                hoverinfo="skip",
            )
        )
        figure.update_layout(
            title="Four post-hoc blends versus equal mean",
            xaxis_title="Equal-mean rel_sol (%)",
            yaxis_title="Candidate rel_sol (%)",
            template=self.request.theme,
            width=980,
            height=760,
        )
        base = self.request.outdir / "figures" / "aggregate" / "four_way_rel_sol"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".html").relative_to(self.request.outdir))

    def _write_weak_selected_figures(
        self,
        geometry: WeakResidualBlendEvaluation,
        seam: FixedSmoothBlendEvaluation,
        weak: FixedSmoothBlendEvaluation,
        selected: Sequence[int],
    ) -> list[str]:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(geometry.sample_ids.tolist())
        }
        paths: list[str] = []
        for sample_id in selected:
            offset = sample_to_offset[sample_id]
            sol = self._numpy(geometry.sol[offset])
            phi_residual = self._numpy(self.weak_fields.phi_full_residual[offset])
            psi_residual = self._numpy(self.weak_fields.psi_full_residual[offset])
            phi_indicator = self._numpy(self.weak_fields.phi_indicator[offset])
            psi_indicator = self._numpy(self.weak_fields.psi_indicator[offset])
            theta = self._numpy(self.weak_fields.theta[offset])
            w_phi = self._numpy(self.weak_fields.w_phi[offset])
            errors = [
                self._numpy(geometry.u_phi[offset]) - sol,
                self._numpy(geometry.u_psi[offset]) - sol,
                self._numpy(geometry.baseline[offset]) - sol,
                self._numpy(weak.blend[offset]) - sol,
                self._numpy(geometry.blend[offset]) - sol,
                self._numpy(seam.blend[offset]) - sol,
            ]
            residual_limit = self._signed_limit(phi_residual, psi_residual)
            theta_limit = self._signed_limit(theta)
            error_limit = self._signed_limit(*errors)
            indicator_limit = max(
                float(phi_indicator.max()),
                float(psi_indicator.max()),
                1.0e-15,
            )
            titles = (
                "R_phi full weak residual",
                "R_psi full weak residual",
                "eta_phi^2",
                "eta_psi^2",
                "weak theta",
                "weak w_phi",
                "u_phi - sol",
                "u_psi - sol",
                "equal mean - sol",
                "weak blend - sol",
                "geometry C2 - sol",
                "seam C2 - sol",
            )
            figure = make_subplots(rows=3, cols=4, subplot_titles=titles)
            panels = (
                (phi_residual, "RdBu", -residual_limit, residual_limit),
                (psi_residual, "RdBu", -residual_limit, residual_limit),
                (phi_indicator, "Viridis", 0.0, indicator_limit),
                (psi_indicator, "Viridis", 0.0, indicator_limit),
                (theta, "RdBu", -theta_limit, theta_limit),
                (w_phi, "Viridis", 0.0, 1.0),
                (errors[0], "RdBu", -error_limit, error_limit),
                (errors[1], "RdBu", -error_limit, error_limit),
                (errors[2], "RdBu", -error_limit, error_limit),
                (errors[3], "RdBu", -error_limit, error_limit),
                (errors[4], "RdBu", -error_limit, error_limit),
                (errors[5], "RdBu", -error_limit, error_limit),
            )
            for index, (values, colorscale, cmin, cmax) in enumerate(panels):
                row = index // 4 + 1
                col = index % 4 + 1
                self._add_scatter(
                    figure,
                    row=row,
                    col=col,
                    values=values,
                    title=titles[index],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    subplot_columns=4,
                    colorbar_column=4,
                    colorbar_y={1: 0.86, 2: 0.5, 3: 0.14}[row],
                    colorbar_length=0.24,
                )
            figure.update_layout(
                title=f"Sample {sample_id}: local weak-residual reliability",
                template=self.request.theme,
                width=1900,
                height=1320,
            )
            for axis_name in [
                key
                for key in figure.layout
                if key.startswith("xaxis") or key.startswith("yaxis")
            ]:
                figure.layout[axis_name].update(scaleanchor=None)
            base = (
                self.request.outdir
                / "figures"
                / "selected"
                / f"sample_{sample_id:04d}_weak_residual_comparison"
            )
            save_plotly_figure(figure, base, logger=self.logger)
            paths.append(
                str(base.with_suffix(".html").relative_to(self.request.outdir))
            )
        return paths

    def _build_weak_summary(
        self,
        *,
        configs: CouplingArtifactConfigs,
        dataset: ComplexCouplingDataset,
        geometry_path: Path,
        test_path: Path,
        coefficient_path: Path,
        aggregate: dict[str, dict[str, float | int]],
        rows: list[dict[str, float | int | str]],
        sweep_rows: list[dict[str, float | int | str]],
        selected: Sequence[int],
        roles: dict[str, int],
        figure_paths: list[str],
    ) -> dict[str, Any]:
        weak_stats = {
            "w_phi_min": float(self.weak_fields.w_phi.min().item()),
            "w_phi_max": float(self.weak_fields.w_phi.max().item()),
            "w_phi_mean": float(self.weak_fields.w_phi.mean().item()),
            "partition_max_abs_residual": float(
                (self.weak_fields.w_phi + self.weak_fields.w_psi - 1.0)
                .abs()
                .max()
                .item()
            ),
            "support_fraction_mean": float(
                self.weak_fields.support_mask.to(dtype=torch.float64).mean().item()
            ),
            "weight_neighbor_jump_max": self._sample_weight_jump_max(
                self.weak_fields.w_phi
            ),
            "phi_indicator_mean": float(self.weak_fields.phi_indicator.mean().item()),
            "psi_indicator_mean": float(self.weak_fields.psi_indicator.mean().item()),
        }
        best_sweep = None if not sweep_rows else sweep_rows[0]
        return {
            "diagnostic": "local_weak_residual_reliability_blend_comparison",
            "status": "posthoc_exploratory",
            "production_code_changed": False,
            "training_or_checkpoint_changed": False,
            "num_samples": len(dataset),
            "geometry_path": str(geometry_path),
            "test_path": str(test_path),
            "coefficients": str(coefficient_path),
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "dtype": str(configs.dataset.dtype),
            "estimators": {
                "equal_mean": {"formula": "0.5*(u_phi+u_psi)"},
                "geometry_c2": {
                    "config": asdict(self.request.blend),
                    "uses_reference_targets": False,
                },
                "mismatch_detected_seam_c2": {
                    "config": asdict(self.request.seam_c2),
                    "uses_reference_targets": False,
                },
                "local_weak_residual_reliability": {
                    "config": asdict(self.request.weak_residual),
                    "formula": (
                        "R_candidate=Rx(u;phi)+Ry(u;psi); "
                        "theta=gamma*(eta_psi^2-eta_phi^2)/"
                        "(eta_phi^2+eta_psi^2+2*floor)"
                    ),
                    "uses_rhs": True,
                    "uses_projected_phi_psi": True,
                    "uses_coefficients": True,
                    "uses_geometry_transition_coordinates": False,
                    "uses_sol": False,
                    "uses_flux_targets": False,
                    "requires_global_matrix_solve": False,
                    "operator_application": "local_P1_element_gather_scatter",
                },
            },
            "aggregate_metrics": aggregate,
            "weak_residual_statistics": weak_stats,
            "weak_parameter_sweep": {
                "enabled": self.request.weak_sweep,
                "row_count": len(sweep_rows),
                "best_rel_sol": best_sweep,
                "exploratory_test_target_sensitivity_only": True,
                "csv": (
                    None
                    if not sweep_rows
                    else "metrics/weak_residual_parameter_sweep.csv"
                ),
            },
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "per_sample_csv": ("metrics/per_sample_weak_residual_blend_comparison.csv"),
            "raw_archive": (
                "data/selected_weak_residual_blend_arrays.npz"
                if self.request.save_generated_data
                else None
            ),
            "figure_paths": figure_paths,
            "figure_count": len(figure_paths),
            "reference_target_policy": (
                "sol is used only after all weights are fixed, for evaluation metrics"
            ),
            "transition_metrics_available": any(
                math.isfinite(float(row["equal_mean_transition_error_rms"]))
                for row in rows
            ),
        }

    def _write_report(self, summary: dict[str, Any]) -> None:
        metrics = summary["aggregate_metrics"]
        lines = [
            "# Local Weak-Residual Reliability Blend Comparison",
            "",
            "## Scope",
            "",
            "This is a frozen-checkpoint post-hoc diagnostic. The candidate weights",
            "use only predicted directional solutions/sources, coefficients, and axial",
            "P1 geometry. Reference `sol` is evaluation-only.",
            "",
            "## Full-Test Comparison",
            "",
            "| Estimator | Mean rel_sol | Change vs equal | Wins vs equal |",
            "| --- | ---: | ---: | ---: |",
        ]
        labels = {
            "equal_mean": "Equal mean",
            "geometry_c2": "Geometry-only C2",
            "mismatch_seam_c2": "Mismatch-detected seam C2",
            "weak_residual_reliability": "Local weak-residual reliability",
        }
        for name, label in labels.items():
            row = metrics[name]
            lines.append(
                f"| {label} | {100.0 * float(row['rel_sol_mean']):.6f}% | "
                f"{100.0 * float(row['rel_sol_relative_change_vs_equal']):+.3f}% | "
                f"{int(row['rel_sol_win_count_vs_equal'])}/{int(row['sample_count'])} |"
            )
        lines.extend(
            [
                "",
                "## Weak Reliability Contract",
                "",
                "For each candidate, the diagnostic assembles the full weak defect",
                "`R=Rx(u;phi)+Ry(u;psi)` from local P1 element contributions. It does",
                "not solve a global matrix equation. Smoothed mass-normalized residual",
                "indicators define a partition of unity, and equal residual evidence",
                "returns the equal mean.",
                "",
                "## Interpretation Boundary",
                "",
                "This local residual is a prediction-only reliability indicator, not a",
                "proved a posteriori bound for the learned axial reconstruction. The",
                "parameter sweep, when enabled, is same-test sensitivity analysis and",
                "must not be treated as independent model selection.",
            ]
        )
        (self.request.outdir / "diagnosis_report.md").write_text(
            "\n".join(lines) + "\n"
        )


def run_weak_residual_blend_comparison(
    request: WeakResidualBlendComparisonRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run the general frozen-checkpoint four-estimator comparison."""

    return WeakResidualBlendComparison(request, logger=logger).run()
