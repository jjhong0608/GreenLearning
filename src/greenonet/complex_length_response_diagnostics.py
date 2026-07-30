from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from greenonet.coefficients import CoefficientFunctions, load_coefficient_functions
from greenonet.complex_coupling_artifacts import ComplexCouplingArtifactExporter
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import (
    ComplexCouplingEvaluator,
    ComplexPredictionBatch,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    CouplingArtifactConfigs,
    load_coupling_artifact_configs,
)
from greenonet.config import (
    BalanceProjectionConfig,
    ComplexPreProjectionFusionConfig,
)
from greenonet.greens import ExactGreenFunction, select_green_reference_policy
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class ComplexLengthResponseDiagnosticRequest:
    """Inputs for the complex line-length response diagnostic."""

    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    geometry: Path | None = None
    test_path: Path | None = None
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    selected_samples: tuple[int, ...] = (47,)
    include_rel_sol_quantiles: bool = True
    transition_coordinate: float | None = None
    transition_zone_radius: float | None = None
    cardinal_radius_grid_steps: int = 2
    equivalence_tolerance: float = 1.0e-10
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.cardinal_radius_grid_steps < 1:
            raise ValueError("cardinal_radius_grid_steps must be positive.")
        if self.equivalence_tolerance <= 0.0:
            raise ValueError("equivalence_tolerance must be positive.")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive.")
        if self.transition_coordinate is not None and self.transition_coordinate <= 0:
            raise ValueError("transition_coordinate must be positive when provided.")
        if self.transition_zone_radius is not None and self.transition_zone_radius <= 0:
            raise ValueError("transition_zone_radius must be positive when provided.")
        if any(index < 0 for index in self.selected_samples):
            raise ValueError("selected_samples must contain non-negative indices.")


@dataclass(frozen=True)
class ExactGreenReconstructionResult:
    """Exact-reference reconstruction in both equivalent coordinate forms."""

    u_phi_valid: torch.Tensor
    u_psi_valid: torch.Tensor
    u_phi_physical_integral_valid: torch.Tensor
    u_psi_physical_integral_valid: torch.Tensor
    equivalence_max_abs: float
    equivalence_max_relative: float
    reference_kinds: tuple[str, ...]

    @property
    def u_mean_valid(self) -> torch.Tensor:
        return 0.5 * (self.u_phi_valid + self.u_psi_valid)

    @property
    def u_physical_integral_mean_valid(self) -> torch.Tensor:
        return 0.5 * (
            self.u_phi_physical_integral_valid + self.u_psi_physical_integral_valid
        )


@dataclass(frozen=True)
class TransitionGeometryInfo:
    inner_radius: float
    horizontal_split_coordinate: float
    horizontal_one_segment_coordinate: float
    vertical_split_coordinate: float
    vertical_one_segment_coordinate: float
    cardinal_radius: float
    horizontal_length_jump_ratio: float
    horizontal_length_squared_jump_ratio: float
    vertical_length_jump_ratio: float
    vertical_length_squared_jump_ratio: float


@dataclass(frozen=True)
class DiagnosticTensors:
    physical_source_error: torch.Tensor
    response_source_error: torch.Tensor
    predicted_exact: ExactGreenReconstructionResult
    target_exact: ExactGreenReconstructionResult
    exact_source_response: torch.Tensor
    exact_solution_error: torch.Tensor
    learned_solution_error: torch.Tensor
    learned_minus_exact: torch.Tensor
    target_exact_closure: torch.Tensor
    decomposition_residual: torch.Tensor


class ExactGreenReconstructionMixin:
    """Diagnostic-only exact Green reconstruction on geometry segment nodes."""

    geometry: ComplexGeometryMetadata
    coeffs: CoefficientFunctions
    device: torch.device
    dtype: torch.dtype
    request: ComplexLengthResponseDiagnosticRequest

    @staticmethod
    def reconstruct_segment_with_kernel(
        *,
        source_physical: torch.Tensor,
        node_weight_unit: torch.Tensor,
        kernel_unit: torch.Tensor,
        length: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unit-form and physical-form integrals for one segment."""

        length_tensor = torch.as_tensor(
            length,
            dtype=source_physical.dtype,
            device=source_physical.device,
        )
        source_unit = source_physical * length_tensor.square()
        unit_solution = torch.matmul(
            source_unit * node_weight_unit.unsqueeze(0),
            kernel_unit.T,
        )
        kernel_physical = length_tensor * kernel_unit
        weight_physical = length_tensor * node_weight_unit
        physical_solution = torch.matmul(
            source_physical * weight_physical.unsqueeze(0),
            kernel_physical.T,
        )
        return unit_solution, physical_solution

    def _reconstruct_exact(
        self,
        source_physical: torch.Tensor,
    ) -> ExactGreenReconstructionResult:
        if source_physical.dim() != 3 or source_physical.shape[1] != 2:
            raise ValueError("source_physical must have shape (B, 2, P).")
        if source_physical.shape[-1] != self.geometry.num_points:
            raise ValueError("source_physical point count does not match geometry.")

        outputs: list[torch.Tensor] = []
        physical_outputs: list[torch.Tensor] = []
        references: set[str] = set()
        max_abs = 0.0
        max_relative = 0.0
        axes: tuple[Literal["x", "y"], ...] = ("x", "y")
        for axis_index, axis in enumerate(axes):
            output, output_physical, axis_abs, axis_relative, axis_references = (
                self._reconstruct_exact_axis(
                    source_valid=source_physical[:, axis_index],
                    axis=axis,
                )
            )
            outputs.append(output)
            physical_outputs.append(output_physical)
            max_abs = max(max_abs, axis_abs)
            max_relative = max(max_relative, axis_relative)
            references.update(axis_references)

        return ExactGreenReconstructionResult(
            u_phi_valid=outputs[0],
            u_psi_valid=outputs[1],
            u_phi_physical_integral_valid=physical_outputs[0],
            u_psi_physical_integral_valid=physical_outputs[1],
            equivalence_max_abs=max_abs,
            equivalence_max_relative=max_relative,
            reference_kinds=tuple(sorted(references)),
        )

    def _reconstruct_exact_axis(
        self,
        *,
        source_valid: torch.Tensor,
        axis: Literal["x", "y"],
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, set[str]]:
        geometry = self.geometry
        if axis == "x":
            ptr = geometry.x_recon_ptr
            node_t_all = geometry.x_recon_t
            node_weight_all = geometry.x_recon_weight
            valid_index_all = geometry.x_recon_valid_index
            lengths = geometry.x_segment_length
        else:
            ptr = geometry.y_recon_ptr
            node_t_all = geometry.y_recon_t
            node_weight_all = geometry.y_recon_weight
            valid_index_all = geometry.y_recon_valid_index
            lengths = geometry.y_segment_length

        output = torch.zeros_like(source_valid)
        physical_output = torch.zeros_like(source_valid)
        max_abs = 0.0
        max_relative = 0.0
        references: set[str] = set()
        for segment_index in range(int(ptr.numel()) - 1):
            start = int(ptr[segment_index].item())
            end = int(ptr[segment_index + 1].item())
            node_t = node_t_all[start:end].to(
                device=self.device,
                dtype=self.dtype,
            )
            node_weight = node_weight_all[start:end].to(
                device=self.device,
                dtype=self.dtype,
            )
            valid_index = valid_index_all[start:end].to(self.device)
            node_source = torch.zeros(
                (source_valid.shape[0], end - start),
                device=self.device,
                dtype=self.dtype,
            )
            interior = valid_index >= 0
            if torch.any(interior):
                node_source[:, interior] = source_valid[:, valid_index[interior]]

            kernel, reference = self._exact_kernel(
                axis=axis,
                segment_index=segment_index,
                node_t=node_t,
            )
            references.add(reference)
            unit_solution, physical_solution = self.reconstruct_segment_with_kernel(
                source_physical=node_source,
                node_weight_unit=node_weight,
                kernel_unit=kernel,
                length=lengths[segment_index],
            )
            difference = (unit_solution - physical_solution).abs()
            segment_abs = float(difference.max().item())
            denominator = max(
                float(unit_solution.abs().max().item()),
                self.request.eps,
            )
            max_abs = max(max_abs, segment_abs)
            max_relative = max(max_relative, segment_abs / denominator)
            if torch.any(interior):
                output[:, valid_index[interior]] = unit_solution[:, interior]
                physical_output[:, valid_index[interior]] = physical_solution[
                    :, interior
                ]
        return output, physical_output, max_abs, max_relative, references

    def _exact_kernel(
        self,
        *,
        axis: Literal["x", "y"],
        segment_index: int,
        node_t: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        geometry = self.geometry
        if axis == "x":
            left = geometry.x_segment_left[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            length = geometry.x_segment_length[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            fixed = geometry.x_segment_y[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            x = left + length * node_t
            y = torch.full_like(x, float(fixed.item()))
            b_phys = self.coeffs.bx_fun(x, y)
        else:
            bottom = geometry.y_segment_bottom[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            length = geometry.y_segment_length[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            fixed = geometry.y_segment_x[segment_index].to(
                device=self.device,
                dtype=self.dtype,
            )
            y = bottom + length * node_t
            x = torch.full_like(y, float(fixed.item()))
            b_phys = self.coeffs.by_fun(x, y)

        a_unit = self.coeffs.a_fun(x, y).to(device=self.device, dtype=self.dtype)
        b_unit = length.to(device=self.device, dtype=self.dtype) * b_phys.to(
            device=self.device,
            dtype=self.dtype,
        )
        c_unit = length.to(device=self.device, dtype=self.dtype).square() * (
            self.coeffs.c_fun(x, y).to(device=self.device, dtype=self.dtype)
        )
        if torch.any(a_unit <= 0.0):
            raise ValueError("Exact Green diagnostic requires positive diffusion a.")
        policy = select_green_reference_policy(b_unit, c_unit)
        if not policy.valid or policy.reference is None:
            raise ValueError(
                "Exact Green diagnostic supports only reaction-free diffusion or "
                f"convection-diffusion coefficients. {policy.skip_reason}"
            )
        green = ExactGreenFunction(node_t, a=a_unit)
        if policy.reference == "diffusion":
            return green().to(device=self.device, dtype=self.dtype), "diffusion"
        return (
            green.convection_diffusion(b_unit).to(
                device=self.device,
                dtype=self.dtype,
            ),
            "convection_diffusion",
        )


class LengthResponseMetricsMixin:
    geometry: ComplexGeometryMetadata
    request: ComplexLengthResponseDiagnosticRequest

    @staticmethod
    def _rms(values: torch.Tensor, eps: float = 0.0) -> float:
        if values.numel() == 0:
            return math.nan
        result = torch.mean(values.square()).sqrt()
        if eps > 0.0 and float(result.item()) < eps:
            return 0.0
        return float(result.item())

    def _relative_l2(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        numerator = torch.linalg.vector_norm(prediction - target)
        denominator = torch.linalg.vector_norm(target).clamp_min(self.request.eps)
        return float((numerator / denominator).item())

    def _segment_weighted_rms(
        self,
        values_valid: torch.Tensor,
        *,
        axis: Literal["x", "y"],
        segment_index: int,
    ) -> float:
        if axis == "x":
            ptr = self.geometry.x_recon_ptr
            weights = self.geometry.x_recon_weight
            valid_index = self.geometry.x_recon_valid_index
        else:
            ptr = self.geometry.y_recon_ptr
            weights = self.geometry.y_recon_weight
            valid_index = self.geometry.y_recon_valid_index
        start = int(ptr[segment_index].item())
        end = int(ptr[segment_index + 1].item())
        segment_index_values = valid_index[start:end].to(values_valid.device)
        segment_weights = weights[start:end].to(
            device=values_valid.device,
            dtype=values_valid.dtype,
        )
        node_values = torch.zeros_like(segment_weights)
        interior = segment_index_values >= 0
        if torch.any(interior):
            node_values[interior] = values_valid[segment_index_values[interior]]
        denominator = segment_weights.sum().clamp_min(self.request.eps)
        return float(
            torch.sqrt(torch.sum(segment_weights * node_values.square()) / denominator)
            .detach()
            .cpu()
            .item()
        )

    def _sample_metric_rows(
        self,
        prediction: ComplexPredictionBatch,
        diagnostics: DiagnosticTensors,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        learned = torch.stack(
            (
                prediction.reconstruction.u_phi_valid,
                prediction.reconstruction.u_psi_valid,
            ),
            dim=1,
        )
        exact = torch.stack(
            (
                diagnostics.predicted_exact.u_phi_valid,
                diagnostics.predicted_exact.u_psi_valid,
            ),
            dim=1,
        )
        target_exact = torch.stack(
            (
                diagnostics.target_exact.u_phi_valid,
                diagnostics.target_exact.u_psi_valid,
            ),
            dim=1,
        )
        for offset, sample_index in enumerate(
            prediction.batch.sample_indices.detach().cpu().tolist()
        ):
            sol = prediction.batch.sol_valid[offset]
            learned_mean = learned[offset].mean(dim=0)
            exact_mean = exact[offset].mean(dim=0)
            target_exact_mean = target_exact[offset].mean(dim=0)
            source_response_mean = diagnostics.exact_source_response[offset].mean(dim=0)
            green_contribution_mean = diagnostics.learned_minus_exact[offset].mean(
                dim=0
            )
            closure_mean = diagnostics.target_exact_closure[offset].mean(dim=0)
            decomposition = diagnostics.decomposition_residual[offset].mean(dim=0)
            row: dict[str, float | int | str] = {
                "sample_id": int(sample_index),
                "file_stem": prediction.batch.file_stems[offset],
                "rel_sol": self._relative_l2(learned_mean, sol),
                "rel_sol_exact_predicted_source": self._relative_l2(exact_mean, sol),
                "rel_sol_target_source_exact_green": self._relative_l2(
                    target_exact_mean,
                    sol,
                ),
                "physical_source_error_phi_rms": self._rms(
                    diagnostics.physical_source_error[offset, 0]
                ),
                "physical_source_error_psi_rms": self._rms(
                    diagnostics.physical_source_error[offset, 1]
                ),
                "response_source_error_phi_rms": self._rms(
                    diagnostics.response_source_error[offset, 0]
                ),
                "response_source_error_psi_rms": self._rms(
                    diagnostics.response_source_error[offset, 1]
                ),
                "exact_source_response_mean_rms": self._rms(source_response_mean),
                "learned_solution_error_mean_rms": self._rms(learned_mean - sol),
                "exact_solution_error_mean_rms": self._rms(exact_mean - sol),
                "learned_minus_exact_mean_rms": self._rms(green_contribution_mean),
                "target_exact_closure_mean_rms": self._rms(closure_mean),
                "decomposition_residual_mean_max_abs": float(
                    decomposition.abs().max().item()
                ),
                "unit_physical_equivalence_max_abs": max(
                    diagnostics.predicted_exact.equivalence_max_abs,
                    diagnostics.target_exact.equivalence_max_abs,
                ),
                "unit_physical_equivalence_max_relative": max(
                    diagnostics.predicted_exact.equivalence_max_relative,
                    diagnostics.target_exact.equivalence_max_relative,
                ),
            }
            if bool(prediction.batch.has_flux[offset].item()):
                row["rel_flux"] = self._relative_l2(
                    prediction.projection.projected_physical[offset],
                    prediction.batch.flux_valid[offset],
                )
            for axis_index, axis_name in enumerate(("phi", "psi")):
                source_rms = self._rms(
                    diagnostics.physical_source_error[offset, axis_index]
                )
                response_rms = self._rms(
                    diagnostics.exact_source_response[offset, axis_index]
                )
                row[f"exact_response_gain_{axis_name}"] = response_rms / max(
                    source_rms,
                    self.request.eps,
                )
            rows.append(row)
        return rows

    def _segment_metric_rows(
        self,
        prediction: ComplexPredictionBatch,
        diagnostics: DiagnosticTensors,
    ) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for sample_offset, sample_index in enumerate(
            prediction.batch.sample_indices.detach().cpu().tolist()
        ):
            sol = prediction.batch.sol_valid[sample_offset]
            axes: tuple[Literal["x", "y"], ...] = ("x", "y")
            for axis_index, axis in enumerate(axes):
                if axis == "x":
                    segment_count = self.geometry.num_x_segments
                    lengths = self.geometry.x_segment_length
                    fixed = self.geometry.x_segment_y
                    valid_segment_ids = self.geometry.x_segment_id
                    learned = prediction.reconstruction.u_phi_valid[sample_offset]
                    exact = diagnostics.predicted_exact.u_phi_valid[sample_offset]
                    target_exact = diagnostics.target_exact.u_phi_valid[sample_offset]
                else:
                    segment_count = self.geometry.num_y_segments
                    lengths = self.geometry.y_segment_length
                    fixed = self.geometry.y_segment_x
                    valid_segment_ids = self.geometry.y_segment_id
                    learned = prediction.reconstruction.u_psi_valid[sample_offset]
                    exact = diagnostics.predicted_exact.u_psi_valid[sample_offset]
                    target_exact = diagnostics.target_exact.u_psi_valid[sample_offset]

                for segment_index in range(segment_count):
                    length = float(lengths[segment_index].item())
                    length_squared = length * length
                    physical_source_rms = self._segment_weighted_rms(
                        diagnostics.physical_source_error[
                            sample_offset,
                            axis_index,
                        ],
                        axis=axis,
                        segment_index=segment_index,
                    )
                    response_source_rms = self._segment_weighted_rms(
                        diagnostics.response_source_error[sample_offset, axis_index],
                        axis=axis,
                        segment_index=segment_index,
                    )
                    exact_response_rms = self._segment_weighted_rms(
                        exact - target_exact,
                        axis=axis,
                        segment_index=segment_index,
                    )
                    exact_gain = exact_response_rms / max(
                        physical_source_rms,
                        self.request.eps,
                    )
                    rows.append(
                        {
                            "sample_id": int(sample_index),
                            "file_stem": prediction.batch.file_stems[sample_offset],
                            "axis": axis,
                            "segment_index": segment_index,
                            "fixed_coordinate": float(fixed[segment_index].item()),
                            "length": length,
                            "length_squared": length_squared,
                            "point_count": int(
                                torch.sum(valid_segment_ids == segment_index).item()
                            ),
                            "physical_source_error_rms": physical_source_rms,
                            "response_source_error_rms": response_source_rms,
                            "exact_source_response_rms": exact_response_rms,
                            "exact_solution_error_rms": self._segment_weighted_rms(
                                exact - sol,
                                axis=axis,
                                segment_index=segment_index,
                            ),
                            "learned_solution_error_rms": self._segment_weighted_rms(
                                learned - sol,
                                axis=axis,
                                segment_index=segment_index,
                            ),
                            "learned_minus_exact_rms": self._segment_weighted_rms(
                                learned - exact,
                                axis=axis,
                                segment_index=segment_index,
                            ),
                            "target_exact_closure_rms": self._segment_weighted_rms(
                                target_exact - sol,
                                axis=axis,
                                segment_index=segment_index,
                            ),
                            "response_gain": exact_gain,
                            "response_gain_divided_by_length_squared": (
                                exact_gain / max(length_squared, self.request.eps)
                            ),
                        }
                    )
        return rows


class TransitionDiagnosticMixin:
    geometry: ComplexGeometryMetadata
    geometry_path: Path
    request: ComplexLengthResponseDiagnosticRequest
    device: torch.device

    @staticmethod
    def _rms(values: torch.Tensor, eps: float = 0.0) -> float:
        raise NotImplementedError

    def _infer_transition_geometry(self) -> TransitionGeometryInfo:
        with np.load(self.geometry_path, allow_pickle=False) as raw:
            if self.request.transition_coordinate is not None:
                inner_radius = float(self.request.transition_coordinate)
            elif "inner_radius" in raw.files:
                inner_radius = float(np.asarray(raw["inner_radius"]).reshape(()))
            else:
                raise ValueError(
                    "Transition coordinate could not be inferred. Provide "
                    "--transition-coordinate or annulus inner_radius metadata."
                )
            step_size = (
                float(np.asarray(raw["step_size"]).reshape(()))
                if "step_size" in raw.files
                else max(float(self.geometry.hx.item()), float(self.geometry.hy.item()))
            )

        horizontal_split, horizontal_outer = self._transition_coordinates(
            self.geometry.x_segment_y,
            inner_radius,
        )
        vertical_split, vertical_outer = self._transition_coordinates(
            self.geometry.y_segment_x,
            inner_radius,
        )
        horizontal_ratio = self._line_length_jump_ratio(
            fixed=self.geometry.x_segment_y,
            lengths=self.geometry.x_segment_length,
            split_coordinate=horizontal_split,
            outer_coordinate=horizontal_outer,
        )
        vertical_ratio = self._line_length_jump_ratio(
            fixed=self.geometry.y_segment_x,
            lengths=self.geometry.y_segment_length,
            split_coordinate=vertical_split,
            outer_coordinate=vertical_outer,
        )
        cardinal_radius = self.request.transition_zone_radius or (
            self.request.cardinal_radius_grid_steps * step_size
        )
        return TransitionGeometryInfo(
            inner_radius=inner_radius,
            horizontal_split_coordinate=horizontal_split,
            horizontal_one_segment_coordinate=horizontal_outer,
            vertical_split_coordinate=vertical_split,
            vertical_one_segment_coordinate=vertical_outer,
            cardinal_radius=cardinal_radius,
            horizontal_length_jump_ratio=horizontal_ratio,
            horizontal_length_squared_jump_ratio=horizontal_ratio * horizontal_ratio,
            vertical_length_jump_ratio=vertical_ratio,
            vertical_length_squared_jump_ratio=vertical_ratio * vertical_ratio,
        )

    @staticmethod
    def _transition_coordinates(
        fixed_coordinates: torch.Tensor,
        inner_radius: float,
    ) -> tuple[float, float]:
        values = np.abs(fixed_coordinates.detach().cpu().numpy())
        unique, counts = np.unique(np.round(values, decimals=14), return_counts=True)
        split_candidates = unique[(unique < inner_radius) & (counts > 1)]
        outer_candidates = unique[unique > inner_radius]
        if split_candidates.size == 0 or outer_candidates.size == 0:
            raise ValueError(
                "Could not infer split and one-segment axial lines around the "
                f"transition coordinate {inner_radius}."
            )
        return float(split_candidates.max()), float(outer_candidates.min())

    @staticmethod
    def _line_length_jump_ratio(
        *,
        fixed: torch.Tensor,
        lengths: torch.Tensor,
        split_coordinate: float,
        outer_coordinate: float,
    ) -> float:
        tolerance = 1.0e-10
        split_mask = torch.isclose(
            fixed.abs(),
            fixed.new_tensor(split_coordinate),
            atol=tolerance,
            rtol=0.0,
        )
        outer_mask = torch.isclose(
            fixed.abs(),
            fixed.new_tensor(outer_coordinate),
            atol=tolerance,
            rtol=0.0,
        )
        split_mean = float(lengths[split_mask].mean().item())
        outer_mean = float(lengths[outer_mask].mean().item())
        if split_mean <= 0.0:
            raise ValueError("Transition split-line segment length must be positive.")
        return outer_mean / split_mean

    def _zone_masks(
        self,
        transition: TransitionGeometryInfo,
    ) -> dict[str, torch.Tensor]:
        coords = self.geometry.coords_valid
        x = coords[:, 0]
        y = coords[:, 1]
        line_tolerance = 0.25 * min(
            float(self.geometry.hx.item()),
            float(self.geometry.hy.item()),
        )
        masks = {
            "global": torch.ones_like(x, dtype=torch.bool),
            "horizontal_split_lines": (
                (y.abs() - transition.horizontal_split_coordinate).abs()
                <= line_tolerance
            ),
            "horizontal_one_segment_lines": (
                (y.abs() - transition.horizontal_one_segment_coordinate).abs()
                <= line_tolerance
            ),
            "vertical_split_lines": (
                (x.abs() - transition.vertical_split_coordinate).abs() <= line_tolerance
            ),
            "vertical_one_segment_lines": (
                (x.abs() - transition.vertical_one_segment_coordinate).abs()
                <= line_tolerance
            ),
        }
        centers = {
            "cardinal_east": (transition.inner_radius, 0.0),
            "cardinal_west": (-transition.inner_radius, 0.0),
            "cardinal_north": (0.0, transition.inner_radius),
            "cardinal_south": (0.0, -transition.inner_radius),
        }
        for name, (center_x, center_y) in centers.items():
            masks[name] = (x - center_x).square() + (
                y - center_y
            ).square() <= transition.cardinal_radius**2
        return masks

    def _transition_zone_rows(
        self,
        prediction: ComplexPredictionBatch,
        diagnostics: DiagnosticTensors,
        transition: TransitionGeometryInfo,
    ) -> list[dict[str, float | int | str]]:
        masks = self._zone_masks(transition)
        learned_mean_error = diagnostics.learned_solution_error.mean(dim=1)
        exact_mean_error = diagnostics.exact_solution_error.mean(dim=1)
        exact_source_response_mean = diagnostics.exact_source_response.mean(dim=1)
        learned_minus_exact_mean = diagnostics.learned_minus_exact.mean(dim=1)
        target_closure_mean = diagnostics.target_exact_closure.mean(dim=1)
        rows: list[dict[str, float | int | str]] = []
        for sample_offset, sample_index in enumerate(
            prediction.batch.sample_indices.detach().cpu().tolist()
        ):
            for zone_name, raw_mask in masks.items():
                mask = raw_mask.to(self.device)
                rows.append(
                    {
                        "sample_id": int(sample_index),
                        "file_stem": prediction.batch.file_stems[sample_offset],
                        "zone": zone_name,
                        "point_count": int(mask.sum().item()),
                        "physical_source_error_phi_rms": self._rms(
                            diagnostics.physical_source_error[
                                sample_offset,
                                0,
                                mask,
                            ]
                        ),
                        "physical_source_error_psi_rms": self._rms(
                            diagnostics.physical_source_error[
                                sample_offset,
                                1,
                                mask,
                            ]
                        ),
                        "response_source_error_phi_rms": self._rms(
                            diagnostics.response_source_error[sample_offset, 0, mask]
                        ),
                        "response_source_error_psi_rms": self._rms(
                            diagnostics.response_source_error[sample_offset, 1, mask]
                        ),
                        "exact_source_response_mean_rms": self._rms(
                            exact_source_response_mean[sample_offset, mask]
                        ),
                        "exact_solution_error_mean_rms": self._rms(
                            exact_mean_error[sample_offset, mask]
                        ),
                        "learned_solution_error_mean_rms": self._rms(
                            learned_mean_error[sample_offset, mask]
                        ),
                        "learned_minus_exact_mean_rms": self._rms(
                            learned_minus_exact_mean[sample_offset, mask]
                        ),
                        "target_exact_closure_mean_rms": self._rms(
                            target_closure_mean[sample_offset, mask]
                        ),
                    }
                )
        return rows


class LengthResponsePlotMixin:
    request: ComplexLengthResponseDiagnosticRequest
    logger: logging.Logger | None

    _SIGNED_FIELDS = {
        "physical_source_error_phi",
        "physical_source_error_psi",
        "response_source_error_phi",
        "response_source_error_psi",
        "learned_u_pred_error",
        "exact_u_pred_error",
        "exact_source_response_mean",
        "learned_minus_exact_mean",
        "target_exact_closure_mean",
        "decomposition_residual_mean",
        "raw_difference",
        "projected_difference",
        "raw_response_constraint_residual",
        "response_constraint_residual",
        "fusion_base_difference",
        "fusion_residual_physical",
        "fusion_fused_difference",
    }

    def _write_selected_figures(
        self,
        selected_arrays: dict[int, dict[str, np.ndarray]],
        file_stems: dict[int, str],
    ) -> list[str]:
        figure_paths: list[str] = []
        groups = {
            "source_stages": (
                "physical_source_error_phi",
                "physical_source_error_psi",
                "response_source_error_phi",
                "response_source_error_psi",
                "log10_response_gain_phi",
                "log10_response_gain_psi",
            ),
            "reconstruction_decomposition": (
                "learned_u_pred_error",
                "exact_u_pred_error",
                "exact_source_response_mean",
                "learned_minus_exact_mean",
                "target_exact_closure_mean",
                "decomposition_residual_mean",
            ),
            "geometry_and_projection": (
                "x_length_squared",
                "y_length_squared",
                "raw_difference",
                "projected_difference",
                "raw_response_constraint_residual",
                "response_constraint_residual",
            ),
            "pre_projection_fusion": (
                "fusion_base_difference",
                "fusion_residual_physical",
                "fusion_fused_difference",
            ),
        }
        for sample_id, arrays in selected_arrays.items():
            coords = arrays["coords_valid"]
            stem = f"sample_{sample_id:04d}_{file_stems[sample_id]}"
            for group_name, fields in groups.items():
                present = tuple(field for field in fields if field in arrays)
                if not present:
                    continue
                figure = self._multi_panel_scatter(
                    title=f"{stem} {group_name.replace('_', ' ')}",
                    coords=coords,
                    arrays=arrays,
                    fields=present,
                )
                base = self.request.outdir / "figures" / stem / group_name
                save_plotly_figure(figure, base, logger=self.logger)
                figure_paths.append(str(base.with_suffix(".json")))
        return figure_paths

    def _multi_panel_scatter(
        self,
        *,
        title: str,
        coords: np.ndarray,
        arrays: dict[str, np.ndarray],
        fields: tuple[str, ...],
    ) -> go.Figure:
        columns = 2
        rows = math.ceil(len(fields) / columns)
        subplot_titles = []
        for field in fields:
            values = np.asarray(arrays[field])
            subplot_titles.append(
                f"{field}<br>min={np.nanmin(values):.3e}, max={np.nanmax(values):.3e}"
            )
        figure = make_subplots(
            rows=rows,
            cols=columns,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.10,
        )
        for index, field in enumerate(fields):
            row = index // columns + 1
            column = index % columns + 1
            values = np.asarray(arrays[field])
            signed = field in self._SIGNED_FIELDS
            marker: dict[str, Any] = {
                "color": values,
                "colorscale": "RdBu" if signed else "Viridis",
                "showscale": False,
                "size": 3,
            }
            finite = values[np.isfinite(values)]
            if signed and finite.size:
                maximum = float(np.max(np.abs(finite)))
                if maximum > 0.0:
                    marker.update(cmin=-maximum, cmax=maximum)
            figure.add_trace(
                go.Scattergl(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode="markers",
                    marker=marker,
                    customdata=values,
                    hovertemplate=(
                        "x=%{x:.6g}<br>y=%{y:.6g}<br>value=%{customdata:.6e}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
            figure.update_xaxes(title_text="x", row=row, col=column)
            figure.update_yaxes(
                title_text="y",
                scaleanchor=f"x{index + 1}" if index > 0 else "x",
                scaleratio=1,
                row=row,
                col=column,
            )
        figure.update_layout(
            template=self.request.theme,
            title=title,
            width=1200,
            height=max(650, rows * 520),
        )
        return figure

    def _write_segment_response_figure(
        self,
        segment_rows: list[dict[str, float | int | str]],
    ) -> str:
        figure = go.Figure()
        for axis, symbol in (("x", "circle"), ("y", "diamond")):
            rows = [row for row in segment_rows if row["axis"] == axis]
            figure.add_trace(
                go.Scattergl(
                    x=[float(row["length_squared"]) for row in rows],
                    y=[float(row["response_gain"]) for row in rows],
                    mode="markers",
                    name=axis,
                    marker={
                        "symbol": symbol,
                        "size": 5,
                        "color": [int(row["sample_id"]) for row in rows],
                        "colorscale": "Viridis",
                        "showscale": axis == "y",
                    },
                    customdata=[
                        [
                            int(row["sample_id"]),
                            int(row["segment_index"]),
                            float(row["fixed_coordinate"]),
                            float(row["response_gain_divided_by_length_squared"]),
                        ]
                        for row in rows
                    ],
                    hovertemplate=(
                        "sample=%{customdata[0]}<br>segment=%{customdata[1]}"
                        "<br>fixed=%{customdata[2]:.6g}<br>L^2=%{x:.6e}"
                        "<br>gain=%{y:.6e}<br>gain/L^2=%{customdata[3]:.6e}"
                        "<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            template=self.request.theme,
            title="Exact source-error response gain versus segment length squared",
            width=1000,
            height=750,
            xaxis_title="segment length squared",
            yaxis_title="exact response gain",
        )
        base = self.request.outdir / "figures" / "segment_response_gain"
        save_plotly_figure(figure, base, logger=self.logger)
        return str(base.with_suffix(".json"))


class ComplexLengthResponseDiagnostic(
    ExactGreenReconstructionMixin,
    LengthResponseMetricsMixin,
    TransitionDiagnosticMixin,
    LengthResponsePlotMixin,
):
    """Run a checkpoint-backed audit of complex line-length amplification."""

    def __init__(
        self,
        request: ComplexLengthResponseDiagnosticRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.coupling_model: ComplexCouplingNet | None = None
        self.request.outdir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        configs = load_coupling_artifact_configs(self.request.config)
        self._validate_configs(configs)
        geometry_path = self.request.geometry or configs.dataset.geometry_path
        if geometry_path is None:
            raise ValueError("A complex geometry path is required.")
        self.geometry_path = geometry_path
        test_path = self.request.test_path or configs.dataset.test_path
        if test_path is None:
            raise ValueError("A complex test dataset path is required.")
        coefficient_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        self.device = torch.device(
            self.request.device or configs.coupling_training.device
        )
        self.dtype = configs.dataset.dtype
        self.geometry = load_complex_geometry(
            self.geometry_path,
            dtype=self.dtype,
        )
        self.coeffs = load_coefficient_functions(coefficient_path)
        dataset = ComplexCouplingDataset(
            test_path,
            self.geometry,
            self.coeffs,
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=self.dtype,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        self._validate_selected_samples(len(dataset))
        prediction = self._evaluate_dataset(dataset, configs)
        if not bool(torch.all(prediction.batch.has_flux).item()):
            missing = (
                prediction.batch.sample_indices[~prediction.batch.has_flux]
                .detach()
                .cpu()
                .tolist()
            )
            raise ValueError(
                "Length-response diagnostics require phi/psi flux targets for "
                f"every sample; missing targets for sample indices {missing}."
            )

        diagnostics = self._build_diagnostic_tensors(prediction)
        self._validate_equivalence(diagnostics)
        sample_rows = self._sample_metric_rows(prediction, diagnostics)
        segment_rows = self._segment_metric_rows(prediction, diagnostics)
        transition = self._infer_transition_geometry()
        zone_rows = self._transition_zone_rows(prediction, diagnostics, transition)
        selected, roles = self._select_samples(sample_rows)
        selected_arrays = self._selected_arrays(
            prediction,
            diagnostics,
            selected,
        )
        file_stems = {
            int(index): stem
            for index, stem in zip(
                prediction.batch.sample_indices.detach().cpu().tolist(),
                prediction.batch.file_stems,
                strict=True,
            )
        }

        self._write_csv(
            self.request.outdir / "metrics" / "per_sample_length_response.csv",
            sample_rows,
        )
        self._write_csv(
            self.request.outdir / "metrics" / "per_segment_length_response.csv",
            segment_rows,
        )
        self._write_csv(
            self.request.outdir / "metrics" / "transition_zone_metrics.csv",
            zone_rows,
        )
        self._write_selected_npz(selected_arrays, file_stems)
        figure_paths = self._write_selected_figures(selected_arrays, file_stems)
        figure_paths.append(self._write_segment_response_figure(segment_rows))

        summary = self._build_summary(
            configs=configs,
            dataset=dataset,
            coefficient_path=coefficient_path,
            sample_rows=sample_rows,
            transition=transition,
            selected=selected,
            roles=roles,
            diagnostics=diagnostics,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        if self.logger is not None:
            self.logger.info(
                "Completed length-response diagnostic for %d samples; selected=%s",
                len(dataset),
                selected,
            )
        return summary

    @staticmethod
    def _validate_configs(configs: CouplingArtifactConfigs) -> None:
        if configs.dataset.geometry_mode != "complex":
            raise ValueError("Length-response diagnostic requires complex geometry.")
        projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        if projection.mode != "physical_symmetric":
            raise ValueError(
                "Length-response diagnostic requires output-contract-v6 "
                "physical_symmetric projection."
            )

    def _validate_selected_samples(self, dataset_size: int) -> None:
        invalid = [
            index for index in self.request.selected_samples if index >= dataset_size
        ]
        if invalid:
            raise IndexError(
                f"Selected sample indices are outside dataset size {dataset_size}: "
                f"{invalid}."
            )

    def _evaluate_dataset(
        self,
        dataset: ComplexCouplingDataset,
        configs: CouplingArtifactConfigs,
    ) -> ComplexPredictionBatch:
        loader_request = CouplingArtifactRequest(
            config=self.request.config,
            coupling_checkpoint=self.request.coupling_checkpoint,
            green_checkpoint=self.request.green_checkpoint,
            outdir=self.request.outdir,
            coefficients=self.request.coefficients,
            device=str(self.device),
            theme=self.request.theme,
            selected_samples=self.request.selected_samples,
        )
        model_loader = ComplexCouplingArtifactExporter(
            loader_request,
            logger=self.logger,
        )
        coupling_model = model_loader._load_complex_model(configs, self.device)
        self.coupling_model = coupling_model
        green_model = model_loader._load_green_model(configs, self.device)
        evaluator = ComplexCouplingEvaluator(
            model=coupling_model,
            green_model=green_model,
            config=configs.coupling_training,
            device=self.device,
            work_dir=self.request.outdir / "_evaluator",
        )
        batch = complex_coupling_collate_fn(
            [dataset[index] for index in range(len(dataset))]
        ).to(self.device)
        with torch.no_grad():
            return evaluator.predict_batch(batch)

    def _build_diagnostic_tensors(
        self,
        prediction: ComplexPredictionBatch,
    ) -> DiagnosticTensors:
        target_physical = prediction.batch.flux_valid
        physical_source_error = (
            prediction.projection.projected_physical - target_physical
        )
        x_length_squared = (
            self.geometry.x_lengths_for_valid_points()
            .to(device=self.device, dtype=self.dtype)
            .square()
        )
        y_length_squared = (
            self.geometry.y_lengths_for_valid_points()
            .to(device=self.device, dtype=self.dtype)
            .square()
        )
        target_unit = torch.stack(
            (
                target_physical[:, 0] * x_length_squared.unsqueeze(0),
                target_physical[:, 1] * y_length_squared.unsqueeze(0),
            ),
            dim=1,
        )
        response_source_error = prediction.projection.projected_response - target_unit
        predicted_exact = self._reconstruct_exact(
            prediction.projection.projected_physical
        )
        target_exact = self._reconstruct_exact(target_physical)
        predicted_exact_axes = torch.stack(
            (predicted_exact.u_phi_valid, predicted_exact.u_psi_valid),
            dim=1,
        )
        target_exact_axes = torch.stack(
            (target_exact.u_phi_valid, target_exact.u_psi_valid),
            dim=1,
        )
        learned_axes = torch.stack(
            (
                prediction.reconstruction.u_phi_valid,
                prediction.reconstruction.u_psi_valid,
            ),
            dim=1,
        )
        sol_axes = prediction.batch.sol_valid.unsqueeze(1).expand_as(learned_axes)
        exact_source_response = predicted_exact_axes - target_exact_axes
        exact_solution_error = predicted_exact_axes - sol_axes
        learned_solution_error = learned_axes - sol_axes
        learned_minus_exact = learned_axes - predicted_exact_axes
        target_exact_closure = target_exact_axes - sol_axes
        decomposition_residual = learned_solution_error - (
            exact_source_response + target_exact_closure + learned_minus_exact
        )
        return DiagnosticTensors(
            physical_source_error=physical_source_error,
            response_source_error=response_source_error,
            predicted_exact=predicted_exact,
            target_exact=target_exact,
            exact_source_response=exact_source_response,
            exact_solution_error=exact_solution_error,
            learned_solution_error=learned_solution_error,
            learned_minus_exact=learned_minus_exact,
            target_exact_closure=target_exact_closure,
            decomposition_residual=decomposition_residual,
        )

    def _validate_equivalence(self, diagnostics: DiagnosticTensors) -> None:
        max_abs = max(
            diagnostics.predicted_exact.equivalence_max_abs,
            diagnostics.target_exact.equivalence_max_abs,
        )
        max_relative = max(
            diagnostics.predicted_exact.equivalence_max_relative,
            diagnostics.target_exact.equivalence_max_relative,
        )
        tolerance = self.request.equivalence_tolerance
        if max_abs > tolerance or max_relative > tolerance:
            raise RuntimeError(
                "Unit/physical Green reconstruction equivalence failed: "
                f"max_abs={max_abs:.6e}, max_relative={max_relative:.6e}, "
                f"tolerance={tolerance:.6e}."
            )

    def _select_samples(
        self,
        sample_rows: list[dict[str, float | int | str]],
    ) -> tuple[tuple[int, ...], dict[str, int]]:
        selected = list(dict.fromkeys(self.request.selected_samples))
        roles: dict[str, int] = {}
        if self.request.include_rel_sol_quantiles:
            sorted_rows = sorted(sample_rows, key=lambda row: float(row["rel_sol"]))
            positions = {
                "min": 0,
                "q25": round(0.25 * (len(sorted_rows) - 1)),
                "q50": round(0.50 * (len(sorted_rows) - 1)),
                "q75": round(0.75 * (len(sorted_rows) - 1)),
                "max": len(sorted_rows) - 1,
            }
            for role, position in positions.items():
                sample_id = int(sorted_rows[position]["sample_id"])
                roles[role] = sample_id
                if sample_id not in selected:
                    selected.append(sample_id)
        return tuple(selected), roles

    def _selected_arrays(
        self,
        prediction: ComplexPredictionBatch,
        diagnostics: DiagnosticTensors,
        selected: Sequence[int],
    ) -> dict[int, dict[str, np.ndarray]]:
        sample_to_offset = {
            int(sample_id): offset
            for offset, sample_id in enumerate(
                prediction.batch.sample_indices.detach().cpu().tolist()
            )
        }
        coords = self.geometry.coords_valid.detach().cpu().numpy()
        x_length_squared = (
            self.geometry.x_lengths_for_valid_points().square().detach().cpu().numpy()
        )
        y_length_squared = (
            self.geometry.y_lengths_for_valid_points().square().detach().cpu().numpy()
        )
        selected_arrays: dict[int, dict[str, np.ndarray]] = {}
        for sample_id in selected:
            offset = sample_to_offset[sample_id]
            source_response_mean = diagnostics.exact_source_response[offset].mean(dim=0)
            learned_minus_exact_mean = diagnostics.learned_minus_exact[offset].mean(
                dim=0
            )
            target_closure_mean = diagnostics.target_exact_closure[offset].mean(dim=0)
            decomposition_mean = diagnostics.decomposition_residual[offset].mean(dim=0)
            physical_error = diagnostics.physical_source_error[offset]
            exact_response = diagnostics.exact_source_response[offset]
            arrays = {
                "coords_valid": coords,
                "rhs": self._numpy(prediction.batch.rhs_valid[offset]),
                "sol": self._numpy(prediction.batch.sol_valid[offset]),
                "target_phi": self._numpy(prediction.batch.flux_valid[offset, 0]),
                "target_psi": self._numpy(prediction.batch.flux_valid[offset, 1]),
                "raw_response_phi": self._numpy(prediction.raw_response[offset, 0]),
                "raw_response_psi": self._numpy(prediction.raw_response[offset, 1]),
                "raw_physical_phi": self._numpy(
                    prediction.projection.raw_physical[offset, 0]
                ),
                "raw_physical_psi": self._numpy(
                    prediction.projection.raw_physical[offset, 1]
                ),
                "projected_response_phi": self._numpy(
                    prediction.projection.projected_response[offset, 0]
                ),
                "projected_response_psi": self._numpy(
                    prediction.projection.projected_response[offset, 1]
                ),
                "projected_physical_phi": self._numpy(
                    prediction.projection.projected_physical[offset, 0]
                ),
                "projected_physical_psi": self._numpy(
                    prediction.projection.projected_physical[offset, 1]
                ),
                "physical_source_error_phi": self._numpy(physical_error[0]),
                "physical_source_error_psi": self._numpy(physical_error[1]),
                "response_source_error_phi": self._numpy(
                    diagnostics.response_source_error[offset, 0]
                ),
                "response_source_error_psi": self._numpy(
                    diagnostics.response_source_error[offset, 1]
                ),
                "x_length_squared": x_length_squared,
                "y_length_squared": y_length_squared,
                "raw_difference": self._numpy(
                    prediction.projection.raw_difference[offset]
                ),
                "projected_difference": self._numpy(
                    prediction.projection.projected_difference[offset]
                ),
                "raw_response_constraint_residual": self._numpy(
                    prediction.projection.raw_response_constraint_residual[offset]
                ),
                "response_constraint_residual": self._numpy(
                    prediction.projection.response_constraint_residual[offset]
                ),
                "learned_u_phi": self._numpy(
                    prediction.reconstruction.u_phi_valid[offset]
                ),
                "learned_u_psi": self._numpy(
                    prediction.reconstruction.u_psi_valid[offset]
                ),
                "learned_u_pred": self._numpy(
                    prediction.reconstruction.u_mean_valid[offset]
                ),
                "exact_u_phi": self._numpy(
                    diagnostics.predicted_exact.u_phi_valid[offset]
                ),
                "exact_u_psi": self._numpy(
                    diagnostics.predicted_exact.u_psi_valid[offset]
                ),
                "exact_u_pred": self._numpy(
                    diagnostics.predicted_exact.u_mean_valid[offset]
                ),
                "target_exact_u_phi": self._numpy(
                    diagnostics.target_exact.u_phi_valid[offset]
                ),
                "target_exact_u_psi": self._numpy(
                    diagnostics.target_exact.u_psi_valid[offset]
                ),
                "target_exact_u_pred": self._numpy(
                    diagnostics.target_exact.u_mean_valid[offset]
                ),
                "learned_u_pred_error": self._numpy(
                    diagnostics.learned_solution_error[offset].mean(dim=0)
                ),
                "exact_u_pred_error": self._numpy(
                    diagnostics.exact_solution_error[offset].mean(dim=0)
                ),
                "exact_source_response_mean": self._numpy(source_response_mean),
                "learned_minus_exact_mean": self._numpy(learned_minus_exact_mean),
                "target_exact_closure_mean": self._numpy(target_closure_mean),
                "decomposition_residual_mean": self._numpy(decomposition_mean),
                "log10_response_gain_phi": self._numpy(
                    torch.log10(
                        exact_response[0].abs()
                        / (physical_error[0].abs() + self.request.eps)
                        + self.request.eps
                    )
                ),
                "log10_response_gain_psi": self._numpy(
                    torch.log10(
                        exact_response[1].abs()
                        / (physical_error[1].abs() + self.request.eps)
                        + self.request.eps
                    )
                ),
            }
            if prediction.pre_projection_fusion is not None:
                fusion = prediction.pre_projection_fusion
                arrays.update(
                    {
                        "base_response_phi": self._numpy(
                            fusion.base_response[offset, 0]
                        ),
                        "base_response_psi": self._numpy(
                            fusion.base_response[offset, 1]
                        ),
                        "fusion_base_physical_p": self._numpy(
                            fusion.base_physical[offset, 0]
                        ),
                        "fusion_base_physical_q": self._numpy(
                            fusion.base_physical[offset, 1]
                        ),
                        "fusion_base_difference": self._numpy(
                            fusion.base_difference[offset]
                        ),
                        "fusion_normalized_difference": self._numpy(
                            fusion.normalized_difference[offset]
                        ),
                        "fusion_normalized_rhs": self._numpy(
                            fusion.normalized_rhs[offset]
                        ),
                        "fusion_residual_normalized": self._numpy(
                            fusion.normalized_residual[offset]
                        ),
                        "fusion_residual_physical": self._numpy(
                            fusion.physical_residual[offset]
                        ),
                        "fusion_fused_difference": self._numpy(
                            fusion.fused_difference[offset]
                        ),
                        "fusion_pre_projection_phi": self._numpy(
                            fusion.fused_physical[offset, 0]
                        ),
                        "fusion_pre_projection_psi": self._numpy(
                            fusion.fused_physical[offset, 1]
                        ),
                        "fusion_source_scale": self._numpy(fusion.source_scale[offset]),
                        "fusion_safe_source_scale": self._numpy(
                            fusion.safe_source_scale[offset]
                        ),
                        "fusion_pre_projection_balance_residual": self._numpy(
                            fusion.pre_projection_balance_residual[offset]
                        ),
                    }
                )
            selected_arrays[sample_id] = arrays
        return selected_arrays

    @staticmethod
    def _numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy()

    @staticmethod
    def _write_csv(
        path: Path,
        rows: list[dict[str, float | int | str]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("")
            return
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _write_selected_npz(
        self,
        selected_arrays: dict[int, dict[str, np.ndarray]],
        file_stems: dict[int, str],
    ) -> None:
        payload: dict[str, np.ndarray] = {}
        for sample_id, arrays in selected_arrays.items():
            prefix = f"sample_{sample_id:04d}_{file_stems[sample_id]}"
            for field, values in arrays.items():
                payload[f"{prefix}_{field}"] = values
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_dir / "selected_diagnostic_arrays.npz",
            **payload,  # type: ignore[arg-type]
        )

    def _build_summary(
        self,
        *,
        configs: CouplingArtifactConfigs,
        dataset: ComplexCouplingDataset,
        coefficient_path: Path | None,
        sample_rows: list[dict[str, float | int | str]],
        transition: TransitionGeometryInfo,
        selected: tuple[int, ...],
        roles: dict[str, int],
        diagnostics: DiagnosticTensors,
        figure_paths: list[str],
    ) -> dict[str, Any]:
        rel_sol = np.asarray([float(row["rel_sol"]) for row in sample_rows])
        projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        fusion_config = ComplexPreProjectionFusionConfig.from_raw(
            configs.coupling_model.pre_projection_fusion
        )
        if self.coupling_model is None:
            raise RuntimeError("Coupling model must be loaded before summary export.")
        return {
            "diagnostic": "complex_length_response",
            "uses_reference_targets_for_training": False,
            "reference_fields_role": "evaluation_only",
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "geometry_path": str(self.geometry_path),
            "test_path": str(dataset.data_dir),
            "coefficients": None if coefficient_path is None else str(coefficient_path),
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "num_samples": len(dataset),
            "num_valid_points": self.geometry.num_points,
            "num_x_segments": self.geometry.num_x_segments,
            "num_y_segments": self.geometry.num_y_segments,
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "projection_mode": projection.mode,
            "raw_output_space": "reference_response",
            "output_contract_version": ComplexCouplingNet.OUTPUT_CONTRACT_VERSION,
            "reconstruction_response_input": {
                "phi": "projected Phi is used directly",
                "psi": "projected Psi is used directly",
                "additional_length_scaling": False,
            },
            "projection_flow": {
                "pre_projection": (
                    "p0=P0/Lx^2; q0=Q0/Ly^2; optional antisymmetric "
                    "difference fusion produces p,q"
                ),
                "physical_projection": ("d=p-q; phi=(rhs+d)/2; psi=(rhs-d)/2"),
                "post_projection": "Phi=Lx^2*phi; Psi=Ly^2*psi",
            },
            "pre_projection_fusion": {
                "enabled": fusion_config.enabled,
                "architecture": "single_nonlinear_residual_mlp",
                "space": "physical_directional_source",
                "input": [
                    "base_difference_over_safe_source_scale",
                    "rhs_over_safe_source_scale",
                ],
                "hidden_dim": fusion_config.hidden_dim,
                "depth": fusion_config.depth,
                "activation": configs.coupling_model.activation,
                "use_bias": configs.coupling_model.use_bias,
                "identity_skip": True,
                "final_layer_initialization": "zeros",
                "explicit_geometry_features": False,
                "learned_linear_branch": False,
                "learned_gate": False,
                "source_scale": "sqrt((A_x^2+A_y^2)/2)",
                "formula": (
                    "d_fused=d_base+A_safe*r_theta([d_base/A_safe,rhs/A_safe])"
                ),
                "pre_projection_balance_constructed": fusion_config.enabled,
                "uses_reference_targets": False,
            },
            "exact_green_reference_kinds": sorted(
                set(diagnostics.predicted_exact.reference_kinds)
                | set(diagnostics.target_exact.reference_kinds)
            ),
            "unit_physical_equivalence": {
                "tolerance": self.request.equivalence_tolerance,
                "max_absolute_difference": max(
                    diagnostics.predicted_exact.equivalence_max_abs,
                    diagnostics.target_exact.equivalence_max_abs,
                ),
                "max_relative_difference": max(
                    diagnostics.predicted_exact.equivalence_max_relative,
                    diagnostics.target_exact.equivalence_max_relative,
                ),
                "passed": True,
            },
            "error_decomposition": (
                "learned_total_error = exact_source_response + "
                "target_exact_closure + learned_minus_exact"
            ),
            "transition": asdict(transition),
            "rel_sol": {
                "min": float(np.min(rel_sol)),
                "q25": float(np.quantile(rel_sol, 0.25)),
                "median": float(np.median(rel_sol)),
                "q75": float(np.quantile(rel_sol, 0.75)),
                "max": float(np.max(rel_sol)),
                "mean": float(np.mean(rel_sol)),
            },
            "figure_count": len(figure_paths),
            "figure_paths": figure_paths,
            "outputs": {
                "per_sample_metrics": "metrics/per_sample_length_response.csv",
                "per_segment_metrics": "metrics/per_segment_length_response.csv",
                "transition_zone_metrics": "metrics/transition_zone_metrics.csv",
                "selected_arrays": "data/selected_diagnostic_arrays.npz",
            },
        }


def run_complex_length_response_diagnostics(
    request: ComplexLengthResponseDiagnosticRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexLengthResponseDiagnostic(request, logger=logger).run()
