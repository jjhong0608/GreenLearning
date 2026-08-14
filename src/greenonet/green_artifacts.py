from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor

from greenonet.axial import make_square_axial_lines
from greenonet.backward_sampler import BackwardSampler
from greenonet.coefficients import CoefficientFunctions, load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_data import (
    ComplexGreenDataset,
    generate_complex_green_data,
)
from greenonet.config import CompileConfig, DatasetConfig, ModelConfig, TrainingConfig
from greenonet.data import AxialDataset
from greenonet.greens import (
    GreenReferenceKind,
    exact_green_kernel_from_coefficients,
    exact_green_kernel_from_unit_coefficients,
    select_green_reference_policy,
)
from greenonet.green_quadrature import (
    build_split_pair_coords,
    evaluate_unit_line_coefficients,
    reconstruct_split_gauss_legendre,
    split_gauss_legendre_nodes,
)
from greenonet.green_lr_scheduler import GreenLearningRateSchedule
from greenonet.green_optimizer import GreenOptimizerFactory
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.numerics import IntegrationRule, integrate
from greenonet.plotly_io import save_plotly_figure
from greenonet.reproducibility import TrainingSeedContext
from greenonet.sampler import ForwardSampler


ScaleLength = float | tuple[float, float]
EvalSplit = Literal["train_like", "validation_like", "custom"]
SamplerMode = Literal["forward", "backward"]


def _training_reproducibility_summary(
    training: TrainingConfig,
) -> dict[str, object]:
    if training.seed is None:
        return {
            "available": False,
            "reason": "legacy_config_without_training_seed",
        }
    return {
        "available": True,
        **TrainingSeedContext(
            stage="green",
            base_seed=training.seed,
            deterministic_algorithms=training.deterministic_algorithms,
            device=training.device,
        ).as_dict(),
    }


@dataclass(frozen=True)
class EvaluationSamplingConfig:
    samples_per_line: int
    sampler_mode: SamplerMode
    scale_length: ScaleLength


@dataclass(frozen=True)
class SelectedXi:
    index: int
    value: float
    label: str


@dataclass(frozen=True)
class MetricStats:
    mean: Tensor
    min: Tensor
    max: Tensor
    std: Tensor


@dataclass(frozen=True)
class GreenArtifactRequest:
    checkpoint: Path
    config: Path
    outdir: Path
    coefficients: Path | None = None
    device: str | None = None
    eval_seed: int = 12345
    eval_split: EvalSplit = "validation_like"
    eval_samples_per_line: int | None = None
    eval_sampler_mode: SamplerMode | None = None
    eval_scale_length: ScaleLength | None = None
    line_indices: tuple[int, ...] | None = None
    xi_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    include_boundary_xi: bool = False
    theme: str = "plotly_white"
    save_generated_data: bool = True


def _parse_dtype(raw: object | None) -> torch.dtype:
    if raw is None:
        return torch.float64
    if not isinstance(raw, str):
        raise TypeError("dtype must be a string.")
    dtype = getattr(torch, raw.replace("torch.", ""))
    return cast(torch.dtype, dtype)


def _jsonify(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    return value


def _tensor_to_numpy(tensor: Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def load_green_artifact_configs(
    config_path: Path,
) -> tuple[DatasetConfig, ModelConfig, TrainingConfig, dict[str, Any]]:
    raw_payload = json.loads(config_path.read_text())
    if not isinstance(raw_payload, dict):
        raise TypeError("Config JSON must contain an object at top level.")

    raw_dataset = raw_payload.get("dataset")
    if not isinstance(raw_dataset, dict):
        raise TypeError("Config must contain dataset object.")
    dataset_cfg = DatasetConfig.from_raw(raw_dataset)

    raw_model = raw_payload.get("model", {})
    if not isinstance(raw_model, dict):
        raise TypeError("model config section must be an object when provided.")
    model_kwargs = dict(raw_model)
    model_kwargs["dtype"] = _parse_dtype(model_kwargs.pop("dtype", "float64"))
    model_cfg = ModelConfig(**model_kwargs)

    raw_training = raw_payload.get("training", {})
    if not isinstance(raw_training, dict):
        raise TypeError("training config section must be an object when provided.")
    training_kwargs = dict(raw_training)
    compile_raw = training_kwargs.pop("compile", None)
    if compile_raw is None:
        compile_cfg = CompileConfig()
    elif isinstance(compile_raw, dict):
        compile_cfg = CompileConfig(**compile_raw)
    else:
        raise TypeError("training.compile must be an object.")
    training_cfg = TrainingConfig(compile=compile_cfg, **training_kwargs)
    GreenOptimizerFactory(training_cfg)
    GreenLearningRateSchedule.validate_config(training_cfg)

    return dataset_cfg, model_cfg, training_cfg, raw_payload


class GreenArtifactExporter:
    """Generate paper-oriented GreenNet artifacts from a saved checkpoint."""

    ZERO_TOL = 1.0e-12

    def __init__(
        self,
        request: GreenArtifactRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger

    def export(self) -> dict[str, object]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        dataset_cfg, model_cfg, training_cfg, raw_config = load_green_artifact_configs(
            self.request.config
        )
        sampling_cfg = self._resolve_sampling_config(dataset_cfg)
        coeff_path = self.request.coefficients or dataset_cfg.coefficient_functions_path
        coeffs = load_coefficient_functions(coeff_path)

        device_name = self.request.device or training_cfg.device
        device = self._resolve_device(device_name)
        model = self._load_model(model_cfg)
        model.to(device)
        model.eval()

        random.seed(self.request.eval_seed)
        torch.manual_seed(self.request.eval_seed)
        if dataset_cfg.geometry_mode == "complex":
            return self._export_complex(
                dataset_cfg=dataset_cfg,
                model_cfg=model_cfg,
                training_cfg=training_cfg,
                raw_config=raw_config,
                sampling_cfg=sampling_cfg,
                coeffs=coeffs,
                coeff_path=coeff_path,
                model=model,
                device=device,
            )

        dataset = self._generate_dataset(
            dataset_cfg=dataset_cfg,
            training_cfg=training_cfg,
            sampling_cfg=sampling_cfg,
            coeffs=coeffs,
        )

        coords = dataset.coords.to(device)
        solution = dataset.solutions.to(device)
        source = dataset.sources.to(device)
        a_vals_all = dataset.a_vals.to(device)
        ap_vals_all = dataset.ap_vals.to(device)
        b_vals_all = dataset.b_vals.to(device)
        c_vals_all = dataset.c_vals.to(device)
        a_vals = a_vals_all[0]
        ap_vals = ap_vals_all[0]
        b_vals = b_vals_all[0]
        c_vals = c_vals_all[0]
        trunk_grid = self._build_trunk_grid(
            m_points=coords.shape[2],
            device=device,
            dtype=coords.dtype,
        )

        with torch.no_grad():
            kernel = cast(
                Tensor,
                model(
                    trunk_grid=trunk_grid,
                    a_vals=a_vals,
                    ap_vals=ap_vals,
                    b_vals=b_vals,
                    c_vals=c_vals,
                ),
            )
            reconstruction = self._reconstruct_solution(
                kernel=kernel,
                source=source,
                trunk_grid=trunk_grid,
                integration_rule=training_cfg.integration_rule,
            )
            rel_sol_by_line = self._relative_solution_error_by_line(
                reconstruction=reconstruction,
                solution=solution,
                trunk_grid=trunk_grid,
                integration_rule=training_cfg.integration_rule,
            )

        rel_green_policy = select_green_reference_policy(
            b_vals_all,
            c_vals_all,
            zero_tol=self.ZERO_TOL,
        )
        rel_green_valid = rel_green_policy.valid
        rel_green_reference = rel_green_policy.reference
        rel_green_skip_reason = rel_green_policy.skip_reason
        exact_kernel: Tensor | None = None
        rel_green_by_line: Tensor | None = None
        if rel_green_valid:
            assert rel_green_reference is not None
            exact_kernel = self._exact_green_kernel(
                coords=coords,
                a_vals=a_vals,
                b_vals=b_vals,
                reference=rel_green_reference,
            )
            rel_green_by_line = self._relative_green_error_by_line(
                prediction=kernel,
                exact_kernel=exact_kernel,
                x_axis=trunk_grid[:, 0, 0],
                integration_rule=training_cfg.integration_rule,
            )

        selected_lines = self._select_line_indices(
            n_lines=int(coords.shape[1]),
            requested=self.request.line_indices,
        )
        selected_xi = self._select_xi(
            x_axis=trunk_grid[0, :, 1],
            fractions=self.request.xi_fractions,
            include_boundary_xi=self.request.include_boundary_xi,
        )
        selected_samples = (0,)

        rel_sol_stats = self._aggregate_stats(rel_sol_by_line)
        rel_green_stats = (
            self._static_line_stats(rel_green_by_line)
            if rel_green_by_line is not None
            else None
        )

        self._write_metrics(
            coords=coords,
            rel_sol_by_line=rel_sol_by_line,
            rel_sol_stats=rel_sol_stats,
            rel_green_by_line=rel_green_by_line,
            rel_green_stats=rel_green_stats,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_lines=selected_lines,
            selected_xi=selected_xi,
            integration_rule=training_cfg.integration_rule,
        )
        self._write_figures(
            coords=coords,
            source=source,
            solution=solution,
            reconstruction=reconstruction,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_lines=selected_lines,
            selected_xi=selected_xi,
            selected_samples=selected_samples,
        )
        if self.request.save_generated_data:
            self._write_raw_data(
                dataset=dataset,
                kernel=kernel,
                exact_kernel=exact_kernel,
                reconstruction=reconstruction,
                selected_lines=selected_lines,
                selected_samples=selected_samples,
            )

        rel_sol_flat = rel_sol_by_line.detach().cpu().reshape(-1)
        rel_green_flat = (
            rel_green_by_line.detach().cpu().reshape(-1)
            if rel_green_by_line is not None
            else None
        )
        summary: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(self.request.checkpoint),
            "config": str(self.request.config),
            "coefficients": str(coeff_path) if coeff_path is not None else None,
            "outdir": str(self.request.outdir),
            "device": str(device),
            "theme": self.request.theme,
            "integration_rule": training_cfg.integration_rule,
            "training_reproducibility": _training_reproducibility_summary(training_cfg),
            **self._optimization_summary(training_cfg),
            "eval_seed": self.request.eval_seed,
            "eval_split": self.request.eval_split,
            "eval_sampling": _jsonify(asdict(sampling_cfg)),
            "selected_line_indices": list(selected_lines),
            "selected_xi": [asdict(item) for item in selected_xi],
            "selected_sample_indices": list(selected_samples),
            "rel_green_valid": rel_green_valid,
            "rel_green_reference": rel_green_reference,
            "rel_green_skip_reason": rel_green_skip_reason,
            "rel_sol": {
                "mean": float(rel_sol_flat.mean().item()),
                "max": float(rel_sol_flat.max().item()),
                "min": float(rel_sol_flat.min().item()),
            },
            "rel_green": (
                None
                if rel_green_flat is None
                else {
                    "mean": float(rel_green_flat.mean().item()),
                    "max": float(rel_green_flat.max().item()),
                    "min": float(rel_green_flat.min().item()),
                }
            ),
            "raw_config": _jsonify(raw_config),
        }
        summary_path = self.request.outdir / "summary.json"
        summary_path.write_text(json.dumps(_jsonify(summary), indent=2) + "\n")
        if self.logger is not None:
            self.logger.info("Saved GreenNet artifact summary to %s", summary_path)
        return summary

    def _resolve_sampling_config(
        self, dataset_cfg: DatasetConfig
    ) -> EvaluationSamplingConfig:
        if self.request.eval_split == "validation_like":
            default_count = (
                dataset_cfg.validation_samples_per_line
                if dataset_cfg.validation_samples_per_line > 0
                else dataset_cfg.samples_per_line
            )
            default_mode = (
                dataset_cfg.validation_sampler_mode or dataset_cfg.sampler_mode
            )
            default_scale = (
                dataset_cfg.validation_scale_length
                if dataset_cfg.validation_scale_length is not None
                else dataset_cfg.scale_length
            )
        else:
            default_count = dataset_cfg.samples_per_line
            default_mode = dataset_cfg.sampler_mode
            default_scale = dataset_cfg.scale_length

        samples_per_line = self.request.eval_samples_per_line or default_count
        if samples_per_line <= 0:
            raise ValueError("Evaluation samples per line must be positive.")
        sampler_mode = self.request.eval_sampler_mode or default_mode
        scale_length = self.request.eval_scale_length or default_scale
        return EvaluationSamplingConfig(
            samples_per_line=int(samples_per_line),
            sampler_mode=sampler_mode,
            scale_length=scale_length,
        )

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Config requests CUDA, but CUDA is not available. "
                    "Use a CPU config for artifact export in this environment."
                )
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Config requests {device}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available."
                )
        return device

    def _load_model(self, model_cfg: ModelConfig) -> GreenONetModel:
        try:
            loaded_model, _loaded_cfg = load_model_with_config(self.request.checkpoint)
        except Exception:
            model = GreenONetModel(model_cfg)
            load_state_dict_auto(model, self.request.checkpoint)
            return model
        if not isinstance(loaded_model, GreenONetModel):
            raise TypeError("Checkpoint metadata does not describe a GreenONet model.")
        return loaded_model

    @staticmethod
    def _sampler_cls(mode: SamplerMode) -> type[ForwardSampler] | type[BackwardSampler]:
        if mode == "forward":
            return ForwardSampler
        if mode == "backward":
            return BackwardSampler
        raise ValueError(f"Unsupported sampler mode: {mode}")

    def _generate_dataset(
        self,
        dataset_cfg: DatasetConfig,
        training_cfg: TrainingConfig,
        sampling_cfg: EvaluationSamplingConfig,
        coeffs: CoefficientFunctions,
    ) -> AxialDataset:
        axial_lines = make_square_axial_lines(
            step_size=dataset_cfg.step_size,
            n_points_per_line=dataset_cfg.n_points_per_line,
        )
        sampler = self._sampler_cls(sampling_cfg.sampler_mode)(
            axial_lines=axial_lines,
            data_size_per_each_line=sampling_cfg.samples_per_line,
            scale_length=sampling_cfg.scale_length,
            deterministic=dataset_cfg.deterministic,
            integration_rule=training_cfg.integration_rule,
            dtype=dataset_cfg.dtype,
        )
        data = sampler.generate_dataset(
            a_fun=coeffs.a_fun,
            ap_fun=coeffs.apx_fun,
            bx_fun=coeffs.bx_fun,
            by_fun=coeffs.by_fun,
            c_fun=coeffs.c_fun,
            a_fun_y=coeffs.a_fun,
            ap_fun_y=coeffs.apy_fun,
            c_fun_y=coeffs.c_fun,
        )
        return AxialDataset(data)

    def _export_complex(
        self,
        *,
        dataset_cfg: DatasetConfig,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        raw_config: dict[str, Any],
        sampling_cfg: EvaluationSamplingConfig,
        coeffs: CoefficientFunctions,
        coeff_path: Path | None,
        model: GreenONetModel,
        device: torch.device,
    ) -> dict[str, object]:
        if dataset_cfg.geometry_path is None:
            raise ValueError(
                "dataset.geometry_path is required for complex Green artifact export."
            )
        geometry = load_complex_geometry(
            dataset_cfg.geometry_path, dtype=model_cfg.dtype
        )
        data = generate_complex_green_data(
            geometry,
            coeffs,
            branch_input_dim=model_cfg.branch_input_dim,
            samples_per_interval=sampling_cfg.samples_per_line,
            sampler_mode=sampling_cfg.sampler_mode,
            scale_length=sampling_cfg.scale_length,
            deterministic=dataset_cfg.deterministic,
            integration_rule=training_cfg.integration_rule,
            source_sampling_factor=(
                training_cfg.green_quadrature.source_sampling_factor
                if training_cfg.green_quadrature.enabled
                else 1
            ),
            dtype=model_cfg.dtype,
        )
        dataset = ComplexGreenDataset(data)
        unit_grid = data.unit_grid.to(device)
        solution = data.solution.to(device)
        source = data.source.to(device)
        source_fine = None if data.source_fine is None else data.source_fine.to(device)
        source_fine_grid = (
            None if data.source_fine_grid is None else data.source_fine_grid.to(device)
        )
        a_vals = data.a_vals.to(device)
        ap_vals = data.ap_vals.to(device)
        b_vals = data.b_vals.to(device)
        c_vals = data.c_vals.to(device)
        trunk_grid = self._build_unit_trunk_grid(unit_grid)

        with torch.no_grad():
            kernel = self._forward_pairs_model(
                model=model,
                trunk_grid=trunk_grid,
                a_vals=a_vals,
                ap_vals=ap_vals,
                b_vals=b_vals,
                c_vals=c_vals,
            )
            if training_cfg.green_quadrature.enabled:
                split_kernel = self._complex_split_kernel_nodes(
                    model=model,
                    coeffs=coeffs,
                    dataset=dataset,
                    unit_grid=unit_grid,
                    a_vals=a_vals,
                    ap_vals=ap_vals,
                    b_vals=b_vals,
                    c_vals=c_vals,
                    order=training_cfg.green_quadrature.order,
                )
                reconstruction = reconstruct_split_gauss_legendre(
                    kernel_nodes=split_kernel,
                    source=source_fine if source_fine is not None else source,
                    source_grid=(
                        source_fine_grid if source_fine_grid is not None else unit_grid
                    ),
                    target_grid=unit_grid,
                    order=training_cfg.green_quadrature.order,
                    source_interpolation=training_cfg.green_quadrature.source_interpolation,
                )
            else:
                reconstruction = self._reconstruct_solution(
                    kernel=kernel,
                    source=source,
                    trunk_grid=trunk_grid,
                    integration_rule=training_cfg.integration_rule,
                )
            rel_sol_by_interval = self._relative_solution_error_by_line(
                reconstruction=reconstruction,
                solution=solution,
                trunk_grid=trunk_grid,
                integration_rule=training_cfg.integration_rule,
            )

        rel_green_policy = select_green_reference_policy(
            b_vals,
            c_vals,
            zero_tol=self.ZERO_TOL,
        )
        exact_kernel: Tensor | None = None
        rel_green_by_interval: Tensor | None = None
        if rel_green_policy.valid:
            assert rel_green_policy.reference is not None
            exact_kernel = exact_green_kernel_from_unit_coefficients(
                unit_grid,
                a_vals,
                b_vals,
                rel_green_policy.reference,
            )
            rel_green_by_interval = self._relative_green_error_by_line(
                prediction=kernel,
                exact_kernel=exact_kernel,
                x_axis=unit_grid,
                integration_rule=training_cfg.integration_rule,
            )

        selected_intervals = self._select_line_indices(
            n_lines=data.num_intervals,
            requested=self.request.line_indices,
        )
        selected_xi = self._select_xi(
            x_axis=unit_grid,
            fractions=self.request.xi_fractions,
            include_boundary_xi=self.request.include_boundary_xi,
        )
        selected_samples = (0,)

        rel_sol_stats = self._aggregate_stats(rel_sol_by_interval)
        rel_green_stats = (
            self._static_line_stats(rel_green_by_interval)
            if rel_green_by_interval is not None
            else None
        )
        self._write_complex_metrics(
            dataset=dataset,
            rel_sol_by_interval=rel_sol_by_interval,
            rel_sol_stats=rel_sol_stats,
            rel_green_by_interval=rel_green_by_interval,
            rel_green_stats=rel_green_stats,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_intervals=selected_intervals,
            selected_xi=selected_xi,
            unit_grid=unit_grid,
            integration_rule=training_cfg.integration_rule,
        )
        self._write_complex_figures(
            dataset=dataset,
            source=source,
            solution=solution,
            reconstruction=reconstruction,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_intervals=selected_intervals,
            selected_xi=selected_xi,
            selected_samples=selected_samples,
            unit_grid=unit_grid,
        )
        if self.request.save_generated_data:
            self._write_complex_raw_data(
                dataset=dataset,
                kernel=kernel,
                exact_kernel=exact_kernel,
                reconstruction=reconstruction,
                selected_intervals=selected_intervals,
                selected_samples=selected_samples,
            )

        rel_sol_flat = rel_sol_by_interval.detach().cpu().reshape(-1)
        rel_green_flat = (
            None
            if rel_green_by_interval is None
            else rel_green_by_interval.detach().cpu().reshape(-1)
        )
        summary: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(self.request.checkpoint),
            "config": str(self.request.config),
            "coefficients": str(coeff_path) if coeff_path is not None else None,
            "outdir": str(self.request.outdir),
            "device": str(device),
            "theme": self.request.theme,
            "integration_rule": training_cfg.integration_rule,
            "training_reproducibility": _training_reproducibility_summary(training_cfg),
            **self._optimization_summary(training_cfg),
            "eval_seed": self.request.eval_seed,
            "eval_split": self.request.eval_split,
            "eval_sampling": _jsonify(asdict(sampling_cfg)),
            "green_quadrature": {
                "enabled": training_cfg.green_quadrature.enabled,
                "rule": training_cfg.green_quadrature.rule,
                "order": training_cfg.green_quadrature.order,
                "source_sampling_factor": training_cfg.green_quadrature.source_sampling_factor,
                "source_interpolation": training_cfg.green_quadrature.source_interpolation,
                "applies_to": "reconstruction_and_rel_sol",
                "rel_green": "uniform_grid_existing",
            },
            "geometry_mode": "complex",
            "geometry_path": str(dataset_cfg.geometry_path),
            "num_intervals": data.num_intervals,
            "num_x_segments": geometry.num_x_segments,
            "num_y_segments": geometry.num_y_segments,
            "intervals": self._complex_interval_summary(dataset),
            "selected_interval_indices": list(selected_intervals),
            "selected_line_indices": list(selected_intervals),
            "selected_xi": [asdict(item) for item in selected_xi],
            "selected_sample_indices": list(selected_samples),
            "rel_green_valid": rel_green_policy.valid,
            "rel_green_reference": rel_green_policy.reference,
            "rel_green_skip_reason": rel_green_policy.skip_reason,
            "rel_sol": {
                "mean": float(rel_sol_flat.mean().item()),
                "max": float(rel_sol_flat.max().item()),
                "min": float(rel_sol_flat.min().item()),
            },
            "rel_green": (
                None
                if rel_green_flat is None
                else {
                    "mean": float(rel_green_flat.mean().item()),
                    "max": float(rel_green_flat.max().item()),
                    "min": float(rel_green_flat.min().item()),
                }
            ),
            "raw_config": _jsonify(raw_config),
        }
        summary_path = self.request.outdir / "summary.json"
        summary_path.write_text(json.dumps(_jsonify(summary), indent=2) + "\n")
        if self.logger is not None:
            self.logger.info(
                "Saved complex GreenNet artifact summary to %s", summary_path
            )
        return summary

    @staticmethod
    def _optimization_summary(
        training_cfg: TrainingConfig,
    ) -> dict[str, object]:
        factory = GreenOptimizerFactory(training_cfg)
        return {
            "green_optimizer_provenance": factory.provenance().as_dict(),
            "green_learning_rate_schedule": (
                GreenLearningRateSchedule.configured_config(training_cfg)
            ),
            "optimizer_state_resume": "not_supported_model_only_checkpoint",
            "lbfgs_scheduler": "disabled",
        }

    @staticmethod
    def _build_unit_trunk_grid(unit_grid: Tensor) -> Tensor:
        return torch.stack(torch.meshgrid(unit_grid, unit_grid, indexing="ij"), dim=-1)

    @staticmethod
    def _forward_pairs_model(
        *,
        model: GreenONetModel,
        trunk_grid: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        a_eval: Tensor | None = None,
        ap_eval: Tensor | None = None,
        b_eval: Tensor | None = None,
    ) -> Tensor:
        pair_forward = getattr(model, "forward_pairs", None)
        if not callable(pair_forward):
            raise TypeError("Complex Green artifact export requires forward_pairs().")
        kwargs: dict[str, Tensor] = {}
        if a_eval is not None:
            kwargs["a_eval"] = a_eval
        if ap_eval is not None:
            kwargs["ap_eval"] = ap_eval
        if b_eval is not None:
            kwargs["b_eval"] = b_eval
        return cast(
            Tensor,
            pair_forward(trunk_grid, a_vals, ap_vals, b_vals, c_vals, **kwargs),
        )

    def _complex_split_kernel_nodes(
        self,
        *,
        model: GreenONetModel,
        coeffs: CoefficientFunctions,
        dataset: ComplexGreenDataset,
        unit_grid: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        order: int,
    ) -> Tensor:
        eta_nodes, _weights = split_gauss_legendre_nodes(unit_grid, order)
        pair_coords = build_split_pair_coords(unit_grid, eta_nodes)
        data = dataset.data
        a_eval, ap_eval, b_eval = evaluate_unit_line_coefficients(
            coeffs,
            axis_id=data.axis_id.to(device=unit_grid.device),
            left=data.left.to(device=unit_grid.device, dtype=unit_grid.dtype),
            fixed=data.fixed.to(device=unit_grid.device, dtype=unit_grid.dtype),
            length=data.length.to(device=unit_grid.device, dtype=unit_grid.dtype),
            t_nodes=pair_coords[..., 0],
        )
        return self._forward_pairs_model(
            model=model,
            trunk_grid=pair_coords,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            a_eval=a_eval,
            ap_eval=ap_eval,
            b_eval=b_eval,
        )

    @staticmethod
    def _complex_axis_name(axis_id: int) -> str:
        return "x" if axis_id == 0 else "y"

    @staticmethod
    def _complex_interval_summary(
        dataset: ComplexGreenDataset,
    ) -> list[dict[str, object]]:
        data = dataset.data
        rows: list[dict[str, object]] = []
        for interval_idx in range(data.num_intervals):
            axis_id = int(data.axis_id[interval_idx].item())
            rows.append(
                {
                    "interval_index": interval_idx,
                    "axis_id": axis_id,
                    "axis": GreenArtifactExporter._complex_axis_name(axis_id),
                    "segment_index": int(data.segment_id[interval_idx].item()),
                    "left": float(data.left[interval_idx].item()),
                    "right": float(data.right[interval_idx].item()),
                    "fixed": float(data.fixed[interval_idx].item()),
                    "length": float(data.length[interval_idx].item()),
                }
            )
        return rows

    def _write_complex_metrics(
        self,
        *,
        dataset: ComplexGreenDataset,
        rel_sol_by_interval: Tensor,
        rel_sol_stats: MetricStats,
        rel_green_by_interval: Tensor | None,
        rel_green_stats: MetricStats | None,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_intervals: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self._write_complex_per_interval_metrics(
            metrics_dir / "per_interval_metrics.csv",
            dataset=dataset,
            rel_sol_stats=rel_sol_stats,
            rel_green_stats=rel_green_stats,
        )
        self._write_complex_sample_metrics(
            metrics_dir / "sample_metrics.csv",
            dataset=dataset,
            rel_sol_by_interval=rel_sol_by_interval,
            rel_green_by_interval=rel_green_by_interval,
        )
        self._write_complex_boundary_and_slice_metrics(
            metrics_dir=metrics_dir,
            dataset=dataset,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_intervals=selected_intervals,
            selected_xi=selected_xi,
            unit_grid=unit_grid,
            integration_rule=integration_rule,
        )

    def _write_complex_per_interval_metrics(
        self,
        path: Path,
        *,
        dataset: ComplexGreenDataset,
        rel_sol_stats: MetricStats,
        rel_green_stats: MetricStats | None,
    ) -> None:
        data = dataset.data
        with path.open("w", newline="") as fp:
            fieldnames = [
                "interval_index",
                "axis_id",
                "axis",
                "segment_index",
                "left",
                "right",
                "fixed",
                "length",
                "rel_sol_interval_mean",
                "rel_sol_interval_min",
                "rel_sol_interval_max",
                "rel_sol_interval_std",
                "rel_green_interval_mean",
                "rel_green_interval_min",
                "rel_green_interval_max",
                "rel_green_interval_std",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for interval_idx in range(data.num_intervals):
                axis_id = int(data.axis_id[interval_idx].item())
                row: dict[str, object] = {
                    "interval_index": interval_idx,
                    "axis_id": axis_id,
                    "axis": self._complex_axis_name(axis_id),
                    "segment_index": int(data.segment_id[interval_idx].item()),
                    "left": float(data.left[interval_idx].item()),
                    "right": float(data.right[interval_idx].item()),
                    "fixed": float(data.fixed[interval_idx].item()),
                    "length": float(data.length[interval_idx].item()),
                    "rel_sol_interval_mean": float(
                        rel_sol_stats.mean[interval_idx].item()
                    ),
                    "rel_sol_interval_min": float(
                        rel_sol_stats.min[interval_idx].item()
                    ),
                    "rel_sol_interval_max": float(
                        rel_sol_stats.max[interval_idx].item()
                    ),
                    "rel_sol_interval_std": float(
                        rel_sol_stats.std[interval_idx].item()
                    ),
                }
                if rel_green_stats is None:
                    row.update(
                        {
                            "rel_green_interval_mean": "",
                            "rel_green_interval_min": "",
                            "rel_green_interval_max": "",
                            "rel_green_interval_std": "",
                        }
                    )
                else:
                    row.update(
                        {
                            "rel_green_interval_mean": float(
                                rel_green_stats.mean[interval_idx].item()
                            ),
                            "rel_green_interval_min": float(
                                rel_green_stats.min[interval_idx].item()
                            ),
                            "rel_green_interval_max": float(
                                rel_green_stats.max[interval_idx].item()
                            ),
                            "rel_green_interval_std": float(
                                rel_green_stats.std[interval_idx].item()
                            ),
                        }
                    )
                writer.writerow(row)

    def _write_complex_sample_metrics(
        self,
        path: Path,
        *,
        dataset: ComplexGreenDataset,
        rel_sol_by_interval: Tensor,
        rel_green_by_interval: Tensor | None,
    ) -> None:
        data = dataset.data
        with path.open("w", newline="") as fp:
            fieldnames = [
                "sample_index",
                "interval_index",
                "axis_id",
                "axis",
                "segment_index",
                "rel_sol_interval",
                "rel_green_interval",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for sample_idx in range(rel_sol_by_interval.shape[0]):
                for interval_idx in range(rel_sol_by_interval.shape[1]):
                    axis_id = int(data.axis_id[interval_idx].item())
                    writer.writerow(
                        {
                            "sample_index": sample_idx,
                            "interval_index": interval_idx,
                            "axis_id": axis_id,
                            "axis": self._complex_axis_name(axis_id),
                            "segment_index": int(data.segment_id[interval_idx].item()),
                            "rel_sol_interval": float(
                                rel_sol_by_interval[sample_idx, interval_idx].item()
                            ),
                            "rel_green_interval": (
                                ""
                                if rel_green_by_interval is None
                                else float(rel_green_by_interval[interval_idx].item())
                            ),
                        }
                    )

    def _write_complex_boundary_and_slice_metrics(
        self,
        *,
        metrics_dir: Path,
        dataset: ComplexGreenDataset,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_intervals: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        unit_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> None:
        boundary_path = metrics_dir / "boundary_diagnostics.csv"
        slice_path = metrics_dir / "green_slice_metrics.csv"
        boundary_fields = [
            "interval_index",
            "axis_id",
            "axis",
            "segment_index",
            "xi_index",
            "xi_value",
            "pred_left_boundary",
            "pred_right_boundary",
            "boundary_abs_max",
            "diagonal_value",
            "ref_left_boundary",
            "ref_right_boundary",
        ]
        slice_fields = [*boundary_fields, "slice_rel_error"]
        data = dataset.data
        with (
            boundary_path.open("w", newline="") as bfp,
            slice_path.open("w", newline="") as sfp,
        ):
            boundary_writer = csv.DictWriter(bfp, fieldnames=boundary_fields)
            slice_writer = csv.DictWriter(sfp, fieldnames=slice_fields)
            boundary_writer.writeheader()
            slice_writer.writeheader()
            for interval_idx in selected_intervals:
                axis_id = int(data.axis_id[interval_idx].item())
                for xi_item in selected_xi:
                    pred_slice = kernel[interval_idx, :, xi_item.index]
                    ref_slice = (
                        None
                        if exact_kernel is None
                        else exact_kernel[interval_idx, :, xi_item.index]
                    )
                    row = {
                        "interval_index": interval_idx,
                        "axis_id": axis_id,
                        "axis": self._complex_axis_name(axis_id),
                        "segment_index": int(data.segment_id[interval_idx].item()),
                        "xi_index": xi_item.index,
                        "xi_value": xi_item.value,
                        "pred_left_boundary": float(pred_slice[0].item()),
                        "pred_right_boundary": float(pred_slice[-1].item()),
                        "boundary_abs_max": max(
                            abs(float(pred_slice[0].item())),
                            abs(float(pred_slice[-1].item())),
                        ),
                        "diagonal_value": float(pred_slice[xi_item.index].item()),
                        "ref_left_boundary": "",
                        "ref_right_boundary": "",
                        "slice_rel_error": "",
                    }
                    if ref_slice is not None:
                        residual = pred_slice - ref_slice
                        num = integrate(
                            residual.pow(2),
                            x=unit_grid,
                            dim=-1,
                            rule=integration_rule,
                        )
                        den = integrate(
                            ref_slice.pow(2),
                            x=unit_grid,
                            dim=-1,
                            rule=integration_rule,
                        ).clamp_min(1.0e-12)
                        row.update(
                            {
                                "ref_left_boundary": float(ref_slice[0].item()),
                                "ref_right_boundary": float(ref_slice[-1].item()),
                                "slice_rel_error": float(torch.sqrt(num / den).item()),
                            }
                        )
                    boundary_writer.writerow({key: row[key] for key in boundary_fields})
                    slice_writer.writerow(row)

    def _write_complex_figures(
        self,
        *,
        dataset: ComplexGreenDataset,
        source: Tensor,
        solution: Tensor,
        reconstruction: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_intervals: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        selected_samples: tuple[int, ...],
        unit_grid: Tensor,
    ) -> None:
        for interval_idx in selected_intervals:
            self._save_complex_green_heatmap(
                kernel=kernel,
                exact_kernel=exact_kernel,
                interval_idx=interval_idx,
                unit_grid=unit_grid,
            )
            self._save_complex_coefficient_figure(
                dataset=dataset,
                a_vals=a_vals,
                ap_vals=ap_vals,
                b_vals=b_vals,
                c_vals=c_vals,
                interval_idx=interval_idx,
                unit_grid=unit_grid,
            )
            for xi_item in selected_xi:
                self._save_complex_green_slice(
                    kernel=kernel,
                    exact_kernel=exact_kernel,
                    interval_idx=interval_idx,
                    xi_item=xi_item,
                    unit_grid=unit_grid,
                )
            for sample_idx in selected_samples:
                self._save_complex_reconstruction_figure(
                    source=source,
                    solution=solution,
                    reconstruction=reconstruction,
                    sample_idx=sample_idx,
                    interval_idx=interval_idx,
                    unit_grid=unit_grid,
                )

    def _save_complex_green_heatmap(
        self,
        *,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        interval_idx: int,
        unit_grid: Tensor,
    ) -> None:
        base = self.request.outdir / "figures" / "green_heatmaps"
        pred = kernel[interval_idx]
        self._save_heatmap_figure(
            z=pred,
            x_axis=unit_grid,
            title=f"Predicted Green kernel interval={interval_idx}",
            base_path=base / f"interval{interval_idx:03d}_green_heatmap_pred",
        )
        if exact_kernel is not None:
            ref = exact_kernel[interval_idx]
            self._save_heatmap_figure(
                z=ref,
                x_axis=unit_grid,
                title=f"Reference Green kernel interval={interval_idx}",
                base_path=base / f"interval{interval_idx:03d}_green_heatmap_ref",
            )
            self._save_heatmap_figure(
                z=pred - ref,
                x_axis=unit_grid,
                title=f"Green kernel error interval={interval_idx}",
                base_path=base / f"interval{interval_idx:03d}_green_heatmap_error",
            )

    def _save_complex_green_slice(
        self,
        *,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        interval_idx: int,
        xi_item: SelectedXi,
        unit_grid: Tensor,
    ) -> None:
        pred_slice = kernel[interval_idx, :, xi_item.index]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=_tensor_to_numpy(unit_grid),
                y=_tensor_to_numpy(pred_slice),
                mode="lines+markers",
                name="predicted",
            )
        )
        if exact_kernel is not None:
            ref_slice = exact_kernel[interval_idx, :, xi_item.index]
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(unit_grid),
                    y=_tensor_to_numpy(ref_slice),
                    mode="lines",
                    name="reference",
                    line=dict(dash="dash"),
                )
            )
        fig.add_vline(
            x=xi_item.value,
            line=dict(color="black", dash="dot"),
            annotation_text="t=eta",
        )
        fig.update_layout(
            title=f"Fixed-eta Green slice interval={interval_idx} eta={xi_item.value:.4f}",
            xaxis_title="t",
            yaxis_title="G_unit(t, eta)",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "green_slices"
            / f"interval{interval_idx:03d}_xi{xi_item.index:03d}_green_slice"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _save_complex_coefficient_figure(
        self,
        *,
        dataset: ComplexGreenDataset,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        interval_idx: int,
        unit_grid: Tensor,
    ) -> None:
        axis_id = int(dataset.data.axis_id[interval_idx].item())
        axis_name = self._complex_axis_name(axis_id)
        ap_name = "apx_unit" if axis_name == "x" else "apy_unit"
        b_name = "bx_unit" if axis_name == "x" else "by_unit"
        fig = go.Figure()
        for name, values in (
            ("a_unit", a_vals[interval_idx]),
            (ap_name, ap_vals[interval_idx]),
            (b_name, b_vals[interval_idx]),
            ("c_unit", c_vals[interval_idx]),
        ):
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(unit_grid),
                    y=_tensor_to_numpy(values),
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=f"Unit coefficient slices interval={interval_idx}",
            xaxis_title="t",
            yaxis_title="coefficient value",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "coefficients"
            / f"interval{interval_idx:03d}_coefficients"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _save_complex_reconstruction_figure(
        self,
        *,
        source: Tensor,
        solution: Tensor,
        reconstruction: Tensor,
        sample_idx: int,
        interval_idx: int,
        unit_grid: Tensor,
    ) -> None:
        exact = solution[sample_idx, interval_idx]
        pred = reconstruction[sample_idx, interval_idx]
        fig = go.Figure()
        for name, values in (
            ("source f_unit", source[sample_idx, interval_idx]),
            ("reference v", exact),
            ("reconstructed v", pred),
            ("error v-v_hat", exact - pred),
        ):
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(unit_grid),
                    y=_tensor_to_numpy(values),
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=f"Unit Green reconstruction sample={sample_idx} interval={interval_idx}",
            xaxis_title="t",
            yaxis_title="value",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "reconstruction"
            / f"sample{sample_idx:03d}_interval{interval_idx:03d}_reconstruction"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _write_complex_raw_data(
        self,
        *,
        dataset: ComplexGreenDataset,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        reconstruction: Tensor,
        selected_intervals: tuple[int, ...],
        selected_samples: tuple[int, ...],
    ) -> None:
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data = dataset.data
        generated_payload: dict[str, np.ndarray] = {
            "unit_grid": _tensor_to_numpy(data.unit_grid),
            "physical_coords": _tensor_to_numpy(data.physical_coords),
            "axis_id": _tensor_to_numpy(data.axis_id),
            "segment_id": _tensor_to_numpy(data.segment_id),
            "left": _tensor_to_numpy(data.left),
            "right": _tensor_to_numpy(data.right),
            "fixed": _tensor_to_numpy(data.fixed),
            "length": _tensor_to_numpy(data.length),
            "solution": _tensor_to_numpy(data.solution),
            "source": _tensor_to_numpy(data.source),
            "a_vals": _tensor_to_numpy(data.a_vals),
            "ap_vals": _tensor_to_numpy(data.ap_vals),
            "b_vals": _tensor_to_numpy(data.b_vals),
            "c_vals": _tensor_to_numpy(data.c_vals),
        }
        if data.source_fine is not None and data.source_fine_grid is not None:
            generated_payload["source_fine"] = _tensor_to_numpy(data.source_fine)
            generated_payload["source_fine_grid"] = _tensor_to_numpy(
                data.source_fine_grid
            )
        savez_compressed = cast(Any, np.savez_compressed)
        savez_compressed(data_dir / "generated_eval_data.npz", **generated_payload)

        selected_pred = torch.stack([kernel[idx] for idx in selected_intervals], dim=0)
        if exact_kernel is None:
            np.savez_compressed(
                data_dir / "selected_green_kernels.npz",
                interval_indices=np.array(selected_intervals, dtype=np.int64),
                predicted=_tensor_to_numpy(selected_pred),
            )
        else:
            selected_ref = torch.stack(
                [exact_kernel[idx] for idx in selected_intervals],
                dim=0,
            )
            np.savez_compressed(
                data_dir / "selected_green_kernels.npz",
                interval_indices=np.array(selected_intervals, dtype=np.int64),
                predicted=_tensor_to_numpy(selected_pred),
                reference=_tensor_to_numpy(selected_ref),
                error=_tensor_to_numpy(selected_pred - selected_ref),
            )

        sample_pairs = [
            (sample_idx, interval_idx)
            for sample_idx in selected_samples
            for interval_idx in selected_intervals
        ]
        selected_reconstruction = torch.stack(
            [reconstruction[sample, interval] for sample, interval in sample_pairs],
            dim=0,
        )
        selected_solution = torch.stack(
            [data.solution[sample, interval] for sample, interval in sample_pairs],
            dim=0,
        )
        selected_source = torch.stack(
            [data.source[sample, interval] for sample, interval in sample_pairs],
            dim=0,
        )
        np.savez_compressed(
            data_dir / "selected_reconstructions.npz",
            sample_indices=np.array(
                [sample for sample, _interval in sample_pairs],
                dtype=np.int64,
            ),
            interval_indices=np.array(
                [interval for _sample, interval in sample_pairs],
                dtype=np.int64,
            ),
            reconstruction=_tensor_to_numpy(selected_reconstruction),
            solution=_tensor_to_numpy(selected_solution),
            source=_tensor_to_numpy(selected_source),
            error=_tensor_to_numpy(selected_solution - selected_reconstruction.cpu()),
        )

    @staticmethod
    def _build_trunk_grid(
        m_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        return torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, m_points, device=device, dtype=dtype),
                torch.linspace(0.0, 1.0, m_points, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        )

    @staticmethod
    def _reconstruct_solution(
        kernel: Tensor,
        source: Tensor,
        trunk_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        xi = trunk_grid[0, :, 1]
        rhs = source.unsqueeze(-2) * kernel.unsqueeze(0)
        return integrate(rhs, x=xi, dim=-1, rule=integration_rule)

    @staticmethod
    def _relative_solution_error_by_line(
        reconstruction: Tensor,
        solution: Tensor,
        trunk_grid: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        x_axis = trunk_grid[:, 0, 0]
        residual = solution - reconstruction
        residual_energy = integrate(
            residual.pow(2), x=x_axis, dim=-1, rule=integration_rule
        )
        solution_energy = integrate(
            solution.pow(2), x=x_axis, dim=-1, rule=integration_rule
        ).clamp_min(1.0e-12)
        return torch.sqrt(residual_energy / solution_energy)

    @staticmethod
    def _relative_green_error_by_line(
        prediction: Tensor,
        exact_kernel: Tensor,
        x_axis: Tensor,
        integration_rule: IntegrationRule,
    ) -> Tensor:
        num = (prediction - exact_kernel).pow(2)
        den = exact_kernel.pow(2)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule).clamp_min(1.0e-12)
        return torch.sqrt(num / den)

    @staticmethod
    def _exact_green_kernel(
        coords: Tensor,
        a_vals: Tensor,
        b_vals: Tensor,
        reference: GreenReferenceKind,
    ) -> Tensor:
        return exact_green_kernel_from_coefficients(
            coords=coords,
            a_vals=a_vals,
            b_vals=b_vals,
            reference=reference,
        )

    @staticmethod
    def _aggregate_stats(values: Tensor) -> MetricStats:
        values_cpu = values.detach().cpu().to(torch.float64)
        mean = values_cpu.mean(dim=0)
        min_val = values_cpu.min(dim=0).values
        max_val = values_cpu.max(dim=0).values
        if values_cpu.shape[0] > 1:
            std = values_cpu.std(dim=0, unbiased=True)
        else:
            std = torch.zeros_like(mean)
        return MetricStats(mean=mean, min=min_val, max=max_val, std=std)

    @staticmethod
    def _static_line_stats(values: Tensor) -> MetricStats:
        values_cpu = values.detach().cpu().to(torch.float64)
        zeros = torch.zeros_like(values_cpu)
        return MetricStats(
            mean=values_cpu,
            min=values_cpu,
            max=values_cpu,
            std=zeros,
        )

    @staticmethod
    def _select_line_indices(
        n_lines: int,
        requested: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if n_lines <= 0:
            raise ValueError("No axial lines are available.")
        if requested:
            selected = list(requested)
        else:
            selected = [0, n_lines // 2, n_lines - 1]
        deduped: list[int] = []
        for index in selected:
            if index < 0 or index >= n_lines:
                raise ValueError(
                    f"Line index {index} is out of range for {n_lines} line(s)."
                )
            if index not in deduped:
                deduped.append(index)
        return tuple(deduped)

    @staticmethod
    def _select_xi(
        x_axis: Tensor,
        fractions: Sequence[float],
        include_boundary_xi: bool,
    ) -> tuple[SelectedXi, ...]:
        if x_axis.numel() == 0:
            raise ValueError("Cannot select xi values on an empty grid.")
        candidates: list[tuple[int, str]] = []
        for fraction in fractions:
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("xi fractions must be between 0 and 1.")
            distances = (x_axis - fraction).abs()
            index = int(torch.argmin(distances).item())
            candidates.append((index, f"{fraction:.6g}"))
        if include_boundary_xi and x_axis.numel() >= 3:
            candidates.extend(
                [
                    (1, "near_left_boundary"),
                    (x_axis.numel() - 2, "near_right_boundary"),
                ]
            )

        selected: list[SelectedXi] = []
        seen: set[int] = set()
        for index, label in candidates:
            if index in seen:
                continue
            seen.add(index)
            selected.append(
                SelectedXi(
                    index=index,
                    value=float(x_axis[index].detach().cpu().item()),
                    label=label,
                )
            )
        return tuple(selected)

    @staticmethod
    def _axis_name(axis: int) -> str:
        return "x" if axis == 0 else "y"

    @staticmethod
    def _line_coordinate(coords: Tensor, axis: int, line_idx: int) -> float:
        transverse_dim = 1 if axis == 0 else 0
        return float(coords[axis, line_idx, 0, transverse_dim].detach().cpu().item())

    def _write_metrics(
        self,
        coords: Tensor,
        rel_sol_by_line: Tensor,
        rel_sol_stats: MetricStats,
        rel_green_by_line: Tensor | None,
        rel_green_stats: MetricStats | None,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_lines: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        integration_rule: IntegrationRule,
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self._write_per_line_metrics(
            metrics_dir / "per_line_metrics.csv",
            coords=coords,
            rel_sol_stats=rel_sol_stats,
            rel_green_stats=rel_green_stats,
        )
        self._write_sample_metrics(
            metrics_dir / "sample_metrics.csv",
            rel_sol_by_line=rel_sol_by_line,
            rel_green_by_line=rel_green_by_line,
        )
        self._write_boundary_and_slice_metrics(
            metrics_dir=metrics_dir,
            coords=coords,
            kernel=kernel,
            exact_kernel=exact_kernel,
            selected_lines=selected_lines,
            selected_xi=selected_xi,
            integration_rule=integration_rule,
        )

    def _write_per_line_metrics(
        self,
        path: Path,
        coords: Tensor,
        rel_sol_stats: MetricStats,
        rel_green_stats: MetricStats | None,
    ) -> None:
        with path.open("w", newline="") as fp:
            fieldnames = [
                "axis_id",
                "axis_name",
                "line_index",
                "line_coordinate",
                "rel_sol_line_mean",
                "rel_sol_line_min",
                "rel_sol_line_max",
                "rel_sol_line_std",
                "rel_green_line_mean",
                "rel_green_line_min",
                "rel_green_line_max",
                "rel_green_line_std",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for axis in range(rel_sol_stats.mean.shape[0]):
                for line_idx in range(rel_sol_stats.mean.shape[1]):
                    row: dict[str, object] = {
                        "axis_id": axis,
                        "axis_name": self._axis_name(axis),
                        "line_index": line_idx,
                        "line_coordinate": self._line_coordinate(
                            coords, axis, line_idx
                        ),
                        "rel_sol_line_mean": float(
                            rel_sol_stats.mean[axis, line_idx].item()
                        ),
                        "rel_sol_line_min": float(
                            rel_sol_stats.min[axis, line_idx].item()
                        ),
                        "rel_sol_line_max": float(
                            rel_sol_stats.max[axis, line_idx].item()
                        ),
                        "rel_sol_line_std": float(
                            rel_sol_stats.std[axis, line_idx].item()
                        ),
                    }
                    if rel_green_stats is None:
                        row.update(
                            {
                                "rel_green_line_mean": "",
                                "rel_green_line_min": "",
                                "rel_green_line_max": "",
                                "rel_green_line_std": "",
                            }
                        )
                    else:
                        row.update(
                            {
                                "rel_green_line_mean": float(
                                    rel_green_stats.mean[axis, line_idx].item()
                                ),
                                "rel_green_line_min": float(
                                    rel_green_stats.min[axis, line_idx].item()
                                ),
                                "rel_green_line_max": float(
                                    rel_green_stats.max[axis, line_idx].item()
                                ),
                                "rel_green_line_std": float(
                                    rel_green_stats.std[axis, line_idx].item()
                                ),
                            }
                        )
                    writer.writerow(row)

    def _write_sample_metrics(
        self,
        path: Path,
        rel_sol_by_line: Tensor,
        rel_green_by_line: Tensor | None,
    ) -> None:
        with path.open("w", newline="") as fp:
            fieldnames = [
                "sample_index",
                "axis_id",
                "axis_name",
                "line_index",
                "rel_sol_line",
                "rel_green_line",
            ]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for sample_idx in range(rel_sol_by_line.shape[0]):
                for axis in range(rel_sol_by_line.shape[1]):
                    for line_idx in range(rel_sol_by_line.shape[2]):
                        writer.writerow(
                            {
                                "sample_index": sample_idx,
                                "axis_id": axis,
                                "axis_name": self._axis_name(axis),
                                "line_index": line_idx,
                                "rel_sol_line": float(
                                    rel_sol_by_line[sample_idx, axis, line_idx].item()
                                ),
                                "rel_green_line": (
                                    ""
                                    if rel_green_by_line is None
                                    else float(rel_green_by_line[axis, line_idx].item())
                                ),
                            }
                        )

    def _write_boundary_and_slice_metrics(
        self,
        metrics_dir: Path,
        coords: Tensor,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_lines: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        integration_rule: IntegrationRule,
    ) -> None:
        boundary_path = metrics_dir / "boundary_diagnostics.csv"
        slice_path = metrics_dir / "green_slice_metrics.csv"
        boundary_fields = [
            "axis_id",
            "axis_name",
            "line_index",
            "line_coordinate",
            "xi_index",
            "xi_value",
            "pred_left_boundary",
            "pred_right_boundary",
            "boundary_abs_max",
            "diagonal_value",
            "ref_left_boundary",
            "ref_right_boundary",
        ]
        slice_fields = [
            *boundary_fields,
            "slice_rel_error",
        ]
        with (
            boundary_path.open("w", newline="") as bfp,
            slice_path.open("w", newline="") as sfp,
        ):
            boundary_writer = csv.DictWriter(bfp, fieldnames=boundary_fields)
            slice_writer = csv.DictWriter(sfp, fieldnames=slice_fields)
            boundary_writer.writeheader()
            slice_writer.writeheader()
            x_axis = coords[0, 0, :, 0]
            for axis in range(kernel.shape[0]):
                for line_idx in selected_lines:
                    for xi_item in selected_xi:
                        pred_slice = kernel[axis, line_idx, :, xi_item.index]
                        ref_slice = (
                            None
                            if exact_kernel is None
                            else exact_kernel[axis, line_idx, :, xi_item.index]
                        )
                        row = self._slice_metric_row(
                            coords=coords,
                            axis=axis,
                            line_idx=line_idx,
                            xi_item=xi_item,
                            pred_slice=pred_slice,
                            ref_slice=ref_slice,
                            x_axis=x_axis,
                            integration_rule=integration_rule,
                        )
                        boundary_writer.writerow(
                            {key: row[key] for key in boundary_fields}
                        )
                        slice_writer.writerow(row)

    def _slice_metric_row(
        self,
        coords: Tensor,
        axis: int,
        line_idx: int,
        xi_item: SelectedXi,
        pred_slice: Tensor,
        ref_slice: Tensor | None,
        x_axis: Tensor,
        integration_rule: IntegrationRule,
    ) -> dict[str, object]:
        pred_left = float(pred_slice[0].detach().cpu().item())
        pred_right = float(pred_slice[-1].detach().cpu().item())
        row: dict[str, object] = {
            "axis_id": axis,
            "axis_name": self._axis_name(axis),
            "line_index": line_idx,
            "line_coordinate": self._line_coordinate(coords, axis, line_idx),
            "xi_index": xi_item.index,
            "xi_value": xi_item.value,
            "pred_left_boundary": pred_left,
            "pred_right_boundary": pred_right,
            "boundary_abs_max": max(abs(pred_left), abs(pred_right)),
            "diagonal_value": float(pred_slice[xi_item.index].detach().cpu().item()),
            "ref_left_boundary": "",
            "ref_right_boundary": "",
            "slice_rel_error": "",
        }
        if ref_slice is not None:
            residual = pred_slice - ref_slice
            num = integrate(residual.pow(2), x=x_axis, dim=-1, rule=integration_rule)
            den = integrate(
                ref_slice.pow(2), x=x_axis, dim=-1, rule=integration_rule
            ).clamp_min(1.0e-12)
            row.update(
                {
                    "ref_left_boundary": float(ref_slice[0].detach().cpu().item()),
                    "ref_right_boundary": float(ref_slice[-1].detach().cpu().item()),
                    "slice_rel_error": float(torch.sqrt(num / den).item()),
                }
            )
        return row

    def _write_figures(
        self,
        coords: Tensor,
        source: Tensor,
        solution: Tensor,
        reconstruction: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        selected_lines: tuple[int, ...],
        selected_xi: tuple[SelectedXi, ...],
        selected_samples: tuple[int, ...],
    ) -> None:
        for axis in range(kernel.shape[0]):
            for line_idx in selected_lines:
                self._save_green_heatmap(
                    kernel=kernel,
                    exact_kernel=exact_kernel,
                    axis=axis,
                    line_idx=line_idx,
                )
                self._save_coefficient_figure(
                    coords=coords,
                    a_vals=a_vals,
                    ap_vals=ap_vals,
                    b_vals=b_vals,
                    c_vals=c_vals,
                    axis=axis,
                    line_idx=line_idx,
                )
                for xi_item in selected_xi:
                    self._save_green_slice(
                        coords=coords,
                        kernel=kernel,
                        exact_kernel=exact_kernel,
                        axis=axis,
                        line_idx=line_idx,
                        xi_item=xi_item,
                    )
                for sample_idx in selected_samples:
                    self._save_reconstruction_figure(
                        coords=coords,
                        source=source,
                        solution=solution,
                        reconstruction=reconstruction,
                        sample_idx=sample_idx,
                        axis=axis,
                        line_idx=line_idx,
                    )

    def _save_green_heatmap(
        self,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        axis: int,
        line_idx: int,
    ) -> None:
        x_axis = torch.linspace(
            0.0,
            1.0,
            kernel.shape[-1],
            dtype=kernel.dtype,
            device=kernel.device,
        )
        base = self.request.outdir / "figures" / "green_heatmaps"
        pred = kernel[axis, line_idx]
        self._save_heatmap_figure(
            z=pred,
            x_axis=x_axis,
            title=(
                f"Predicted Green kernel axis={axis} line={line_idx} "
                "(diagonal/kink behavior)"
            ),
            base_path=base / f"axis{axis}_line{line_idx:03d}_green_heatmap_pred",
        )
        if exact_kernel is not None:
            ref = exact_kernel[axis, line_idx]
            self._save_heatmap_figure(
                z=ref,
                x_axis=x_axis,
                title=f"Reference Green kernel axis={axis} line={line_idx}",
                base_path=base / f"axis{axis}_line{line_idx:03d}_green_heatmap_ref",
            )
            self._save_heatmap_figure(
                z=pred - ref,
                x_axis=x_axis,
                title=f"Green kernel error axis={axis} line={line_idx}",
                base_path=base / f"axis{axis}_line{line_idx:03d}_green_heatmap_error",
            )

    def _save_heatmap_figure(
        self,
        z: Tensor,
        x_axis: Tensor,
        title: str,
        base_path: Path,
    ) -> None:
        axis_np = _tensor_to_numpy(x_axis)
        fig = go.Figure(
            data=go.Heatmap(
                z=_tensor_to_numpy(z),
                x=axis_np,
                y=axis_np,
                colorscale="Viridis",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=axis_np,
                y=axis_np,
                mode="lines",
                name="x=xi",
                line=dict(color="white", dash="dash", width=2),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title="xi",
            yaxis_title="x",
            template=self.request.theme,
            xaxis=dict(constrain="domain"),
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
                constrain="domain",
            ),
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _save_green_slice(
        self,
        coords: Tensor,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        axis: int,
        line_idx: int,
        xi_item: SelectedXi,
    ) -> None:
        x_axis = coords[axis, line_idx, :, axis]
        pred_slice = kernel[axis, line_idx, :, xi_item.index]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=_tensor_to_numpy(x_axis),
                y=_tensor_to_numpy(pred_slice),
                mode="lines+markers",
                name="predicted",
            )
        )
        if exact_kernel is not None:
            ref_slice = exact_kernel[axis, line_idx, :, xi_item.index]
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(x_axis),
                    y=_tensor_to_numpy(ref_slice),
                    mode="lines",
                    name="reference",
                    line=dict(dash="dash"),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[float(x_axis[0].item()), float(x_axis[-1].item())],
                y=[float(pred_slice[0].item()), float(pred_slice[-1].item())],
                mode="markers",
                name="boundary values",
                marker=dict(size=10, symbol="x"),
            )
        )
        fig.add_vline(
            x=xi_item.value,
            line=dict(color="black", dash="dot"),
            annotation_text="x=xi",
        )
        fig.update_layout(
            title=(
                f"Fixed-xi Green slice axis={axis} line={line_idx} "
                f"xi={xi_item.value:.4f}"
            ),
            xaxis_title="x",
            yaxis_title="G(x, xi)",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "green_slices"
            / f"axis{axis}_line{line_idx:03d}_xi{xi_item.index:03d}_green_slice"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _save_coefficient_figure(
        self,
        coords: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        axis: int,
        line_idx: int,
    ) -> None:
        line_coord = coords[axis, line_idx, :, axis]
        b_name = "bx" if axis == 0 else "by"
        ap_name = "apx" if axis == 0 else "apy"
        fig = go.Figure()
        for name, values in (
            ("a", a_vals[axis, line_idx]),
            (ap_name, ap_vals[axis, line_idx]),
            (b_name, b_vals[axis, line_idx]),
            ("c", c_vals[axis, line_idx]),
        ):
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(line_coord),
                    y=_tensor_to_numpy(values),
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=f"Coefficient slices axis={axis} line={line_idx}",
            xaxis_title=self._axis_name(axis),
            yaxis_title="coefficient value",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "coefficients"
            / f"axis{axis}_line{line_idx:03d}_coefficients"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _save_reconstruction_figure(
        self,
        coords: Tensor,
        source: Tensor,
        solution: Tensor,
        reconstruction: Tensor,
        sample_idx: int,
        axis: int,
        line_idx: int,
    ) -> None:
        line_coord = coords[axis, line_idx, :, axis]
        exact = solution[sample_idx, axis, line_idx]
        pred = reconstruction[sample_idx, axis, line_idx]
        fig = go.Figure()
        for name, values in (
            ("source f", source[sample_idx, axis, line_idx]),
            ("reference u", exact),
            ("reconstructed u", pred),
            ("error u-u_hat", exact - pred),
        ):
            fig.add_trace(
                go.Scatter(
                    x=_tensor_to_numpy(line_coord),
                    y=_tensor_to_numpy(values),
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=(
                f"Green reconstruction sample={sample_idx} axis={axis} line={line_idx}"
            ),
            xaxis_title=self._axis_name(axis),
            yaxis_title="value",
            template=self.request.theme,
        )
        base_path = (
            self.request.outdir
            / "figures"
            / "reconstruction"
            / f"sample{sample_idx:03d}_axis{axis}_line{line_idx:03d}_reconstruction"
        )
        save_plotly_figure(fig, base_path, self.logger)

    def _write_raw_data(
        self,
        dataset: AxialDataset,
        kernel: Tensor,
        exact_kernel: Tensor | None,
        reconstruction: Tensor,
        selected_lines: tuple[int, ...],
        selected_samples: tuple[int, ...],
    ) -> None:
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_dir / "generated_eval_data.npz",
            coords=_tensor_to_numpy(dataset.coords),
            solution=_tensor_to_numpy(dataset.solutions),
            source=_tensor_to_numpy(dataset.sources),
            a_vals=_tensor_to_numpy(dataset.a_vals),
            ap_vals=_tensor_to_numpy(dataset.ap_vals),
            b_vals=_tensor_to_numpy(dataset.b_vals),
            c_vals=_tensor_to_numpy(dataset.c_vals),
        )

        selected_pairs = [
            (axis, line_idx)
            for axis in range(kernel.shape[0])
            for line_idx in selected_lines
        ]
        selected_pred = torch.stack(
            [kernel[axis, line_idx] for axis, line_idx in selected_pairs], dim=0
        )
        selected_ref = (
            None
            if exact_kernel is None
            else torch.stack(
                [exact_kernel[axis, line_idx] for axis, line_idx in selected_pairs],
                dim=0,
            )
        )
        kernel_axes = np.array([axis for axis, _line in selected_pairs], dtype=np.int64)
        kernel_line_indices = np.array(
            [line for _axis, line in selected_pairs], dtype=np.int64
        )
        if selected_ref is None:
            np.savez_compressed(
                data_dir / "selected_green_kernels.npz",
                axes=kernel_axes,
                line_indices=kernel_line_indices,
                predicted=_tensor_to_numpy(selected_pred),
            )
        else:
            np.savez_compressed(
                data_dir / "selected_green_kernels.npz",
                axes=kernel_axes,
                line_indices=kernel_line_indices,
                predicted=_tensor_to_numpy(selected_pred),
                reference=_tensor_to_numpy(selected_ref),
                error=_tensor_to_numpy(selected_pred - selected_ref),
            )

        sample_pairs = [
            (sample_idx, axis, line_idx)
            for sample_idx in selected_samples
            for axis in range(reconstruction.shape[1])
            for line_idx in selected_lines
        ]
        selected_reconstruction = _tensor_to_numpy(
            torch.stack(
                [
                    reconstruction[sample, axis, line]
                    for sample, axis, line in sample_pairs
                ],
                dim=0,
            )
        )
        selected_solution = _tensor_to_numpy(
            torch.stack(
                [
                    dataset.solutions[sample, axis, line]
                    for sample, axis, line in sample_pairs
                ],
                dim=0,
            )
        )
        selected_source = _tensor_to_numpy(
            torch.stack(
                [
                    dataset.sources[sample, axis, line]
                    for sample, axis, line in sample_pairs
                ],
                dim=0,
            )
        )
        np.savez_compressed(
            data_dir / "selected_reconstructions.npz",
            sample_indices=np.array(
                [sample for sample, _axis, _line in sample_pairs], dtype=np.int64
            ),
            axes=np.array(
                [axis for _sample, axis, _line in sample_pairs], dtype=np.int64
            ),
            line_indices=np.array(
                [line for _sample, _axis, line in sample_pairs], dtype=np.int64
            ),
            reconstruction=selected_reconstruction,
            solution=selected_solution,
            source=selected_source,
            error=selected_solution - selected_reconstruction,
        )


def export_green_artifacts(
    request: GreenArtifactRequest,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    return GreenArtifactExporter(request=request, logger=logger).export()
