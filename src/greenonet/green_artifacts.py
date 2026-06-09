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
from greenonet.config import CompileConfig, DatasetConfig, ModelConfig, TrainingConfig
from greenonet.data import AxialDataset
from greenonet.greens import ExactGreenFunction
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.numerics import IntegrationRule, integrate
from greenonet.plotly_io import save_plotly_figure
from greenonet.sampler import ForwardSampler


ScaleLength = float | tuple[float, float]
EvalSplit = Literal["train_like", "validation_like", "custom"]
SamplerMode = Literal["forward", "backward"]


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


def _parse_scale_length(raw: object, field_name: str) -> ScaleLength:
    if isinstance(raw, bool):
        raise TypeError(f"{field_name} must be a number or two-number list.")
    if isinstance(raw, int | float):
        return float(raw)
    if isinstance(raw, list | tuple) and len(raw) == 2:
        left, right = raw
        if isinstance(left, bool) or isinstance(right, bool):
            raise TypeError(f"{field_name} entries must be numbers.")
        if not isinstance(left, int | float) or not isinstance(right, int | float):
            raise TypeError(f"{field_name} entries must be numbers.")
        return (float(left), float(right))
    raise TypeError(f"{field_name} must be a number or two-number list.")


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
    dataset_kwargs = dict(raw_dataset)
    dataset_kwargs.pop("domain", None)
    dataset_kwargs["dtype"] = _parse_dtype(dataset_kwargs.pop("dtype", "float64"))
    dataset_kwargs["scale_length"] = _parse_scale_length(
        dataset_kwargs.get("scale_length", 0.1), "dataset.scale_length"
    )
    validation_scale = dataset_kwargs.get("validation_scale_length")
    if validation_scale is not None:
        dataset_kwargs["validation_scale_length"] = _parse_scale_length(
            validation_scale, "dataset.validation_scale_length"
        )
    for key in ("training_path", "validation_path", "test_path"):
        if dataset_kwargs.get(key) is not None:
            dataset_kwargs[key] = Path(cast(str, dataset_kwargs[key]))
    if dataset_kwargs.get("coefficient_functions_path") is not None:
        dataset_kwargs["coefficient_functions_path"] = Path(
            cast(str, dataset_kwargs["coefficient_functions_path"])
        )
    dataset_cfg = DatasetConfig(**dataset_kwargs)

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

        rel_green_valid = self._is_diffusion_only(b_vals_all, c_vals_all)
        rel_green_skip_reason = None
        exact_kernel: Tensor | None = None
        rel_green_by_line: Tensor | None = None
        if rel_green_valid:
            exact_kernel = self._exact_green_kernel(coords=coords, a_vals=a_vals)
            rel_green_by_line = self._relative_green_error_by_line(
                prediction=kernel,
                exact_kernel=exact_kernel,
                x_axis=trunk_grid[:, 0, 0],
                integration_rule=training_cfg.integration_rule,
            )
        else:
            rel_green_skip_reason = (
                "Skipped because sampled convection b_vals or reaction c_vals "
                "are nonzero; rel_green is treated as a diffusion-only metric."
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
            "eval_seed": self.request.eval_seed,
            "eval_split": self.request.eval_split,
            "eval_sampling": _jsonify(asdict(sampling_cfg)),
            "selected_line_indices": list(selected_lines),
            "selected_xi": [asdict(item) for item in selected_xi],
            "selected_sample_indices": list(selected_samples),
            "rel_green_valid": rel_green_valid,
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
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule).clamp_min(
            1.0e-12
        )
        return torch.sqrt(num / den)

    @staticmethod
    def _exact_green_kernel(coords: Tensor, a_vals: Tensor) -> Tensor:
        exact = torch.zeros(
            (
                a_vals.shape[0],
                a_vals.shape[1],
                a_vals.shape[2],
                a_vals.shape[2],
            ),
            device=a_vals.device,
            dtype=a_vals.dtype,
        )
        for axis in range(a_vals.shape[0]):
            for line_idx in range(a_vals.shape[1]):
                x_axis = coords[axis, line_idx, :, axis]
                exact[axis, line_idx] = ExactGreenFunction(
                    x_axis,
                    a=a_vals[axis, line_idx],
                )()
        return exact

    def _is_diffusion_only(self, b_vals: Tensor, c_vals: Tensor) -> bool:
        b_max = float(b_vals.detach().abs().max().item())
        c_max = float(c_vals.detach().abs().max().item())
        return b_max <= self.ZERO_TOL and c_max <= self.ZERO_TOL

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
                                    rel_sol_by_line[
                                        sample_idx, axis, line_idx
                                    ].item()
                                ),
                                "rel_green_line": (
                                    ""
                                    if rel_green_by_line is None
                                    else float(
                                        rel_green_by_line[axis, line_idx].item()
                                    )
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
        with boundary_path.open("w", newline="") as bfp, slice_path.open(
            "w", newline=""
        ) as sfp:
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
            "diagonal_value": float(
                pred_slice[xi_item.index].detach().cpu().item()
            ),
            "ref_left_boundary": "",
            "ref_right_boundary": "",
            "slice_rel_error": "",
        }
        if ref_slice is not None:
            residual = pred_slice - ref_slice
            num = integrate(
                residual.pow(2), x=x_axis, dim=-1, rule=integration_rule
            )
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
                f"Green reconstruction sample={sample_idx} "
                f"axis={axis} line={line_idx}"
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
        kernel_axes = np.array(
            [axis for axis, _line in selected_pairs], dtype=np.int64
        )
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
