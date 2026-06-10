from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
import plotly.graph_objects as go
import torch
from torch import Tensor
from torch.nn.functional import pad

from greenonet.coefficients import load_coefficient_functions
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CompileConfig,
    CouplingBestRelSolCheckpointConfig,
    CouplingCoefficientTermsConfig,
    CouplingLossesConfig,
    CouplingLossTermConfig,
    CouplingModelConfig,
    CouplingPeriodicCheckpointConfig,
    CouplingTrainingConfig,
    CouplingTrunkPositionalEncodingConfig,
    DatasetConfig,
    GreenResponseFeatureConfig,
    ModelConfig,
    SourceStencilLiftConfig,
    TrainingConfig,
)
from greenonet.coupling_data import CouplingDataset, split_coupling_batch
from greenonet.coupling_model import CouplingNet
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.numerics import IntegrationRule, integrate
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class CouplingArtifactRequest:
    config: Path
    coupling_checkpoint: Path
    green_checkpoint: Path
    outdir: Path
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    selected_samples: tuple[int, ...] | None = None
    max_samples: int | None = 3
    plot_workers: int = 1
    save_generated_data: bool = True


@dataclass(frozen=True)
class CouplingArtifactConfigs:
    dataset: DatasetConfig
    green_model: ModelConfig
    green_training: TrainingConfig
    coupling_model: CouplingModelConfig
    coupling_training: CouplingTrainingConfig
    raw: dict[str, Any]


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: int
    file_stem: str
    source_grid: Tensor
    source_grid_policy: str
    solution_grids: dict[str, Tensor]
    flux_grids: dict[str, Tensor]
    coefficient_grids: dict[str, Tensor]
    pred_sol_components: Tensor
    sol_components: Tensor
    pred_flux_components: Tensor
    flux_components: Tensor


def _parse_dtype(raw: object | None) -> torch.dtype:
    if raw is None:
        return torch.float64
    if not isinstance(raw, str):
        raise TypeError("dtype must be a string.")
    return cast(torch.dtype, getattr(torch, raw.replace("torch.", "")))


def _parse_scale_length(raw: object, field_name: str) -> float | tuple[float, float]:
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


def _build_compile_config(raw: object | None, section_name: str) -> CompileConfig:
    if raw is None:
        return CompileConfig()
    if isinstance(raw, dict):
        return CompileConfig(**raw)
    raise TypeError(f"{section_name}.compile must be an object.")


def _build_coupling_losses_config(raw: object | None) -> CouplingLossesConfig:
    if raw is None:
        return CouplingLossesConfig()
    if not isinstance(raw, dict):
        raise TypeError("coupling_training.losses must be an object.")
    loss_kwargs = dict(raw)
    defaults = CouplingLossesConfig()
    parsed: dict[str, CouplingLossTermConfig] = {}
    for key in (
        "l2_consistency",
        "energy_consistency",
        "cross_consistency",
        "balance_loss",
        "symmetric_boundary_loss",
    ):
        raw_term = loss_kwargs.pop(key, None)
        if raw_term is None:
            parsed[key] = getattr(defaults, key)
        elif isinstance(raw_term, dict):
            parsed[key] = CouplingLossTermConfig(**raw_term)
        else:
            raise TypeError(f"coupling_training.losses.{key} must be an object.")
    if loss_kwargs:
        unknown = ", ".join(sorted(loss_kwargs))
        raise TypeError(f"coupling_training.losses has unknown keys: {unknown}.")
    return CouplingLossesConfig(**parsed)


def load_coupling_artifact_configs(config_path: Path) -> CouplingArtifactConfigs:
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
    green_model_cfg = ModelConfig(**model_kwargs)

    raw_training = raw_payload.get("training", {})
    if not isinstance(raw_training, dict):
        raise TypeError("training config section must be an object when provided.")
    training_kwargs = dict(raw_training)
    compile_cfg = _build_compile_config(training_kwargs.pop("compile", None), "training")
    green_training_cfg = TrainingConfig(compile=compile_cfg, **training_kwargs)

    raw_coupling_model = raw_payload.get("coupling_model", {})
    if not isinstance(raw_coupling_model, dict):
        raise TypeError("coupling_model config section must be an object.")
    coupling_model_kwargs = dict(raw_coupling_model)
    source_lift_raw = coupling_model_kwargs.pop("source_stencil_lift", None)
    coefficient_terms_raw = coupling_model_kwargs.pop("coefficient_terms", None)
    green_response_raw = coupling_model_kwargs.pop("green_response_feature", None)
    positional_raw = coupling_model_kwargs.pop("trunk_positional_encoding", None)
    axis_1d_trunk_raw = coupling_model_kwargs.pop("axis_1d_trunk", None)
    balance_projection_raw = coupling_model_kwargs.pop("balance_projection", None)
    coupling_model_kwargs["dtype"] = _parse_dtype(
        coupling_model_kwargs.pop("dtype", "float64")
    )
    coupling_model_cfg = CouplingModelConfig(
        balance_projection=BalanceProjectionConfig.from_raw(balance_projection_raw),
        source_stencil_lift=(
            SourceStencilLiftConfig()
            if source_lift_raw is None
            else SourceStencilLiftConfig(**cast(dict[str, Any], source_lift_raw))
        ),
        coefficient_terms=(
            CouplingCoefficientTermsConfig()
            if coefficient_terms_raw is None
            else CouplingCoefficientTermsConfig(
                **cast(dict[str, Any], coefficient_terms_raw)
            )
        ),
        green_response_feature=(
            GreenResponseFeatureConfig()
            if green_response_raw is None
            else GreenResponseFeatureConfig(**cast(dict[str, Any], green_response_raw))
        ),
        trunk_positional_encoding=(
            CouplingTrunkPositionalEncodingConfig()
            if positional_raw is None
            else CouplingTrunkPositionalEncodingConfig(
                **cast(dict[str, Any], positional_raw)
            )
        ),
        axis_1d_trunk=Axis1DTrunkConfig.from_raw(axis_1d_trunk_raw),
        **coupling_model_kwargs,
    )

    raw_coupling_training = raw_payload.get("coupling_training", {})
    if not isinstance(raw_coupling_training, dict):
        raise TypeError("coupling_training config section must be an object.")
    coupling_training_kwargs = dict(raw_coupling_training)
    losses_cfg = _build_coupling_losses_config(coupling_training_kwargs.pop("losses", None))
    compile_cfg = _build_compile_config(
        coupling_training_kwargs.pop("compile", None), "coupling_training"
    )
    periodic_raw = coupling_training_kwargs.pop("periodic_checkpoint", None)
    best_raw = coupling_training_kwargs.pop("best_rel_sol_checkpoint", None)
    periodic_cfg = (
        CouplingPeriodicCheckpointConfig()
        if periodic_raw is None
        else CouplingPeriodicCheckpointConfig(**cast(dict[str, Any], periodic_raw))
    )
    best_cfg = (
        CouplingBestRelSolCheckpointConfig()
        if best_raw is None
        else CouplingBestRelSolCheckpointConfig(**cast(dict[str, Any], best_raw))
    )
    coupling_training_cfg = CouplingTrainingConfig(
        losses=losses_cfg,
        compile=compile_cfg,
        periodic_checkpoint=periodic_cfg,
        best_rel_sol_checkpoint=best_cfg,
        **coupling_training_kwargs,
    )

    return CouplingArtifactConfigs(
        dataset=dataset_cfg,
        green_model=green_model_cfg,
        green_training=green_training_cfg,
        coupling_model=coupling_model_cfg,
        coupling_training=coupling_training_cfg,
        raw=raw_payload,
    )


class CouplingArtifactExporter:
    """Export paper-facing CouplingNet metrics and selected sample figures."""

    def __init__(
        self,
        request: CouplingArtifactRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.integration_rule: IntegrationRule = "simpson"

    def export(self) -> dict[str, object]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        self._ensure_output_tree()
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.test_path is None:
            raise ValueError("dataset.test_path must be set for Coupling artifact export.")
        coeff_path = self.request.coefficients or configs.dataset.coefficient_functions_path
        coeffs = load_coefficient_functions(coeff_path)
        device = self._resolve_device(self.request.device or configs.coupling_training.device)

        dataset = CouplingDataset(
            data_dir=configs.dataset.test_path,
            step_size=configs.dataset.step_size,
            n_points_per_line=configs.dataset.n_points_per_line,
            dtype=configs.dataset.dtype,
            integration_rule=configs.coupling_training.integration_rule,
            a_fun=coeffs.a_fun,
            bx_fun=coeffs.bx_fun,
            by_fun=coeffs.by_fun,
            c_fun=coeffs.c_fun,
            ap_fun_x=coeffs.apx_fun,
            ap_fun_y=coeffs.apy_fun,
        )
        selected_indices = self._select_sample_indices(len(dataset))

        coupling_model = self._load_coupling_model(configs.coupling_model)
        green_model = self._load_green_model(configs.green_model)
        coupling_model.to(device)
        green_model.to(device)
        coupling_model.eval()
        green_model.eval()

        first_core, _first_boundary = split_coupling_batch(tuple(dataset[0]))
        sample_coords = first_core[0].to(device)
        sample_a = first_core[6].to(device)
        sample_b = first_core[7].to(device)
        sample_c = first_core[8].to(device)
        sample_ap = first_core[9].to(device)
        green_kernel = self._compute_green_kernel(
            green_model=green_model,
            coords=sample_coords,
            a_vals=sample_a,
            ap_vals=sample_ap,
            b_vals=sample_b,
            c_vals=sample_c,
            dtype=configs.dataset.dtype,
            device=device,
        )

        metric_rows: list[dict[str, object]] = []
        balance_rows: list[dict[str, object]] = []
        boundary_rows: list[dict[str, object]] = []
        selected_records: list[SampleEvaluation] = []
        figure_paths: list[str] = []
        source_policies: set[str] = set()
        was_training = coupling_model.training
        try:
            with torch.no_grad():
                for sample_id in range(len(dataset)):
                    item = tuple(dataset[sample_id])
                    core, _boundary = split_coupling_batch(item)
                    evaluation = self._evaluate_sample(
                        sample_id=sample_id,
                        item=item,
                        dataset=dataset,
                        coupling_model=coupling_model,
                        green_kernel=green_kernel,
                        integration_rule=configs.coupling_training.integration_rule,
                        device=device,
                    )
                    source_policies.add(evaluation.source_grid_policy)
                    sample_metrics = self._sample_metrics(
                        evaluation,
                        x_axis=core[0][0, 0, :, 0].to(device),
                        integration_rule=configs.coupling_training.integration_rule,
                    )
                    metric_rows.append(sample_metrics["sample"])
                    balance_rows.append(sample_metrics["balance"])
                    boundary_rows.append(sample_metrics["boundary"])
                    if sample_id in selected_indices:
                        selected_records.append(evaluation)
                        figure_paths.extend(self._write_sample_figures(evaluation))
        finally:
            coupling_model.train(was_training)

        self._write_metrics(metric_rows, balance_rows, boundary_rows)
        if self.request.save_generated_data:
            self._write_raw_data(selected_records)

        aggregate = self._write_aggregate_metrics(metric_rows)
        balance_projection = BalanceProjectionConfig.from_raw(
            configs.coupling_model.balance_projection
        )
        summary: dict[str, object] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": str(self.request.config),
            "coupling_checkpoint": str(self.request.coupling_checkpoint),
            "green_checkpoint": str(self.request.green_checkpoint),
            "coefficients": str(coeff_path) if coeff_path is not None else None,
            "test_path": str(configs.dataset.test_path),
            "outdir": str(self.request.outdir),
            "device": str(device),
            "theme": self.request.theme,
            "integration_rule": configs.coupling_training.integration_rule,
            "selected_samples": list(selected_indices),
            "plot_workers": self.request.plot_workers,
            "save_generated_data": self.request.save_generated_data,
            "source_grid_policy": sorted(source_policies),
            "figures": figure_paths,
            "aggregate_metrics": aggregate,
            "balance_projection": _jsonify(asdict(balance_projection)),
            "losses": _jsonify(asdict(configs.coupling_training.losses)),
            "coupling_model": _jsonify(asdict(configs.coupling_model)),
            "raw_config": _jsonify(configs.raw),
        }
        summary_path = self.request.outdir / "summary.json"
        summary_path.write_text(json.dumps(_jsonify(summary), indent=2) + "\n")
        if self.logger is not None:
            self.logger.info("Saved CouplingNet artifact summary to %s", summary_path)
        return summary

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is not available.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested {device}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available."
                )
        return device

    def _load_coupling_model(self, model_cfg: CouplingModelConfig) -> torch.nn.Module:
        try:
            loaded_model, _loaded_cfg = load_model_with_config(
                self.request.coupling_checkpoint
            )
        except Exception:
            model = CouplingNet(model_cfg)
            load_state_dict_auto(model, self.request.coupling_checkpoint)
            return model
        if not isinstance(loaded_model, CouplingNet):
            raise TypeError("Coupling checkpoint metadata does not describe CouplingNet.")
        return loaded_model

    def _load_green_model(self, model_cfg: ModelConfig) -> GreenONetModel:
        try:
            loaded_model, _loaded_cfg = load_model_with_config(self.request.green_checkpoint)
        except Exception:
            model = GreenONetModel(model_cfg)
            load_state_dict_auto(model, self.request.green_checkpoint)
            return model
        if not isinstance(loaded_model, GreenONetModel):
            raise TypeError("Green checkpoint metadata does not describe GreenONetModel.")
        return loaded_model

    @staticmethod
    def _compute_green_kernel(
        green_model: GreenONetModel,
        coords: Tensor,
        a_vals: Tensor,
        ap_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        m_points = coords.shape[2]
        trunk_grid = torch.stack(
            torch.meshgrid(
                torch.linspace(0.0, 1.0, m_points, device=device, dtype=dtype),
                torch.linspace(0.0, 1.0, m_points, device=device, dtype=dtype),
                indexing="ij",
            ),
            dim=-1,
        )
        return cast(
            Tensor,
            green_model(
                trunk_grid=trunk_grid,
                a_vals=a_vals.to(device),
                ap_vals=ap_vals.to(device),
                b_vals=b_vals.to(device),
                c_vals=c_vals.to(device),
            ),
        ).detach()

    def _select_sample_indices(self, dataset_len: int) -> tuple[int, ...]:
        if dataset_len <= 0:
            raise ValueError("Coupling artifact export requires at least one sample.")
        if self.request.selected_samples is not None:
            requested = list(self.request.selected_samples)
        else:
            max_samples = (
                self.request.max_samples
                if self.request.max_samples is not None
                else 3
            )
            if max_samples <= 0:
                raise ValueError("--max-samples must be positive when provided.")
            requested = list(range(min(max_samples, dataset_len)))
        selected: list[int] = []
        for index in requested:
            if index < 0 or index >= dataset_len:
                raise ValueError(
                    f"Selected sample index {index} is out of range for "
                    f"{dataset_len} sample(s)."
                )
            if index not in selected:
                selected.append(index)
        return tuple(selected)

    def _integrate(self, green: Tensor, values: Tensor, x_axis: Tensor) -> Tensor:
        weighted = values.unsqueeze(-2) * green.unsqueeze(0)
        return integrate(weighted, x=x_axis, dim=-1, rule=self.integration_rule)

    def _green_response_feature_enabled(self, model: torch.nn.Module) -> bool:
        module = getattr(model, "_orig_mod", model)
        return bool(getattr(module, "green_response_feature_enabled", False))

    def _green_response_tilde(
        self,
        model: torch.nn.Module,
        green_kernel: Tensor,
        rhs_tilde: Tensor,
        x_axis: Tensor,
        y_axis: Tensor,
    ) -> Tensor | None:
        if not self._green_response_feature_enabled(model):
            return None
        response = torch.empty_like(rhs_tilde)
        response[:, 0] = self._integrate(green_kernel[0], rhs_tilde[:, 0], x_axis)
        response[:, 1] = self._integrate(green_kernel[1], rhs_tilde[:, 1], y_axis)
        return response

    def _evaluate_sample(
        self,
        sample_id: int,
        item: tuple[object, ...],
        dataset: CouplingDataset,
        coupling_model: torch.nn.Module,
        green_kernel: Tensor,
        integration_rule: IntegrationRule,
        device: torch.device,
    ) -> SampleEvaluation:
        self.integration_rule = integration_rule
        core, _boundary = split_coupling_batch(item)
        coords, rhs_raw, rhs_tilde, rhs_norm, sol, flux, a_vals, b_vals, c_vals, ap = core
        del ap
        x_axis = coords[0, 0, :, 0].to(device)
        y_axis = coords[1, 0, :, 1].to(device)
        model_inputs: dict[str, Tensor] = {
            "coords": coords.to(device),
            "a_vals": a_vals.unsqueeze(0).to(device),
            "b_vals": b_vals.unsqueeze(0).to(device),
            "c_vals": c_vals.unsqueeze(0).to(device),
            "rhs_raw": rhs_raw.unsqueeze(0).to(device),
            "rhs_tilde": rhs_tilde.unsqueeze(0).to(device),
            "rhs_norm": rhs_norm.unsqueeze(0).to(device),
        }
        green_response = self._green_response_tilde(
            coupling_model,
            green_kernel,
            model_inputs["rhs_tilde"],
            x_axis,
            y_axis,
        )
        if green_response is not None:
            model_inputs["green_response_tilde"] = green_response
        pred_flux_raw = cast(Tensor, coupling_model(**model_inputs))
        phi_lines = pred_flux_raw[:, 0].clone()
        psi_lines = pred_flux_raw[:, 1].clone()
        phi_lines[..., 0] = 0.0
        phi_lines[..., -1] = 0.0
        psi_lines[..., 0] = 0.0
        psi_lines[..., -1] = 0.0

        pred_sol_x = self._integrate(green_kernel[0], phi_lines, x_axis)
        pred_sol_y = self._integrate(green_kernel[1], psi_lines, y_axis)
        pred_sol_x = pad(pred_sol_x[..., 1:-1], pad=(1, 1, 1, 1), value=0.0)
        pred_sol_y = pad(pred_sol_y[..., 1:-1], pad=(1, 1, 1, 1), value=0.0)
        pred_flux = torch.stack((phi_lines, psi_lines), dim=1)
        pred_flux = pad(pred_flux[..., 1:-1], pad=(1, 1, 1, 1), value=0.0)
        exact_flux = pad(flux.unsqueeze(0).to(device)[..., 1:-1], pad=(1, 1, 1, 1), value=0.0)
        exact_sol = pad(sol.unsqueeze(0).to(device)[..., 1:-1], pad=(1, 1, 1, 1), value=0.0)

        file_stem = ""
        sample_path: Path | None = None
        if sample_id < len(dataset.files):
            sample_path = dataset.files[sample_id]
            file_stem = sample_path.stem

        source_grid, source_policy = self._source_grid(
            sample_path=sample_path,
            rhs_raw=rhs_raw,
        )
        source_grid = source_grid.to(dtype=exact_sol.dtype, device=device)
        solution_grids = self._solution_grids(
            sol_components=exact_sol[0],
            pred_sol_x=pred_sol_x[0],
            pred_sol_y=pred_sol_y[0],
        )
        flux_grids = self._flux_grids(
            flux_components=exact_flux[0],
            pred_flux_components=pred_flux[0],
            source_grid=source_grid,
        )
        coefficient_grids = self._coefficient_grids(a_vals, b_vals, c_vals)

        return SampleEvaluation(
            sample_id=sample_id,
            file_stem=file_stem or f"sample_{sample_id}",
            source_grid=source_grid.detach().cpu(),
            source_grid_policy=source_policy,
            solution_grids={key: value.detach().cpu() for key, value in solution_grids.items()},
            flux_grids={key: value.detach().cpu() for key, value in flux_grids.items()},
            coefficient_grids={
                key: value.detach().cpu() for key, value in coefficient_grids.items()
            },
            pred_sol_components=torch.stack((pred_sol_x[0], pred_sol_y[0]), dim=0).detach(),
            sol_components=exact_sol[0].detach(),
            pred_flux_components=pred_flux[0].detach(),
            flux_components=exact_flux[0].detach(),
        )

    @staticmethod
    def _source_grid(
        sample_path: Path | None,
        rhs_raw: Tensor,
    ) -> tuple[Tensor, str]:
        m_points = rhs_raw.shape[-1]
        if sample_path is not None:
            with np.load(sample_path) as data:
                if "rhs" in data:
                    rhs = data["rhs"]
                    if rhs.ndim == 2 and rhs.shape[0] == rhs.shape[1]:
                        idx = np.round(
                            np.linspace(0, rhs.shape[0] - 1, m_points)
                        ).astype(int)
                        grid = rhs[np.ix_(idx, idx)]
                        return torch.from_numpy(grid), "npz_rhs_resampled_to_model_grid"
        interior = 0.5 * (rhs_raw[0, :, 1:-1] + rhs_raw[1, :, 1:-1].transpose(-1, -2))
        grid = torch.full(
            (m_points, m_points),
            float("nan"),
            dtype=rhs_raw.dtype,
            device=rhs_raw.device,
        )
        grid[1:-1, 1:-1] = interior
        return grid, "axis_line_average_with_nan_boundary"

    @staticmethod
    def _solution_grids(
        sol_components: Tensor,
        pred_sol_x: Tensor,
        pred_sol_y: Tensor,
    ) -> dict[str, Tensor]:
        u_x = sol_components[0]
        u_y = sol_components[1].transpose(-1, -2)
        u = 0.5 * (u_x + u_y)
        u_pred_x = pred_sol_x
        u_pred_y = pred_sol_y.transpose(-1, -2)
        u_pred = 0.5 * (u_pred_x + u_pred_y)
        return {
            "u": u,
            "u_pred": u_pred,
            "u_pred_x": u_pred_x,
            "u_pred_y": u_pred_y,
            "u_error": u_pred - u,
            "u_pred_x_error": u_pred_x - u,
            "u_pred_y_error": u_pred_y - u,
            "u_pred_x_minus_u_pred_y": u_pred_x - u_pred_y,
        }

    @staticmethod
    def _flux_grids(
        flux_components: Tensor,
        pred_flux_components: Tensor,
        source_grid: Tensor,
    ) -> dict[str, Tensor]:
        phi = flux_components[0]
        psi = flux_components[1].transpose(-1, -2)
        phi_pred = pred_flux_components[0]
        psi_pred = pred_flux_components[1].transpose(-1, -2)
        phi_plus_psi = phi_pred + psi_pred
        return {
            "phi": phi,
            "psi": psi,
            "phi_pred": phi_pred,
            "psi_pred": psi_pred,
            "phi_error": phi_pred - phi,
            "psi_error": psi_pred - psi,
            "phi_plus_psi": phi_plus_psi,
            "balance_residual": source_grid - phi_plus_psi,
        }

    @staticmethod
    def _interior_grid(grid: Tensor) -> Tensor:
        if grid.ndim != 2:
            return grid
        if grid.shape[0] <= 2 or grid.shape[1] <= 2:
            return grid
        return grid[1:-1, 1:-1]

    @staticmethod
    def _line_grid(values: Tensor, *, transpose: bool = False) -> Tensor:
        grid = pad(values[:, 1:-1], pad=(1, 1, 1, 1), value=0.0)
        return grid.transpose(-1, -2) if transpose else grid

    def _coefficient_grids(
        self,
        a_vals: Tensor,
        b_vals: Tensor,
        c_vals: Tensor,
    ) -> dict[str, Tensor]:
        a_x = self._line_grid(a_vals[0])
        a_y = self._line_grid(a_vals[1], transpose=True)
        c_x = self._line_grid(c_vals[0])
        c_y = self._line_grid(c_vals[1], transpose=True)
        return {
            "a": 0.5 * (a_x + a_y),
            "bx": self._line_grid(b_vals[0]),
            "by": self._line_grid(b_vals[1], transpose=True),
            "c": 0.5 * (c_x + c_y),
        }

    def _relative_l2_integral(
        self,
        pred: Tensor,
        target: Tensor,
        x_axis: Tensor,
        integration_rule: IntegrationRule,
        eps: float = 1e-12,
    ) -> float:
        pred = pred.to(x_axis.device)
        target = target.to(x_axis.device)
        num = integrate((pred - target).pow(2), x=x_axis, dim=-1, rule=integration_rule)
        num = integrate(num, x=x_axis, dim=-1, rule=integration_rule).mean()
        den = integrate(target.pow(2), x=x_axis, dim=-1, rule=integration_rule)
        den = integrate(den, x=x_axis, dim=-1, rule=integration_rule).mean().clamp_min(eps)
        return float(torch.sqrt(num / den).item())

    def _field_l2(self, field: Tensor, x_axis: Tensor, integration_rule: IntegrationRule) -> float:
        clean = torch.nan_to_num(field.to(x_axis.device), nan=0.0)
        energy = integrate(clean.pow(2), x=x_axis, dim=-1, rule=integration_rule)
        energy = integrate(energy, x=x_axis, dim=-1, rule=integration_rule)
        return float(torch.sqrt(energy).item())

    def _sample_metrics(
        self,
        evaluation: SampleEvaluation,
        x_axis: Tensor,
        integration_rule: IntegrationRule,
    ) -> dict[str, dict[str, object]]:
        rel_sol = self._relative_l2_integral(
            evaluation.pred_sol_components.unsqueeze(0),
            evaluation.sol_components.unsqueeze(0),
            x_axis,
            integration_rule,
        )
        rel_flux = self._relative_l2_integral(
            evaluation.pred_flux_components.unsqueeze(0),
            evaluation.flux_components.unsqueeze(0),
            x_axis,
            integration_rule,
        )
        balance = evaluation.flux_grids["balance_residual"]
        balance_l2 = self._field_l2(balance, x_axis, integration_rule)
        balance_max_abs = float(torch.nan_to_num(balance.abs(), nan=0.0).max().item())
        phi_pred = evaluation.flux_grids["phi_pred"]
        psi_pred = evaluation.flux_grids["psi_pred"]
        boundary_stack = torch.cat(
            [
                phi_pred[0, :],
                phi_pred[-1, :],
                psi_pred[:, 0],
                psi_pred[:, -1],
            ]
        )
        boundary_l2 = float(torch.sqrt(torch.mean(boundary_stack.pow(2))).item())
        boundary_max_abs = float(boundary_stack.abs().max().item())
        base = {
            "sample_id": evaluation.sample_id,
            "file": evaluation.file_stem,
            "rel_sol": rel_sol,
            "rel_flux": rel_flux,
            "balance_l2": balance_l2,
            "balance_max_abs": balance_max_abs,
        }
        return {
            "sample": base,
            "balance": {
                "sample_id": evaluation.sample_id,
                "file": evaluation.file_stem,
                "balance_l2": balance_l2,
                "balance_max_abs": balance_max_abs,
            },
            "boundary": {
                "sample_id": evaluation.sample_id,
                "file": evaluation.file_stem,
                "boundary_l2": boundary_l2,
                "boundary_max_abs": boundary_max_abs,
            },
        }

    @staticmethod
    def _heatmap(
        title: str,
        grid: Tensor,
        theme: str,
        *,
        signed: bool = False,
    ) -> go.Figure:
        z = grid.detach().cpu().numpy()
        z_abs = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.0
        zmin = -z_abs if signed and z_abs > 0.0 else None
        zmax = z_abs if signed and z_abs > 0.0 else None
        return go.Figure(
            data=go.Heatmap(
                z=z,
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu" if signed else "Viridis",
                colorbar=dict(exponentformat="power", showexponent="all"),
            ),
            layout=go.Layout(
                template=theme,
                width=900,
                height=900,
                title=title,
                font={"family": "Times New Roman", "size": 22},
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1),
            ),
        )

    @staticmethod
    def _trace_figure(title: str, y_values: Tensor, theme: str) -> go.Figure:
        x = list(range(y_values.numel()))
        return go.Figure(
            data=go.Scatter(x=x, y=y_values.detach().cpu(), mode="lines"),
            layout=go.Layout(
                template=theme,
                width=900,
                height=500,
                title=title,
                font={"family": "Times New Roman", "size": 22},
                xaxis_title="Grid index",
                yaxis_title="Value",
            ),
        )

    def _write_sample_figures(self, evaluation: SampleEvaluation) -> list[str]:
        stem = f"sample_{evaluation.sample_id:04d}_{evaluation.file_stem}"
        written: list[str] = []
        figure_specs: list[tuple[str, str, Tensor, str, bool, bool]] = [
            ("solution", "f", evaluation.source_grid, "Source f", False, False),
            ("solution", "u", evaluation.solution_grids["u"], "Exact solution u", False, False),
            (
                "solution",
                "u_pred",
                evaluation.solution_grids["u_pred"],
                "Predicted solution u_pred",
                False,
                False,
            ),
            (
                "solution",
                "u_pred_x",
                evaluation.solution_grids["u_pred_x"],
                "Predicted solution u_pred_x",
                False,
                False,
            ),
            (
                "solution",
                "u_pred_y",
                evaluation.solution_grids["u_pred_y"],
                "Predicted solution u_pred_y",
                False,
                False,
            ),
            (
                "solution",
                "u_error",
                evaluation.solution_grids["u_error"],
                "Signed error u_pred - u",
                True,
                False,
            ),
            (
                "solution",
                "u_pred_x_error",
                evaluation.solution_grids["u_pred_x_error"],
                "Signed error u_pred_x - u",
                True,
                False,
            ),
            (
                "solution",
                "u_pred_y_error",
                evaluation.solution_grids["u_pred_y_error"],
                "Signed error u_pred_y - u",
                True,
                False,
            ),
            (
                "solution",
                "u_pred_x_minus_u_pred_y",
                evaluation.solution_grids["u_pred_x_minus_u_pred_y"],
                "Mismatch u_pred_x - u_pred_y",
                True,
                False,
            ),
            ("phi_psi", "phi", evaluation.flux_grids["phi"], "Exact phi", False, True),
            ("phi_psi", "psi", evaluation.flux_grids["psi"], "Exact psi", False, True),
            (
                "phi_psi",
                "phi_pred",
                evaluation.flux_grids["phi_pred"],
                "Predicted phi_pred",
                False,
                True,
            ),
            (
                "phi_psi",
                "psi_pred",
                evaluation.flux_grids["psi_pred"],
                "Predicted psi_pred",
                False,
                True,
            ),
            (
                "phi_psi",
                "phi_error",
                evaluation.flux_grids["phi_error"],
                "Signed error phi_pred - phi",
                True,
                True,
            ),
            (
                "phi_psi",
                "psi_error",
                evaluation.flux_grids["psi_error"],
                "Signed error psi_pred - psi",
                True,
                True,
            ),
            (
                "balance",
                "phi_plus_psi",
                evaluation.flux_grids["phi_plus_psi"],
                "phi + psi",
                False,
                True,
            ),
            (
                "balance",
                "balance_residual",
                evaluation.flux_grids["balance_residual"],
                "Balance residual f - phi - psi",
                True,
                True,
            ),
        ]
        for directory, name, grid, title, signed, crop_boundary in figure_specs:
            plot_grid = self._interior_grid(grid) if crop_boundary else grid
            base_path = self.request.outdir / "figures" / directory / f"{stem}_{name}"
            save_plotly_figure(
                self._heatmap(title, plot_grid, self.request.theme, signed=signed),
                base_path,
                logger=self.logger,
            )
            written.append(str(base_path.with_suffix(".html")))

        for name, grid in evaluation.coefficient_grids.items():
            base_path = self.request.outdir / "figures" / "coefficients" / f"{stem}_{name}"
            save_plotly_figure(
                self._heatmap(f"Coefficient {name}", grid, self.request.theme),
                base_path,
                logger=self.logger,
            )
            written.append(str(base_path.with_suffix(".html")))
        return written

    def _ensure_output_tree(self) -> None:
        for directory in (
            "metrics",
            "data",
            "figures/training_curves",
            "figures/solution",
            "figures/phi_psi",
            "figures/balance",
            "figures/boundary",
            "figures/coefficients",
        ):
            (self.request.outdir / directory).mkdir(parents=True, exist_ok=True)

    def _write_metrics(
        self,
        metric_rows: list[dict[str, object]],
        balance_rows: list[dict[str, object]],
        boundary_rows: list[dict[str, object]],
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(
            metrics_dir / "per_sample_metrics.csv",
            metric_rows,
            ["sample_id", "file", "rel_sol", "rel_flux", "balance_l2", "balance_max_abs"],
        )
        self._write_csv(
            metrics_dir / "balance_residual_metrics.csv",
            balance_rows,
            ["sample_id", "file", "balance_l2", "balance_max_abs"],
        )
        self._write_csv(
            metrics_dir / "boundary_residual_metrics.csv",
            boundary_rows,
            ["sample_id", "file", "boundary_l2", "boundary_max_abs"],
        )

    @staticmethod
    def _write_csv(
        path: Path,
        rows: list[dict[str, object]],
        fieldnames: list[str],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_aggregate_metrics(
        self,
        metric_rows: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        aggregate: dict[str, dict[str, object]] = {}
        rows: list[dict[str, object]] = []
        for metric in ("rel_sol", "rel_flux", "balance_l2", "balance_max_abs"):
            metric_values: list[float] = []
            for row in metric_rows:
                value = row[metric]
                if not isinstance(value, int | float):
                    raise TypeError(f"Metric '{metric}' must be numeric.")
                metric_values.append(float(value))
            values = torch.tensor(metric_values, dtype=torch.float64)
            best_idx = int(torch.argmin(values).item())
            worst_idx = int(torch.argmax(values).item())
            stats = {
                "metric": metric,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=True).item()) if values.numel() > 1 else 0.0,
                "median": float(values.median().item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "p90": float(torch.quantile(values, 0.9).item()),
                "best_sample": metric_rows[best_idx]["sample_id"],
                "worst_sample": metric_rows[worst_idx]["sample_id"],
            }
            aggregate[metric] = dict(stats)
            rows.append(stats)
        self._write_csv(
            metrics_dir / "aggregate_metrics.csv",
            rows,
            [
                "metric",
                "mean",
                "std",
                "median",
                "min",
                "max",
                "p90",
                "best_sample",
                "worst_sample",
            ],
        )
        return aggregate

    def _write_raw_data(self, records: list[SampleEvaluation]) -> None:
        if not records:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        def stack_grid(records_dict: Sequence[dict[str, Tensor]], key: str) -> np.ndarray:
            return torch.stack([record[key] for record in records_dict], dim=0).numpy()

        solution_dicts = [record.solution_grids for record in records]
        flux_dicts = [record.flux_grids for record in records]
        coefficient_dicts = [record.coefficient_grids for record in records]
        np.savez_compressed(
            data_dir / "selected_samples.npz",
            sample_ids=np.array([record.sample_id for record in records], dtype=np.int64),
            source=np.stack([record.source_grid.numpy() for record in records], axis=0),
            u=stack_grid(solution_dicts, "u"),
            phi=stack_grid(flux_dicts, "phi"),
            psi=stack_grid(flux_dicts, "psi"),
            a=stack_grid(coefficient_dicts, "a"),
            bx=stack_grid(coefficient_dicts, "bx"),
            by=stack_grid(coefficient_dicts, "by"),
            c=stack_grid(coefficient_dicts, "c"),
        )
        np.savez_compressed(
            data_dir / "selected_predictions.npz",
            sample_ids=np.array([record.sample_id for record in records], dtype=np.int64),
            u_pred=stack_grid(solution_dicts, "u_pred"),
            u_pred_x=stack_grid(solution_dicts, "u_pred_x"),
            u_pred_y=stack_grid(solution_dicts, "u_pred_y"),
            phi_pred=stack_grid(flux_dicts, "phi_pred"),
            psi_pred=stack_grid(flux_dicts, "psi_pred"),
        )
        np.savez_compressed(
            data_dir / "selected_diagnostics.npz",
            sample_ids=np.array([record.sample_id for record in records], dtype=np.int64),
            u_error=stack_grid(solution_dicts, "u_error"),
            u_pred_x_error=stack_grid(solution_dicts, "u_pred_x_error"),
            u_pred_y_error=stack_grid(solution_dicts, "u_pred_y_error"),
            u_pred_x_minus_u_pred_y=stack_grid(
                solution_dicts, "u_pred_x_minus_u_pred_y"
            ),
            phi_error=stack_grid(flux_dicts, "phi_error"),
            psi_error=stack_grid(flux_dicts, "psi_error"),
            phi_plus_psi=stack_grid(flux_dicts, "phi_plus_psi"),
            balance_residual=stack_grid(flux_dicts, "balance_residual"),
        )


def export_coupling_artifacts(
    request: CouplingArtifactRequest,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    return CouplingArtifactExporter(request=request, logger=logger).export()
