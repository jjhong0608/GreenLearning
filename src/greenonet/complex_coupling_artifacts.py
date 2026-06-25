from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_coupling_evaluator import ComplexCouplingEvaluator
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    CouplingArtifactRequest,
    load_coupling_artifact_configs,
)
from greenonet.config import Axis1DTrunkConfig
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class ComplexSelectedSample:
    sample_id: int
    file_stem: str
    arrays: dict[str, np.ndarray]


class ComplexCouplingArtifactExporter:
    """Export complex-geometry CouplingNet metrics, raw archives, and scatter plots."""

    COLOR_RANGE_POLICY: ClassVar[str] = "shared_reference_prediction_groups"
    COLOR_RANGE_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "solution": ("sol", "u_pred", "u_phi", "u_psi"),
        "phi": ("target_phi", "phi"),
        "psi": ("target_psi", "psi"),
    }
    FIGURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "rhs",
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "u_split_mismatch",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
    )
    SIGNED_FIGURE_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "u_pred_error",
            "u_phi_error",
            "u_psi_error",
            "u_split_mismatch",
            "phi_error",
            "psi_error",
        }
    )
    FIGURE_TITLES: ClassVar[dict[str, str]] = {
        "rhs": "Source rhs",
        "sol": "Exact solution sol",
        "u_pred": "Predicted solution u_pred",
        "u_phi": "Reconstructed solution u_phi",
        "u_psi": "Reconstructed solution u_psi",
        "u_pred_error": "Signed error u_pred - sol",
        "u_phi_error": "Signed error u_phi - sol",
        "u_psi_error": "Signed error u_psi - sol",
        "u_split_mismatch": "Mismatch u_phi - u_psi",
        "phi": "Projected phi",
        "psi": "Projected psi",
        "target_phi": "Target phi",
        "target_psi": "Target psi",
        "phi_error": "Signed error phi - target_phi",
        "psi_error": "Signed error psi - target_psi",
    }

    def __init__(
        self,
        request: CouplingArtifactRequest,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger
        self.request.outdir.mkdir(parents=True, exist_ok=True)

    def export(self) -> dict[str, Any]:
        configs = load_coupling_artifact_configs(self.request.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError(
                "Complex artifact exporter requires geometry_mode='complex'."
            )
        if configs.dataset.geometry_path is None:
            raise ValueError("dataset.geometry_path is required for complex artifacts.")
        if configs.dataset.test_path is None:
            raise ValueError("dataset.test_path is required for complex artifacts.")

        device = torch.device(self.request.device or configs.coupling_training.device)
        coeff_path = (
            self.request.coefficients or configs.dataset.coefficient_functions_path
        )
        coeffs = load_coefficient_functions(coeff_path)
        geometry = load_complex_geometry(
            configs.dataset.geometry_path,
            dtype=configs.dataset.dtype,
        )
        dataset = ComplexCouplingDataset(
            configs.dataset.test_path,
            geometry,
            coeffs,
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            coefficient_terms=configs.coupling_model.coefficient_terms,
            integration_rule=configs.coupling_training.integration_rule,
        )
        coupling_model = self._load_complex_model(configs, device)
        green_model = self._load_green_model(configs, device)
        evaluator = ComplexCouplingEvaluator(
            model=coupling_model,
            green_model=green_model,
            device=device,
            work_dir=self.request.outdir,
        )
        metric_rows = self._evaluate_rows(dataset, evaluator, configs)
        selected, roles, policy = self._select_sample_indices(metric_rows)
        selected_samples = self._evaluate_selected(dataset, evaluator, selected, device)
        self._write_metric_csv(metric_rows)
        self._write_selected_npz(selected_samples)
        figure_fields = self._figure_fields(selected_samples)
        figure_paths = self._write_figures(
            selected_samples,
            self.request.theme,
        )
        aggregate = self._aggregate_metrics(metric_rows)
        axis_1d_trunk = Axis1DTrunkConfig.from_raw(configs.coupling_model.axis_1d_trunk)
        summary = {
            "geometry_mode": "complex",
            "device": str(device),
            "coefficients": None if coeff_path is None else str(coeff_path),
            "geometry_path": str(configs.dataset.geometry_path),
            "test_path": str(configs.dataset.test_path),
            "selected_samples": list(selected),
            "selected_sample_roles": roles,
            "selected_sample_policy": policy,
            "plot_workers": self.request.plot_workers,
            "save_generated_data": self.request.save_generated_data,
            "aggregate_metrics": aggregate,
            "figure_count": len(figure_paths),
            "figure_fields": list(figure_fields),
            "error_convention": "signed_difference",
            "solution_prediction": "u_pred=0.5*(u_phi+u_psi)",
            "non_error_color_range_policy": self.COLOR_RANGE_POLICY,
            "non_error_color_range_groups": {
                name: list(fields) for name, fields in self.COLOR_RANGE_GROUPS.items()
            },
            "optional_flux_targets_exported": self._has_flux_target_artifacts(
                selected_samples
            ),
            "source_branch": {
                "enabled": True,
                "scaling": "f_unit=L^2*f_phys",
                "normalization": "segment_unit_l2",
            },
            "coefficient_terms": {
                "diffusion": configs.coupling_model.coefficient_terms.diffusion,
                "convection": configs.coupling_model.coefficient_terms.convection,
                "reaction": configs.coupling_model.coefficient_terms.reaction,
            },
            "transverse_encoding": {
                "coordinate": "global_normalized_transverse",
                "num_frequencies": axis_1d_trunk.num_frequencies,
                "max_frequency": axis_1d_trunk.max_frequency,
            },
        }
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        return summary

    def _load_complex_model(
        self,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> ComplexCouplingNet:
        model = ComplexCouplingNet(configs.coupling_model)
        load_state_dict_auto(model, self.request.coupling_checkpoint)
        model.to(device)
        model.eval()
        return model

    def _load_green_model(
        self,
        configs: CouplingArtifactConfigs,
        device: torch.device,
    ) -> torch.nn.Module:
        model: torch.nn.Module
        try:
            model, _loaded_config = load_model_with_config(
                self.request.green_checkpoint
            )
        except Exception:
            model = GreenONetModel(configs.green_model)
            load_state_dict_auto(model, self.request.green_checkpoint)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _evaluate_rows(
        dataset: ComplexCouplingDataset,
        evaluator: ComplexCouplingEvaluator,
        configs: CouplingArtifactConfigs,
    ) -> list[dict[str, float | int | str]]:
        loader = DataLoader(
            dataset,
            batch_size=configs.coupling_training.batch_size,
            shuffle=False,
            collate_fn=complex_coupling_collate_fn,
        )
        rows: list[dict[str, float | int | str]] = []
        with torch.no_grad():
            for batch in loader:
                prediction = evaluator.predict_batch(batch.to(evaluator.device))
                for offset, sample_index in enumerate(
                    prediction.batch.sample_indices.cpu().tolist()
                ):
                    row = evaluator._sample_metric_row(prediction, offset)
                    row["sample_id"] = int(sample_index)
                    row["file_stem"] = prediction.batch.file_stems[offset]
                    rows.append(row)
        return rows

    def _select_sample_indices(
        self,
        metric_rows: list[dict[str, float | int | str]],
    ) -> tuple[tuple[int, ...], dict[str, int], str]:
        if self.request.selected_samples is not None:
            seen: set[int] = set()
            explicit_selected: list[int] = []
            for value in self.request.selected_samples:
                if value not in seen:
                    explicit_selected.append(int(value))
                    seen.add(int(value))
            return tuple(explicit_selected), {}, "explicit"
        if not metric_rows:
            return (), {}, "empty"
        sorted_rows = sorted(metric_rows, key=lambda row: float(row["rel_sol"]))
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
        quantile_selected = tuple(dict.fromkeys(roles.values()))
        return quantile_selected, roles, "rel_sol_quantiles"

    @staticmethod
    def _evaluate_selected(
        dataset: ComplexCouplingDataset,
        evaluator: ComplexCouplingEvaluator,
        selected: tuple[int, ...],
        device: torch.device,
    ) -> list[ComplexSelectedSample]:
        samples: list[ComplexSelectedSample] = []
        coords = dataset.geometry.coords_valid.detach().cpu().numpy()
        with torch.no_grad():
            for sample_id in selected:
                batch = complex_coupling_collate_fn([dataset[sample_id]]).to(device)
                prediction = evaluator.predict_batch(batch)
                rhs = prediction.batch.rhs_valid[0].detach().cpu().numpy()
                sol = prediction.batch.sol_valid[0].detach().cpu().numpy()
                phi = (
                    prediction.projection.projected_physical[0, 0]
                    .detach()
                    .cpu()
                    .numpy()
                )
                psi = (
                    prediction.projection.projected_physical[0, 1]
                    .detach()
                    .cpu()
                    .numpy()
                )
                u_phi = prediction.reconstruction.u_phi_valid[0].detach().cpu().numpy()
                u_psi = prediction.reconstruction.u_psi_valid[0].detach().cpu().numpy()
                u_pred = (
                    prediction.reconstruction.u_mean_valid[0].detach().cpu().numpy()
                )
                arrays = {
                    "coords_valid": coords,
                    "rhs": rhs,
                    "sol": sol,
                    "raw_unit_phi": prediction.raw_unit[0, 0].detach().cpu().numpy(),
                    "raw_unit_psi": prediction.raw_unit[0, 1].detach().cpu().numpy(),
                    "phi": phi,
                    "psi": psi,
                    "u_pred": u_pred,
                    "u_phi": u_phi,
                    "u_psi": u_psi,
                    "u_pred_error": u_pred - sol,
                    "u_phi_error": u_phi - sol,
                    "u_psi_error": u_psi - sol,
                    "u_split_mismatch": u_phi - u_psi,
                }
                if bool(prediction.batch.has_flux[0].item()):
                    target_phi = (
                        prediction.batch.flux_valid[0, 0].detach().cpu().numpy()
                    )
                    target_psi = (
                        prediction.batch.flux_valid[0, 1].detach().cpu().numpy()
                    )
                    arrays["target_phi"] = target_phi
                    arrays["target_psi"] = target_psi
                    arrays["phi_error"] = phi - target_phi
                    arrays["psi_error"] = psi - target_psi
                samples.append(
                    ComplexSelectedSample(
                        sample_id=sample_id,
                        file_stem=prediction.batch.file_stems[0],
                        arrays=arrays,
                    )
                )
        return samples

    def _write_metric_csv(
        self,
        metric_rows: list[dict[str, float | int | str]],
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        if not metric_rows:
            return
        fieldnames = list(metric_rows[0].keys())
        for row in metric_rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with (metrics_dir / "per_sample_metrics.csv").open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)

    def _write_selected_npz(
        self,
        selected_samples: list[ComplexSelectedSample],
    ) -> None:
        if not self.request.save_generated_data:
            return
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {}
        for sample in selected_samples:
            prefix = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            for key, value in sample.arrays.items():
                payload[f"{prefix}_{key}"] = value
        np.savez(data_dir / "selected_raw_arrays.npz", **payload)  # type: ignore[arg-type]

    def _write_figures(
        self,
        selected_samples: list[ComplexSelectedSample],
        theme: str,
    ) -> list[str]:
        figure_paths: list[str] = []
        for sample in selected_samples:
            stem = f"sample_{sample.sample_id:04d}_{sample.file_stem}"
            color_ranges = self._color_ranges_for_sample(sample.arrays)
            for field in self._figure_fields_for_sample(sample.arrays):
                fig = self._scatter_figure(
                    title=f"{stem} {self.FIGURE_TITLES[field]}",
                    coords=sample.arrays["coords_valid"],
                    values=sample.arrays[field],
                    theme=theme,
                    signed=field in self.SIGNED_FIGURE_FIELDS,
                    color_range=color_ranges.get(field),
                )
                base_path = self.request.outdir / "figures" / field / f"{stem}_{field}"
                save_plotly_figure(fig, base_path, logger=self.logger)
                figure_paths.append(str(base_path.with_suffix(".json")))
        return figure_paths

    @classmethod
    def _color_ranges_for_sample(
        cls,
        arrays: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        ranges: dict[str, dict[str, float]] = {}
        for fields in cls.COLOR_RANGE_GROUPS.values():
            color_range = cls._shared_color_range(arrays, fields)
            if not color_range:
                continue
            for field in fields:
                if field in arrays:
                    ranges[field] = color_range
        return ranges

    @staticmethod
    def _shared_color_range(
        arrays: dict[str, np.ndarray],
        fields: tuple[str, ...],
    ) -> dict[str, float]:
        finite_values: list[np.ndarray] = []
        for field in fields:
            if field not in arrays:
                continue
            values = np.asarray(arrays[field])
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_values.append(finite)
        if not finite_values:
            return {}
        joined = np.concatenate(finite_values)
        return {
            "cmin": float(np.min(joined)),
            "cmax": float(np.max(joined)),
        }

    @classmethod
    def _figure_fields_for_sample(
        cls, arrays: dict[str, np.ndarray]
    ) -> tuple[str, ...]:
        return tuple(field for field in cls.FIGURE_FIELDS if field in arrays)

    @classmethod
    def _figure_fields(
        cls,
        selected_samples: list[ComplexSelectedSample],
    ) -> tuple[str, ...]:
        fields: list[str] = []
        seen: set[str] = set()
        for sample in selected_samples:
            for field in cls._figure_fields_for_sample(sample.arrays):
                if field not in seen:
                    fields.append(field)
                    seen.add(field)
        return tuple(fields)

    @staticmethod
    def _has_flux_target_artifacts(
        selected_samples: list[ComplexSelectedSample],
    ) -> bool:
        return any(
            {"target_phi", "target_psi", "phi_error", "psi_error"}.issubset(
                sample.arrays
            )
            for sample in selected_samples
        )

    @staticmethod
    def _scatter_figure(
        *,
        title: str,
        coords: np.ndarray,
        values: np.ndarray,
        theme: str,
        signed: bool = False,
        color_range: dict[str, float] | None = None,
    ) -> go.Figure:
        finite_values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        marker_color_range: dict[str, float] = dict(color_range or {})
        if signed and max_abs > 0.0:
            marker_color_range = {"cmin": -max_abs, "cmax": max_abs}
        return go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": "RdBu" if signed else "Viridis",
                    "showscale": True,
                    "size": 6,
                    "colorbar": {"exponentformat": "power", "showexponent": "all"},
                    **marker_color_range,
                },
            ),
            layout=go.Layout(
                template=theme,
                width=900,
                height=800,
                title=title,
                xaxis_title="x",
                yaxis_title="y",
                yaxis={"scaleanchor": "x", "scaleratio": 1},
            ),
        )

    @staticmethod
    def _aggregate_metrics(
        metric_rows: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        for key in ("loss", "loss_energy_consistency", "rel_sol", "rel_flux"):
            values = [float(row[key]) for row in metric_rows if key in row]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_max"] = float(np.max(values))
        return aggregate


def export_complex_coupling_artifacts(
    request: CouplingArtifactRequest,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexCouplingArtifactExporter(request, logger=logger).export()
