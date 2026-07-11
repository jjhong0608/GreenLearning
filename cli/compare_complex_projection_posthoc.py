from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Sequence

import numpy as np
import plotly.graph_objects as go
import torch
from rich.logging import RichHandler

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import ComplexGeometryMetadata, load_complex_geometry
from greenonet.complex_projection import (
    ComplexProjectionResult,
    apply_geometry_weighted_projection,
    apply_hard_symmetric_projection,
)
from greenonet.complex_reconstruction import reconstruct_from_projected_unit
from greenonet.coupling_artifacts import (
    CouplingArtifactConfigs,
    load_coupling_artifact_configs,
)
from greenonet.green_interval import build_segment_branch_samples
from greenonet.io import load_model_with_config, load_state_dict_auto
from greenonet.model import GreenONetModel
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class ProjectionPosthocConfig:
    artifact_root: Path
    geometry: Path
    green_checkpoint: Path
    config: Path
    outdir: Path
    coefficients: Path | None = None
    device: str | None = None
    theme: str = "plotly_white"
    balance_tol: float = 1.0e-10
    transition_coordinate: float | None = None


@dataclass(frozen=True)
class RawSelectedSample:
    key: str
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ProjectionEvaluation:
    projection_name: str
    projection: ComplexProjectionResult
    arrays: dict[str, np.ndarray]
    balance_max_abs: float


class GreenReconstructionMixin:
    @staticmethod
    def _load_green_model(
        configs: CouplingArtifactConfigs,
        checkpoint: Path,
        device: torch.device,
    ) -> torch.nn.Module:
        model: torch.nn.Module
        try:
            model, _loaded_config = load_model_with_config(checkpoint)
        except Exception:
            model = GreenONetModel(configs.green_model)
            load_state_dict_auto(model, checkpoint)
        model.to(device)
        model.eval()
        return model

    @staticmethod
    def _build_green_branches(
        *,
        geometry: ComplexGeometryMetadata,
        configs: CouplingArtifactConfigs,
        coeff_path: Path | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coeffs = load_coefficient_functions(coeff_path)
        x_coeffs = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="x",
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            device=device,
        )
        y_coeffs = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="y",
            branch_input_dim=configs.coupling_model.branch_input_dim,
            dtype=configs.dataset.dtype,
            device=device,
        )
        x_branch = torch.stack(
            (x_coeffs.a_unit, x_coeffs.ap_unit, x_coeffs.b_unit, x_coeffs.c_unit),
            dim=1,
        ).unsqueeze(0)
        y_branch = torch.stack(
            (y_coeffs.a_unit, y_coeffs.ap_unit, y_coeffs.b_unit, y_coeffs.c_unit),
            dim=1,
        ).unsqueeze(0)
        return x_branch, y_branch


class RawArchiveMixin:
    REQUIRED_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "coords_valid",
        "rhs",
        "sol",
        "raw_unit_phi",
        "raw_unit_psi",
    )
    KNOWN_SUFFIXES: ClassVar[tuple[str, ...]] = (
        "coords_valid",
        "raw_unit_phi",
        "raw_unit_psi",
        "u_split_mismatch",
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
        "u_pred",
        "u_phi",
        "u_psi",
        "rhs",
        "sol",
        "phi",
        "psi",
    )

    @classmethod
    def _load_selected_samples(cls, artifact_root: Path) -> list[RawSelectedSample]:
        archive_path = artifact_root / "data" / "selected_raw_arrays.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing selected raw archive: {archive_path}")
        grouped: dict[str, dict[str, np.ndarray]] = {}
        suffixes = sorted(cls.KNOWN_SUFFIXES, key=len, reverse=True)
        with np.load(archive_path, allow_pickle=False) as raw:
            for key in raw.files:
                suffix = cls._matched_suffix(key, suffixes)
                if suffix is None:
                    continue
                prefix = key[: -(len(suffix) + 1)]
                grouped.setdefault(prefix, {})[suffix] = np.array(raw[key])
        samples = [
            RawSelectedSample(key=key, arrays=arrays)
            for key, arrays in sorted(grouped.items())
        ]
        if not samples:
            raise ValueError(f"No selected samples found in {archive_path}.")
        for sample in samples:
            missing = sorted(set(cls.REQUIRED_SUFFIXES) - set(sample.arrays))
            if missing:
                raise KeyError(
                    f"Selected sample '{sample.key}' is missing fields: "
                    f"{', '.join(missing)}."
                )
        return samples

    @staticmethod
    def _matched_suffix(key: str, suffixes: list[str]) -> str | None:
        for suffix in suffixes:
            if key.endswith(f"_{suffix}"):
                return suffix
        return None


class ProjectionMetricMixin:
    ERROR_FIELDS: ClassVar[tuple[str, ...]] = (
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "u_split_mismatch",
    )

    @staticmethod
    def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
        numerator = float(np.linalg.norm(np.asarray(pred) - np.asarray(target)))
        denominator = max(float(np.linalg.norm(np.asarray(target))), 1.0e-12)
        return numerator / denominator

    @classmethod
    def _projection_metrics(
        cls,
        *,
        arrays: dict[str, np.ndarray],
        balance_max_abs: float,
        prefix: str,
        mask: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        if mask is None:
            mask = np.ones_like(arrays["sol"], dtype=bool)
        point_count = int(np.count_nonzero(mask))
        metrics: dict[str, float | int] = {"point_count": point_count}
        if point_count == 0:
            for name in (
                "rel_u_pred",
                "rel_u_phi",
                "rel_u_psi",
                "u_pred_error_rms",
                "u_phi_error_rms",
                "u_psi_error_rms",
                "u_split_mismatch_rms",
                "u_pred_error_mean_abs",
                "u_phi_error_mean_abs",
                "u_psi_error_mean_abs",
                "u_split_mismatch_mean_abs",
                "u_pred_error_max_abs",
                "u_phi_error_max_abs",
                "u_psi_error_max_abs",
                "u_split_mismatch_max_abs",
                "balance_max_abs",
            ):
                metrics[f"{prefix}_{name}"] = float("nan")
            return metrics

        sol = arrays["sol"][mask]
        metrics[f"{prefix}_rel_u_pred"] = cls._relative_l2(arrays["u_pred"][mask], sol)
        metrics[f"{prefix}_rel_u_phi"] = cls._relative_l2(arrays["u_phi"][mask], sol)
        metrics[f"{prefix}_rel_u_psi"] = cls._relative_l2(arrays["u_psi"][mask], sol)
        for field in cls.ERROR_FIELDS:
            values = np.asarray(arrays[field][mask], dtype=np.float64)
            metrics[f"{prefix}_{field}_rms"] = float(np.sqrt(np.mean(values**2)))
            metrics[f"{prefix}_{field}_mean_abs"] = float(np.mean(np.abs(values)))
            metrics[f"{prefix}_{field}_max_abs"] = float(np.max(np.abs(values)))
        metrics[f"{prefix}_balance_max_abs"] = float(balance_max_abs)
        return metrics

    @staticmethod
    def _delta_metrics(
        *,
        weighted: dict[str, np.ndarray],
        symmetric: dict[str, np.ndarray],
        prefix: str = "delta_weighted_minus_symmetric",
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for field in ("u_pred", "u_phi", "u_psi", "u_split_mismatch"):
            delta = np.asarray(weighted[field]) - np.asarray(symmetric[field])
            metrics[f"{prefix}_{field}_rms"] = float(np.sqrt(np.mean(delta**2)))
            metrics[f"{prefix}_{field}_mean"] = float(np.mean(delta))
            metrics[f"{prefix}_{field}_max_abs"] = float(np.max(np.abs(delta)))
        return metrics


class ZoneMetricMixin:
    @staticmethod
    def _read_transition_coordinate(
        geometry_path: Path,
        override: float | None,
    ) -> tuple[float | None, dict[str, Any]]:
        if override is not None:
            if override <= 0:
                raise ValueError("--transition-coordinate must be positive.")
            return float(override), {"source": "cli", "inner_radius": None}
        with np.load(geometry_path, allow_pickle=False) as raw:
            if "inner_radius" not in raw.files:
                return None, {"source": "unavailable", "inner_radius": None}
            inner_radius = float(np.asarray(raw["inner_radius"]).reshape(()))
            grid_values: list[np.ndarray] = []
            if "grid_x" in raw.files:
                grid_values.append(np.asarray(raw["grid_x"], dtype=np.float64))
            if "grid_y" in raw.files:
                grid_values.append(np.asarray(raw["grid_y"], dtype=np.float64))
            if not grid_values:
                return None, {"source": "missing_grid", "inner_radius": inner_radius}
            absolute = np.unique(np.abs(np.concatenate(grid_values)))
            candidates = absolute[absolute > inner_radius]
            if candidates.size == 0:
                return None, {"source": "no_candidate", "inner_radius": inner_radius}
            return float(np.min(candidates)), {
                "source": "inner_radius_grid",
                "inner_radius": inner_radius,
            }

    @staticmethod
    def _zone_masks(
        *,
        coords: np.ndarray,
        geometry: ComplexGeometryMetadata,
        transition_coordinate: float | None,
    ) -> dict[str, np.ndarray]:
        masks: dict[str, np.ndarray] = {
            "global": np.ones(coords.shape[0], dtype=bool),
        }
        if transition_coordinate is None:
            return masks
        line_tol = 0.1 * max(float(geometry.hx.item()), float(geometry.hy.item()))
        line_tol = max(line_tol, 1.0e-10)
        zone_radius = 2.0 * max(float(geometry.hx.item()), float(geometry.hy.item()))
        x = coords[:, 0]
        y = coords[:, 1]
        masks[f"horizontal_abs_y_{transition_coordinate:.8g}"] = np.isclose(
            np.abs(y), transition_coordinate, atol=line_tol, rtol=0.0
        )
        masks[f"vertical_abs_x_{transition_coordinate:.8g}"] = np.isclose(
            np.abs(x), transition_coordinate, atol=line_tol, rtol=0.0
        )
        centers = {
            "cardinal_right": (transition_coordinate, 0.0),
            "cardinal_left": (-transition_coordinate, 0.0),
            "cardinal_top": (0.0, transition_coordinate),
            "cardinal_bottom": (0.0, -transition_coordinate),
        }
        for name, center in centers.items():
            distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
            masks[f"{name}_{transition_coordinate:.8g}"] = distance <= zone_radius
        return masks


class FigureMixin:
    @staticmethod
    def _scatter_figure(
        *,
        title: str,
        coords: np.ndarray,
        values: np.ndarray,
        theme: str,
    ) -> go.Figure:
        finite_values = values[np.isfinite(values)]
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
        marker_range = {"cmin": -max_abs, "cmax": max_abs} if max_abs > 0.0 else {}
        return go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": "RdBu",
                    "showscale": True,
                    "size": 6,
                    "colorbar": {"exponentformat": "power", "showexponent": "all"},
                    **marker_range,
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


class ComplexProjectionPosthocComparator(
    GreenReconstructionMixin,
    RawArchiveMixin,
    ProjectionMetricMixin,
    ZoneMetricMixin,
    FigureMixin,
):
    PROJECTION_NAMES: ClassVar[tuple[str, str]] = ("symmetric", "geometry_weighted")
    FIGURE_FIELDS: ClassVar[tuple[str, ...]] = (
        "u_phi_error",
        "u_psi_error",
        "u_pred_error",
        "u_split_mismatch",
    )
    DELTA_FIELDS: ClassVar[tuple[str, ...]] = (
        "u_phi",
        "u_psi",
        "u_pred",
        "u_phi_error",
        "u_psi_error",
        "u_pred_error",
        "u_split_mismatch",
    )

    def __init__(
        self,
        config: ProjectionPosthocConfig,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger
        self.config.outdir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        configs = load_coupling_artifact_configs(self.config.config)
        if configs.dataset.geometry_mode != "complex":
            raise ValueError("Post-hoc projection comparison requires complex mode.")
        device = torch.device(self.config.device or configs.coupling_training.device)
        coeff_path = (
            self.config.coefficients or configs.dataset.coefficient_functions_path
        )
        geometry = load_complex_geometry(
            self.config.geometry,
            dtype=configs.dataset.dtype,
            device=device,
        )
        samples = self._load_selected_samples(self.config.artifact_root)
        green_model = self._load_green_model(
            configs,
            self.config.green_checkpoint,
            device,
        )
        x_green_branch, y_green_branch = self._build_green_branches(
            geometry=geometry,
            configs=configs,
            coeff_path=coeff_path,
            device=device,
        )
        transition_coordinate, transition_meta = self._read_transition_coordinate(
            self.config.geometry,
            self.config.transition_coordinate,
        )

        per_sample_rows: list[dict[str, float | int | str]] = []
        zone_rows: list[dict[str, float | int | str]] = []
        raw_payload: dict[str, np.ndarray] = {}
        figure_paths: list[str] = []
        for sample in samples:
            evaluations = self._evaluate_sample(
                sample=sample,
                geometry=geometry,
                green_model=green_model,
                x_green_branch=x_green_branch,
                y_green_branch=y_green_branch,
                device=device,
                dtype=configs.dataset.dtype,
            )
            per_sample_rows.append(self._sample_row(sample, evaluations))
            zone_rows.extend(
                self._zone_rows(
                    sample=sample,
                    evaluations=evaluations,
                    geometry=geometry,
                    transition_coordinate=transition_coordinate,
                )
            )
            self._collect_raw_payload(sample, evaluations, raw_payload)
            figure_paths.extend(self._write_figures(sample, evaluations))

        self._write_csv(
            self.config.outdir / "per_sample_projection_comparison.csv", per_sample_rows
        )
        self._write_csv(
            self.config.outdir / "zone_projection_comparison.csv", zone_rows
        )
        data_dir = self.config.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez(data_dir / "selected_projection_raw_arrays.npz", **raw_payload)  # type: ignore[arg-type]
        summary = {
            "artifact_root": str(self.config.artifact_root),
            "geometry_path": str(self.config.geometry),
            "green_checkpoint": str(self.config.green_checkpoint),
            "config": str(self.config.config),
            "coefficients": None if coeff_path is None else str(coeff_path),
            "device": str(device),
            "projection_modes": list(self.PROJECTION_NAMES),
            "geometry_weighted_rule": "direct_length_squared",
            "geometry_weighted_beta": "2*w_phi*w_psi",
            "solution_prediction": "u_pred=0.5*(u_phi+u_psi)",
            "error_convention": "signed_difference",
            "sample_count": len(samples),
            "sample_keys": [sample.key for sample in samples],
            "balance_tol": self.config.balance_tol,
            "transition_coordinate": transition_coordinate,
            "transition_coordinate_metadata": transition_meta,
            "figure_count": len(figure_paths),
            "figure_paths": figure_paths,
            "aggregate_metrics": self._aggregate_per_sample(per_sample_rows),
        }
        (self.config.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        self.logger.info(
            "Completed post-hoc projection comparison for %d selected samples",
            len(samples),
        )
        return summary

    def _evaluate_sample(
        self,
        *,
        sample: RawSelectedSample,
        geometry: ComplexGeometryMetadata,
        green_model: torch.nn.Module,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, ProjectionEvaluation]:
        coords = np.asarray(sample.arrays["coords_valid"], dtype=np.float64)
        expected_coords = geometry.coords_valid.detach().cpu().numpy()
        if coords.shape != expected_coords.shape or not np.allclose(
            coords,
            expected_coords,
            rtol=1.0e-10,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Selected sample '{sample.key}' coords_valid does not match geometry."
            )
        raw_unit = torch.stack(
            (
                torch.as_tensor(
                    sample.arrays["raw_unit_phi"], dtype=dtype, device=device
                ),
                torch.as_tensor(
                    sample.arrays["raw_unit_psi"], dtype=dtype, device=device
                ),
            ),
            dim=0,
        ).unsqueeze(0)
        rhs = torch.as_tensor(
            sample.arrays["rhs"], dtype=dtype, device=device
        ).unsqueeze(0)
        sol_np = np.asarray(sample.arrays["sol"], dtype=np.float64)
        projection_builders = {
            "symmetric": apply_hard_symmetric_projection,
            "geometry_weighted": apply_geometry_weighted_projection,
        }
        evaluations: dict[str, ProjectionEvaluation] = {}
        with torch.no_grad():
            for name, builder in projection_builders.items():
                projection = builder(raw_unit=raw_unit, rhs_phys=rhs, geometry=geometry)
                balance = (
                    projection.projected_physical[:, 0]
                    + projection.projected_physical[:, 1]
                    - rhs
                )
                balance_max_abs = float(balance.abs().max().item())
                if balance_max_abs > self.config.balance_tol:
                    raise ValueError(
                        f"{name} projection balance residual {balance_max_abs:.6e} "
                        f"exceeds tolerance {self.config.balance_tol:.6e}."
                    )
                reconstruction = reconstruct_from_projected_unit(
                    green_model=green_model,
                    geometry=geometry,
                    projected_unit=projection.projected_unit,
                    x_green_branch=x_green_branch,
                    y_green_branch=y_green_branch,
                )
                u_phi = reconstruction.u_phi_valid[0].detach().cpu().numpy()
                u_psi = reconstruction.u_psi_valid[0].detach().cpu().numpy()
                u_pred = reconstruction.u_mean_valid[0].detach().cpu().numpy()
                phi = projection.projected_physical[0, 0].detach().cpu().numpy()
                psi = projection.projected_physical[0, 1].detach().cpu().numpy()
                arrays = {
                    "coords_valid": coords,
                    "rhs": np.asarray(sample.arrays["rhs"], dtype=np.float64),
                    "sol": sol_np,
                    "raw_unit_phi": np.asarray(
                        sample.arrays["raw_unit_phi"], dtype=np.float64
                    ),
                    "raw_unit_psi": np.asarray(
                        sample.arrays["raw_unit_psi"], dtype=np.float64
                    ),
                    "phi": phi,
                    "psi": psi,
                    "u_phi": u_phi,
                    "u_psi": u_psi,
                    "u_pred": u_pred,
                    "u_phi_error": u_phi - sol_np,
                    "u_psi_error": u_psi - sol_np,
                    "u_pred_error": u_pred - sol_np,
                    "u_split_mismatch": u_phi - u_psi,
                }
                evaluations[name] = ProjectionEvaluation(
                    projection_name=name,
                    projection=projection,
                    arrays=arrays,
                    balance_max_abs=balance_max_abs,
                )
        return evaluations

    def _sample_row(
        self,
        sample: RawSelectedSample,
        evaluations: dict[str, ProjectionEvaluation],
    ) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {"sample_key": sample.key}
        for name in self.PROJECTION_NAMES:
            evaluation = evaluations[name]
            row.update(
                self._projection_metrics(
                    arrays=evaluation.arrays,
                    balance_max_abs=evaluation.balance_max_abs,
                    prefix=name,
                )
            )
        row.update(
            self._delta_metrics(
                weighted=evaluations["geometry_weighted"].arrays,
                symmetric=evaluations["symmetric"].arrays,
            )
        )
        return row

    def _zone_rows(
        self,
        *,
        sample: RawSelectedSample,
        evaluations: dict[str, ProjectionEvaluation],
        geometry: ComplexGeometryMetadata,
        transition_coordinate: float | None,
    ) -> list[dict[str, float | int | str]]:
        coords = evaluations["symmetric"].arrays["coords_valid"]
        masks = self._zone_masks(
            coords=coords,
            geometry=geometry,
            transition_coordinate=transition_coordinate,
        )
        rows: list[dict[str, float | int | str]] = []
        for zone_name, mask in masks.items():
            for projection_name in self.PROJECTION_NAMES:
                evaluation = evaluations[projection_name]
                metrics = self._projection_metrics(
                    arrays=evaluation.arrays,
                    balance_max_abs=evaluation.balance_max_abs,
                    prefix="metric",
                    mask=mask,
                )
                rows.append(
                    {
                        "sample_key": sample.key,
                        "zone": zone_name,
                        "projection": projection_name,
                        **metrics,
                    }
                )
        return rows

    def _collect_raw_payload(
        self,
        sample: RawSelectedSample,
        evaluations: dict[str, ProjectionEvaluation],
        raw_payload: dict[str, np.ndarray],
    ) -> None:
        base = sample.key
        for shared in ("coords_valid", "rhs", "sol", "raw_unit_phi", "raw_unit_psi"):
            raw_payload[f"{base}_{shared}"] = np.asarray(sample.arrays[shared])
        for projection_name, evaluation in evaluations.items():
            for field, value in evaluation.arrays.items():
                if field in {
                    "coords_valid",
                    "rhs",
                    "sol",
                    "raw_unit_phi",
                    "raw_unit_psi",
                }:
                    continue
                raw_payload[f"{base}_{projection_name}_{field}"] = value
        weighted = evaluations["geometry_weighted"].arrays
        symmetric = evaluations["symmetric"].arrays
        for field in self.DELTA_FIELDS:
            raw_payload[f"{base}_weighted_minus_symmetric_{field}"] = (
                weighted[field] - symmetric[field]
            )

    def _write_figures(
        self,
        sample: RawSelectedSample,
        evaluations: dict[str, ProjectionEvaluation],
    ) -> list[str]:
        paths: list[str] = []
        coords = evaluations["symmetric"].arrays["coords_valid"]
        for projection_name, evaluation in evaluations.items():
            for field in self.FIGURE_FIELDS:
                base_path = (
                    self.config.outdir
                    / "figures"
                    / projection_name
                    / field
                    / f"{sample.key}_{projection_name}_{field}"
                )
                fig = self._scatter_figure(
                    title=f"{sample.key} {projection_name} {field}",
                    coords=coords,
                    values=evaluation.arrays[field],
                    theme=self.config.theme,
                )
                save_plotly_figure(fig, base_path, logger=self.logger)
                paths.append(str(base_path.with_suffix(".json")))

        weighted = evaluations["geometry_weighted"].arrays
        symmetric = evaluations["symmetric"].arrays
        for field in self.DELTA_FIELDS:
            base_path = (
                self.config.outdir
                / "figures"
                / "weighted_minus_symmetric"
                / field
                / f"{sample.key}_weighted_minus_symmetric_{field}"
            )
            fig = self._scatter_figure(
                title=f"{sample.key} weighted-minus-symmetric {field}",
                coords=coords,
                values=weighted[field] - symmetric[field],
                theme=self.config.theme,
            )
            save_plotly_figure(fig, base_path, logger=self.logger)
            paths.append(str(base_path.with_suffix(".json")))
        return paths

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("")
            return
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _aggregate_per_sample(
        rows: list[dict[str, float | int | str]],
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        numeric_keys = [
            key
            for row in rows
            for key, value in row.items()
            if key != "sample_key" and isinstance(value, int | float)
        ]
        for key in sorted(set(numeric_keys)):
            values = [
                float(row[key])
                for row in rows
                if key in row and np.isfinite(float(row[key]))
            ]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(values))
                aggregate[f"{key}_max"] = float(np.max(values))
        return aggregate


class CompareComplexProjectionPosthocCLI:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare symmetric and geometry-weighted projections using an "
                "existing complex CouplingNet selected_raw_arrays.npz archive."
            )
        )
        parser.add_argument("--artifact-root", type=Path, required=True)
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--green-checkpoint", type=Path, required=True)
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--outdir", type=Path, required=True)
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument("--balance-tol", type=float, default=1.0e-10)
        parser.add_argument("--transition-coordinate", type=float, default=None)
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("CompareComplexProjectionPosthoc")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)
        logging.root.handlers.clear()

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            outdir / "compare_complex_projection_posthoc.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def parse_config(
        self,
        argv: Sequence[str] | None = None,
    ) -> ProjectionPosthocConfig:
        args = self.parser.parse_args(argv)
        if args.balance_tol <= 0:
            raise ValueError("--balance-tol must be positive.")
        return ProjectionPosthocConfig(
            artifact_root=args.artifact_root,
            geometry=args.geometry,
            green_checkpoint=args.green_checkpoint,
            config=args.config,
            outdir=args.outdir,
            coefficients=args.coefficients,
            device=args.device,
            theme=args.theme,
            balance_tol=float(args.balance_tol),
            transition_coordinate=args.transition_coordinate,
        )

    def run(self, argv: Sequence[str] | None = None) -> dict[str, Any]:
        config = self.parse_config(argv)
        logger = self._build_logger(config.outdir)
        logger.info("Starting post-hoc projection comparison")
        return ComplexProjectionPosthocComparator(config, logger=logger).run()


def main() -> None:
    CompareComplexProjectionPosthocCLI().run()


if __name__ == "__main__":
    main()
