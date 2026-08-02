from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_mismatch_blend_diagnostics import (
    MismatchGradientBlendConfig,
    MismatchGradientBlendMixin,
    MismatchSeamC2BlendConfig,
)


PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
PRESENTATION_ROOT: Final = Path(__file__).resolve().parent
DEFAULT_OUTDIR: Final = PRESENTATION_ROOT / "assets"
FONT_FAMILY: Final = "Aptos, Segoe UI, Helvetica, Arial, sans-serif"
INK: Final = "#172026"
MUTED: Final = "#66737a"
TEAL: Final = "#0a7c86"
CORAL: Final = "#d95f49"
AMBER: Final = "#c9851e"
STEEL: Final = "#4d6f80"
PANEL: Final = "#ffffff"
GRID: Final = "#d7dedc"
EPSILON: Final = 1e-12
BUILDER_VERSION: Final = 3


@dataclass(frozen=True)
class MeetingArtifactPaths:
    legacy_artifact_root: Path = Path("checkpoints/Annulus_poisson/coupling/artifacts")
    length_response_root: Path = Path(
        "checkpoints/Annulus_poisson/coupling4/length_response_diagnostics"
    )
    poisson_artifact_root: Path = Path(
        "checkpoints/Annulus_poisson/coupling15/artifacts"
    )
    poisson_weak_root: Path = Path(
        "checkpoints/Annulus_poisson/coupling15/"
        "weak_residual_reliability_blend_comparison"
    )
    poisson_geometry_path: Path = Path("data/geometry/annulus_02_05_1_128.npz")
    poisson_geometry_blend_root: Path = Path(
        "checkpoints/Annulus_poisson/coupling15/compact_c2_cross_axis_blend"
    )
    cdr_artifact_root: Path = Path("checkpoints/annulus_CDR/coupling5/artifacts")
    cdr_weak_root: Path = Path(
        "checkpoints/annulus_CDR/coupling5/weak_residual_reliability_blend_comparison"
    )

    def resolved(self, project_root: Path) -> MeetingArtifactPaths:
        return MeetingArtifactPaths(
            legacy_artifact_root=self._resolve(project_root, self.legacy_artifact_root),
            length_response_root=self._resolve(project_root, self.length_response_root),
            poisson_artifact_root=self._resolve(
                project_root, self.poisson_artifact_root
            ),
            poisson_weak_root=self._resolve(project_root, self.poisson_weak_root),
            poisson_geometry_path=self._resolve(
                project_root, self.poisson_geometry_path
            ),
            poisson_geometry_blend_root=self._resolve(
                project_root, self.poisson_geometry_blend_root
            ),
            cdr_artifact_root=self._resolve(project_root, self.cdr_artifact_root),
            cdr_weak_root=self._resolve(project_root, self.cdr_weak_root),
        )

    @staticmethod
    def _resolve(project_root: Path, path: Path) -> Path:
        return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class MeetingAssetConfig:
    project_root: Path = PROJECT_ROOT
    outdir: Path = DEFAULT_OUTDIR
    paths: MeetingArtifactPaths = MeetingArtifactPaths()
    overwrite: bool = False

    @property
    def resolved_paths(self) -> MeetingArtifactPaths:
        return self.paths.resolved(self.project_root)


@dataclass(frozen=True)
class SelectedSample:
    sample_id: int
    prefix: str
    arrays: dict[str, np.ndarray]
    metrics: dict[str, float | str]


@dataclass(frozen=True)
class WeakBlendSample:
    sample_id: int
    position: int
    arrays: dict[str, np.ndarray]
    metrics: dict[str, float | str]


@dataclass(frozen=True)
class CoefficientFields:
    coords_valid: np.ndarray
    a: np.ndarray
    bx: np.ndarray
    by: np.ndarray
    b_magnitude: np.ndarray
    c: np.ndarray
    quiver_indices: np.ndarray


@dataclass(frozen=True)
class GeometryC2DetailFields:
    coords_valid: np.ndarray
    distance_phi: np.ndarray
    distance_psi: np.ndarray
    influence_phi: np.ndarray
    influence_psi: np.ndarray
    theta: np.ndarray
    w_phi: np.ndarray
    w_psi: np.ndarray


@dataclass(frozen=True)
class MismatchSeamDetailFields:
    coords_valid: np.ndarray
    mismatch: np.ndarray
    x_profile: np.ndarray
    y_profile: np.ndarray
    x_midpoints: np.ndarray
    y_midpoints: np.ndarray
    x_seams: np.ndarray
    y_seams: np.ndarray
    theta: np.ndarray
    w_phi: np.ndarray
    w_psi: np.ndarray


class LoggingMixin:
    logger: logging.Logger

    @staticmethod
    def build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("AnnulusMeetingAssetBuilder")
        logger.handlers.clear()

        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()
        return logger


class ProvenanceMixin:
    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def file_provenance(cls, path: Path) -> dict[str, str | int]:
        if not path.is_file():
            raise FileNotFoundError(f"Provenance input is missing: {path}")
        return {
            "sha256": cls.sha256(path),
            "size_bytes": path.stat().st_size,
        }


class ArtifactLoaderMixin:
    logger: logging.Logger

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required presentation artifact is missing: {path}"
            )

    @staticmethod
    def _validate_finite(name: str, values: np.ndarray) -> None:
        if not np.issubdtype(values.dtype, np.number):
            return
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    @classmethod
    def _require_keys(
        cls,
        archive: np.lib.npyio.NpzFile,
        required: Iterable[str],
        path: Path,
    ) -> None:
        missing = sorted(set(required).difference(archive.files))
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")

    @staticmethod
    def _sample_prefix(archive: np.lib.npyio.NpzFile, sample_id: int) -> str:
        suffix = f"sample_{sample_id:06d}_coords_valid"
        matches = [
            key[: -len("_coords_valid")]
            for key in archive.files
            if key.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one selected prefix ending in {suffix!r}; "
                f"found {matches}"
            )
        return matches[0]

    @staticmethod
    def _metrics_row(path: Path, sample_id: int) -> dict[str, float | str]:
        if not path.is_file():
            return {}
        frame = pd.read_csv(path)
        if "sample_id" not in frame.columns:
            raise ValueError(f"{path} does not contain a sample_id column")
        rows = frame.loc[frame["sample_id"].astype(int) == sample_id]
        if len(rows) != 1:
            raise ValueError(
                f"Expected one metrics row for sample {sample_id} in {path}; "
                f"found {len(rows)}"
            )
        row = rows.iloc[0]
        return {
            str(key): (
                float(value)
                if isinstance(value, (int, float, np.number))
                else str(value)
            )
            for key, value in row.items()
        }

    def load_selected_sample(
        self,
        artifact_root: Path,
        sample_id: int,
        required_fields: Iterable[str],
    ) -> SelectedSample:
        archive_path = artifact_root / "data" / "selected_raw_arrays.npz"
        self._require_file(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            prefix = self._sample_prefix(archive, sample_id)
            keys = [f"{prefix}_{field}" for field in required_fields]
            self._require_keys(archive, keys, archive_path)
            arrays = {
                field: np.asarray(archive[f"{prefix}_{field}"])
                for field in required_fields
            }
        for field, values in arrays.items():
            self._validate_finite(f"{prefix}_{field}", values)
        coords = arrays["coords_valid"]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"{prefix}_coords_valid must have shape (P, 2), got {coords.shape}"
            )
        point_count = coords.shape[0]
        for field, values in arrays.items():
            if field == "coords_valid":
                continue
            if values.shape != (point_count,):
                raise ValueError(
                    f"{prefix}_{field} must have shape ({point_count},), "
                    f"got {values.shape}"
                )
        metrics = self._metrics_row(
            artifact_root / "metrics" / "per_sample_metrics.csv", sample_id
        )
        self.logger.info("Loaded sample %s from %s", sample_id, archive_path)
        return SelectedSample(
            sample_id=sample_id,
            prefix=prefix,
            arrays=arrays,
            metrics=metrics,
        )

    def load_weak_sample(self, weak_root: Path, sample_id: int) -> WeakBlendSample:
        archive_path = weak_root / "data" / "selected_weak_residual_blend_arrays.npz"
        self._require_file(archive_path)
        required = (
            "selected_sample_ids",
            "coords_valid",
            "sol",
            "rhs",
            "projected_physical",
            "u_phi",
            "u_psi",
            "u_equal_mean",
            "u_geometry_c2",
            "u_mismatch_seam_c2",
            "u_weak_residual_reliability",
            "weak_phi_indicator",
            "weak_psi_indicator",
            "weak_theta",
            "weak_w_phi",
            "weak_w_psi",
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            self._require_keys(archive, required, archive_path)
            sample_ids = np.asarray(archive["selected_sample_ids"], dtype=np.int64)
            matches = np.flatnonzero(sample_ids == sample_id)
            if matches.size != 1:
                raise ValueError(
                    f"Expected sample {sample_id} once in {archive_path}; "
                    f"available ids are {sample_ids.tolist()}"
                )
            position = int(matches[0])
            arrays = {"coords_valid": np.asarray(archive["coords_valid"])}
            for field in required[2:]:
                values = np.asarray(archive[field])
                arrays[field] = values[position]
        coords = arrays["coords_valid"]
        point_count = coords.shape[0]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Weak coords_valid must have shape (P, 2), got {coords.shape}"
            )
        if arrays["projected_physical"].shape != (2, point_count):
            raise ValueError(
                "Weak projected_physical must have shape (2, P), got "
                f"{arrays['projected_physical'].shape}"
            )
        for field, values in arrays.items():
            self._validate_finite(f"weak sample {sample_id} {field}", values)
            if field not in {"coords_valid", "projected_physical"} and values.shape != (
                point_count,
            ):
                raise ValueError(
                    f"Weak field {field} must have shape ({point_count},), "
                    f"got {values.shape}"
                )
        metrics = self._metrics_row(
            weak_root / "metrics" / "per_sample_weak_residual_blend_comparison.csv",
            sample_id,
        )
        self.logger.info("Loaded weak sample %s from %s", sample_id, archive_path)
        return WeakBlendSample(
            sample_id=sample_id,
            position=position,
            arrays=arrays,
            metrics=metrics,
        )

    def load_geometry_c2_detail(self) -> GeometryC2DetailFields:
        archive_path = (
            self.paths.poisson_geometry_blend_root
            / "data"
            / "selected_fixed_smooth_blend_arrays.npz"
        )
        self._require_file(archive_path)
        required = (
            "coords_valid",
            "distance_phi",
            "distance_psi",
            "influence_phi",
            "influence_psi",
            "theta",
            "w_phi",
            "w_psi",
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            self._require_keys(archive, required, archive_path)
            arrays = {field: np.asarray(archive[field]) for field in required}
        coords = arrays["coords_valid"]
        point_count = coords.shape[0]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Geometry C2 coords must have shape (P, 2), got {coords.shape}"
            )
        for field in required[1:]:
            values = arrays[field]
            self._validate_finite(f"geometry C2 {field}", values)
            if values.shape != (point_count,):
                raise ValueError(
                    f"Geometry C2 field {field} must have shape ({point_count},), "
                    f"got {values.shape}"
                )
        if not np.allclose(
            arrays["w_phi"] + arrays["w_psi"], 1.0, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("Geometry C2 weights must sum to one")
        return GeometryC2DetailFields(**arrays)

    def build_mismatch_seam_detail(
        self,
        weak: WeakBlendSample,
    ) -> MismatchSeamDetailFields:
        geometry = load_complex_geometry(
            self.paths.poisson_geometry_path,
            dtype=torch.float64,
            device="cpu",
        )
        coords = weak.arrays["coords_valid"]
        geometry_coords = geometry.coords_valid.detach().cpu().numpy()
        if not np.allclose(coords, geometry_coords, atol=1e-12, rtol=1e-12):
            raise ValueError("Poisson weak sample coordinates do not match geometry")
        u_phi = torch.as_tensor(weak.arrays["u_phi"][None, :], dtype=torch.float64)
        u_psi = torch.as_tensor(weak.arrays["u_psi"][None, :], dtype=torch.float64)
        mismatch = MismatchGradientBlendMixin.build_mismatch_gradient_fields(
            geometry,
            u_phi,
            u_psi,
            MismatchGradientBlendConfig(),
        )
        seam = MismatchGradientBlendMixin.build_mismatch_seam_c2_fields(
            geometry,
            mismatch,
            MismatchSeamC2BlendConfig(
                gamma=0.3,
                ramp_width=0.09375,
                max_seams_per_axis=2,
                peak_relative_threshold=0.25,
                profile_smoothing_steps=1,
            ),
        )

        def numpy(values: torch.Tensor) -> np.ndarray:
            return values.detach().cpu().numpy()

        return MismatchSeamDetailFields(
            coords_valid=coords,
            mismatch=numpy(mismatch.mismatch[0]),
            x_profile=numpy(seam.x_edge_profile[0]),
            y_profile=numpy(seam.y_edge_profile[0]),
            x_midpoints=numpy(seam.x_edge_midpoints),
            y_midpoints=numpy(seam.y_edge_midpoints),
            x_seams=numpy(seam.x_seam_coordinates[0]),
            y_seams=numpy(seam.y_seam_coordinates[0]),
            theta=numpy(seam.theta[0]),
            w_phi=numpy(seam.w_phi[0]),
            w_psi=numpy(seam.w_psi[0]),
        )

    def load_coefficient_fields(self, artifact_root: Path) -> CoefficientFields:
        archive_path = artifact_root / "data" / "coefficient_fields.npz"
        self._require_file(archive_path)
        required = (
            "coords_valid",
            "a",
            "bx",
            "by",
            "b_magnitude",
            "c",
            "quiver_indices",
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            self._require_keys(archive, required, archive_path)
            arrays = {key: np.asarray(archive[key]) for key in required}
        coords = arrays["coords_valid"]
        point_count = coords.shape[0]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Coefficient coords must have shape (P, 2), got {coords.shape}"
            )
        for field in ("a", "bx", "by", "b_magnitude", "c"):
            if arrays[field].shape != (point_count,):
                raise ValueError(
                    f"Coefficient field {field} must have shape ({point_count},), "
                    f"got {arrays[field].shape}"
                )
            self._validate_finite(f"coefficient {field}", arrays[field])
        indices = np.asarray(arrays["quiver_indices"], dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= point_count):
            raise ValueError(
                "quiver_indices must be valid one-dimensional point indices"
            )
        expected_magnitude = np.hypot(arrays["bx"], arrays["by"])
        if not np.allclose(
            arrays["b_magnitude"], expected_magnitude, atol=1e-12, rtol=1e-12
        ):
            raise ValueError("b_magnitude is inconsistent with bx/by")
        return CoefficientFields(
            coords_valid=coords,
            a=arrays["a"],
            bx=arrays["bx"],
            by=arrays["by"],
            b_magnitude=arrays["b_magnitude"],
            c=arrays["c"],
            quiver_indices=indices,
        )

    @staticmethod
    def load_metrics_frame(weak_root: Path) -> pd.DataFrame:
        path = weak_root / "metrics" / "per_sample_weak_residual_blend_comparison.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required comparison metrics are missing: {path}")
        frame = pd.read_csv(path)
        required = {
            "sample_id",
            "equal_mean_rel_sol",
            "geometry_c2_rel_sol",
            "mismatch_seam_c2_rel_sol",
            "weak_residual_reliability_rel_sol",
            "equal_mean_transition_error_rms",
            "geometry_c2_transition_error_rms",
            "mismatch_seam_c2_transition_error_rms",
            "weak_residual_reliability_transition_error_rms",
            "equal_mean_transition_trace_error_jump_rms",
            "geometry_c2_transition_trace_error_jump_rms",
            "mismatch_seam_c2_transition_trace_error_jump_rms",
            "weak_residual_reliability_transition_trace_error_jump_rms",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{path} contains no samples")
        return frame

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Required JSON artifact is missing: {path}")
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return raw


class DiagnosticsMixin:
    @staticmethod
    def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
        return float(
            np.linalg.norm((prediction - reference).ravel())
            / max(float(np.linalg.norm(reference.ravel())), EPSILON)
        )

    @staticmethod
    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    @staticmethod
    def relative_change(candidate: pd.Series, baseline: pd.Series) -> float:
        return float(candidate.mean() / baseline.mean() - 1.0)

    @classmethod
    def comparison_summary(cls, frame: pd.DataFrame) -> dict[str, Any]:
        specifications = {
            "equal_mean": "equal_mean",
            "geometry_c2": "geometry_c2",
            "mismatch_seam_c2": "mismatch_seam_c2",
            "weak_residual_reliability": "weak_residual_reliability",
        }
        baseline_rel = frame["equal_mean_rel_sol"]
        baseline_transition = frame["equal_mean_transition_error_rms"]
        baseline_trace = frame["equal_mean_transition_trace_error_jump_rms"]
        summary: dict[str, Any] = {"num_samples": int(len(frame)), "estimators": {}}
        for name, prefix in specifications.items():
            rel = frame[f"{prefix}_rel_sol"]
            transition = frame[f"{prefix}_transition_error_rms"]
            trace = frame[f"{prefix}_transition_trace_error_jump_rms"]
            summary["estimators"][name] = {
                "mean_rel_sol": float(rel.mean()),
                "median_rel_sol": float(rel.median()),
                "rel_sol_change_vs_equal": (
                    0.0
                    if name == "equal_mean"
                    else cls.relative_change(rel, baseline_rel)
                ),
                "win_count_vs_equal": (
                    0 if name == "equal_mean" else int((rel < baseline_rel).sum())
                ),
                "transition_rms_change_vs_equal": (
                    0.0
                    if name == "equal_mean"
                    else cls.relative_change(transition, baseline_transition)
                ),
                "trace_jump_rms_change_vs_equal": (
                    0.0
                    if name == "equal_mean"
                    else cls.relative_change(trace, baseline_trace)
                ),
            }
        return summary

    @staticmethod
    def validate_standard_weak_alignment(
        label: str,
        standard: SelectedSample,
        weak: WeakBlendSample,
    ) -> None:
        pairs = (
            (standard.arrays["coords_valid"], weak.arrays["coords_valid"], "coords"),
            (standard.arrays["rhs"], weak.arrays["rhs"], "rhs"),
            (standard.arrays["sol"], weak.arrays["sol"], "sol"),
            (standard.arrays["u_phi"], weak.arrays["u_phi"], "u_phi"),
            (standard.arrays["u_psi"], weak.arrays["u_psi"], "u_psi"),
            (
                standard.arrays["phi"],
                weak.arrays["projected_physical"][0],
                "phi",
            ),
            (
                standard.arrays["psi"],
                weak.arrays["projected_physical"][1],
                "psi",
            ),
        )
        for standard_values, weak_values, field in pairs:
            if not np.allclose(standard_values, weak_values, atol=1e-12, rtol=1e-12):
                maximum = float(np.max(np.abs(standard_values - weak_values)))
                raise ValueError(
                    f"{label} standard/weak {field} mismatch; max abs={maximum:.3e}"
                )

    @classmethod
    def selected_result_metrics(
        cls,
        standard: SelectedSample,
        weak: WeakBlendSample,
    ) -> dict[str, float]:
        arrays = standard.arrays
        weak_prediction = weak.arrays["u_weak_residual_reliability"]
        source_balance = arrays["phi"] + arrays["psi"] - arrays["rhs"]
        return {
            "sample_id": float(standard.sample_id),
            "u_phi_rel_sol": cls.relative_l2(arrays["u_phi"], arrays["sol"]),
            "u_psi_rel_sol": cls.relative_l2(arrays["u_psi"], arrays["sol"]),
            "weak_rel_sol": cls.relative_l2(weak_prediction, arrays["sol"]),
            "balance_max_abs": float(np.max(np.abs(source_balance))),
            "phi_error_rms": cls.rms(arrays["phi_error"]),
            "psi_error_rms": cls.rms(arrays["psi_error"]),
            "u_phi_error_rms": cls.rms(arrays["u_phi"] - arrays["sol"]),
            "u_psi_error_rms": cls.rms(arrays["u_psi"] - arrays["sol"]),
            "weak_error_rms": cls.rms(weak_prediction - arrays["sol"]),
            "phi_relative_diagnostic": cls.relative_l2(
                arrays["phi"], arrays["target_phi"]
            ),
            "psi_relative_diagnostic": cls.relative_l2(
                arrays["psi"], arrays["target_psi"]
            ),
        }


class PlotlyFigureMixin:
    @staticmethod
    def _range(values: np.ndarray, padding_fraction: float = 0.04) -> list[float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        width = max(maximum - minimum, EPSILON)
        padding = padding_fraction * width
        return [minimum - padding, maximum + padding]

    @staticmethod
    def _symmetric_limit(*values: np.ndarray) -> float:
        maximum = max(float(np.max(np.abs(value))) for value in values)
        return max(maximum, EPSILON)

    @staticmethod
    def _sequential_range(values: np.ndarray) -> tuple[float, float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if np.isclose(minimum, maximum):
            padding = max(abs(minimum), 1.0) * 0.02
            return minimum - padding, maximum + padding
        return minimum, maximum

    @staticmethod
    def _field_trace(
        coords: np.ndarray,
        values: np.ndarray,
        label: str,
        coloraxis: str,
        point_size: float = 3.1,
    ) -> go.Scattergl:
        return go.Scattergl(
            x=coords[:, 0],
            y=coords[:, 1],
            mode="markers",
            marker={
                "color": values,
                "coloraxis": coloraxis,
                "size": point_size,
                "line": {"width": 0},
            },
            customdata=np.column_stack((values,)),
            hovertemplate=(
                f"x=%{{x:.5f}}<br>y=%{{y:.5f}}<br>{label}=%{{customdata[0]:.6e}}"
                "<extra></extra>"
            ),
            name=label,
            showlegend=False,
        )

    @staticmethod
    def _base_layout(width: int, height: int) -> dict[str, Any]:
        return {
            "template": "plotly_white",
            "width": width,
            "height": height,
            "autosize": True,
            "paper_bgcolor": PANEL,
            "plot_bgcolor": PANEL,
            "font": {"family": FONT_FAMILY, "size": 17, "color": INK},
            "margin": {"l": 36, "r": 88, "t": 54, "b": 30},
            "hoverlabel": {"font": {"family": FONT_FAMILY}},
        }

    @classmethod
    def _style_field_axes(cls, figure: go.Figure, coords: np.ndarray) -> None:
        x_range = cls._range(coords[:, 0])
        y_range = cls._range(coords[:, 1])
        x_keys = sorted(
            (key for key in figure.layout if key.startswith("xaxis")),
            key=lambda key: (len(key), key),
        )
        for x_key in x_keys:
            suffix = x_key.removeprefix("xaxis")
            x_ref = f"x{suffix}" if suffix else "x"
            y_key = f"yaxis{suffix}"
            if y_key not in figure.layout:
                continue
            figure.layout[x_key].update(
                range=x_range,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=GRID,
                mirror=True,
                tickfont={"size": 11},
                nticks=3,
                constrain="domain",
            )
            figure.layout[y_key].update(
                range=y_range,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=GRID,
                mirror=True,
                tickfont={"size": 11},
                nticks=3,
                scaleanchor=x_ref,
                scaleratio=1,
                constrain="domain",
            )

    @classmethod
    def build_legacy_error_matrix(
        cls,
        sample: SelectedSample,
        *,
        show_cardinal_markers: bool = False,
    ) -> go.Figure:
        arrays = sample.arrays
        coords = arrays["coords_valid"]
        source_limit = cls._symmetric_limit(arrays["phi_error"], arrays["psi_error"])
        solution_limit = cls._symmetric_limit(
            arrays["u_phi_error"],
            arrays["u_pred_error"],
            arrays["u_psi_error"],
        )
        figure = make_subplots(
            rows=2,
            cols=6,
            specs=[
                [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
                [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            ],
            subplot_titles=(
                "phi - phi*",
                "psi - psi*",
                "u_phi - u",
                "u_pred - u",
                "u_psi - u",
            ),
            horizontal_spacing=0.05,
            vertical_spacing=0.16,
        )
        for field, label, row, col, coloraxis in (
            ("phi_error", "phi - phi*", 1, 1, "coloraxis"),
            ("psi_error", "psi - psi*", 1, 4, "coloraxis"),
            ("u_phi_error", "u_phi - u", 2, 1, "coloraxis2"),
            ("u_pred_error", "u_pred - u", 2, 3, "coloraxis2"),
            ("u_psi_error", "u_psi - u", 2, 5, "coloraxis2"),
        ):
            figure.add_trace(
                cls._field_trace(coords, arrays[field], label, coloraxis),
                row=row,
                col=col,
            )
        if show_cardinal_markers:
            marker_x = np.asarray([0.2, -0.2, 0.0, 0.0])
            marker_y = np.asarray([0.0, 0.0, 0.2, -0.2])
            figure.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode="markers",
                    marker={
                        "symbol": "circle-open",
                        "size": 12,
                        "color": INK,
                        "line": {"width": 2},
                    },
                    hovertemplate="axial transition at r=0.2<extra></extra>",
                    showlegend=False,
                ),
                row=2,
                col=3,
            )
        layout = cls._base_layout(1600, 850)
        layout.update(
            coloraxis={
                "colorscale": "RdBu",
                "cmin": -source_limit,
                "cmax": source_limit,
                "colorbar": {"title": "source error", "len": 0.37, "y": 0.79},
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -solution_limit,
                "cmax": solution_limit,
                "colorbar": {"title": "solution error", "len": 0.37, "y": 0.23},
            },
            annotations=list(figure.layout.annotations)
            + [
                {
                    "text": "Directional-source errors against numerical FEniCSx targets",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": 1.08,
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 18, "color": STEEL},
                },
                {
                    "text": "Solution-reconstruction errors",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.0,
                    "y": 0.47,
                    "showarrow": False,
                    "xanchor": "left",
                    "font": {"size": 18, "color": STEEL},
                },
            ]
            + (
                [
                    {
                        "text": "axial transition at r=0.2",
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.50,
                        "y": -0.04,
                        "showarrow": False,
                        "font": {"size": 14, "color": INK},
                    }
                ]
                if show_cardinal_markers
                else []
            ),
        )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @classmethod
    def build_four_way_scatter(cls, frame: pd.DataFrame, equation: str) -> go.Figure:
        baseline = 100.0 * frame["equal_mean_rel_sol"].to_numpy(float)
        methods = (
            ("Geometry C2", "geometry_c2_rel_sol", AMBER),
            ("Mismatch seam C2", "mismatch_seam_c2_rel_sol", CORAL),
            ("Local weak residual", "weak_residual_reliability_rel_sol", TEAL),
        )
        values = [baseline]
        figure = go.Figure()
        for label, column, color in methods:
            candidate = 100.0 * frame[column].to_numpy(float)
            values.append(candidate)
            figure.add_trace(
                go.Scatter(
                    x=baseline,
                    y=candidate,
                    mode="markers",
                    marker={"size": 9, "color": color, "opacity": 0.78},
                    name=label,
                    customdata=np.asarray(frame["sample_id"], dtype=int),
                    hovertemplate=(
                        "sample=%{customdata}<br>equal=%{x:.3f}%"
                        "<br>candidate=%{y:.3f}%<extra></extra>"
                    ),
                )
            )
        combined = np.concatenate(values)
        lower = float(np.min(combined)) * 0.94
        upper = float(np.max(combined)) * 1.04
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"color": MUTED, "dash": "dash", "width": 2},
                name="no change",
                hoverinfo="skip",
            )
        )
        layout = cls._base_layout(980, 650)
        layout.update(
            title={
                "text": f"{equation}: paired test-sample relative solution error",
                "x": 0.02,
            },
            xaxis={"title": "Equal mean rel. sol. error (%)", "range": [lower, upper]},
            yaxis={
                "title": "Candidate rel. sol. error (%)",
                "range": [lower, upper],
            },
            showlegend=False,
            margin={"l": 78, "r": 24, "t": 68, "b": 66},
        )
        figure.update_layout(**layout)
        return figure

    @classmethod
    def build_weak_inset(cls, weak: WeakBlendSample) -> go.Figure:
        coords = weak.arrays["coords_valid"]
        equal_error = weak.arrays["u_equal_mean"] - weak.arrays["sol"]
        weak_error = weak.arrays["u_weak_residual_reliability"] - weak.arrays["sol"]
        error_limit = cls._symmetric_limit(equal_error, weak_error)
        figure = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=("weak w_phi", "equal-mean error", "weak-blend error"),
            horizontal_spacing=0.045,
        )
        figure.add_trace(
            cls._field_trace(coords, weak.arrays["weak_w_phi"], "w_phi", "coloraxis"),
            row=1,
            col=1,
        )
        figure.add_trace(
            cls._field_trace(coords, equal_error, "equal error", "coloraxis2"),
            row=1,
            col=2,
        )
        figure.add_trace(
            cls._field_trace(coords, weak_error, "weak error", "coloraxis2"),
            row=1,
            col=3,
        )
        layout = cls._base_layout(1180, 410)
        layout.update(
            coloraxis={
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": False,
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -error_limit,
                "cmax": error_limit,
                "colorbar": {"title": "signed error", "len": 0.8},
            },
        )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @classmethod
    def build_geometry_c2_detail(cls, fields: GeometryC2DetailFields) -> go.Figure:
        coords = fields.coords_valid
        figure = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "B(d_phi / delta)",
                "B(d_psi / delta)",
                "theta_geom",
                "final w_phi",
            ),
            horizontal_spacing=0.035,
        )
        for values, label, column, coloraxis in (
            (fields.influence_phi, "B_phi", 1, "coloraxis"),
            (fields.influence_psi, "B_psi", 2, "coloraxis"),
            (fields.theta, "theta_geom", 3, "coloraxis2"),
            (fields.w_phi, "w_phi", 4, "coloraxis3"),
        ):
            figure.add_trace(
                cls._field_trace(
                    coords,
                    values,
                    label,
                    coloraxis,
                    point_size=2.8,
                ),
                row=1,
                col=column,
            )
        theta_limit = cls._symmetric_limit(fields.theta)
        weight_limit = max(
            float(np.max(np.abs(fields.w_phi - 0.5))),
            float(np.max(np.abs(fields.w_psi - 0.5))),
            EPSILON,
        )
        layout = cls._base_layout(1500, 510)
        layout.update(
            title={
                "text": "Known transition coordinates -> compact C2 influence -> partition weight",
                "x": 0.02,
            },
            coloraxis={
                "colorscale": "YlOrBr",
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": False,
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -theta_limit,
                "cmax": theta_limit,
                "showscale": False,
            },
            coloraxis3={
                "colorscale": "Teal",
                "cmin": 0.5 - weight_limit,
                "cmax": 0.5 + weight_limit,
                "colorbar": {"title": "w_phi", "len": 0.72},
            },
            margin={"l": 30, "r": 82, "t": 76, "b": 32},
        )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @classmethod
    def build_mismatch_seam_detail_figure(
        cls,
        fields: MismatchSeamDetailFields,
    ) -> go.Figure:
        coords = fields.coords_valid
        figure = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "m = u_phi - u_psi",
                "x/y profiles + detected seams",
                "theta_seam",
                "final w_phi",
            ),
            horizontal_spacing=0.045,
        )
        mismatch_limit = cls._symmetric_limit(fields.mismatch)
        theta_limit = cls._symmetric_limit(fields.theta)
        figure.add_trace(
            cls._field_trace(
                coords,
                fields.mismatch,
                "mismatch",
                "coloraxis",
                point_size=2.8,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=fields.x_midpoints,
                y=fields.x_profile,
                mode="lines",
                line={"color": CORAL, "width": 3},
                name="x-edge profile",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=fields.y_midpoints,
                y=fields.y_profile,
                mode="lines",
                line={"color": STEEL, "width": 3},
                name="y-edge profile",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        for seam, color in (
            (fields.x_seams, CORAL),
            (fields.y_seams, STEEL),
        ):
            for coordinate in seam:
                figure.add_vline(
                    x=float(coordinate),
                    line={"color": color, "dash": "dot", "width": 1.5},
                    row=1,
                    col=2,
                )
        figure.add_trace(
            cls._field_trace(
                coords,
                fields.theta,
                "theta_seam",
                "coloraxis2",
                point_size=2.8,
            ),
            row=1,
            col=3,
        )
        figure.add_trace(
            cls._field_trace(
                coords,
                fields.w_phi,
                "w_phi",
                "coloraxis3",
                point_size=2.8,
            ),
            row=1,
            col=4,
        )
        layout = cls._base_layout(1500, 510)
        layout.update(
            title={
                "text": "Mismatch -> detected seams -> fixed compact C2 weight",
                "x": 0.02,
            },
            coloraxis={
                "colorscale": "RdBu",
                "cmin": -mismatch_limit,
                "cmax": mismatch_limit,
                "showscale": False,
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -theta_limit,
                "cmax": theta_limit,
                "showscale": False,
            },
            coloraxis3={
                "colorscale": "Teal",
                "cmin": 0.35,
                "cmax": 0.65,
                "colorbar": {"title": "w_phi", "len": 0.72},
            },
            margin={"l": 30, "r": 82, "t": 76, "b": 36},
        )
        figure.update_layout(**layout)
        x_range = cls._range(coords[:, 0])
        y_range = cls._range(coords[:, 1])
        for column in (1, 3, 4):
            figure.update_xaxes(
                range=x_range,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=GRID,
                mirror=True,
                nticks=3,
                constrain="domain",
                row=1,
                col=column,
            )
            figure.update_yaxes(
                range=y_range,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor=GRID,
                mirror=True,
                nticks=3,
                scaleanchor=f"x{column}" if column > 1 else "x",
                scaleratio=1,
                constrain="domain",
                row=1,
                col=column,
            )
        figure.update_xaxes(tickfont={"size": 13}, row=1, col=2)
        figure.update_yaxes(tickfont={"size": 13}, row=1, col=2)
        return figure

    @classmethod
    def build_weak_reliability_detail(cls, weak: WeakBlendSample) -> go.Figure:
        coords = weak.arrays["coords_valid"]
        phi_indicator = weak.arrays["weak_phi_indicator"]
        psi_indicator = weak.arrays["weak_psi_indicator"]
        indicator_floor = max(
            EPSILON,
            0.1 * 0.5 * (float(np.mean(phi_indicator)) + float(np.mean(psi_indicator))),
        )
        phi_display = np.log10(phi_indicator + indicator_floor)
        psi_display = np.log10(psi_indicator + indicator_floor)
        indicator_min = float(min(np.min(phi_display), np.min(psi_display)))
        indicator_max = float(max(np.max(phi_display), np.max(psi_display)))
        theta = weak.arrays["weak_theta"]
        theta_limit = cls._symmetric_limit(theta)
        figure = make_subplots(
            rows=1,
            cols=4,
            subplot_titles=(
                "log10 local eta_phi^2",
                "log10 local eta_psi^2",
                "theta_weak",
                "final w_phi",
            ),
            horizontal_spacing=0.045,
        )
        for values, label, column, coloraxis in (
            (phi_display, "eta_phi", 1, "coloraxis"),
            (psi_display, "eta_psi", 2, "coloraxis"),
            (theta, "theta_weak", 3, "coloraxis2"),
            (weak.arrays["weak_w_phi"], "w_phi", 4, "coloraxis3"),
        ):
            figure.add_trace(
                cls._field_trace(
                    coords,
                    values,
                    label,
                    coloraxis,
                    point_size=2.8,
                ),
                row=1,
                col=column,
            )
        layout = cls._base_layout(1500, 510)
        layout.update(
            title={
                "text": "Weak defect -> candidate reliability -> partition weight",
                "x": 0.02,
            },
            coloraxis={
                "colorscale": "Viridis",
                "cmin": indicator_min,
                "cmax": indicator_max,
                "showscale": False,
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -theta_limit,
                "cmax": theta_limit,
                "showscale": False,
            },
            coloraxis3={
                "colorscale": "Teal",
                "cmin": 0.25,
                "cmax": 0.75,
                "colorbar": {"title": "w_phi", "len": 0.72},
            },
            margin={"l": 30, "r": 82, "t": 76, "b": 32},
        )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @classmethod
    def build_cross_equation_comparison(
        cls,
        poisson_summary: dict[str, Any],
        cdr_summary: dict[str, Any],
    ) -> go.Figure:
        labels = ["Geometry C2", "Mismatch seam C2", "Weak residual"]
        keys = ["geometry_c2", "mismatch_seam_c2", "weak_residual_reliability"]
        poisson = [
            100.0 * poisson_summary["estimators"][key]["rel_sol_change_vs_equal"]
            for key in keys
        ]
        cdr = [
            100.0 * cdr_summary["estimators"][key]["rel_sol_change_vs_equal"]
            for key in keys
        ]
        figure = go.Figure(
            data=[
                go.Bar(
                    name="Pure Poisson",
                    x=labels,
                    y=poisson,
                    marker_color=TEAL,
                    text=[f"{value:.2f}%" for value in poisson],
                    textposition="outside",
                ),
                go.Bar(
                    name="CDR",
                    x=labels,
                    y=cdr,
                    marker_color=CORAL,
                    text=[f"{value:.2f}%" for value in cdr],
                    textposition="outside",
                ),
            ]
        )
        layout = cls._base_layout(980, 560)
        layout.update(
            barmode="group",
            title={
                "text": "Relative solution-error change versus equal mean",
                "x": 0.02,
            },
            yaxis={"title": "Change (%)", "zeroline": True, "zerolinecolor": INK},
            showlegend=False,
            margin={"l": 72, "r": 28, "t": 66, "b": 66},
        )
        figure.update_layout(**layout)
        return figure

    @staticmethod
    def _result_colorbar(title: str, y: float) -> dict[str, Any]:
        return {
            "title": {
                "text": title,
                "side": "right",
                "font": {"size": 13, "color": INK},
            },
            "len": 0.24,
            "y": y,
            "x": 1.012,
            "xanchor": "left",
            "xpad": 10,
            "thickness": 13,
            "tickfont": {"size": 11, "color": INK},
        }

    @classmethod
    def build_result_fields(
        cls,
        standard: SelectedSample,
        weak: WeakBlendSample,
        coefficients: CoefficientFields | None = None,
    ) -> go.Figure:
        arrays = standard.arrays
        coords = arrays["coords_valid"]
        weak_prediction = weak.arrays["u_weak_residual_reliability"]
        source_component_limit = cls._symmetric_limit(arrays["phi"], arrays["psi"])
        rhs_min, rhs_max = cls._sequential_range(arrays["rhs"])
        solution_min = min(
            float(np.min(arrays["sol"])),
            float(np.min(arrays["u_phi"])),
            float(np.min(arrays["u_psi"])),
            float(np.min(weak_prediction)),
        )
        solution_max = max(
            float(np.max(arrays["sol"])),
            float(np.max(arrays["u_phi"])),
            float(np.max(arrays["u_psi"])),
            float(np.max(weak_prediction)),
        )
        has_coefficients = coefficients is not None
        rows = 3 if has_coefficients else 2
        titles: list[str]
        specs: list[list[dict[str, Any] | None]]
        if has_coefficients:
            titles = [
                "diffusion a",
                "convection b",
                "reaction c",
                "coefficient context",
                "source f",
                "directional phi",
                "directional psi",
                "weak w_phi",
                "reference u",
                "u_phi",
                "u_psi",
                "weak u_pred",
            ]
            specs = [[{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}]]
        else:
            titles = [
                "source f",
                "directional phi",
                "directional psi",
                "weak w_phi",
                "reference u",
                "u_phi",
                "u_psi",
                "weak u_pred",
            ]
            specs = [[{}, {}, {}, {}], [{}, {}, {}, {}]]
        figure = make_subplots(
            rows=rows,
            cols=4,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.035,
            vertical_spacing=0.105 if has_coefficients else 0.13,
        )
        source_row = 2 if has_coefficients else 1
        solution_row = source_row + 1

        if coefficients is not None:
            if not np.allclose(
                coefficients.coords_valid, coords, atol=1e-12, rtol=1e-12
            ):
                raise ValueError(
                    "CDR coefficient coordinates do not match sample coordinates"
                )
            a_min, a_max = cls._sequential_range(coefficients.a)
            c_min, c_max = cls._sequential_range(coefficients.c)
            b_min, b_max = cls._sequential_range(coefficients.b_magnitude)
            for values, label, column, coloraxis in (
                (coefficients.a, "a", 1, "coloraxis5"),
                (coefficients.b_magnitude, "|b|", 2, "coloraxis6"),
                (coefficients.c, "c", 3, "coloraxis7"),
            ):
                figure.add_trace(
                    cls._field_trace(coords, values, label, coloraxis, point_size=2.8),
                    row=1,
                    col=column,
                )
            quiver = cls._quiver_trace(coefficients)
            for trace in quiver:
                figure.add_trace(trace, row=1, col=2)
            figure.add_annotation(
                text="physical coefficients<br>direct at coords_valid",
                x=0.875,
                y=0.86,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="center",
                font={"size": 14, "color": STEEL},
            )
        else:
            a_min = a_max = c_min = c_max = b_min = b_max = 0.0

        for values, label, column, coloraxis in (
            (arrays["rhs"], "f", 1, "coloraxis"),
            (arrays["phi"], "phi", 2, "coloraxis2"),
            (arrays["psi"], "psi", 3, "coloraxis2"),
            (weak.arrays["weak_w_phi"], "w_phi", 4, "coloraxis3"),
        ):
            figure.add_trace(
                cls._field_trace(coords, values, label, coloraxis, point_size=2.9),
                row=source_row,
                col=column,
            )
        for values, label, column in (
            (arrays["sol"], "u", 1),
            (arrays["u_phi"], "u_phi", 2),
            (arrays["u_psi"], "u_psi", 3),
            (weak_prediction, "u_pred weak", 4),
        ):
            figure.add_trace(
                cls._field_trace(coords, values, label, "coloraxis4", point_size=2.9),
                row=solution_row,
                col=column,
            )
        height = 900 if has_coefficients else 720
        layout = cls._base_layout(1600, height)
        layout.update(
            coloraxis={
                "colorscale": "Viridis",
                "cmin": rhs_min,
                "cmax": rhs_max,
                "colorbar": cls._result_colorbar(
                    "f", 0.67 if has_coefficients else 0.77
                ),
            },
            coloraxis2={
                "colorscale": "RdBu"
                if np.min([arrays["phi"], arrays["psi"]]) < 0
                else "Viridis",
                "cmin": -source_component_limit,
                "cmax": source_component_limit,
                "showscale": False,
            },
            coloraxis3={
                "colorscale": "Viridis",
                "cmin": 0.0,
                "cmax": 1.0,
                "showscale": False,
            },
            coloraxis4={
                "colorscale": "Viridis",
                "cmin": solution_min,
                "cmax": solution_max,
                "colorbar": cls._result_colorbar(
                    "u", 0.19 if has_coefficients else 0.25
                ),
            },
            margin={"l": 30, "r": 118, "t": 54, "b": 30},
        )
        if coefficients is not None:
            layout.update(
                coloraxis5={
                    "colorscale": "Viridis",
                    "cmin": a_min,
                    "cmax": a_max,
                    "showscale": False,
                },
                coloraxis6={
                    "colorscale": "Viridis",
                    "cmin": b_min,
                    "cmax": b_max,
                    "showscale": False,
                },
                coloraxis7={
                    "colorscale": "Viridis",
                    "cmin": c_min,
                    "cmax": c_max,
                    "showscale": False,
                },
            )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @classmethod
    def build_result_errors(
        cls,
        standard: SelectedSample,
        weak: WeakBlendSample,
    ) -> go.Figure:
        arrays = standard.arrays
        coords = arrays["coords_valid"]
        weak_error = weak.arrays["u_weak_residual_reliability"] - arrays["sol"]
        source_limit = cls._symmetric_limit(arrays["phi_error"], arrays["psi_error"])
        solution_errors = (
            arrays["u_phi"] - arrays["sol"],
            arrays["u_psi"] - arrays["sol"],
            weak_error,
        )
        solution_limit = cls._symmetric_limit(*solution_errors)
        titles = (
            f"phi - phi* | RMS {cls.rms(arrays['phi_error']):.2e}",
            f"psi - psi* | RMS {cls.rms(arrays['psi_error']):.2e}",
            f"u_phi - u | RMS {cls.rms(solution_errors[0]):.2e}",
            f"u_psi - u | RMS {cls.rms(solution_errors[1]):.2e}",
            f"weak u_pred - u | RMS {cls.rms(weak_error):.2e}",
        )
        figure = make_subplots(
            rows=2,
            cols=6,
            specs=[
                [{"colspan": 3}, None, None, {"colspan": 3}, None, None],
                [{"colspan": 2}, None, {"colspan": 2}, None, {"colspan": 2}, None],
            ],
            subplot_titles=titles,
            horizontal_spacing=0.05,
            vertical_spacing=0.17,
        )
        for values, label, row, col, coloraxis in (
            (arrays["phi_error"], "phi - phi*", 1, 1, "coloraxis"),
            (arrays["psi_error"], "psi - psi*", 1, 4, "coloraxis"),
            (solution_errors[0], "u_phi - u", 2, 1, "coloraxis2"),
            (solution_errors[1], "u_psi - u", 2, 3, "coloraxis2"),
            (weak_error, "weak u_pred - u", 2, 5, "coloraxis2"),
        ):
            figure.add_trace(
                cls._field_trace(coords, values, label, coloraxis), row=row, col=col
            )
        layout = cls._base_layout(1600, 820)
        layout.update(
            coloraxis={
                "colorscale": "RdBu",
                "cmin": -source_limit,
                "cmax": source_limit,
                "colorbar": {"title": "source error", "len": 0.36, "y": 0.79},
            },
            coloraxis2={
                "colorscale": "RdBu",
                "cmin": -solution_limit,
                "cmax": solution_limit,
                "colorbar": {"title": "solution error", "len": 0.36, "y": 0.22},
            },
        )
        figure.update_layout(**layout)
        cls._style_field_axes(figure, coords)
        return figure

    @staticmethod
    def _quiver_trace(fields: CoefficientFields) -> tuple[go.BaseTraceType, ...]:
        indices = fields.quiver_indices
        if indices.size == 0 or float(np.max(fields.b_magnitude)) <= EPSILON:
            return ()
        coords = fields.coords_valid[indices]
        magnitude = fields.b_magnitude[indices]
        positive_differences: list[float] = []
        for axis in range(2):
            unique = np.unique(coords[:, axis])
            differences = np.diff(unique)
            positive_differences.extend(differences[differences > EPSILON].tolist())
        grid_spacing = min(positive_differences) if positive_differences else 0.03
        scale = 0.65 * grid_spacing / max(float(np.max(magnitude)), EPSILON)
        quiver = ff.create_quiver(
            coords[:, 0],
            coords[:, 1],
            scale * fields.bx[indices],
            scale * fields.by[indices],
            scale=1.0,
            arrow_scale=0.25,
            line={"color": INK, "width": 1.0},
        )
        for trace in quiver.data:
            trace.update(showlegend=False, hoverinfo="skip")
        return tuple(quiver.data)


class FigureSaveMixin:
    logger: logging.Logger

    def save_figure(
        self,
        figure: go.Figure,
        output_path: Path,
        overwrite: bool,
    ) -> None:
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}; pass --overwrite to replace it"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        responsive_figure = go.Figure(figure)
        responsive_figure.update_layout(width=None, height=None, autosize=True)
        pio.write_html(
            responsive_figure,
            file=output_path,
            include_plotlyjs="directory",
            full_html=True,
            auto_open=False,
            default_width="100vw",
            default_height="100vh",
            config={
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": False,
                "scrollZoom": False,
            },
        )
        html = output_path.read_text()
        if "cdn.plot.ly" in html or "https://" in html or "http://" in html:
            raise ValueError(f"Generated HTML is not offline-safe: {output_path}")
        self.logger.info("Saved offline Plotly asset %s", output_path)


class AnnulusMeetingAssetBuilder(
    LoggingMixin,
    ProvenanceMixin,
    ArtifactLoaderMixin,
    DiagnosticsMixin,
    PlotlyFigureMixin,
    FigureSaveMixin,
):
    def __init__(self, config: MeetingAssetConfig) -> None:
        self.config = config
        self.paths = config.resolved_paths
        self.logger = self.build_logger(config.outdir.parent / "build_assets.log")

    def run(self) -> dict[str, Any]:
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        data = self._load_and_validate()
        figures = self._build_figures(data)
        assets: dict[str, Any] = {}
        for filename, (figure, metadata) in figures.items():
            output_path = self.config.outdir / filename
            self.save_figure(figure, output_path, self.config.overwrite)
            assets[filename] = {
                **metadata,
                "generated_sha256": self.sha256(output_path),
                "generated_size_bytes": output_path.stat().st_size,
            }

        manifest = self._build_manifest(data, assets)
        manifest_path = self.config.outdir / "manifest.json"
        if manifest_path.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Output already exists: {manifest_path}; pass --overwrite to replace it"
            )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.logger.info("Saved provenance manifest %s", manifest_path)
        return manifest

    def _load_and_validate(self) -> dict[str, Any]:
        common_fields = (
            "coords_valid",
            "rhs",
            "sol",
            "phi",
            "psi",
            "u_phi",
            "u_psi",
            "target_phi",
            "target_psi",
            "phi_error",
            "psi_error",
        )
        legacy = self.load_selected_sample(
            self.paths.legacy_artifact_root,
            47,
            (
                "coords_valid",
                "phi_error",
                "psi_error",
                "u_phi_error",
                "u_pred_error",
                "u_psi_error",
            ),
        )
        poisson_standard = self.load_selected_sample(
            self.paths.poisson_artifact_root, 0, common_fields
        )
        poisson_weak = self.load_weak_sample(self.paths.poisson_weak_root, 0)
        poisson_weak_47 = self.load_weak_sample(self.paths.poisson_weak_root, 47)
        geometry_c2_detail = self.load_geometry_c2_detail()
        mismatch_seam_detail = self.build_mismatch_seam_detail(poisson_weak)
        cdr_standard = self.load_selected_sample(
            self.paths.cdr_artifact_root, 9, common_fields
        )
        cdr_weak = self.load_weak_sample(self.paths.cdr_weak_root, 9)
        coefficients = self.load_coefficient_fields(self.paths.cdr_artifact_root)
        self.validate_standard_weak_alignment("Poisson", poisson_standard, poisson_weak)
        self.validate_standard_weak_alignment("CDR", cdr_standard, cdr_weak)
        if not np.allclose(
            coefficients.coords_valid,
            cdr_standard.arrays["coords_valid"],
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("CDR coefficient coordinates do not match sample 9")

        poisson_frame = self.load_metrics_frame(self.paths.poisson_weak_root)
        cdr_frame = self.load_metrics_frame(self.paths.cdr_weak_root)
        poisson_summary = self.comparison_summary(poisson_frame)
        cdr_summary = self.comparison_summary(cdr_frame)
        poisson_selected_metrics = self.selected_result_metrics(
            poisson_standard, poisson_weak
        )
        cdr_selected_metrics = self.selected_result_metrics(cdr_standard, cdr_weak)
        length_summary = self.load_json(
            self.paths.length_response_root / "summary.json"
        )
        transition = length_summary.get("transition")
        equivalence = length_summary.get("unit_physical_equivalence")
        if not isinstance(transition, dict) or not isinstance(equivalence, dict):
            raise ValueError(
                "Length-response summary is missing transition/equivalence data"
            )
        return {
            "legacy": legacy,
            "poisson_standard": poisson_standard,
            "poisson_weak": poisson_weak,
            "poisson_weak_47": poisson_weak_47,
            "geometry_c2_detail": geometry_c2_detail,
            "mismatch_seam_detail": mismatch_seam_detail,
            "cdr_standard": cdr_standard,
            "cdr_weak": cdr_weak,
            "coefficients": coefficients,
            "poisson_frame": poisson_frame,
            "cdr_frame": cdr_frame,
            "poisson_summary": poisson_summary,
            "cdr_summary": cdr_summary,
            "poisson_selected_metrics": poisson_selected_metrics,
            "cdr_selected_metrics": cdr_selected_metrics,
            "length_transition": transition,
            "unit_physical_equivalence": equivalence,
        }

    def _build_figures(
        self, data: dict[str, Any]
    ) -> dict[str, tuple[go.Figure, dict[str, Any]]]:
        return {
            "annulus_transition_sample47_error_matrix.html": (
                self.build_legacy_error_matrix(data["legacy"]),
                {
                    "slides": [1],
                    "sample_id": 47,
                    "source_files": [
                        self._rel(
                            self.paths.legacy_artifact_root
                            / "data"
                            / "selected_raw_arrays.npz"
                        )
                    ],
                    "field_keys": [
                        "phi_error",
                        "psi_error",
                        "u_phi_error",
                        "u_pred_error",
                        "u_psi_error",
                    ],
                },
            ),
            "annulus_transition_sample47_error_matrix_marked.html": (
                self.build_legacy_error_matrix(
                    data["legacy"], show_cardinal_markers=True
                ),
                {
                    "slides": [1],
                    "sample_id": 47,
                    "source_files": [
                        self._rel(
                            self.paths.legacy_artifact_root
                            / "data"
                            / "selected_raw_arrays.npz"
                        )
                    ],
                    "field_keys": [
                        "phi_error",
                        "psi_error",
                        "u_phi_error",
                        "u_pred_error",
                        "u_psi_error",
                    ],
                    "animation_state": "cardinal transition markers visible",
                },
            ),
            "geometry_c2_method_sample0.html": (
                self.build_geometry_c2_detail(data["geometry_c2_detail"]),
                {
                    "slides": [6],
                    "sample_id": 0,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_geometry_blend_root
                            / "data"
                            / "selected_fixed_smooth_blend_arrays.npz"
                        ),
                        self._rel(
                            self.paths.poisson_geometry_blend_root / "summary.json"
                        ),
                    ],
                    "field_keys": [
                        "distance_phi",
                        "distance_psi",
                        "influence_phi",
                        "influence_psi",
                        "theta",
                        "w_phi",
                        "w_psi",
                    ],
                    "construction": "known topology compact C2",
                },
            ),
            "mismatch_seam_c2_method_sample0.html": (
                self.build_mismatch_seam_detail_figure(data["mismatch_seam_detail"]),
                {
                    "slides": [7],
                    "sample_id": 0,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_weak_root
                            / "data"
                            / "selected_weak_residual_blend_arrays.npz"
                        ),
                        self._rel(self.paths.poisson_geometry_path),
                        self._rel(self.paths.poisson_weak_root / "summary.json"),
                    ],
                    "field_keys": [
                        "u_phi",
                        "u_psi",
                        "mismatch",
                        "x_edge_profile",
                        "y_edge_profile",
                        "x_seam_coordinates",
                        "y_seam_coordinates",
                        "theta",
                        "w_phi",
                    ],
                    "construction": "prediction mismatch detected seam compact C2",
                },
            ),
            "weak_residual_reliability_method_sample0.html": (
                self.build_weak_reliability_detail(data["poisson_weak"]),
                {
                    "slides": [8],
                    "sample_id": 0,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_weak_root
                            / "data"
                            / "selected_weak_residual_blend_arrays.npz"
                        ),
                        self._rel(self.paths.poisson_weak_root / "summary.json"),
                    ],
                    "field_keys": [
                        "weak_phi_indicator",
                        "weak_psi_indicator",
                        "weak_theta",
                        "weak_w_phi",
                        "weak_w_psi",
                    ],
                    "construction": "local full PDE weak residual reliability",
                },
            ),
            "poisson_four_way_rel_sol.html": (
                self.build_four_way_scatter(data["poisson_frame"], "Pure Poisson"),
                {
                    "slides": [9],
                    "sample_id": None,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_weak_root
                            / "metrics"
                            / "per_sample_weak_residual_blend_comparison.csv"
                        )
                    ],
                    "field_keys": [
                        "equal_mean_rel_sol",
                        "geometry_c2_rel_sol",
                        "mismatch_seam_c2_rel_sol",
                        "weak_residual_reliability_rel_sol",
                    ],
                },
            ),
            "poisson_weak_sample47_inset.html": (
                self.build_weak_inset(data["poisson_weak_47"]),
                {
                    "slides": [9],
                    "sample_id": 47,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_weak_root
                            / "data"
                            / "selected_weak_residual_blend_arrays.npz"
                        )
                    ],
                    "field_keys": [
                        "weak_w_phi",
                        "u_equal_mean",
                        "u_weak_residual_reliability",
                        "sol",
                    ],
                },
            ),
            "poisson_cdr_rule_comparison.html": (
                self.build_cross_equation_comparison(
                    data["poisson_summary"], data["cdr_summary"]
                ),
                {
                    "slides": [10],
                    "sample_id": None,
                    "source_files": [
                        self._rel(
                            self.paths.poisson_weak_root
                            / "metrics"
                            / "per_sample_weak_residual_blend_comparison.csv"
                        ),
                        self._rel(
                            self.paths.cdr_weak_root
                            / "metrics"
                            / "per_sample_weak_residual_blend_comparison.csv"
                        ),
                    ],
                    "field_keys": ["rel_sol_change_vs_equal"],
                },
            ),
            "poisson_weak_result_fields_sample0.html": (
                self.build_result_fields(
                    data["poisson_standard"], data["poisson_weak"]
                ),
                self._result_metadata("Poisson", 0, [11], include_coefficients=False),
            ),
            "poisson_weak_result_errors_sample0.html": (
                self.build_result_errors(
                    data["poisson_standard"], data["poisson_weak"]
                ),
                self._error_metadata("Poisson", 0, [12]),
            ),
            "cdr_weak_result_fields_sample9.html": (
                self.build_result_fields(
                    data["cdr_standard"], data["cdr_weak"], data["coefficients"]
                ),
                self._result_metadata("CDR", 9, [13], include_coefficients=True),
            ),
            "cdr_weak_result_errors_sample9.html": (
                self.build_result_errors(data["cdr_standard"], data["cdr_weak"]),
                self._error_metadata("CDR", 9, [14]),
            ),
        }

    def _result_metadata(
        self,
        equation: str,
        sample_id: int,
        slides: list[int],
        include_coefficients: bool,
    ) -> dict[str, Any]:
        standard_root = (
            self.paths.poisson_artifact_root
            if equation == "Poisson"
            else self.paths.cdr_artifact_root
        )
        weak_root = (
            self.paths.poisson_weak_root
            if equation == "Poisson"
            else self.paths.cdr_weak_root
        )
        source_files = [
            self._rel(standard_root / "data" / "selected_raw_arrays.npz"),
            self._rel(weak_root / "data" / "selected_weak_residual_blend_arrays.npz"),
        ]
        if include_coefficients:
            source_files.append(
                self._rel(standard_root / "data" / "coefficient_fields.npz")
            )
        return {
            "slides": slides,
            "sample_id": sample_id,
            "source_files": source_files,
            "field_keys": [
                "rhs",
                "phi",
                "psi",
                "sol",
                "u_phi",
                "u_psi",
                "u_weak_residual_reliability",
                "weak_w_phi",
            ]
            + (
                ["a", "bx", "by", "b_magnitude", "c", "quiver_indices"]
                if include_coefficients
                else []
            ),
            "final_prediction_contract": "u_weak_residual_reliability",
        }

    def _error_metadata(
        self, equation: str, sample_id: int, slides: list[int]
    ) -> dict[str, Any]:
        metadata = self._result_metadata(
            equation, sample_id, slides, include_coefficients=False
        )
        metadata["field_keys"] = [
            "phi_error",
            "psi_error",
            "u_phi",
            "u_psi",
            "sol",
            "u_weak_residual_reliability",
        ]
        metadata["error_convention"] = "signed_difference"
        return metadata

    def _build_manifest(
        self,
        data: dict[str, Any],
        assets: dict[str, Any],
    ) -> dict[str, Any]:
        legacy = data["legacy"].arrays
        coefficient_fields: CoefficientFields = data["coefficients"]
        source_provenance = {
            self._rel(path): self.file_provenance(path) for path in self._source_paths()
        }
        plotly_bundle_path = self.config.outdir / "plotly.min.js"
        return {
            "builder_version": BUILDER_VERSION,
            "project_root": str(self.config.project_root),
            "offline_plotly": True,
            "plotly_bundle": "plotly.min.js",
            "plotly_bundle_provenance": self.file_provenance(plotly_bundle_path),
            "source_provenance": source_provenance,
            "final_prediction_contract": (
                "u_weak_residual_reliability; standard artifact u_pred is equal mean"
            ),
            "reference_fields_role": "evaluation_only",
            "method_detail_contract": {
                "sample_id": 0,
                "geometry_c2": "known topology; sample independent",
                "mismatch_seam_c2": "u_phi-u_psi only; no reference target",
                "weak_residual_reliability": (
                    "local full PDE defect; no reference target or global solve"
                ),
            },
            "assets": assets,
            "legacy_sample_47": {
                "sample_id": 47,
                "source_error_limit": self._symmetric_limit(
                    legacy["phi_error"], legacy["psi_error"]
                ),
                "solution_error_limit": self._symmetric_limit(
                    legacy["u_phi_error"],
                    legacy["u_pred_error"],
                    legacy["u_psi_error"],
                ),
                "field_rms": {
                    field: self.rms(legacy[field])
                    for field in (
                        "phi_error",
                        "psi_error",
                        "u_phi_error",
                        "u_pred_error",
                        "u_psi_error",
                    )
                },
            },
            "poisson": {
                "comparison": data["poisson_summary"],
                "representative_sample": data["poisson_selected_metrics"],
                "rel_flux": float(
                    data["poisson_standard"].metrics.get("rel_flux", float("nan"))
                ),
            },
            "cdr": {
                "comparison": data["cdr_summary"],
                "representative_sample": data["cdr_selected_metrics"],
                "rel_flux": float(
                    data["cdr_standard"].metrics.get("rel_flux", float("nan"))
                ),
                "coefficient_statistics": {
                    field: {
                        "min": float(np.min(getattr(coefficient_fields, field))),
                        "max": float(np.max(getattr(coefficient_fields, field))),
                        "mean": float(np.mean(getattr(coefficient_fields, field))),
                    }
                    for field in ("a", "bx", "by", "b_magnitude", "c")
                },
                "quiver_point_count": int(coefficient_fields.quiver_indices.size),
            },
            "length_response_diagnostic": {
                "source_files": [
                    self._rel(self.paths.length_response_root / "summary.json"),
                    self._rel(self.paths.length_response_root / "diagnosis_report.md"),
                ],
                "transition": data["length_transition"],
                "unit_physical_equivalence": data["unit_physical_equivalence"],
            },
        }

    def _source_paths(self) -> list[Path]:
        paths = [
            self.paths.legacy_artifact_root / "data" / "selected_raw_arrays.npz",
            self.paths.legacy_artifact_root / "metrics" / "per_sample_metrics.csv",
            self.paths.length_response_root / "summary.json",
            self.paths.length_response_root / "diagnosis_report.md",
            self.paths.poisson_artifact_root / "data" / "selected_raw_arrays.npz",
            self.paths.poisson_artifact_root / "metrics" / "per_sample_metrics.csv",
            self.paths.poisson_weak_root
            / "data"
            / "selected_weak_residual_blend_arrays.npz",
            self.paths.poisson_weak_root
            / "metrics"
            / "per_sample_weak_residual_blend_comparison.csv",
            self.paths.poisson_weak_root / "summary.json",
            self.paths.poisson_geometry_path,
            self.paths.poisson_geometry_blend_root
            / "data"
            / "selected_fixed_smooth_blend_arrays.npz",
            self.paths.poisson_geometry_blend_root / "summary.json",
            self.paths.cdr_artifact_root / "data" / "selected_raw_arrays.npz",
            self.paths.cdr_artifact_root / "data" / "coefficient_fields.npz",
            self.paths.cdr_artifact_root / "metrics" / "per_sample_metrics.csv",
            self.paths.cdr_weak_root
            / "data"
            / "selected_weak_residual_blend_arrays.npz",
            self.paths.cdr_weak_root
            / "metrics"
            / "per_sample_weak_residual_blend_comparison.csv",
        ]
        return sorted(set(paths), key=self._rel)

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.config.project_root))
        except ValueError:
            return str(path)


class BuildAssetsCLI:
    @staticmethod
    def parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Build offline Plotly assets for the Annulus transition-error "
                "Reveal.js meeting deck without loading model checkpoints."
            )
        )
        parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
        parser.add_argument(
            "--legacy-artifact-root",
            type=Path,
            default=MeetingArtifactPaths.legacy_artifact_root,
        )
        parser.add_argument(
            "--length-response-root",
            type=Path,
            default=MeetingArtifactPaths.length_response_root,
        )
        parser.add_argument(
            "--poisson-artifact-root",
            type=Path,
            default=MeetingArtifactPaths.poisson_artifact_root,
        )
        parser.add_argument(
            "--poisson-weak-root",
            type=Path,
            default=MeetingArtifactPaths.poisson_weak_root,
        )
        parser.add_argument(
            "--poisson-geometry-path",
            type=Path,
            default=MeetingArtifactPaths.poisson_geometry_path,
        )
        parser.add_argument(
            "--poisson-geometry-blend-root",
            type=Path,
            default=MeetingArtifactPaths.poisson_geometry_blend_root,
        )
        parser.add_argument(
            "--cdr-artifact-root",
            type=Path,
            default=MeetingArtifactPaths.cdr_artifact_root,
        )
        parser.add_argument(
            "--cdr-weak-root",
            type=Path,
            default=MeetingArtifactPaths.cdr_weak_root,
        )
        parser.add_argument("--overwrite", action="store_true")
        return parser

    @classmethod
    def run(cls) -> None:
        args = cls.parser().parse_args()
        paths = MeetingArtifactPaths(
            legacy_artifact_root=args.legacy_artifact_root,
            length_response_root=args.length_response_root,
            poisson_artifact_root=args.poisson_artifact_root,
            poisson_weak_root=args.poisson_weak_root,
            poisson_geometry_path=args.poisson_geometry_path,
            poisson_geometry_blend_root=args.poisson_geometry_blend_root,
            cdr_artifact_root=args.cdr_artifact_root,
            cdr_weak_root=args.cdr_weak_root,
        )
        config = MeetingAssetConfig(
            project_root=args.project_root.resolve(),
            outdir=args.outdir.resolve(),
            paths=paths,
            overwrite=bool(args.overwrite),
        )
        AnnulusMeetingAssetBuilder(config).run()


if __name__ == "__main__":
    BuildAssetsCLI.run()
