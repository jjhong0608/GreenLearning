from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.logging import RichHandler

from greenonet.plotly_io import save_plotly_figure


DEFAULT_ARTIFACT_ROOT: Final = Path("checkpoints/Diffusion/coupling/artifacts")
DEFAULT_OUTDIR: Final = Path("docs/presentations/wccm_eccomas_2026/assets")
EPSILON: Final = 1e-12
PLOT_FONT_FAMILY: Final = "Aptos, Segoe UI, Helvetica, Arial, sans-serif"
ROLE_ORDER: Final = ("min", "q25", "q50", "q75", "max")
FIELD_ORDER: Final = ("rhs", "sol", "u_pred", "u_pred_error")
FIELD_LABELS: Final = {
    "rhs": "Source",
    "sol": "Reference",
    "u_pred": "Prediction",
    "u_pred_error": "Signed error",
}


@dataclass(frozen=True)
class CouplingEvidencePanelConfig:
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    outdir: Path = DEFAULT_OUTDIR
    basename: str = "coupling_evidence_rel_sol_quantiles"
    metric: str = "rel_sol"
    point_size: float = 4.4
    image_size: int = 360
    theme: str = "plotly_white"
    overwrite: bool = False


@dataclass(frozen=True)
class SelectedSample:
    role: str
    sample_id: int
    file_stem: str
    key_prefix: str
    metrics: dict[str, float]
    arrays: dict[str, np.ndarray]


class LoggingMixin:
    logger: logging.Logger

    @staticmethod
    def build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("WccmCouplingEvidencePanel")
        logger.handlers.clear()

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()
        return logger


class ArtifactLoadMixin:
    logger: logging.Logger

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact file does not exist: {path}")

    def load_selected_samples(
        self, config: CouplingEvidencePanelConfig
    ) -> list[SelectedSample]:
        summary_path = config.artifact_root / "summary.json"
        metrics_path = config.artifact_root / "metrics" / "per_sample_metrics.csv"
        raw_path = config.artifact_root / "data" / "selected_raw_arrays.npz"
        for path in (summary_path, metrics_path, raw_path):
            self._require_file(path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected_roles = summary.get("selected_sample_roles")
        if not isinstance(selected_roles, dict):
            raise ValueError(
                f"{summary_path} must contain a selected_sample_roles object."
            )

        metrics_df = pd.read_csv(metrics_path)
        raw = np.load(raw_path)
        samples: list[SelectedSample] = []
        for role in ROLE_ORDER:
            if role not in selected_roles:
                raise ValueError(f"Missing selected role {role!r} in {summary_path}")
            sample_id = int(selected_roles[role])
            metrics_row = metrics_df.loc[
                metrics_df["sample_id"].astype(int) == sample_id
            ]
            if metrics_row.empty:
                raise ValueError(
                    f"Sample {sample_id} for role {role!r} is missing from {metrics_path}"
                )
            row = metrics_row.iloc[0]
            file_stem = str(row["file_stem"])
            key_prefix = f"sample_{sample_id:04d}_{file_stem}"
            sample_arrays = self._load_sample_arrays(raw, key_prefix)
            samples.append(
                SelectedSample(
                    role=role,
                    sample_id=sample_id,
                    file_stem=file_stem,
                    key_prefix=key_prefix,
                    metrics={
                        "loss": float(row["loss"]),
                        "loss_energy_consistency": float(
                            row["loss_energy_consistency"]
                        ),
                        "rel_sol": float(row["rel_sol"]),
                        "rel_flux": float(row["rel_flux"]),
                    },
                    arrays=sample_arrays,
                )
            )
        self.logger.info(
            "Loaded %s selected CouplingNet samples from %s",
            len(samples),
            raw_path,
        )
        return samples

    @staticmethod
    def _load_sample_arrays(
        raw: np.lib.npyio.NpzFile,
        key_prefix: str,
    ) -> dict[str, np.ndarray]:
        required = ("coords_valid", *FIELD_ORDER)
        arrays: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for field in required:
            key = f"{key_prefix}_{field}"
            if key not in raw.files:
                missing.append(key)
                continue
            arrays[field] = np.asarray(raw[key], dtype=float)
        if missing:
            raise ValueError(f"Selected raw arrays are missing keys: {missing}")
        coords = arrays["coords_valid"]
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"{key_prefix}_coords_valid must have shape (P, 2); got {coords.shape}"
            )
        point_count = coords.shape[0]
        for field in FIELD_ORDER:
            values = arrays[field]
            if values.shape != (point_count,):
                raise ValueError(
                    f"{key_prefix}_{field} must have shape ({point_count},); "
                    f"got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{key_prefix}_{field} contains non-finite values")
        return arrays


class ColorScaleMixin:
    @staticmethod
    def compute_color_ranges(
        samples: list[SelectedSample],
    ) -> dict[str, tuple[float, float]]:
        error_values = np.concatenate(
            [sample.arrays["u_pred_error"] for sample in samples]
        )
        error_abs_max = float(np.max(np.abs(error_values)))
        ranges = {"u_pred_error": (-error_abs_max, error_abs_max)}
        for sample in samples:
            ranges[f"{sample.role}:rhs"] = ColorScaleMixin._finite_range(
                sample.arrays["rhs"]
            )
            solution_values = np.concatenate(
                [sample.arrays[field] for field in ("sol", "u_pred")]
            )
            ranges[f"{sample.role}:solution"] = ColorScaleMixin._finite_range(
                solution_values
            )
        return ranges

    @staticmethod
    def _finite_range(values: np.ndarray) -> tuple[float, float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum <= EPSILON:
            return (minimum - 1.0, maximum + 1.0)
        return (minimum, maximum)

    @staticmethod
    def color_range_key(sample: SelectedSample, field: str) -> str:
        if field in {"sol", "u_pred"}:
            return f"{sample.role}:solution"
        if field == "rhs":
            return f"{sample.role}:rhs"
        return field


class FigureBuildMixin(ColorScaleMixin):
    @staticmethod
    def _axis_range(samples: list[SelectedSample]) -> tuple[list[float], list[float]]:
        coords = np.concatenate([sample.arrays["coords_valid"] for sample in samples])
        x_min = float(np.min(coords[:, 0]))
        x_max = float(np.max(coords[:, 0]))
        y_min = float(np.min(coords[:, 1]))
        y_max = float(np.max(coords[:, 1]))
        pad = 0.02 * max(x_max - x_min, y_max - y_min)
        return [x_min - pad, x_max + pad], [y_min - pad, y_max + pad]

    def build_field_figures(
        self,
        config: CouplingEvidencePanelConfig,
        samples: list[SelectedSample],
    ) -> dict[tuple[str, str], go.Figure]:
        color_ranges = self.compute_color_ranges(samples)
        x_range, y_range = self._axis_range(samples)
        figures: dict[tuple[str, str], go.Figure] = {}
        for sample in samples:
            coords = sample.arrays["coords_valid"]
            for field in FIELD_ORDER:
                zmin, zmax = color_ranges[self.color_range_key(sample, field)]
                signed = field == "u_pred_error"
                figures[(sample.role, field)] = self._build_clean_field_figure(
                    coords=coords,
                    values=sample.arrays[field],
                    zmin=zmin,
                    zmax=zmax,
                    signed=signed,
                    x_range=x_range,
                    y_range=y_range,
                    config=config,
                )
        return figures

    @staticmethod
    def _build_clean_field_figure(
        *,
        coords: np.ndarray,
        values: np.ndarray,
        zmin: float,
        zmax: float,
        signed: bool,
        x_range: list[float],
        y_range: list[float],
        config: CouplingEvidencePanelConfig,
    ) -> go.Figure:
        fig = go.Figure(
            data=go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "color": values,
                    "colorscale": "RdBu_r" if signed else "Viridis",
                    "cmin": zmin,
                    "cmax": zmax,
                    "showscale": False,
                    "size": config.point_size,
                    "line": {"width": 0},
                },
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            template=config.theme,
            width=config.image_size,
            height=config.image_size,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"family": PLOT_FONT_FAMILY, "size": 1},
            margin={"l": 1, "r": 1, "t": 1, "b": 1},
            xaxis={
                "visible": False,
                "range": x_range,
                "constrain": "domain",
            },
            yaxis={
                "visible": False,
                "range": y_range,
                "scaleanchor": "x",
                "scaleratio": 1,
                "constrain": "domain",
            },
        )
        return fig


class DiagnosticsMixin:
    @staticmethod
    def build_summary(
        config: CouplingEvidencePanelConfig,
        samples: list[SelectedSample],
    ) -> dict[str, Any]:
        return {
            "artifact_root": str(config.artifact_root),
            "metric": config.metric,
            "roles": [
                {
                    "role": sample.role,
                    "sample_id": sample.sample_id,
                    "file_stem": sample.file_stem,
                    "rel_sol": sample.metrics["rel_sol"],
                    "rel_flux": sample.metrics["rel_flux"],
                    "loss_energy_consistency": sample.metrics[
                        "loss_energy_consistency"
                    ],
                }
                for sample in samples
            ],
            "field_labels": FIELD_LABELS,
            "row_scale_policy": {
                "rhs": "independent per selected sample",
                "sol_u_pred": (
                    "per selected sample; reference and prediction share the "
                    "same scale within each sample"
                ),
                "u_pred_error": (
                    "zero-centered shared scale over selected signed errors"
                ),
            },
            "visible_panel_policy": (
                "panel images omit titles, axes, and colorbars; slide-native "
                "labels and metric tables provide context"
            ),
        }


class FigureSaveMixin:
    logger: logging.Logger

    @staticmethod
    def _output_bases(config: CouplingEvidencePanelConfig) -> list[Path]:
        bases: list[Path] = []
        for role in ROLE_ORDER:
            for field in FIELD_ORDER:
                bases.append(config.outdir / f"{config.basename}_{role}_{field}")
        return bases

    def check_overwrite(self, config: CouplingEvidencePanelConfig) -> None:
        if config.overwrite:
            return
        paths = [
            output_base.with_suffix(extension)
            for output_base in self._output_bases(config)
            for extension in (".html", ".json", ".png", ".pdf")
        ]
        paths.append(config.outdir / f"{config.basename}_summary.json")
        conflicts = [path for path in paths if path.exists()]
        if conflicts:
            formatted = ", ".join(str(path) for path in conflicts[:5])
            raise FileExistsError(
                "Output files already exist. Pass --overwrite to replace them: "
                f"{formatted}"
            )

    def save_outputs(
        self,
        config: CouplingEvidencePanelConfig,
        figures: dict[tuple[str, str], go.Figure],
        summary: dict[str, Any],
    ) -> None:
        config.outdir.mkdir(parents=True, exist_ok=True)
        for (role, field), fig in figures.items():
            base = config.outdir / f"{config.basename}_{role}_{field}"
            save_plotly_figure(fig, base, logger=self.logger)
        summary_path = config.outdir / f"{config.basename}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.logger.info(
            "Saved %s separated CouplingNet evidence field panels.",
            len(figures),
        )
        self.logger.info("Saved CouplingNet evidence summary to %s", summary_path)


class CouplingEvidencePanelBuilder(
    LoggingMixin,
    ArtifactLoadMixin,
    FigureBuildMixin,
    DiagnosticsMixin,
    FigureSaveMixin,
):
    def __init__(self, config: CouplingEvidencePanelConfig) -> None:
        self.config = config
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self.logger = self.build_logger(
            self.config.outdir / f"{self.config.basename}.log"
        )

    def run(self) -> None:
        self.check_overwrite(self.config)
        samples = self.load_selected_samples(self.config)
        field_figures = self.build_field_figures(self.config, samples)
        summary = self.build_summary(self.config, samples)
        self.save_outputs(self.config, field_figures, summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build WCCM-ECCOMAS slide-ready CouplingNet evidence panels from "
            "export_coupling_artifacts.py outputs."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root directory containing CouplingNet artifact data and metrics.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for slide-ready panels and summary JSON.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="coupling_evidence_rel_sol_quantiles",
        help="Output basename prefix without suffix.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="rel_sol",
        help="Metric used to describe the selected quantile roles.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.4,
        help="Marker size for clean field panels.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=360,
        help="Width and height in pixels for each clean field panel.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="plotly_white",
        help="Plotly template name.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = CouplingEvidencePanelConfig(
        artifact_root=args.artifact_root,
        outdir=args.outdir,
        basename=args.basename,
        metric=args.metric,
        point_size=args.point_size,
        image_size=args.image_size,
        theme=args.theme,
        overwrite=args.overwrite,
    )
    CouplingEvidencePanelBuilder(config).run()


if __name__ == "__main__":
    main()
