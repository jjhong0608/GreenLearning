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
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from greenonet.plotly_io import save_plotly_figure


DEFAULT_ARTIFACT_ROOT: Final = Path("checkpoints/Diffusion/green/artifacts")
DEFAULT_OUTDIR: Final = Path("docs/presentations/wccm_eccomas_2026/assets")
EPSILON: Final = 1e-12
PLOT_FONT_FAMILY: Final = "Aptos, Segoe UI, Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class GreenEvidencePanelConfig:
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    outdir: Path = DEFAULT_OUTDIR
    interval_index: int = 109
    eta: float = 0.5
    diagonal_band_width: int = 5
    basename: str | None = None
    theme: str = "plotly_white"
    overwrite: bool = False

    @property
    def output_basename(self) -> str:
        if self.basename is not None:
            return self.basename
        eta_tag = f"{int(round(self.eta * 100)):03d}"
        return f"greennet_evidence_interval{self.interval_index:03d}_eta{eta_tag}"


@dataclass(frozen=True)
class GreenKernelSelection:
    interval_index: int
    selected_position: int
    grid: np.ndarray
    predicted: np.ndarray
    reference: np.ndarray
    error: np.ndarray


@dataclass(frozen=True)
class SliceSelection:
    eta: float
    eta_index: int
    predicted: np.ndarray
    reference: np.ndarray
    error: np.ndarray
    slice_rel_error: float
    boundary_abs_max: float


@dataclass(frozen=True)
class IntervalMetadata:
    interval_index: int
    axis: str
    segment_index: int
    fixed: float
    length: float


class LoggingMixin:
    logger: logging.Logger

    @staticmethod
    def build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("WccmGreenEvidencePanel")
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

    def load_kernel_selection(
        self, config: GreenEvidencePanelConfig
    ) -> GreenKernelSelection:
        kernel_path = config.artifact_root / "data" / "selected_green_kernels.npz"
        self._require_file(kernel_path)
        data = np.load(kernel_path)
        required = ("interval_indices", "predicted", "reference")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Kernel NPZ is missing required keys: {missing}")

        interval_indices = data["interval_indices"].astype(int)
        matches = np.flatnonzero(interval_indices == config.interval_index)
        if matches.size == 0:
            available = ", ".join(str(int(item)) for item in interval_indices)
            raise ValueError(
                f"Interval {config.interval_index} is not present in {kernel_path}; "
                f"available intervals: {available}"
            )
        selected_position = int(matches[0])
        predicted = np.asarray(data["predicted"][selected_position], dtype=float)
        reference = np.asarray(data["reference"][selected_position], dtype=float)
        if predicted.shape != reference.shape:
            raise ValueError(
                "Predicted and reference kernels must have matching shapes; "
                f"got {predicted.shape} and {reference.shape}"
            )
        if predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1]:
            raise ValueError(
                "Selected kernels must be square 2D arrays indexed by (t, eta); "
                f"got shape {predicted.shape}"
            )
        if "error" in data.files:
            error = np.asarray(data["error"][selected_position], dtype=float)
        else:
            error = predicted - reference
        if error.shape != predicted.shape:
            raise ValueError(
                f"Kernel error shape {error.shape} does not match {predicted.shape}"
            )
        self._validate_finite("predicted kernel", predicted)
        self._validate_finite("reference kernel", reference)
        self._validate_finite("kernel error", error)

        grid = np.linspace(0.0, 1.0, predicted.shape[0])
        self.logger.info(
            "Loaded interval %s kernels from %s with shape %s",
            config.interval_index,
            kernel_path,
            predicted.shape,
        )
        return GreenKernelSelection(
            interval_index=config.interval_index,
            selected_position=selected_position,
            grid=grid,
            predicted=predicted,
            reference=reference,
            error=error,
        )

    @staticmethod
    def _validate_finite(name: str, values: np.ndarray) -> None:
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    def load_interval_metadata(
        self, config: GreenEvidencePanelConfig
    ) -> IntervalMetadata:
        metrics_path = config.artifact_root / "metrics" / "per_interval_metrics.csv"
        if not metrics_path.exists():
            self.logger.warning("Interval metrics not found: %s", metrics_path)
            return IntervalMetadata(
                interval_index=config.interval_index,
                axis="unknown",
                segment_index=config.interval_index,
                fixed=float("nan"),
                length=float("nan"),
            )
        df = pd.read_csv(metrics_path)
        row_df = df.loc[df["interval_index"].astype(int) == config.interval_index]
        if row_df.empty:
            self.logger.warning(
                "Interval %s not found in %s",
                config.interval_index,
                metrics_path,
            )
            return IntervalMetadata(
                interval_index=config.interval_index,
                axis="unknown",
                segment_index=config.interval_index,
                fixed=float("nan"),
                length=float("nan"),
            )
        row = row_df.iloc[0]
        return IntervalMetadata(
            interval_index=int(row["interval_index"]),
            axis=str(row["axis"]),
            segment_index=int(row["segment_index"]),
            fixed=float(row["fixed"]),
            length=float(row["length"]),
        )

    def load_slice_selection(
        self,
        config: GreenEvidencePanelConfig,
        kernels: GreenKernelSelection,
    ) -> SliceSelection:
        if not 0.0 <= config.eta <= 1.0:
            raise ValueError(f"--eta must be in [0, 1], got {config.eta}")

        eta_index = int(np.argmin(np.abs(kernels.grid - config.eta)))
        eta = float(kernels.grid[eta_index])
        predicted = kernels.predicted[:, eta_index]
        reference = kernels.reference[:, eta_index]
        error = predicted - reference
        slice_rel_error = self._relative_l2(error, reference)
        boundary_abs_max = float(
            max(
                abs(predicted[0]),
                abs(predicted[-1]),
                abs(reference[0]),
                abs(reference[-1]),
            )
        )

        metrics_path = config.artifact_root / "metrics" / "green_slice_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            candidates = metrics.loc[
                metrics["interval_index"].astype(int) == config.interval_index
            ].copy()
            if not candidates.empty:
                candidates["distance"] = (
                    candidates["xi_value"].astype(float) - eta
                ).abs()
                metric_row = candidates.sort_values("distance", kind="mergesort").iloc[
                    0
                ]
                slice_rel_error = float(metric_row["slice_rel_error"])
                boundary_abs_max = float(metric_row["boundary_abs_max"])

        self.logger.info(
            "Selected fixed eta %.6f at grid index %s for interval %s",
            eta,
            eta_index,
            config.interval_index,
        )
        return SliceSelection(
            eta=eta,
            eta_index=eta_index,
            predicted=predicted,
            reference=reference,
            error=error,
            slice_rel_error=slice_rel_error,
            boundary_abs_max=boundary_abs_max,
        )

    @staticmethod
    def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
        denominator = float(np.linalg.norm(reference.ravel()))
        return float(np.linalg.norm(error.ravel()) / max(denominator, EPSILON))


class DiagnosticsMixin:
    @staticmethod
    def compute_diagnostics(
        config: GreenEvidencePanelConfig,
        kernels: GreenKernelSelection,
        slice_selection: SliceSelection,
        metadata: IntervalMetadata,
    ) -> dict[str, Any]:
        abs_error = np.abs(kernels.error)
        total_abs_error = float(abs_error.sum())
        reference_norm = float(np.linalg.norm(kernels.reference.ravel()))
        kernel_rel_error = float(
            np.linalg.norm(kernels.error.ravel()) / max(reference_norm, EPSILON)
        )

        if config.diagonal_band_width < 0:
            raise ValueError(
                "--diagonal-band-width must be non-negative; "
                f"got {config.diagonal_band_width}"
            )
        grid_diffs = np.diff(kernels.grid.astype(float))
        if grid_diffs.size == 0:
            raise ValueError("Kernel grid must contain at least two points.")
        grid_step = float(np.median(grid_diffs))
        if not np.allclose(grid_diffs, grid_step, rtol=1e-6, atol=1e-10):
            raise ValueError(
                "Kernel grid must be uniformly spaced for band diagnostics."
            )
        diagonal_band_radius = float(config.diagonal_band_width * grid_step)
        grid_step_denominator = int(round(1.0 / grid_step))
        row_index, col_index = np.indices(kernels.error.shape)
        band_mask = np.abs(row_index - col_index) <= config.diagonal_band_width
        off_band_mask = ~band_mask
        band_abs_error = abs_error[band_mask]
        off_band_abs_error = abs_error[off_band_mask]
        band_error_mass = float(band_abs_error.sum() / max(total_abs_error, EPSILON))
        band_area_ratio = float(band_mask.mean())
        band_mean_error = float(band_abs_error.mean())
        off_band_mean_error = float(off_band_abs_error.mean())
        band_mean_ratio = float(band_mean_error / max(off_band_mean_error, EPSILON))
        band_max_error = float(band_abs_error.max())
        off_band_max_error = float(off_band_abs_error.max())
        band_max_ratio = float(band_max_error / max(off_band_max_error, EPSILON))

        return {
            "artifact_root": str(config.artifact_root),
            "interval_index": config.interval_index,
            "axis": metadata.axis,
            "segment_index": metadata.segment_index,
            "fixed_coordinate": metadata.fixed,
            "segment_length": metadata.length,
            "eta_requested": config.eta,
            "eta_selected": slice_selection.eta,
            "eta_index": slice_selection.eta_index,
            "diagonal_band_width": config.diagonal_band_width,
            "unit_grid_step": grid_step,
            "unit_grid_step_denominator": grid_step_denominator,
            "diagonal_band_radius": diagonal_band_radius,
            "kernel_relative_l2_error": kernel_rel_error,
            "kernel_max_abs_error": float(abs_error.max()),
            "reference_max_abs": float(np.abs(kernels.reference).max()),
            "predicted_max_abs": float(np.abs(kernels.predicted).max()),
            "slice_relative_l2_error": slice_selection.slice_rel_error,
            "boundary_abs_max": slice_selection.boundary_abs_max,
            "diagonal_band_error_mass": band_error_mass,
            "diagonal_band_area_ratio": band_area_ratio,
            "diagonal_band_mean_abs_error": band_mean_error,
            "off_diagonal_band_mean_abs_error": off_band_mean_error,
            "diagonal_band_mean_error_ratio": band_mean_ratio,
            "diagonal_band_max_abs_error": band_max_error,
            "off_diagonal_band_max_abs_error": off_band_max_error,
            "diagonal_band_max_error_ratio": band_max_ratio,
        }


class FigureBuildMixin:
    @staticmethod
    def _kernel_color_limits(kernels: GreenKernelSelection) -> tuple[float, float]:
        return (
            float(min(kernels.reference.min(), kernels.predicted.min())),
            float(max(kernels.reference.max(), kernels.predicted.max())),
        )

    @staticmethod
    def _error_abs_max(kernels: GreenKernelSelection) -> float:
        error_abs_max = float(np.abs(kernels.error).max())
        if error_abs_max <= EPSILON:
            return EPSILON
        return error_abs_max

    def build_figure(
        self,
        config: GreenEvidencePanelConfig,
        kernels: GreenKernelSelection,
        slice_selection: SliceSelection,
        diagnostics: dict[str, Any],
    ) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "xy", "colspan": 2}, None, {"type": "table"}],
            ],
            subplot_titles=(
                "Reference kernel",
                "Learned kernel",
                "Signed error",
                f"Fixed-η slice at η={slice_selection.eta:.2f}",
                "",
            ),
            column_widths=[0.32, 0.32, 0.36],
            row_heights=[0.58, 0.42],
            horizontal_spacing=0.085,
            vertical_spacing=0.14,
        )

        kernel_min, kernel_max = self._kernel_color_limits(kernels)
        error_abs_max = self._error_abs_max(kernels)

        self._add_kernel_heatmap(
            fig,
            kernels.grid,
            kernels.reference,
            row=1,
            col=1,
            zmin=kernel_min,
            zmax=kernel_max,
            showscale=False,
        )
        self._add_kernel_heatmap(
            fig,
            kernels.grid,
            kernels.predicted,
            row=1,
            col=2,
            zmin=kernel_min,
            zmax=kernel_max,
            showscale=True,
        )
        self._add_error_heatmap(
            fig,
            kernels.grid,
            kernels.error,
            row=1,
            col=3,
            zmax=error_abs_max,
        )
        self._add_diagonal_guides(fig)
        self._add_slice_plot(fig, kernels.grid, slice_selection)
        self._add_metric_table(fig, diagnostics)
        self._style_figure(fig, config)
        return fig

    def build_separate_figures(
        self,
        kernels: GreenKernelSelection,
        slice_selection: SliceSelection,
    ) -> dict[str, go.Figure]:
        kernel_min, kernel_max = self._kernel_color_limits(kernels)
        error_abs_max = self._error_abs_max(kernels)
        return {
            "reference_kernel": self._build_single_heatmap_figure(
                grid=kernels.grid,
                values=kernels.reference,
                zmin=kernel_min,
                zmax=kernel_max,
                colorscale="Viridis",
                colorbar_title="G",
            ),
            "learned_kernel": self._build_single_heatmap_figure(
                grid=kernels.grid,
                values=kernels.predicted,
                zmin=kernel_min,
                zmax=kernel_max,
                colorscale="Viridis",
                colorbar_title="G",
            ),
            "signed_error": self._build_single_heatmap_figure(
                grid=kernels.grid,
                values=kernels.error,
                zmin=-error_abs_max,
                zmax=error_abs_max,
                colorscale="RdBu_r",
                colorbar_title="Error",
            ),
            "fixed_eta_slice": self._build_single_slice_figure(
                grid=kernels.grid,
                slice_selection=slice_selection,
            ),
        }

    @staticmethod
    def _build_single_heatmap_figure(
        *,
        grid: np.ndarray,
        values: np.ndarray,
        zmin: float,
        zmax: float,
        colorscale: str,
        colorbar_title: str,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=grid,
                y=grid,
                z=values,
                zmin=zmin,
                zmax=zmax,
                colorscale=colorscale,
                colorbar={
                    "title": {"text": colorbar_title, "font": {"size": 19}},
                    "tickfont": {"size": 17},
                    "thickness": 14,
                    "len": 0.78,
                },
                hovertemplate=(
                    "η=%{x:.3f}<br>t=%{y:.3f}<br>value=%{z:.3e}<extra></extra>"
                ),
            )
        )
        fig.add_shape(
            type="line",
            x0=0.0,
            y0=0.0,
            x1=1.0,
            y1=1.0,
            xref="x",
            yref="y",
            line={"color": "rgba(255,255,255,0.92)", "width": 2, "dash": "dash"},
        )
        fig.update_layout(
            template="plotly_white",
            width=560,
            height=440,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={
                "family": PLOT_FONT_FAMILY,
                "size": 21,
            },
            margin={"l": 55, "r": 65, "t": 20, "b": 55},
        )
        fig.update_xaxes(
            title_text="η",
            title_font={"size": 23},
            tickfont={"size": 18},
            range=[0.0, 1.0],
        )
        fig.update_yaxes(
            title_text="t",
            title_font={"size": 23},
            tickfont={"size": 18},
            range=[0.0, 1.0],
        )
        return fig

    @staticmethod
    def _build_single_slice_figure(
        *,
        grid: np.ndarray,
        slice_selection: SliceSelection,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=slice_selection.predicted,
                mode="lines",
                name="learned",
                line={"color": "#0a7c86", "width": 4},
                opacity=0.88,
                legendrank=2,
                hovertemplate="t=%{x:.3f}<br>G_theta=%{y:.3e}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=slice_selection.reference,
                mode="lines",
                name="reference",
                line={"color": "#d95f49", "width": 5, "dash": "dash"},
                legendrank=1,
                hovertemplate="t=%{x:.3f}<br>G_ref=%{y:.3e}<extra></extra>",
            )
        )
        fig.add_vline(
            x=slice_selection.eta,
            line={"color": "#172026", "dash": "dot", "width": 2},
        )
        fig.update_layout(
            template="plotly_white",
            width=900,
            height=310,
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={
                "family": PLOT_FONT_FAMILY,
                "size": 21,
            },
            legend={
                "orientation": "h",
                "x": 0.02,
                "y": -0.28,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 19},
            },
            margin={"l": 65, "r": 30, "t": 20, "b": 75},
        )
        fig.update_xaxes(
            title_text="t",
            title_font={"size": 23},
            tickfont={"size": 18},
            range=[0.0, 1.0],
        )
        fig.update_yaxes(
            title_text="G(t, η₀)",
            title_font={"size": 23},
            tickfont={"size": 18},
            zeroline=True,
        )
        return fig

    @staticmethod
    def _add_kernel_heatmap(
        fig: go.Figure,
        grid: np.ndarray,
        values: np.ndarray,
        *,
        row: int,
        col: int,
        zmin: float,
        zmax: float,
        showscale: bool,
    ) -> None:
        fig.add_trace(
            go.Heatmap(
                x=grid,
                y=grid,
                z=values,
                zmin=zmin,
                zmax=zmax,
                colorscale="Viridis",
                showscale=showscale,
                colorbar={
                    "title": {"text": "G", "font": {"size": 16}},
                    "tickfont": {"size": 14},
                    "len": 0.46,
                    "thickness": 14,
                    "x": 0.64,
                    "y": 0.77,
                },
                hovertemplate=("η=%{x:.3f}<br>t=%{y:.3f}<br>G=%{z:.3e}<extra></extra>"),
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def _add_error_heatmap(
        fig: go.Figure,
        grid: np.ndarray,
        values: np.ndarray,
        *,
        row: int,
        col: int,
        zmax: float,
    ) -> None:
        fig.add_trace(
            go.Heatmap(
                x=grid,
                y=grid,
                z=values,
                zmin=-zmax,
                zmax=zmax,
                colorscale="RdBu_r",
                colorbar={
                    "title": {"text": "Error", "font": {"size": 16}},
                    "tickfont": {"size": 14},
                    "len": 0.46,
                    "thickness": 14,
                    "x": 1.02,
                    "y": 0.77,
                },
                hovertemplate=(
                    "η=%{x:.3f}<br>t=%{y:.3f}<br>error=%{z:.3e}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    @staticmethod
    def _add_diagonal_guides(fig: go.Figure) -> None:
        for axis_suffix in ("", "2", "3"):
            fig.add_shape(
                type="line",
                x0=0.0,
                y0=0.0,
                x1=1.0,
                y1=1.0,
                xref=f"x{axis_suffix}",
                yref=f"y{axis_suffix}",
                line={"color": "rgba(255,255,255,0.92)", "width": 2, "dash": "dash"},
            )

    @staticmethod
    def _add_slice_plot(
        fig: go.Figure,
        grid: np.ndarray,
        slice_selection: SliceSelection,
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=slice_selection.predicted,
                mode="lines",
                name="learned",
                line={"color": "#0a7c86", "width": 3},
                opacity=0.88,
                legendrank=2,
                hovertemplate="t=%{x:.3f}<br>G_theta=%{y:.3e}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=slice_selection.reference,
                mode="lines",
                name="reference",
                line={"color": "#d95f49", "width": 4, "dash": "dash"},
                legendrank=1,
                hovertemplate="t=%{x:.3f}<br>G_ref=%{y:.3e}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.add_vline(
            x=slice_selection.eta,
            line={"color": "#172026", "dash": "dot", "width": 2},
            row=2,
            col=1,
        )

    @staticmethod
    def _add_metric_table(fig: go.Figure, diagnostics: dict[str, Any]) -> None:
        rows = [
            ("η", f"{diagnostics['eta_selected']:.2f}"),
            ("Kernel rel. L2", f"{diagnostics['kernel_relative_l2_error']:.2%}"),
            ("Slice rel. L2", f"{diagnostics['slice_relative_l2_error']:.2%}"),
            (
                "Diagonal band",
                (
                    f"|t-η| ≤ {diagnostics['diagonal_band_width']}/"
                    f"{diagnostics['unit_grid_step_denominator']} "
                    f"= {diagnostics['diagonal_band_radius']:.4f}"
                ),
            ),
            (
                "Error mass / band area",
                (
                    f"{diagnostics['diagonal_band_error_mass']:.1%} / "
                    f"{diagnostics['diagonal_band_area_ratio']:.1%}"
                ),
            ),
            (
                "Mean error / off-band mean",
                f"{diagnostics['diagonal_band_mean_error_ratio']:.2f}x",
            ),
        ]
        fig.add_trace(
            go.Table(
                header={
                    "values": ["Diagnostic", "Value"],
                    "fill_color": "#0a7c86",
                    "font": {"color": "white", "size": 15},
                    "align": "left",
                    "height": 28,
                },
                cells={
                    "values": [[item[0] for item in rows], [item[1] for item in rows]],
                    "fill_color": [["#f7f8f4"] * len(rows), ["#ffffff"] * len(rows)],
                    "font": {"color": "#172026", "size": 14},
                    "align": "left",
                    "height": 26,
                },
            ),
            row=2,
            col=3,
        )

    @staticmethod
    def _style_figure(fig: go.Figure, config: GreenEvidencePanelConfig) -> None:
        fig.update_layout(
            title={
                "text": "GreenNet evidence: learned axial Green-kernel structure",
                "x": 0.02,
                "xanchor": "left",
            },
            template=config.theme,
            width=1500,
            height=900,
            paper_bgcolor="#f7f8f4",
            plot_bgcolor="#ffffff",
            font={
                "family": PLOT_FONT_FAMILY,
                "size": 16,
            },
            legend={
                "orientation": "h",
                "x": 0.02,
                "y": -0.04,
                "xanchor": "left",
                "yanchor": "top",
            },
            margin={"l": 70, "r": 95, "t": 95, "b": 90},
        )
        for axis_name in ("xaxis", "xaxis2", "xaxis3"):
            fig.layout[axis_name].update(title="η", range=[0.0, 1.0])
        for axis_name in ("yaxis", "yaxis2", "yaxis3"):
            fig.layout[axis_name].update(title="t", range=[0.0, 1.0])
        fig.update_xaxes(title_text="t", row=2, col=1, range=[0.0, 1.0])
        fig.update_yaxes(title_text="G(t, η₀)", row=2, col=1, zeroline=True)


class FigureSaveMixin:
    logger: logging.Logger

    @staticmethod
    def _output_bases(config: GreenEvidencePanelConfig) -> list[Path]:
        base = config.outdir / config.output_basename
        separate_suffixes = (
            "reference_kernel",
            "learned_kernel",
            "signed_error",
            "fixed_eta_slice",
        )
        return [
            base,
            *[
                config.outdir / f"{config.output_basename}_{suffix}"
                for suffix in separate_suffixes
            ],
        ]

    def check_overwrite(self, config: GreenEvidencePanelConfig) -> None:
        if config.overwrite:
            return
        paths = [
            output_base.with_suffix(extension)
            for output_base in self._output_bases(config)
            for extension in (".html", ".json", ".png", ".pdf")
        ]
        paths.extend(
            [
                config.outdir / f"{config.output_basename}_summary.json",
            ]
        )
        conflicts = [path for path in paths if path.exists()]
        if conflicts:
            formatted = ", ".join(str(path) for path in conflicts[:5])
            raise FileExistsError(
                "Output files already exist. Pass --overwrite to replace them: "
                f"{formatted}"
            )

    def save_outputs(
        self,
        config: GreenEvidencePanelConfig,
        fig: go.Figure,
        separate_figures: dict[str, go.Figure],
        diagnostics: dict[str, Any],
    ) -> None:
        config.outdir.mkdir(parents=True, exist_ok=True)
        base = config.outdir / config.output_basename
        save_plotly_figure(fig, base, logger=self.logger)
        for suffix, separate_fig in separate_figures.items():
            separate_base = config.outdir / f"{config.output_basename}_{suffix}"
            save_plotly_figure(separate_fig, separate_base, logger=self.logger)
        summary_path = config.outdir / f"{config.output_basename}_summary.json"
        summary_path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.logger.info("Saved GreenNet evidence panel to %s.*", base)
        self.logger.info(
            "Saved %s separated GreenNet evidence assets.", len(separate_figures)
        )
        self.logger.info("Saved GreenNet evidence summary to %s", summary_path)


class GreenEvidencePanelBuilder(
    LoggingMixin,
    ArtifactLoadMixin,
    DiagnosticsMixin,
    FigureBuildMixin,
    FigureSaveMixin,
):
    def __init__(self, config: GreenEvidencePanelConfig) -> None:
        self.config = config
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self.logger = self.build_logger(
            self.config.outdir / f"{self.config.output_basename}.log"
        )

    def run(self) -> None:
        self.check_overwrite(self.config)
        kernels = self.load_kernel_selection(self.config)
        metadata = self.load_interval_metadata(self.config)
        slice_selection = self.load_slice_selection(self.config, kernels)
        diagnostics = self.compute_diagnostics(
            self.config,
            kernels,
            slice_selection,
            metadata,
        )
        fig = self.build_figure(self.config, kernels, slice_selection, diagnostics)
        separate_figures = self.build_separate_figures(kernels, slice_selection)
        self.save_outputs(self.config, fig, separate_figures, diagnostics)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a WCCM-ECCOMAS slide-ready GreenNet evidence panel from "
            "export_green_artifacts.py outputs."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
        help="Root directory containing GreenNet artifact data and metrics.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Output directory for the slide-ready panel and summary JSON.",
    )
    parser.add_argument(
        "--interval-index",
        type=int,
        default=109,
        help="Selected interval index present in selected_green_kernels.npz.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Fixed eta value for the Green-kernel slice.",
    )
    parser.add_argument(
        "--diagonal-band-width",
        type=int,
        default=5,
        help="Half-width in grid cells used for diagonal-band error diagnostics.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="Optional output basename without suffix.",
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
    config = GreenEvidencePanelConfig(
        artifact_root=args.artifact_root,
        outdir=args.outdir,
        interval_index=args.interval_index,
        eta=args.eta,
        diagonal_band_width=args.diagonal_band_width,
        basename=args.basename,
        theme=args.theme,
        overwrite=args.overwrite,
    )
    GreenEvidencePanelBuilder(config).run()


if __name__ == "__main__":
    main()
