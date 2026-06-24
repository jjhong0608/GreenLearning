from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.logging import RichHandler

from greenonet.plotly_io import save_plotly_figure


AxisName = Literal["x", "y"]

LOG_Y_FLOOR: Final = 1e-16
AXIS_COLORS: Final[dict[AxisName, str]] = {"x": "#2563eb", "y": "#d97706"}
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
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
)
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "interval_index",
    "axis_id",
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
)
FIGURE_NAMES: Final[tuple[str, ...]] = (
    "rel_sol_by_fixed",
    "rel_sol_by_fixed_log",
    "rel_sol_by_length",
    "rel_sol_by_length_log",
    "rel_sol_distribution",
    "rel_green_by_fixed",
    "rel_green_distribution",
    "rel_sol_chord_map_x",
    "rel_sol_chord_map_y",
)


@dataclass(frozen=True)
class ComplexGreenIntervalMetricsPlotConfig:
    csv_path: Path
    outdir: Path
    theme: str = "plotly_white"
    overwrite: bool = False


class LoggingMixin:
    logger: logging.Logger

    @staticmethod
    def build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("ComplexGreenIntervalMetricsPlotter")
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


class MetricsCsvMixin:
    logger: logging.Logger

    def load_metrics_csv(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Metrics CSV does not exist: {csv_path}")

        df = pd.read_csv(csv_path)
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"Metrics CSV is missing required columns: {missing}")
        if df.empty:
            raise ValueError(f"Metrics CSV contains no rows: {csv_path}")

        axis_values = set(df["axis"].astype(str).unique())
        invalid_axis_values = sorted(axis_values - {"x", "y"})
        if invalid_axis_values:
            raise ValueError(
                "Metrics CSV axis column must contain only 'x' or 'y'; "
                f"got {invalid_axis_values}"
            )

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="raise")
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Metrics CSV column contains non-finite values: {column}"
                )

        self.logger.info("Loaded %s interval rows from %s", len(df), csv_path)
        return df


class FigureSaveMixin:
    logger: logging.Logger

    def save_figure(self, fig: go.Figure, outdir: Path, name: str) -> None:
        save_plotly_figure(fig, outdir / name, logger=self.logger)
        self.logger.info("Saved Plotly figure set for %s", name)


class ComplexGreenIntervalMetricsPlotter(
    LoggingMixin,
    MetricsCsvMixin,
    FigureSaveMixin,
):
    def __init__(self, config: ComplexGreenIntervalMetricsPlotConfig) -> None:
        self.config = config
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self.logger = self.build_logger(
            self.config.outdir / "plot_complex_green_interval_metrics.log"
        )

    @staticmethod
    def _ordered_axis_df(df: pd.DataFrame, axis: AxisName) -> pd.DataFrame:
        return (
            df.loc[df["axis"] == axis]
            .copy()
            .sort_values(["fixed", "segment_index"], kind="mergesort")
        )

    @staticmethod
    def _floored(values: Iterable[float]) -> list[float]:
        return [max(float(value), LOG_Y_FLOOR) for value in values]

    def _check_overwrite(self) -> None:
        if self.config.overwrite:
            return
        existing = [
            self.config.outdir / "metrics_summary.json",
        ]
        existing.extend(self.config.outdir / f"{name}.html" for name in FIGURE_NAMES)
        conflicts = [path for path in existing if path.exists()]
        if conflicts:
            formatted = ", ".join(str(path) for path in conflicts[:5])
            suffix = "" if len(conflicts) <= 5 else f", ... ({len(conflicts)} total)"
            raise FileExistsError(
                "Output files already exist. Pass --overwrite to replace them: "
                f"{formatted}{suffix}"
            )

    def _base_layout(
        self,
        fig: go.Figure,
        *,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
    ) -> go.Figure:
        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=1100,
            height=700,
            font={"family": "Times New Roman", "size": 20},
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            legend_title_text="Axis",
            margin={"l": 90, "r": 40, "t": 90, "b": 80},
        )
        return fig

    def build_rel_sol_by_fixed(self, df: pd.DataFrame, *, log_y: bool) -> go.Figure:
        fig = go.Figure()
        for axis in ("x", "y"):
            axis_df = self._ordered_axis_df(df, axis=axis)
            y_values = axis_df["rel_sol_interval_mean"].to_list()
            if log_y:
                y_values = self._floored(y_values)
            fig.add_trace(
                go.Scatter(
                    x=axis_df["fixed"],
                    y=y_values,
                    mode="lines+markers",
                    name=f"{axis}-segments",
                    line={"color": AXIS_COLORS[axis], "width": 2},
                    marker={"size": 7},
                    error_y={
                        "type": "data",
                        "array": axis_df["rel_sol_interval_std"],
                        "visible": True,
                    },
                    hovertemplate=(
                        "axis=%{fullData.name}<br>"
                        "fixed=%{x:.6f}<br>"
                        "rel_sol_mean=%{y:.3e}<extra></extra>"
                    ),
                )
            )
        fig = self._base_layout(
            fig,
            title="Complex GreenNet interval solution error by transverse coordinate",
            xaxis_title="Transverse fixed coordinate",
            yaxis_title="Mean relative solution reconstruction error",
        )
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    def build_rel_sol_by_length(self, df: pd.DataFrame, *, log_y: bool) -> go.Figure:
        fig = go.Figure()
        for axis in ("x", "y"):
            axis_df = self._ordered_axis_df(df, axis=axis)
            y_values = axis_df["rel_sol_interval_mean"].to_list()
            if log_y:
                y_values = self._floored(y_values)
            fig.add_trace(
                go.Scatter(
                    x=axis_df["length"],
                    y=y_values,
                    mode="markers",
                    name=f"{axis}-segments",
                    marker={
                        "size": 9,
                        "color": AXIS_COLORS[axis],
                        "opacity": 0.82,
                        "line": {"color": "#111827", "width": 0.5},
                    },
                    customdata=np.stack(
                        [
                            axis_df["fixed"].to_numpy(dtype=float),
                            axis_df["segment_index"].to_numpy(dtype=float),
                        ],
                        axis=1,
                    ),
                    hovertemplate=(
                        "axis=%{fullData.name}<br>"
                        "segment=%{customdata[1]:.0f}<br>"
                        "length=%{x:.6f}<br>"
                        "fixed=%{customdata[0]:.6f}<br>"
                        "rel_sol_mean=%{y:.3e}<extra></extra>"
                    ),
                )
            )
        fig = self._base_layout(
            fig,
            title="Complex GreenNet interval solution error by chord length",
            xaxis_title="Segment length",
            yaxis_title="Mean relative solution reconstruction error",
        )
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    def build_distribution(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        title: str,
        yaxis_title: str,
    ) -> go.Figure:
        fig = go.Figure()
        for axis in ("x", "y"):
            axis_df = self._ordered_axis_df(df, axis=axis)
            fig.add_trace(
                go.Box(
                    y=axis_df[metric_column],
                    name=f"{axis}-segments",
                    boxpoints=False,
                    marker={"color": AXIS_COLORS[axis]},
                    line={"color": AXIS_COLORS[axis]},
                )
            )
        return self._base_layout(
            fig,
            title=title,
            xaxis_title="Axis",
            yaxis_title=yaxis_title,
        )

    def build_rel_green_by_fixed(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for axis in ("x", "y"):
            axis_df = self._ordered_axis_df(df, axis=axis)
            fig.add_trace(
                go.Scatter(
                    x=axis_df["fixed"],
                    y=axis_df["rel_green_interval_mean"],
                    mode="lines+markers",
                    name=f"{axis}-segments",
                    line={"color": AXIS_COLORS[axis], "width": 2},
                    marker={"size": 7},
                    hovertemplate=(
                        "axis=%{fullData.name}<br>"
                        "fixed=%{x:.6f}<br>"
                        "rel_green_mean=%{y:.3e}<extra></extra>"
                    ),
                )
            )
        return self._base_layout(
            fig,
            title="Complex GreenNet interval Green-kernel error by transverse coordinate",
            xaxis_title="Transverse fixed coordinate",
            yaxis_title="Mean relative Green-kernel error",
        )

    @staticmethod
    def _continuous_color(value: float, vmin: float, vmax: float) -> str:
        if vmax <= vmin:
            normalized = 0.5
        else:
            normalized = (value - vmin) / (vmax - vmin)
        r, g, b = _viridis_rgb(normalized)
        return f"rgb({r},{g},{b})"

    def build_chord_map(self, df: pd.DataFrame, *, axis: AxisName) -> go.Figure:
        axis_df = self._ordered_axis_df(df, axis=axis)
        metric = axis_df["rel_sol_interval_mean"].to_numpy(dtype=float)
        vmin = float(metric.min())
        vmax = float(metric.max())

        fig = go.Figure()
        for row in axis_df.itertuples(index=False):
            value = float(getattr(row, "rel_sol_interval_mean"))
            color = self._continuous_color(value, vmin, vmax)
            if axis == "x":
                x_values = [float(row.left), float(row.right)]
                y_values = [float(row.fixed), float(row.fixed)]
            else:
                x_values = [float(row.fixed), float(row.fixed)]
                y_values = [float(row.left), float(row.right)]
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    line={"color": color, "width": 3},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        if axis == "x":
            midpoint_x = (axis_df["left"] + axis_df["right"]) / 2.0
            midpoint_y = axis_df["fixed"]
        else:
            midpoint_x = axis_df["fixed"]
            midpoint_y = (axis_df["left"] + axis_df["right"]) / 2.0
        fig.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                marker={
                    "size": 5,
                    "color": axis_df["rel_sol_interval_mean"],
                    "colorscale": "Viridis",
                    "cmin": vmin,
                    "cmax": vmax,
                    "colorbar": {"title": "rel_sol mean"},
                    "opacity": 0.01,
                },
                showlegend=False,
                customdata=np.stack(
                    [
                        axis_df["segment_index"].to_numpy(dtype=float),
                        axis_df["length"].to_numpy(dtype=float),
                        axis_df["fixed"].to_numpy(dtype=float),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "segment=%{customdata[0]:.0f}<br>"
                    "length=%{customdata[1]:.6f}<br>"
                    "fixed=%{customdata[2]:.6f}<br>"
                    "rel_sol_mean=%{marker.color:.3e}<extra></extra>"
                ),
            )
        )
        fig = self._base_layout(
            fig,
            title=f"Complex GreenNet {axis}-segment solution error chord map",
            xaxis_title="x",
            yaxis_title="y",
        )
        fig.update_layout(width=850, height=850)
        fig.update_xaxes(scaleanchor="y", scaleratio=1.0)
        return fig

    @staticmethod
    def _metric_summary(df: pd.DataFrame, column: str) -> dict[str, float]:
        values = df[column].to_numpy(dtype=float)
        return {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
            "std": float(values.std(ddof=0)),
        }

    def write_summary(self, df: pd.DataFrame) -> None:
        rel_green_values = df["rel_green_interval_mean"].to_numpy(dtype=float)
        summary = {
            "source_csv": str(self.config.csv_path.resolve()),
            "row_count": int(len(df)),
            "axis_counts": {
                str(axis): int(count)
                for axis, count in df.groupby("axis", sort=True)
                .size()
                .to_dict()
                .items()
            },
            "metric_summary": {
                "length": self._metric_summary(df, "length"),
                "rel_sol_interval_mean": self._metric_summary(
                    df, "rel_sol_interval_mean"
                ),
                "rel_green_interval_mean": self._metric_summary(
                    df, "rel_green_interval_mean"
                ),
            },
            "rel_green_interval_mean_is_constant": bool(
                np.allclose(rel_green_values, rel_green_values[0], rtol=0.0, atol=0.0)
            ),
            "generated_figures": list(FIGURE_NAMES),
        }
        summary_path = self.config.outdir / "metrics_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.logger.info("Saved summary JSON to %s", summary_path)

    def run(self) -> None:
        self._check_overwrite()
        df = self.load_metrics_csv(self.config.csv_path)
        figures = {
            "rel_sol_by_fixed": self.build_rel_sol_by_fixed(df, log_y=False),
            "rel_sol_by_fixed_log": self.build_rel_sol_by_fixed(df, log_y=True),
            "rel_sol_by_length": self.build_rel_sol_by_length(df, log_y=False),
            "rel_sol_by_length_log": self.build_rel_sol_by_length(df, log_y=True),
            "rel_sol_distribution": self.build_distribution(
                df,
                metric_column="rel_sol_interval_mean",
                title="Complex GreenNet interval solution error distribution",
                yaxis_title="Mean relative solution reconstruction error",
            ),
            "rel_green_by_fixed": self.build_rel_green_by_fixed(df),
            "rel_green_distribution": self.build_distribution(
                df,
                metric_column="rel_green_interval_mean",
                title="Complex GreenNet interval Green-kernel error distribution",
                yaxis_title="Mean relative Green-kernel error",
            ),
            "rel_sol_chord_map_x": self.build_chord_map(df, axis="x"),
            "rel_sol_chord_map_y": self.build_chord_map(df, axis="y"),
        }
        for name, fig in figures.items():
            self.save_figure(fig, self.config.outdir, name)
        self.write_summary(df)


def _viridis_rgb(value: float) -> tuple[int, int, int]:
    scale = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    clipped = min(1.0, max(0.0, value))
    position = clipped * (len(scale) - 1)
    idx = min(int(position), len(scale) - 2)
    frac = position - idx
    start = scale[idx]
    end = scale[idx + 1]
    return tuple(
        int(round(start[channel] + frac * (end[channel] - start[channel])))
        for channel in range(3)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize complex GreenNet per_interval_metrics.csv with Plotly "
            "figures and a compact JSON summary."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        required=True,
        help="Path to per_interval_metrics.csv.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for Plotly figures, summary JSON, and log file.",
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
        help="Allow replacing existing output files in --outdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ComplexGreenIntervalMetricsPlotConfig(
        csv_path=args.csv_path,
        outdir=args.outdir,
        theme=args.theme,
        overwrite=args.overwrite,
    )
    ComplexGreenIntervalMetricsPlotter(config).run()


if __name__ == "__main__":
    main()
