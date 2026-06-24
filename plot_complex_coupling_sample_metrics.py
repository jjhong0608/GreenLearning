from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rich.logging import RichHandler

from greenonet.plotly_io import save_plotly_figure


LOG_Y_FLOOR: Final = 1e-16
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "file_stem",
    "loss",
    "loss_energy_consistency",
    "rel_sol",
    "rel_flux",
)
METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "loss",
    "loss_energy_consistency",
    "rel_sol",
    "rel_flux",
)
RELATIVE_METRIC_COLUMNS: Final[tuple[str, ...]] = ("rel_sol", "rel_flux")
FIGURE_NAMES: Final[tuple[str, ...]] = (
    "rel_sol_by_sample",
    "rel_flux_by_sample",
    "loss_by_sample",
    "loss_by_sample_log",
    "metric_distributions",
    "rel_sol_vs_rel_flux",
    "best_rel_sol_samples",
    "best_rel_flux_samples",
    "best_loss_samples",
    "worst_rel_sol_samples",
    "worst_rel_flux_samples",
    "worst_loss_samples",
)
METRIC_COLORS: Final[dict[str, str]] = {
    "rel_sol": "#2563eb",
    "rel_flux": "#d97706",
    "loss": "#047857",
    "loss_energy_consistency": "#7c3aed",
}


@dataclass(frozen=True)
class ComplexCouplingSampleMetricsPlotConfig:
    csv_path: Path
    outdir: Path
    theme: str = "plotly_white"
    top_n: int = 10
    overwrite: bool = False


class LoggingMixin:
    logger: logging.Logger

    @staticmethod
    def build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("ComplexCouplingSampleMetricsPlotter")
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

        df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
        sample_ids = pd.to_numeric(df["sample_id"], errors="raise")
        sample_id_values = sample_ids.to_numpy(dtype=float)
        if not np.isfinite(sample_id_values).all():
            raise ValueError("Metrics CSV column contains non-finite values: sample_id")
        if not np.allclose(sample_id_values, np.round(sample_id_values)):
            raise ValueError("Metrics CSV sample_id values must be integer-like.")
        if (sample_id_values < 0.0).any():
            raise ValueError("Metrics CSV sample_id values must be nonnegative.")
        df["sample_id"] = sample_id_values.astype(np.int64)
        if df["sample_id"].duplicated().any():
            duplicates = sorted(
                df.loc[df["sample_id"].duplicated(), "sample_id"].tolist()
            )
            raise ValueError(
                f"Metrics CSV contains duplicate sample_id values: {duplicates}"
            )

        file_stems = df["file_stem"].astype(str)
        if file_stems.str.strip().eq("").any():
            raise ValueError("Metrics CSV file_stem column contains empty values.")
        df["file_stem"] = file_stems

        for column in METRIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="raise")
            values = df[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"Metrics CSV column contains non-finite values: {column}"
                )
            if (values < 0.0).any():
                raise ValueError(
                    f"Metrics CSV column contains negative metric values: {column}"
                )

        df = df.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
        self.logger.info("Loaded %s sample metric rows from %s", len(df), csv_path)
        return df


class FigureSaveMixin:
    logger: logging.Logger

    def save_figure(self, fig: go.Figure, outdir: Path, name: str) -> None:
        save_plotly_figure(fig, outdir / name, logger=self.logger)
        self.logger.info("Saved Plotly figure set for %s", name)


class ComplexCouplingSampleMetricsPlotter(
    LoggingMixin,
    MetricsCsvMixin,
    FigureSaveMixin,
):
    def __init__(self, config: ComplexCouplingSampleMetricsPlotConfig) -> None:
        if config.top_n < 1:
            raise ValueError("top_n must be at least 1.")
        self.config = config
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        self.logger = self.build_logger(
            self.config.outdir / "plot_complex_coupling_sample_metrics.log"
        )

    def _check_overwrite(self) -> None:
        if self.config.overwrite:
            return
        existing = [self.config.outdir / "metrics_summary.json"]
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
        width: int = 1100,
        height: int = 700,
    ) -> go.Figure:
        fig.update_layout(
            title=title,
            template=self.config.theme,
            width=width,
            height=height,
            font={"family": "Times New Roman", "size": 20},
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            margin={"l": 90, "r": 50, "t": 90, "b": 80},
        )
        return fig

    @staticmethod
    def _percent_values(df: pd.DataFrame, column: str) -> np.ndarray:
        return df[column].to_numpy(dtype=float) * 100.0

    @staticmethod
    def _floored(values: np.ndarray) -> np.ndarray:
        return np.maximum(values.astype(float), LOG_Y_FLOOR)

    @staticmethod
    def _sample_customdata(df: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                df["sample_id"].to_numpy(dtype=object),
                df["file_stem"].to_numpy(dtype=object),
            ],
            axis=1,
        )

    def build_metric_by_sample(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        percent: bool,
        log_y: bool = False,
    ) -> go.Figure:
        y_values = (
            self._percent_values(df, metric_column)
            if percent
            else df[metric_column].to_numpy(dtype=float)
        )
        if log_y:
            y_values = self._floored(y_values)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["sample_id"],
                y=y_values,
                mode="lines+markers",
                name=metric_column,
                line={"color": METRIC_COLORS[metric_column], "width": 2},
                marker={
                    "size": 8,
                    "color": METRIC_COLORS[metric_column],
                    "line": {"color": "#111827", "width": 0.5},
                },
                customdata=self._sample_customdata(df),
                hovertemplate=(
                    "sample_id=%{customdata[0]}<br>"
                    "file=%{customdata[1]}<br>"
                    f"{metric_column}=%{{y:.4g}}"
                    "<extra></extra>"
                ),
            )
        )
        yaxis_title = f"{metric_column} (%)" if percent else metric_column
        fig = self._base_layout(
            fig,
            title=f"Complex CouplingNet test {metric_column} by sample",
            xaxis_title="Sample ID",
            yaxis_title=yaxis_title,
        )
        if percent:
            fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        if log_y:
            fig.update_yaxes(type="log")
        return fig

    def build_metric_distributions(self, df: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        for column in RELATIVE_METRIC_COLUMNS:
            fig.add_trace(
                go.Box(
                    y=self._percent_values(df, column),
                    name=column,
                    boxpoints=False,
                    marker={"color": METRIC_COLORS[column]},
                    line={"color": METRIC_COLORS[column]},
                    hovertemplate=f"{column}=%{{y:.4g}}%<extra></extra>",
                )
            )
        fig = self._base_layout(
            fig,
            title="Complex CouplingNet relative metric distributions",
            xaxis_title="Metric",
            yaxis_title="Relative error (%)",
        )
        fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        fig.update_layout(boxmode="group")
        return fig

    def build_rel_sol_vs_rel_flux(self, df: pd.DataFrame) -> go.Figure:
        customdata = np.stack(
            [
                df["sample_id"].to_numpy(dtype=object),
                df["file_stem"].to_numpy(dtype=object),
                df["loss"].to_numpy(dtype=object),
            ],
            axis=1,
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self._percent_values(df, "rel_sol"),
                y=self._percent_values(df, "rel_flux"),
                mode="markers",
                marker={
                    "size": 10,
                    "color": df["loss"],
                    "colorscale": "Viridis",
                    "colorbar": {"title": "loss"},
                    "line": {"color": "#111827", "width": 0.5},
                },
                customdata=customdata,
                hovertemplate=(
                    "sample_id=%{customdata[0]}<br>"
                    "file=%{customdata[1]}<br>"
                    "loss=%{customdata[2]:.4g}<br>"
                    "rel_sol=%{x:.4g}%<br>"
                    "rel_flux=%{y:.4g}%<extra></extra>"
                ),
                name="test samples",
            )
        )
        fig = self._base_layout(
            fig,
            title="Complex CouplingNet rel_sol versus rel_flux",
            xaxis_title="rel_sol (%)",
            yaxis_title="rel_flux (%)",
        )
        fig.update_xaxes(tickformat=".2f", ticksuffix="%")
        fig.update_yaxes(tickformat=".2f", ticksuffix="%")
        return fig

    def build_ranked_samples(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        percent: bool,
        rank_kind: str,
    ) -> go.Figure:
        if rank_kind not in {"best", "worst"}:
            raise ValueError(f"Unsupported rank kind: {rank_kind}")
        ascending = rank_kind == "best"
        ranked = (
            df.sort_values(metric_column, ascending=ascending, kind="mergesort")
            .head(self.config.top_n)
            .copy()
        )
        x_values = (
            self._percent_values(ranked, metric_column)
            if percent
            else ranked[metric_column].to_numpy(dtype=float)
        )
        hover_metric_name = f"{metric_column} (%)" if percent else metric_column
        customdata = np.stack(
            [
                ranked["sample_id"].to_numpy(dtype=object),
                ranked["file_stem"].to_numpy(dtype=object),
            ],
            axis=1,
        )
        label = "Best" if rank_kind == "best" else "Worst"
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=ranked["file_stem"],
                orientation="h",
                marker={"color": METRIC_COLORS[metric_column]},
                customdata=customdata,
                hovertemplate=(
                    "sample_id=%{customdata[0]}<br>"
                    "file=%{customdata[1]}<br>"
                    f"{hover_metric_name}=%{{x:.4g}}"
                    "<extra></extra>"
                ),
                name=metric_column,
            )
        )
        xaxis_title = hover_metric_name
        fig = self._base_layout(
            fig,
            title=f"{label} Complex CouplingNet test samples by {metric_column}",
            xaxis_title=xaxis_title,
            yaxis_title="Sample",
            height=max(520, 120 + 46 * len(ranked)),
        )
        fig.update_yaxes(autorange="reversed")
        if percent:
            fig.update_xaxes(tickformat=".2f", ticksuffix="%")
        return fig

    def build_best_samples(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        percent: bool,
    ) -> go.Figure:
        return self.build_ranked_samples(
            df,
            metric_column=metric_column,
            percent=percent,
            rank_kind="best",
        )

    def build_worst_samples(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        percent: bool,
    ) -> go.Figure:
        return self.build_ranked_samples(
            df,
            metric_column=metric_column,
            percent=percent,
            rank_kind="worst",
        )

    @staticmethod
    def _metric_summary(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values, ddof=0)),
            "q25": float(np.percentile(values, 25)),
            "median": float(np.percentile(values, 50)),
            "q75": float(np.percentile(values, 75)),
        }

    @staticmethod
    def _json_float(value: float) -> float | None:
        if np.isfinite(value):
            return float(value)
        return None

    def _metric_correlation(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, float | None]]:
        corr = df.loc[:, list(METRIC_COLUMNS)].corr(method="pearson")
        return {
            str(row_key): {
                str(column_key): self._json_float(float(value))
                for column_key, value in row.items()
            }
            for row_key, row in corr.iterrows()
        }

    def _ranked_sample_records(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
        rank_kind: str,
    ) -> list[dict[str, int | str | float]]:
        if rank_kind not in {"best", "worst"}:
            raise ValueError(f"Unsupported rank kind: {rank_kind}")
        ascending = rank_kind == "best"
        ranked = (
            df.sort_values(metric_column, ascending=ascending, kind="mergesort")
            .head(self.config.top_n)
            .copy()
        )
        records: list[dict[str, int | str | float]] = []
        for row in ranked.itertuples(index=False):
            record: dict[str, int | str | float] = {
                "sample_id": int(row.sample_id),
                "file_stem": str(row.file_stem),
                metric_column: float(getattr(row, metric_column)),
            }
            if metric_column in RELATIVE_METRIC_COLUMNS:
                record[f"{metric_column}_percent"] = (
                    float(getattr(row, metric_column)) * 100.0
                )
            records.append(record)
        return records

    def _best_sample_records(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
    ) -> list[dict[str, int | str | float]]:
        return self._ranked_sample_records(
            df,
            metric_column=metric_column,
            rank_kind="best",
        )

    def _worst_sample_records(
        self,
        df: pd.DataFrame,
        *,
        metric_column: str,
    ) -> list[dict[str, int | str | float]]:
        return self._ranked_sample_records(
            df,
            metric_column=metric_column,
            rank_kind="worst",
        )

    def write_summary(self, df: pd.DataFrame) -> None:
        metric_summary_raw = {
            column: self._metric_summary(df[column].to_numpy(dtype=float))
            for column in METRIC_COLUMNS
        }
        metric_summary_percent = {
            column: self._metric_summary(self._percent_values(df, column))
            for column in RELATIVE_METRIC_COLUMNS
        }
        summary = {
            "source_csv": str(self.config.csv_path.resolve()),
            "row_count": int(len(df)),
            "sample_id_min": int(df["sample_id"].min()),
            "sample_id_max": int(df["sample_id"].max()),
            "top_n": int(self.config.top_n),
            "generated_figures": list(FIGURE_NAMES),
            "relative_metric_display": "percent_in_figures_raw_fraction_in_csv",
            "metric_summary_raw": metric_summary_raw,
            "metric_summary_percent": metric_summary_percent,
            "metric_correlation_pearson": self._metric_correlation(df),
            "loss_equals_loss_energy_consistency": bool(
                np.allclose(
                    df["loss"].to_numpy(dtype=float),
                    df["loss_energy_consistency"].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=0.0,
                )
            ),
            "best_samples": {
                "rel_sol": self._best_sample_records(df, metric_column="rel_sol"),
                "rel_flux": self._best_sample_records(df, metric_column="rel_flux"),
                "loss": self._best_sample_records(df, metric_column="loss"),
            },
            "worst_samples": {
                "rel_sol": self._worst_sample_records(df, metric_column="rel_sol"),
                "rel_flux": self._worst_sample_records(df, metric_column="rel_flux"),
                "loss": self._worst_sample_records(df, metric_column="loss"),
            },
        }
        summary_path = self.config.outdir / "metrics_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.logger.info("Saved summary JSON to %s", summary_path)

    def run(self) -> None:
        self._check_overwrite()
        df = self.load_metrics_csv(self.config.csv_path)
        figures = {
            "rel_sol_by_sample": self.build_metric_by_sample(
                df,
                metric_column="rel_sol",
                percent=True,
            ),
            "rel_flux_by_sample": self.build_metric_by_sample(
                df,
                metric_column="rel_flux",
                percent=True,
            ),
            "loss_by_sample": self.build_metric_by_sample(
                df,
                metric_column="loss",
                percent=False,
            ),
            "loss_by_sample_log": self.build_metric_by_sample(
                df,
                metric_column="loss",
                percent=False,
                log_y=True,
            ),
            "metric_distributions": self.build_metric_distributions(df),
            "rel_sol_vs_rel_flux": self.build_rel_sol_vs_rel_flux(df),
            "best_rel_sol_samples": self.build_best_samples(
                df,
                metric_column="rel_sol",
                percent=True,
            ),
            "best_rel_flux_samples": self.build_best_samples(
                df,
                metric_column="rel_flux",
                percent=True,
            ),
            "best_loss_samples": self.build_best_samples(
                df,
                metric_column="loss",
                percent=False,
            ),
            "worst_rel_sol_samples": self.build_worst_samples(
                df,
                metric_column="rel_sol",
                percent=True,
            ),
            "worst_rel_flux_samples": self.build_worst_samples(
                df,
                metric_column="rel_flux",
                percent=True,
            ),
            "worst_loss_samples": self.build_worst_samples(
                df,
                metric_column="loss",
                percent=False,
            ),
        }
        for name, fig in figures.items():
            self.save_figure(fig, self.config.outdir, name)
        self.write_summary(df)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize complex CouplingNet test_per_sample_metrics.csv with "
            "Plotly figures and a compact JSON summary."
        )
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=Path,
        required=True,
        help="Path to test_per_sample_metrics.csv.",
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
        "--top-n",
        type=int,
        default=10,
        help="Number of best/worst samples to include in ranking figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files in --outdir.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ComplexCouplingSampleMetricsPlotConfig(
        csv_path=args.csv_path,
        outdir=args.outdir,
        theme=args.theme,
        top_n=args.top_n,
        overwrite=args.overwrite,
    )
    ComplexCouplingSampleMetricsPlotter(config).run()


if __name__ == "__main__":
    main()
