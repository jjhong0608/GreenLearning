from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import plotly.graph_objects as go
from rich.logging import RichHandler


AxisName = Literal["x", "y"]


@dataclass(frozen=True)
class LineMetricRow:
    axis_name: str
    line_coordinate: float
    rel_sol_line_mean: float
    rel_sol_line_std: float
    rel_green_line_mean: float
    val_rel_sol_line_mean: float | None
    val_rel_sol_line_std: float | None


class PerLineBarPlotter:
    REQUIRED_COLUMNS = (
        "axis_name",
        "line_coordinate",
        "rel_sol_line_mean",
        "rel_sol_line_std",
        "rel_green_line_mean",
    )

    @staticmethod
    def _parse_optional_float(value: str | None) -> float | None:
        if value is None:
            return None
        stripped = value.strip()
        if stripped == "":
            return None
        return float(stripped)

    def __init__(
        self,
        csv_a: Path,
        csv_b: Path,
        outdir: Path,
        label_a: str | None = None,
        label_b: str | None = None,
    ) -> None:
        self.csv_a = csv_a
        self.csv_b = csv_b
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.label_a = label_a or csv_a.stem
        self.label_b = label_b or csv_b.stem
        self.logger = self._build_logger(self.outdir / "plot_per_line_bars.log")

    @staticmethod
    def _build_logger(log_path: Path) -> logging.Logger:
        logger = logging.getLogger("PerLineBarPlotter")
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

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()
        return logger

    def _read_rows(self, csv_path: Path) -> list[LineMetricRow]:
        with csv_path.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise ValueError(f"No header found in CSV: {csv_path}")
            missing = [c for c in self.REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing columns {missing}: {csv_path}")

            rows: list[LineMetricRow] = []
            for row in reader:
                rows.append(
                    LineMetricRow(
                        axis_name=row["axis_name"],
                        line_coordinate=float(row["line_coordinate"]),
                        rel_sol_line_mean=float(row["rel_sol_line_mean"]),
                        rel_sol_line_std=float(row["rel_sol_line_std"]),
                        rel_green_line_mean=float(row["rel_green_line_mean"]),
                        val_rel_sol_line_mean=self._parse_optional_float(
                            row.get("val_rel_sol_line_mean")
                        ),
                        val_rel_sol_line_std=self._parse_optional_float(
                            row.get("val_rel_sol_line_std")
                        ),
                    )
                )
        self.logger.info("Loaded %s rows from %s", len(rows), csv_path)
        return rows

    def _aligned_axis_rows(
        self,
        rows_a: list[LineMetricRow],
        rows_b: list[LineMetricRow],
        axis: AxisName,
    ) -> tuple[list[LineMetricRow], list[LineMetricRow]]:
        axis_rows_a = [r for r in rows_a if r.axis_name == axis]
        axis_rows_b = [r for r in rows_b if r.axis_name == axis]
        map_a = {r.line_coordinate: r for r in axis_rows_a}
        map_b = {r.line_coordinate: r for r in axis_rows_b}
        common_coords = sorted(set(map_a).intersection(map_b))
        aligned_a = [map_a[c] for c in common_coords]
        aligned_b = [map_b[c] for c in common_coords]
        self.logger.info(
            "Axis %s aligned rows: %s (csv-a=%s, csv-b=%s)",
            axis,
            len(common_coords),
            len(axis_rows_a),
            len(axis_rows_b),
        )
        return aligned_a, aligned_b

    def _build_sol_figure_for_axis(
        self,
        axis: AxisName,
        rows_a: list[LineMetricRow],
        rows_b: list[LineMetricRow],
    ) -> go.Figure:
        x_a = [r.line_coordinate for r in rows_a]
        sol_a = [r.rel_sol_line_mean for r in rows_a]
        sol_a_std = [r.rel_sol_line_std for r in rows_a]

        x_b = [r.line_coordinate for r in rows_b]
        sol_b = [r.rel_sol_line_mean for r in rows_b]
        sol_b_std = [r.rel_sol_line_std for r in rows_b]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=self.label_a,
                x=x_a,
                y=sol_a,
                error_y=dict(type="data", array=sol_a_std, visible=True),
                # marker_color="#1f77b4",
                marker_color="#17becf",
            )
        )
        fig.add_trace(
            go.Bar(
                name=self.label_b,
                x=x_b,
                y=sol_b,
                error_y=dict(type="data", array=sol_b_std, visible=True),
                # marker_color="#d62728",
                marker_color="#ff9896",
            )
        )
        fig.update_layout(
            title=f"Per-line rel. err. sol. ({axis}-axial lines)",
            # title=f"Per-line rel_sol_line_mean ({axis}-axis lines)",
            xaxis_title="coordinate",
            yaxis_title="Relative L2 error of the solution",
            barmode="group",
            template="plotly_white",
            font=go.layout.Font(family="Times New Roman", weight="bold"),
        )
        return fig

    def _build_val_sol_figure_for_axis(
        self,
        axis: AxisName,
        rows_a: list[LineMetricRow],
        rows_b: list[LineMetricRow],
    ) -> go.Figure:
        x_a = [r.line_coordinate for r in rows_a]
        sol_a = [r.val_rel_sol_line_mean for r in rows_a]
        sol_a_std = [r.val_rel_sol_line_std for r in rows_a]

        x_b = [r.line_coordinate for r in rows_b]
        sol_b = [r.val_rel_sol_line_mean for r in rows_b]
        sol_b_std = [r.val_rel_sol_line_std for r in rows_b]

        if any(value is None for value in sol_a + sol_b + sol_a_std + sol_b_std):
            raise ValueError(
                f"Validation solution metrics are incomplete for axis '{axis}'."
            )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=self.label_a,
                x=x_a,
                y=sol_a,
                error_y=dict(type="data", array=sol_a_std, visible=True),
                marker_color="#17becf",
            )
        )
        fig.add_trace(
            go.Bar(
                name=self.label_b,
                x=x_b,
                y=sol_b,
                error_y=dict(type="data", array=sol_b_std, visible=True),
                marker_color="#ff9896",
            )
        )
        fig.update_layout(
            title=f"Per-line val. rel. err. sol. ({axis}-axial lines)",
            xaxis_title="coordinate",
            yaxis_title="Relative L2 error of the validation solution",
            barmode="group",
            template="plotly_white",
            font=go.layout.Font(family="Times New Roman", weight="bold"),
        )
        return fig

    @staticmethod
    def _has_validation_sol(rows: list[LineMetricRow]) -> bool:
        return any(
            row.val_rel_sol_line_mean is not None and row.val_rel_sol_line_std is not None
            for row in rows
        )

    def _build_green_figure_for_axis(
        self,
        axis: AxisName,
        rows_a: list[LineMetricRow],
        rows_b: list[LineMetricRow],
    ) -> go.Figure:
        x_a = [r.line_coordinate for r in rows_a]
        green_a = [r.rel_green_line_mean for r in rows_a]

        x_b = [r.line_coordinate for r in rows_b]
        green_b = [r.rel_green_line_mean for r in rows_b]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name=self.label_a,
                x=x_a,
                y=green_a,
                marker_color="#17becf",
            )
        )
        fig.add_trace(
            go.Bar(
                name=self.label_b,
                x=x_b,
                y=green_b,
                marker_color="#ff9896",
            )
        )

        fig.update_layout(
            # title=f"Per-line rel_green_line_mean ({axis}-axis lines)",
            title=f"Per-line rel. err. green's function ({axis}-axial lines)",
            xaxis_title="coordinate",
            yaxis_title="Relative L2 error of the Green's function",
            barmode="group",
            template="plotly_white",
            font=go.layout.Font(family="Times New Roman", weight="bold"),
        )
        return fig

    def _save_axis_metric_figure(
        self, fig: go.Figure, axis: AxisName, metric_tag: str
    ) -> None:
        base = self.outdir / f"per_line_compare_{axis}_{metric_tag}"
        png_path = base.with_suffix(".png")
        pdf_path = base.with_suffix(".pdf")
        fig.write_image(str(png_path))
        fig.write_image(str(pdf_path))
        self.logger.info("Saved figure files: %s and %s", png_path, pdf_path)

    def run(self) -> None:
        rows_a = self._read_rows(self.csv_a)
        rows_b = self._read_rows(self.csv_b)
        for axis in ("x", "y"):
            aligned_a, aligned_b = self._aligned_axis_rows(rows_a, rows_b, axis=axis)
            if not aligned_a:
                raise ValueError(
                    f"No common line_coordinate rows for axis '{axis}' "
                    f"between {self.csv_a} and {self.csv_b}"
                )
            fig_sol = self._build_sol_figure_for_axis(
                axis=axis, rows_a=aligned_a, rows_b=aligned_b
            )
            self._save_axis_metric_figure(fig_sol, axis=axis, metric_tag="rel_sol")
            has_val_a = self._has_validation_sol(aligned_a)
            has_val_b = self._has_validation_sol(aligned_b)
            if has_val_a and has_val_b:
                fig_val_sol = self._build_val_sol_figure_for_axis(
                    axis=axis, rows_a=aligned_a, rows_b=aligned_b
                )
                self._save_axis_metric_figure(
                    fig_val_sol, axis=axis, metric_tag="val_rel_sol"
                )
            elif has_val_a != has_val_b:
                raise ValueError(
                    f"Validation solution metrics are present for only one CSV on axis '{axis}'."
                )
            fig_green = self._build_green_figure_for_axis(
                axis=axis, rows_a=aligned_a, rows_b=aligned_b
            )
            self._save_axis_metric_figure(fig_green, axis=axis, metric_tag="rel_green")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two per_line_metrics.csv files with grouped bars and "
            "error bars for training/validation solution metrics."
        )
    )
    parser.add_argument("--csv-a", type=Path, required=True, help="First CSV path")
    parser.add_argument("--csv-b", type=Path, required=True, help="Second CSV path")
    parser.add_argument(
        "--label-a",
        type=str,
        default=None,
        help="Legend label for first CSV (default: file stem)",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default=None,
        help="Legend label for second CSV (default: file stem)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots_per_line_bars"),
        help="Output directory for PNG/PDF figures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plotter = PerLineBarPlotter(
        csv_a=args.csv_a,
        csv_b=args.csv_b,
        outdir=args.outdir,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    plotter.run()
