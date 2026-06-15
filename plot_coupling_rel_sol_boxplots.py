from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go

from greenonet.plotly_io import save_plotly_figure


DEFAULT_INPUT_DIR = Path("checkpoints/For_Workshop/CouplingNetResults")
DEFAULT_BASENAME = "coupling_rel_sol_boxplot"
REQUIRED_EXPORT_SUFFIXES = (".html", ".json", ".png", ".pdf")
PROBLEM_ORDER = (
    "Poisson",
    "Diffusion",
    "Diffusion_Reaction",
    "Convection_Diffusion_Reaction",
)


@dataclass(frozen=True)
class RelSolSeries:
    label: str
    values: list[float]


class CouplingRelSolBoxplotter:
    """Build a workshop comparison boxplot from per-sample CouplingNet CSVs."""

    def __init__(
        self,
        *,
        indir: Path = DEFAULT_INPUT_DIR,
        outdir: Path = DEFAULT_INPUT_DIR,
        basename: str = DEFAULT_BASENAME,
        theme: str = "plotly_white",
        y_log: bool = False,
        rel_sol_percentile: float = 100.0,
    ) -> None:
        self.indir = indir
        self.outdir = outdir
        self.basename = basename
        self.theme = theme
        self.y_log = y_log
        if not math.isfinite(rel_sol_percentile) or not (0.0 < rel_sol_percentile <= 100.0):
            raise ValueError("rel_sol_percentile must be in (0, 100].")
        self.rel_sol_percentile = rel_sol_percentile

    @staticmethod
    def label_from_path(path: Path) -> str:
        stem = path.stem
        suffix = "_per_sample_metrics"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        return stem.replace("_", " ")

    @staticmethod
    def _problem_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        suffix = "_per_sample_metrics"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        try:
            return (PROBLEM_ORDER.index(stem), stem)
        except ValueError:
            return (len(PROBLEM_ORDER), stem)

    def find_csv_files(self) -> list[Path]:
        paths = sorted(
            self.indir.glob("*_per_sample_metrics.csv"),
            key=self._problem_key,
        )
        if len(paths) != 4:
            raise ValueError(
                "Expected exactly 4 '*_per_sample_metrics.csv' files in "
                f"{self.indir}, found {len(paths)}."
            )
        return paths

    @staticmethod
    def read_rel_sol(path: Path) -> list[float]:
        values: list[float] = []
        with path.open(newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None or "rel_sol" not in reader.fieldnames:
                raise ValueError(f"{path} does not contain a 'rel_sol' column.")
            for row_number, row in enumerate(reader, start=2):
                raw_value = row.get("rel_sol")
                try:
                    values.append(float(raw_value) if raw_value is not None else float("nan"))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{row_number} has non-numeric rel_sol={raw_value!r}."
                    ) from exc
        if not values:
            raise ValueError(f"{path} contains no rel_sol values.")
        return values

    def filter_lowest_percent(self, values: Sequence[float]) -> list[float]:
        if self.rel_sol_percentile >= 100.0:
            return list(values)
        sorted_values = sorted(values)
        n_keep = max(1, math.floor(len(sorted_values) * self.rel_sol_percentile / 100.0))
        return sorted_values[:n_keep]

    def load_series(self) -> list[RelSolSeries]:
        return [
            RelSolSeries(
                label=self.label_from_path(path),
                values=self.filter_lowest_percent(self.read_rel_sol(path)),
            )
            for path in self.find_csv_files()
        ]

    def make_figure(self, series: Sequence[RelSolSeries]) -> go.Figure:
        fig = go.Figure()
        for item in series:
            fig.add_trace(
                go.Box(
                    y=[value * 100 for value in item.values],
                    name=item.label,
                    boxpoints=False,
                )
            )
        fig.update_layout(
            title="CouplingNet Test rel_sol Distribution",
            template=self.theme,
            font={"family": "Times New Roman", "size": 22},
            width=1100,
            height=700,
            xaxis_title="Problem",
            yaxis_title="rel_sol (%)",
            yaxis_type="log" if self.y_log else "linear",
            yaxis_tickformat=".2f",
            yaxis_ticksuffix="%",
            boxmode="group",
        )
        return fig

    @staticmethod
    def _require_exports(base_path: Path) -> None:
        missing = [
            suffix
            for suffix in REQUIRED_EXPORT_SUFFIXES
            if not base_path.with_suffix(suffix).exists()
        ]
        if missing:
            raise RuntimeError(
                "Missing required Plotly export files for "
                f"{base_path}: {', '.join(missing)}."
            )

    def run(self) -> Path:
        series = self.load_series()
        fig = self.make_figure(series)
        base_path = self.outdir / self.basename
        save_plotly_figure(fig, base_path)
        self._require_exports(base_path)
        return base_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CouplingNet per-sample rel_sol boxplots by problem."
    )
    parser.add_argument(
        "--indir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing four *_per_sample_metrics.csv files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory where boxplot files will be written.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=DEFAULT_BASENAME,
        help="Output filename stem.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="plotly_white",
        help="Plotly template name.",
    )
    parser.add_argument(
        "--y-log",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--rel-sol-percentile",
        type=float,
        default=100.0,
        help="Keep only the lowest percentile of rel_sol values (1~100, default=100).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    CouplingRelSolBoxplotter(
        indir=args.indir,
        outdir=args.outdir,
        basename=args.basename,
        theme=args.theme,
        y_log=bool(args.y_log),
        rel_sol_percentile=args.rel_sol_percentile,
    ).run()


if __name__ == "__main__":
    main()
