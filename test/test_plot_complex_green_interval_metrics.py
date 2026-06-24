from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from plot_complex_green_interval_metrics import (
    ComplexGreenIntervalMetricsPlotConfig,
    ComplexGreenIntervalMetricsPlotter,
    FIGURE_NAMES,
)


def _write_metrics_csv(path: Path, *, axis_override: str | None = None) -> None:
    header = (
        "interval_index,axis_id,axis,segment_index,left,right,fixed,length,"
        "rel_sol_interval_mean,rel_sol_interval_min,rel_sol_interval_max,"
        "rel_sol_interval_std,rel_green_interval_mean,rel_green_interval_min,"
        "rel_green_interval_max,rel_green_interval_std\n"
    )
    rows = [
        ("0", "0", "x", "0", "-0.5", "0.5", "-0.25", "1.0", "1e-3"),
        ("1", "0", "x", "1", "-0.4", "0.4", "0.25", "0.8", "2e-3"),
        ("2", "1", "y", "0", "-0.5", "0.5", "-0.25", "1.0", "1.5e-3"),
        ("3", "1", "y", "1", "-0.4", "0.4", "0.25", "0.8", "2.5e-3"),
    ]
    lines = [header]
    for row in rows:
        (
            interval_index,
            axis_id,
            axis,
            segment_index,
            left,
            right,
            fixed,
            length,
            sol,
        ) = row
        if axis_override is not None and interval_index == "0":
            axis = axis_override
        lines.append(
            ",".join(
                [
                    interval_index,
                    axis_id,
                    axis,
                    segment_index,
                    left,
                    right,
                    fixed,
                    length,
                    sol,
                    "5e-4",
                    "3e-3",
                    "1e-4",
                    "3.25e-4",
                    "3.25e-4",
                    "3.25e-4",
                    "0.0",
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _plotter(csv_path: Path, outdir: Path) -> ComplexGreenIntervalMetricsPlotter:
    return ComplexGreenIntervalMetricsPlotter(
        ComplexGreenIntervalMetricsPlotConfig(
            csv_path=csv_path,
            outdir=outdir,
            overwrite=True,
        )
    )


def test_run_saves_expected_figures_and_summary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    csv_path = tmp_path / "per_interval_metrics.csv"
    outdir = tmp_path / "figures"
    _write_metrics_csv(csv_path)

    def fake_write_image(self: object, path: str) -> None:
        del self
        Path(path).write_text("stub", encoding="utf-8")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)

    _plotter(csv_path, outdir).run()

    for name in FIGURE_NAMES:
        for suffix in (".html", ".json", ".png", ".pdf"):
            assert (outdir / f"{name}{suffix}").exists()
    assert (outdir / "metrics_summary.json").exists()
    assert (outdir / "plot_complex_green_interval_metrics.log").exists()


def test_missing_required_column_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("axis,rel_sol_interval_mean\nx,1e-3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        _plotter(csv_path, tmp_path / "out").load_metrics_csv(csv_path)


def test_invalid_axis_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad_axis.csv"
    _write_metrics_csv(csv_path, axis_override="z")

    with pytest.raises(ValueError, match="only 'x' or 'y'"):
        _plotter(csv_path, tmp_path / "out").load_metrics_csv(csv_path)


def test_log_distribution_and_chord_map_contracts(tmp_path: Path) -> None:
    csv_path = tmp_path / "per_interval_metrics.csv"
    _write_metrics_csv(csv_path)
    plotter = _plotter(csv_path, tmp_path / "out")
    df = plotter.load_metrics_csv(csv_path)

    fig_log = plotter.build_rel_sol_by_fixed(df, log_y=True)
    assert fig_log.layout.yaxis.type == "log"

    fig_box = plotter.build_distribution(
        df,
        metric_column="rel_sol_interval_mean",
        title="distribution",
        yaxis_title="rel_sol",
    )
    assert fig_box.data[0].boxpoints is False
    assert fig_box.data[1].boxpoints is False

    fig_chord = plotter.build_chord_map(df, axis="x")
    assert fig_chord.layout.xaxis.scaleanchor == "y"
