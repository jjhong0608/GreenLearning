from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pytest import MonkeyPatch

from plot_complex_coupling_sample_metrics import (
    ComplexCouplingSampleMetricsPlotConfig,
    ComplexCouplingSampleMetricsPlotter,
    FIGURE_NAMES,
)


def _write_metrics_csv(
    path: Path,
    *,
    missing_column: bool = False,
    duplicate_sample_id: bool = False,
    negative_metric: bool = False,
    nonfinite_metric: bool = False,
) -> None:
    header = "sample_id,file_stem,loss,loss_energy_consistency,rel_sol,rel_flux\n"
    if missing_column:
        header = "sample_id,file_stem,loss,rel_sol,rel_flux\n"
    rows = [
        [0, "sample_000000", 2.5e-4, 2.5e-4, 0.020, 0.110],
        [1, "sample_000001", 4.5e-4, 4.5e-4, 0.050, 0.090],
        [2, "sample_000002", 3.5e-4, 3.5e-4, 0.030, 0.160],
        [3, "sample_000003", 7.0e-4, 7.0e-4, 0.080, 0.180],
    ]
    if duplicate_sample_id:
        rows[-1][0] = 1
    if negative_metric:
        rows[0][4] = -0.1
    if nonfinite_metric:
        rows[0][5] = float("nan")

    lines = [header]
    for row in rows:
        if missing_column:
            lines.append(f"{row[0]},{row[1]},{row[2]:.8f},{row[4]:.8f},{row[5]:.8f}\n")
        else:
            lines.append(
                f"{row[0]},{row[1]},{row[2]:.8f},{row[3]:.8f},"
                f"{row[4]:.8f},{row[5]:.8f}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


def _plotter(
    csv_path: Path,
    outdir: Path,
    *,
    top_n: int = 2,
) -> ComplexCouplingSampleMetricsPlotter:
    return ComplexCouplingSampleMetricsPlotter(
        ComplexCouplingSampleMetricsPlotConfig(
            csv_path=csv_path,
            outdir=outdir,
            top_n=top_n,
            overwrite=True,
        )
    )


def test_run_saves_expected_figures_and_summary(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    csv_path = tmp_path / "test_per_sample_metrics.csv"
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
    summary = json.loads((outdir / "metrics_summary.json").read_text())
    assert summary["row_count"] == 4
    assert summary["top_n"] == 2
    assert summary["loss_equals_loss_energy_consistency"] is True
    assert summary["metric_summary_percent"]["rel_sol"]["max"] == pytest.approx(8.0)
    assert summary["best_samples"]["rel_sol"][0]["sample_id"] == 0
    assert summary["worst_samples"]["rel_sol"][0]["sample_id"] == 3
    assert (outdir / "plot_complex_coupling_sample_metrics.log").exists()


def test_missing_required_column_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    _write_metrics_csv(csv_path, missing_column=True)

    with pytest.raises(ValueError, match="missing required columns"):
        _plotter(csv_path, tmp_path / "out").load_metrics_csv(csv_path)


def test_nonfinite_and_negative_metrics_fail_fast(tmp_path: Path) -> None:
    nonfinite_path = tmp_path / "nonfinite.csv"
    negative_path = tmp_path / "negative.csv"
    _write_metrics_csv(nonfinite_path, nonfinite_metric=True)
    _write_metrics_csv(negative_path, negative_metric=True)

    with pytest.raises(ValueError, match="non-finite values: rel_flux"):
        _plotter(nonfinite_path, tmp_path / "out_nonfinite").load_metrics_csv(
            nonfinite_path
        )
    with pytest.raises(ValueError, match="negative metric values: rel_sol"):
        _plotter(negative_path, tmp_path / "out_negative").load_metrics_csv(
            negative_path
        )


def test_duplicate_sample_id_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicate.csv"
    _write_metrics_csv(csv_path, duplicate_sample_id=True)

    with pytest.raises(ValueError, match="duplicate sample_id"):
        _plotter(csv_path, tmp_path / "out").load_metrics_csv(csv_path)


def test_figure_contracts_and_top_n(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_per_sample_metrics.csv"
    _write_metrics_csv(csv_path)
    plotter = _plotter(csv_path, tmp_path / "out", top_n=2)
    df = plotter.load_metrics_csv(csv_path)

    fig_log = plotter.build_metric_by_sample(
        df,
        metric_column="loss",
        percent=False,
        log_y=True,
    )
    assert fig_log.layout.yaxis.type == "log"

    fig_box = plotter.build_metric_distributions(df)
    assert fig_box.data[0].boxpoints is False
    assert fig_box.layout.yaxis.title.text == "Relative error (%)"
    assert fig_box.layout.yaxis.ticksuffix == "%"

    fig_scatter = plotter.build_rel_sol_vs_rel_flux(df)
    assert fig_scatter.data[0].marker.colorbar.title.text == "loss"
    assert "sample_id" in fig_scatter.data[0].hovertemplate
    assert fig_scatter.layout.xaxis.ticksuffix == "%"

    fig_best = plotter.build_best_samples(
        df,
        metric_column="rel_flux",
        percent=True,
    )
    best_customdata = np.asarray(fig_best.data[0].customdata)
    assert best_customdata[:, 0].tolist() == [1, 0]
    assert fig_best.layout.title.text == (
        "Best Complex CouplingNet test samples by rel_flux"
    )
    assert fig_best.layout.xaxis.ticksuffix == "%"

    fig_worst = plotter.build_worst_samples(
        df,
        metric_column="rel_flux",
        percent=True,
    )
    customdata = np.asarray(fig_worst.data[0].customdata)
    assert customdata[:, 0].tolist() == [3, 2]
    assert fig_worst.layout.xaxis.ticksuffix == "%"


def test_invalid_top_n_fails_fast(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_per_sample_metrics.csv"
    _write_metrics_csv(csv_path)

    with pytest.raises(ValueError, match="top_n"):
        _plotter(csv_path, tmp_path / "out", top_n=0)
