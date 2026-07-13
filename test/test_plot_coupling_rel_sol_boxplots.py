from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import pytest

from plot_coupling_rel_sol_boxplots import (
    CouplingRelSolBoxplotter,
)


CSV_NAMES = (
    "Convection_Diffusion_Reaction_per_sample_metrics.csv",
    "Poisson_per_sample_metrics.csv",
    "Diffusion_per_sample_metrics.csv",
    "Diffusion_Reaction_per_sample_metrics.csv",
)


def _write_csvs(
    directory: Path,
    *,
    missing_rel_sol: bool = False,
    custom_values: dict[str, list[float]] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(CSV_NAMES):
        path = directory / name
        if missing_rel_sol and name == "Poisson_per_sample_metrics.csv":
            path.write_text("sample_id,file,rel_flux\n0,sample,0.2\n")
            continue
        if custom_values and name in custom_values:
            values = custom_values[name]
        else:
            values = [0.1 + idx, 0.2 + idx]
        rows = ["sample_id,file,rel_sol,rel_flux"]
        for row_idx, rel_sol_value in enumerate(values):
            rows.append(f"{row_idx},sample_{row_idx},{rel_sol_value:.3f},0.2")
        path.write_text("\n".join(rows) + "\n")


def test_load_series_orders_and_labels_workshop_csvs(tmp_path: Path) -> None:
    _write_csvs(tmp_path)
    plotter = CouplingRelSolBoxplotter(indir=tmp_path, outdir=tmp_path)

    series = plotter.load_series()

    assert [item.label for item in series] == [
        "Poisson",
        "Diffusion",
        "Diffusion Reaction",
        "Convection Diffusion Reaction",
    ]
    assert [item.values for item in series] == [
        [1.1, 1.2],
        [2.1, 2.2],
        [3.1, 3.2],
        [0.1, 0.2],
    ]


def test_make_figure_creates_one_box_trace_per_problem(tmp_path: Path) -> None:
    _write_csvs(tmp_path)
    plotter = CouplingRelSolBoxplotter(indir=tmp_path, outdir=tmp_path)

    fig = plotter.make_figure(plotter.load_series())

    assert len(fig.data) == 4
    assert [trace.name for trace in fig.data] == [
        "Poisson",
        "Diffusion",
        "Diffusion Reaction",
        "Convection Diffusion Reaction",
    ]
    assert all(trace.type == "box" for trace in fig.data)
    assert all(trace.boxpoints is False for trace in fig.data)
    assert fig.layout.yaxis.title.text == "rel_sol (%)"
    assert fig.layout.yaxis.ticksuffix == "%"
    assert fig.layout.yaxis.tickformat == ".2f"

    expected_scaled = [
        [110.0, 120.0],
        [210.0, 220.0],
        [310.0, 320.0],
        [10.0, 20.0],
    ]
    for trace, expected in zip(fig.data, expected_scaled, strict=True):
        assert list(trace.y) == pytest.approx(expected)


def test_filter_lowest_percentile_keeps_low_values(tmp_path: Path) -> None:
    custom_values = {
        "Poisson_per_sample_metrics.csv": [0.34, 0.12, 0.78, 0.19, 0.45, 0.06],
        "Diffusion_per_sample_metrics.csv": [1.3, 0.9],
        "Diffusion_Reaction_per_sample_metrics.csv": [4.2, 3.1],
        "Convection_Diffusion_Reaction_per_sample_metrics.csv": [0.5, 0.4],
    }
    _write_csvs(tmp_path, custom_values=custom_values)
    plotter = CouplingRelSolBoxplotter(
        indir=tmp_path,
        outdir=tmp_path,
        rel_sol_percentile=50.0,
    )
    series = plotter.load_series()

    assert series[0].values == [0.06, 0.12, 0.19]
    assert series[1].values == [0.9]
    assert series[2].values == [3.1]
    assert series[3].values == [0.4]


def test_missing_rel_sol_column_fails_fast(tmp_path: Path) -> None:
    _write_csvs(tmp_path, missing_rel_sol=True)
    plotter = CouplingRelSolBoxplotter(indir=tmp_path, outdir=tmp_path)

    with pytest.raises(ValueError, match="rel_sol"):
        plotter.load_series()


def test_run_writes_all_four_plotly_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)
    _write_csvs(tmp_path)
    plotter = CouplingRelSolBoxplotter(
        indir=tmp_path,
        outdir=tmp_path,
        basename="boxplot",
    )

    base_path = plotter.run()

    assert base_path == tmp_path / "boxplot"
    for suffix in (".html", ".json", ".png", ".pdf"):
        assert base_path.with_suffix(suffix).exists()


def test_run_reports_missing_static_exports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_image(self: go.Figure, path: str) -> None:
        del self, path
        raise RuntimeError("no static backend")

    monkeypatch.setattr(go.Figure, "write_image", fail_write_image)
    _write_csvs(tmp_path)
    plotter = CouplingRelSolBoxplotter(
        indir=tmp_path,
        outdir=tmp_path,
        basename="boxplot",
    )

    with pytest.raises(RuntimeError, match=r"\.png.*\.pdf"):
        plotter.run()

    assert (tmp_path / "boxplot.html").exists()
    assert (tmp_path / "boxplot.json").exists()
    assert not (tmp_path / "boxplot.png").exists()
    assert not (tmp_path / "boxplot.pdf").exists()
