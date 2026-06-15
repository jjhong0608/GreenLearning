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


def _write_csvs(directory: Path, *, missing_rel_sol: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(CSV_NAMES):
        path = directory / name
        if missing_rel_sol and name == "Poisson_per_sample_metrics.csv":
            path.write_text("sample_id,file,rel_flux\n0,sample,0.2\n")
            continue
        path.write_text(
            "\n".join(
                [
                    "sample_id,file,rel_sol,rel_flux",
                    f"0,sample_a,{0.1 + idx:.3f},0.2",
                    f"1,sample_b,{0.2 + idx:.3f},0.3",
                ]
            )
            + "\n"
        )


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
    assert [trace.y for trace in fig.data] == expected_scaled


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
