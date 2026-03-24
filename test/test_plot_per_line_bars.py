from __future__ import annotations

from pathlib import Path

from pytest import MonkeyPatch

from plot_per_line_bars import PerLineBarPlotter


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    header = (
        "axis_id,axis_name,line_index,line_coordinate,"
        "rel_sol_line,rel_green_line,"
        "rel_sol_line_mean,rel_sol_line_min,rel_sol_line_max,rel_sol_line_std,"
        "rel_green_line_mean,rel_green_line_min,rel_green_line_max,rel_green_line_std,"
        "val_rel_sol_line,val_rel_sol_line_mean,val_rel_sol_line_min,val_rel_sol_line_max,val_rel_sol_line_std\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            ",".join(
                [
                    row["axis_id"],
                    row["axis_name"],
                    row["line_index"],
                    row["line_coordinate"],
                    row["rel_sol_line"],
                    row["rel_green_line"],
                    row["rel_sol_line_mean"],
                    row["rel_sol_line_min"],
                    row["rel_sol_line_max"],
                    row["rel_sol_line_std"],
                    row["rel_green_line_mean"],
                    row["rel_green_line_min"],
                    row["rel_green_line_max"],
                    row["rel_green_line_std"],
                    row["val_rel_sol_line"],
                    row["val_rel_sol_line_mean"],
                    row["val_rel_sol_line_min"],
                    row["val_rel_sol_line_max"],
                    row["val_rel_sol_line_std"],
                ]
            )
            + "\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _sample_rows(scale: float) -> list[dict[str, str]]:
    return [
        {
            "axis_id": "0",
            "axis_name": "x",
            "line_index": "0",
            "line_coordinate": "0.25",
            "rel_sol_line": f"{0.1 * scale}",
            "rel_green_line": f"{0.2 * scale}",
            "rel_sol_line_mean": f"{0.1 * scale}",
            "rel_sol_line_min": f"{0.05 * scale}",
            "rel_sol_line_max": f"{0.15 * scale}",
            "rel_sol_line_std": f"{0.02 * scale}",
            "rel_green_line_mean": f"{0.2 * scale}",
            "rel_green_line_min": f"{0.2 * scale}",
            "rel_green_line_max": f"{0.2 * scale}",
            "rel_green_line_std": "0.0",
            "val_rel_sol_line": f"{0.15 * scale}",
            "val_rel_sol_line_mean": f"{0.15 * scale}",
            "val_rel_sol_line_min": f"{0.10 * scale}",
            "val_rel_sol_line_max": f"{0.20 * scale}",
            "val_rel_sol_line_std": f"{0.01 * scale}",
        },
        {
            "axis_id": "0",
            "axis_name": "x",
            "line_index": "1",
            "line_coordinate": "0.5",
            "rel_sol_line": f"{0.2 * scale}",
            "rel_green_line": f"{0.3 * scale}",
            "rel_sol_line_mean": f"{0.2 * scale}",
            "rel_sol_line_min": f"{0.1 * scale}",
            "rel_sol_line_max": f"{0.3 * scale}",
            "rel_sol_line_std": f"{0.04 * scale}",
            "rel_green_line_mean": f"{0.3 * scale}",
            "rel_green_line_min": f"{0.3 * scale}",
            "rel_green_line_max": f"{0.3 * scale}",
            "rel_green_line_std": "0.0",
            "val_rel_sol_line": f"{0.25 * scale}",
            "val_rel_sol_line_mean": f"{0.25 * scale}",
            "val_rel_sol_line_min": f"{0.20 * scale}",
            "val_rel_sol_line_max": f"{0.30 * scale}",
            "val_rel_sol_line_std": f"{0.02 * scale}",
        },
        {
            "axis_id": "1",
            "axis_name": "y",
            "line_index": "0",
            "line_coordinate": "0.25",
            "rel_sol_line": f"{0.11 * scale}",
            "rel_green_line": f"{0.21 * scale}",
            "rel_sol_line_mean": f"{0.11 * scale}",
            "rel_sol_line_min": f"{0.08 * scale}",
            "rel_sol_line_max": f"{0.14 * scale}",
            "rel_sol_line_std": f"{0.01 * scale}",
            "rel_green_line_mean": f"{0.21 * scale}",
            "rel_green_line_min": f"{0.21 * scale}",
            "rel_green_line_max": f"{0.21 * scale}",
            "rel_green_line_std": "0.0",
            "val_rel_sol_line": f"{0.16 * scale}",
            "val_rel_sol_line_mean": f"{0.16 * scale}",
            "val_rel_sol_line_min": f"{0.12 * scale}",
            "val_rel_sol_line_max": f"{0.20 * scale}",
            "val_rel_sol_line_std": f"{0.01 * scale}",
        },
        {
            "axis_id": "1",
            "axis_name": "y",
            "line_index": "1",
            "line_coordinate": "0.5",
            "rel_sol_line": f"{0.22 * scale}",
            "rel_green_line": f"{0.31 * scale}",
            "rel_sol_line_mean": f"{0.22 * scale}",
            "rel_sol_line_min": f"{0.12 * scale}",
            "rel_sol_line_max": f"{0.32 * scale}",
            "rel_sol_line_std": f"{0.02 * scale}",
            "rel_green_line_mean": f"{0.31 * scale}",
            "rel_green_line_min": f"{0.31 * scale}",
            "rel_green_line_max": f"{0.31 * scale}",
            "rel_green_line_std": "0.0",
            "val_rel_sol_line": f"{0.26 * scale}",
            "val_rel_sol_line_mean": f"{0.26 * scale}",
            "val_rel_sol_line_min": f"{0.21 * scale}",
            "val_rel_sol_line_max": f"{0.31 * scale}",
            "val_rel_sol_line_std": f"{0.02 * scale}",
        },
    ]


def test_build_figure_traces_and_errorbars(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_csv(csv_a, _sample_rows(scale=1.0))
    _write_csv(csv_b, _sample_rows(scale=2.0))

    plotter = PerLineBarPlotter(
        csv_a=csv_a,
        csv_b=csv_b,
        outdir=tmp_path / "out",
        label_a="run-a",
        label_b="run-b",
    )
    rows_a = plotter._read_rows(csv_a)
    rows_b = plotter._read_rows(csv_b)
    aligned_a, aligned_b = plotter._aligned_axis_rows(rows_a, rows_b, axis="x")
    assert len(aligned_a) == 2
    assert len(aligned_b) == 2

    fig_sol_x = plotter._build_sol_figure_for_axis(
        axis="x", rows_a=aligned_a, rows_b=aligned_b
    )
    assert len(fig_sol_x.data) == 2
    assert fig_sol_x.data[0].name == "run-a"
    assert fig_sol_x.data[1].name == "run-b"
    assert fig_sol_x.data[0].error_y.visible is True
    assert fig_sol_x.data[1].error_y.visible is True

    fig_green_x = plotter._build_green_figure_for_axis(
        axis="x", rows_a=aligned_a, rows_b=aligned_b
    )
    assert len(fig_green_x.data) == 2
    assert fig_green_x.data[0].name == "run-a"
    assert fig_green_x.data[1].name == "run-b"
    assert fig_green_x.data[0].error_y.visible is None
    assert fig_green_x.data[1].error_y.visible is None

    fig_val_sol_x = plotter._build_val_sol_figure_for_axis(
        axis="x", rows_a=aligned_a, rows_b=aligned_b
    )
    assert len(fig_val_sol_x.data) == 2
    assert fig_val_sol_x.data[0].name == "run-a"
    assert fig_val_sol_x.data[1].name == "run-b"
    assert fig_val_sol_x.data[0].error_y.visible is True
    assert fig_val_sol_x.data[1].error_y.visible is True


def test_run_saves_validation_solution_figures(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    _write_csv(csv_a, _sample_rows(scale=1.0))
    _write_csv(csv_b, _sample_rows(scale=2.0))

    saved_paths: list[str] = []

    def fake_write_image(self: object, path: str) -> None:
        del self
        saved_paths.append(path)
        Path(path).write_text("stub", encoding="utf-8")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)

    plotter = PerLineBarPlotter(
        csv_a=csv_a,
        csv_b=csv_b,
        outdir=tmp_path / "out",
        label_a="run-a",
        label_b="run-b",
    )
    plotter.run()

    outdir = tmp_path / "out"
    assert (outdir / "per_line_compare_x_val_rel_sol.png").exists()
    assert (outdir / "per_line_compare_y_val_rel_sol.png").exists()
