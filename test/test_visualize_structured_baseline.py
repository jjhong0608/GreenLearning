from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from greenonet.greens import ExactGreenFunction
from greenonet.numerics import integrate


def _write_sample_npz(path: Path) -> Path:
    grid = np.linspace(0.0, 1.0, 257)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    rhs = xx + 0.25 * yy
    sol = np.zeros_like(rhs)
    uxx = np.ones((255, 255))
    uyy = 2.0 * np.ones((255, 255))
    np.savez(path, sol=sol, rhs=rhs, uxx=uxx, uyy=uyy)
    return path


def test_visualizer_uses_train_cli_coefficient_function(
    tmp_path: Path, monkeypatch
) -> None:
    from visualize_structured_baseline import (
        BaselineVisualizationConfig,
        StructuredBaselineVisualizer,
    )

    monkeypatch.setattr(
        "visualize_structured_baseline.TrainCLI.a_fun",
        staticmethod(lambda x, y: torch.full_like(x, 2.0)),
    )
    npz_path = _write_sample_npz(tmp_path / "sample.npz")
    visualizer = StructuredBaselineVisualizer(
        BaselineVisualizationConfig(
            npz_path=npz_path,
            outdir=tmp_path / "out",
            step_size=0.5,
            n_points_per_line=3,
            integration_rule="trapezoid",
        )
    )

    bundle = visualizer._build_bundle()

    assert torch.allclose(bundle.a_x_lines, torch.full_like(bundle.a_x_lines, 2.0))
    assert torch.allclose(bundle.a_y_lines, torch.full_like(bundle.a_y_lines, 2.0))


def test_visualizer_builds_inverse_structured_baseline_with_exact_green(
    tmp_path: Path, monkeypatch
) -> None:
    from visualize_structured_baseline import (
        BaselineVisualizationConfig,
        StructuredBaselineVisualizer,
    )

    monkeypatch.setattr(
        "visualize_structured_baseline.TrainCLI.a_fun",
        staticmethod(lambda x, y: torch.ones_like(x)),
    )
    npz_path = _write_sample_npz(tmp_path / "sample.npz")
    visualizer = StructuredBaselineVisualizer(
        BaselineVisualizationConfig(
            npz_path=npz_path,
            outdir=tmp_path / "out",
            step_size=0.5,
            n_points_per_line=3,
            integration_rule="trapezoid",
        )
    )

    bundle = visualizer._build_bundle()

    green_x = ExactGreenFunction(bundle.x_axis, bundle.a_x_lines)()
    green_y = ExactGreenFunction(bundle.y_axis, bundle.a_y_lines)()
    response_x = integrate(
        bundle.rhs_x_lines.unsqueeze(-2) * green_x,
        x=bundle.x_axis,
        dim=-1,
        rule=bundle.integration_rule,
    )
    response_y = integrate(
        bundle.rhs_y_lines.unsqueeze(-2) * green_y,
        x=bundle.y_axis,
        dim=-1,
        rule=bundle.integration_rule,
    )
    expected_response_x_local = response_x[:, 1:-1]
    expected_response_y_local = response_y[:, 1:-1].transpose(-1, -2)
    expected_mag_x = torch.sqrt(
        expected_response_x_local.pow(2) + visualizer.BASELINE_EPS**2
    )
    expected_mag_y = torch.sqrt(
        expected_response_y_local.pow(2) + visualizer.BASELINE_EPS**2
    )
    expected_weight = expected_mag_y / (expected_mag_x + expected_mag_y)
    rhs_common = bundle.rhs_x_lines[:, 1:-1]
    expected_phi = expected_weight * rhs_common
    expected_psi_y = ((1.0 - expected_weight) * rhs_common).transpose(-1, -2)

    assert torch.allclose(bundle.response_x_local, expected_response_x_local)
    assert torch.allclose(bundle.response_y_local, expected_response_y_local)
    assert torch.allclose(bundle.magnitude_x, expected_mag_x)
    assert torch.allclose(bundle.magnitude_y, expected_mag_y)
    assert torch.allclose(bundle.weight_grid, expected_weight)
    assert torch.allclose(bundle.phi_baseline_lines[:, 1:-1], expected_phi)
    assert torch.allclose(bundle.psi_baseline_lines[:, 1:-1], expected_psi_y)


def test_visualizer_weight_omits_old_denominator_epsilon(
    tmp_path: Path, monkeypatch
) -> None:
    from visualize_structured_baseline import (
        BaselineVisualizationConfig,
        StructuredBaselineVisualizer,
    )

    monkeypatch.setattr(
        "visualize_structured_baseline.TrainCLI.a_fun",
        staticmethod(lambda x, y: torch.ones_like(x)),
    )
    npz_path = _write_sample_npz(tmp_path / "sample.npz")
    visualizer = StructuredBaselineVisualizer(
        BaselineVisualizationConfig(
            npz_path=npz_path,
            outdir=tmp_path / "out",
            step_size=0.5,
            n_points_per_line=3,
            integration_rule="trapezoid",
        )
    )

    response_x_local = torch.zeros((1, 1), dtype=torch.float64)
    response_y_local = torch.zeros((1, 1), dtype=torch.float64)

    expected_mag_x = torch.sqrt(response_x_local.pow(2) + visualizer.BASELINE_EPS**2)
    expected_mag_y = torch.sqrt(response_y_local.pow(2) + visualizer.BASELINE_EPS**2)
    expected_weight = expected_mag_y / (expected_mag_x + expected_mag_y)
    weight_with_old_denominator_eps = expected_mag_y / (
        expected_mag_x + expected_mag_y + visualizer.BASELINE_EPS
    )

    magnitude_x, magnitude_y = visualizer._compute_smoothed_response_magnitudes(
        response_x_local, response_y_local
    )
    actual_weight = magnitude_y / (magnitude_x + magnitude_y)

    assert torch.allclose(actual_weight, expected_weight)
    assert actual_weight[0, 0] == torch.tensor(0.5, dtype=torch.float64)
    assert weight_with_old_denominator_eps[0, 0] == torch.tensor(
        1.0 / 3.0, dtype=torch.float64
    )
    assert not torch.allclose(actual_weight, weight_with_old_denominator_eps)


def test_visualizer_run_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    from visualize_structured_baseline import (
        BaselineVisualizationConfig,
        StructuredBaselineVisualizer,
    )

    monkeypatch.setattr(
        "visualize_structured_baseline.TrainCLI.a_fun",
        staticmethod(lambda x, y: torch.ones_like(x)),
    )

    def fake_write_html(self: object, path: str, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        Path(path).write_text("stub", encoding="utf-8")

    def fake_write_image(self: object, path: str, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        Path(path).write_text("stub", encoding="utf-8")

    monkeypatch.setattr("plotly.graph_objects.Figure.write_html", fake_write_html)
    monkeypatch.setattr("plotly.graph_objects.Figure.write_image", fake_write_image)

    npz_path = _write_sample_npz(tmp_path / "sample.npz")
    outdir = tmp_path / "out"
    visualizer = StructuredBaselineVisualizer(
        BaselineVisualizationConfig(
            npz_path=npz_path,
            outdir=outdir,
            step_size=0.5,
            n_points_per_line=3,
            integration_rule="trapezoid",
        )
    )

    visualizer.run()

    assert (outdir / "x_axial_lines.html").exists()
    assert (outdir / "y_axial_lines.html").exists()
    assert (outdir / "structured_baseline_weight.html").exists()
    assert (outdir / "structured_baseline_phi.html").exists()
    assert (outdir / "structured_baseline_psi.html").exists()
    assert (outdir / "visualize_structured_baseline.log").exists()


def test_visualizer_line_figure_uses_interior_points_and_correction_trace(
    tmp_path: Path, monkeypatch
) -> None:
    from visualize_structured_baseline import (
        BaselineVisualizationConfig,
        StructuredBaselineVisualizer,
    )

    monkeypatch.setattr(
        "visualize_structured_baseline.TrainCLI.a_fun",
        staticmethod(lambda x, y: torch.ones_like(x)),
    )
    npz_path = _write_sample_npz(tmp_path / "sample.npz")
    visualizer = StructuredBaselineVisualizer(
        BaselineVisualizationConfig(
            npz_path=npz_path,
            outdir=tmp_path / "out",
            step_size=0.5,
            n_points_per_line=3,
            integration_rule="trapezoid",
        )
    )

    bundle = visualizer._build_bundle()
    correction_x_lines = bundle.flux_x_lines[:, 1:-1] - bundle.phi_baseline_lines[:, 1:-1]
    fig = visualizer._build_line_figure(
        axis_name="x",
        axis_coords=bundle.x_axis,
        line_positions=bundle.x_line_positions,
        rhs_lines=bundle.rhs_x_lines,
        flux_lines=bundle.flux_x_lines,
        baseline_lines=bundle.phi_baseline_lines,
        correction_lines=correction_x_lines,
    )

    assert len(fig.data) == 4
    assert fig.data[0].name == "rhs"
    assert fig.data[1].name == "exact flux-div"
    assert fig.data[2].name == "structured baseline"
    assert fig.data[3].name == "correction"
    assert np.allclose(fig.data[0].x, bundle.x_axis[1:-1].cpu().numpy())
    assert np.allclose(fig.data[0].y, bundle.rhs_x_lines[0, 1:-1].cpu().numpy())
    assert np.allclose(fig.data[2].y, bundle.phi_baseline_lines[0, 1:-1].cpu().numpy())
    assert np.allclose(fig.data[3].y, correction_x_lines[0].cpu().numpy())
