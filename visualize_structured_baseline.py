from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from cli.train import TrainCLI
from greenonet.coupling_data import CouplingDataset
from greenonet.greens import ExactGreenFunction
from greenonet.numerics import IntegrationRule, integrate


@dataclass(frozen=True)
class BaselineVisualizationConfig:
    npz_path: Path
    outdir: Path
    step_size: float
    n_points_per_line: int | None = None
    integration_rule: IntegrationRule = "simpson"
    dtype: torch.dtype = torch.float64


@dataclass(frozen=True)
class StructuredBaselineBundle:
    npz_path: Path
    x_axis: torch.Tensor
    y_axis: torch.Tensor
    x_line_positions: torch.Tensor
    y_line_positions: torch.Tensor
    rhs_x_lines: torch.Tensor
    rhs_y_lines: torch.Tensor
    flux_x_lines: torch.Tensor
    flux_y_lines: torch.Tensor
    a_x_lines: torch.Tensor
    a_y_lines: torch.Tensor
    response_x_lines: torch.Tensor
    response_y_lines: torch.Tensor
    response_x_local: torch.Tensor
    response_y_local: torch.Tensor
    magnitude_x: torch.Tensor
    magnitude_y: torch.Tensor
    weight_grid: torch.Tensor
    phi_baseline_grid: torch.Tensor
    psi_baseline_grid: torch.Tensor
    phi_baseline_lines: torch.Tensor
    psi_baseline_lines: torch.Tensor
    integration_rule: IntegrationRule


class StructuredBaselineLoggingMixin:
    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(self.work_dir / "visualize_structured_baseline.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        self.logger.addHandler(handler)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()


class StructuredBaselineDatasetMixin:
    def _build_dataset(self) -> CouplingDataset:
        return CouplingDataset(
            data_dir=self.config.npz_path.parent,
            step_size=self.config.step_size,
            n_points_per_line=self.config.n_points_per_line,
            dtype=self.config.dtype,
            integration_rule=self.config.integration_rule,
            a_fun=TrainCLI.a_fun,
            b_fun=TrainCLI.b_fun,
            c_fun=TrainCLI.c_fun,
            ap_fun_x=TrainCLI.apx_fun,
            ap_fun_y=TrainCLI.apy_fun,
        )

    def _resolve_sample_index(self, dataset: CouplingDataset) -> int:
        target = self.config.npz_path.resolve()
        for index, path in enumerate(dataset.files):
            if path.resolve() == target:
                return index
        raise FileNotFoundError(f"{target} is not present in dataset directory {dataset.files}")

    def _build_line_positions(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_line_positions = coords[0, :, 0, 1].to(dtype=self.config.dtype)
        y_line_positions = coords[1, :, 0, 0].to(dtype=self.config.dtype)
        return x_line_positions, y_line_positions


class StructuredBaselineComputationMixin:
    BASELINE_EPS = 1e-12

    @staticmethod
    def _integrate_green_lines(
        green: torch.Tensor,
        values: torch.Tensor,
        axis_coords: torch.Tensor,
        integration_rule: IntegrationRule,
    ) -> torch.Tensor:
        weighted = values.unsqueeze(-2) * green
        return integrate(weighted, x=axis_coords, dim=-1, rule=integration_rule)

    @staticmethod
    def _pad_interior_lines(interior: torch.Tensor, n_points: int) -> torch.Tensor:
        padded = torch.zeros(
            (interior.shape[0], n_points), dtype=interior.dtype, device=interior.device
        )
        padded[:, 1:-1] = interior
        return padded

    def _compute_local_green_response_fields(
        self,
        response_x_lines: torch.Tensor,
        response_y_lines: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        response_x_local = response_x_lines[:, 1:-1]
        response_y_local = response_y_lines[:, 1:-1].transpose(-1, -2)
        return response_x_local, response_y_local

    def _compute_smoothed_response_magnitudes(
        self,
        response_x_local: torch.Tensor,
        response_y_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.BASELINE_EPS
        magnitude_x = torch.sqrt(response_x_local.pow(2) + delta**2)
        magnitude_y = torch.sqrt(response_y_local.pow(2) + delta**2)
        return magnitude_x, magnitude_y

    def _build_bundle(self) -> StructuredBaselineBundle:
        dataset = self._build_dataset()
        sample_index = self._resolve_sample_index(dataset)
        (
            coords,
            rhs_raw,
            _rhs_tilde,
            _rhs_norm,
            _sol,
            flux,
            a_vals,
            _b_vals,
            _c_vals,
            _ap,
        ) = dataset[sample_index]
        x_axis = coords[0, 0, :, 0]
        y_axis = coords[1, 0, :, 1]
        x_line_positions, y_line_positions = self._build_line_positions(coords)

        rhs_x_lines = rhs_raw[0]
        rhs_y_lines = rhs_raw[1]
        flux_x_lines = flux[0]
        flux_y_lines = flux[1]
        a_x_lines = a_vals[0]
        a_y_lines = a_vals[1]

        green_x = ExactGreenFunction(x_axis, a_x_lines)()
        green_y = ExactGreenFunction(y_axis, a_y_lines)()
        response_x_lines = self._integrate_green_lines(
            green_x, rhs_x_lines, x_axis, self.config.integration_rule
        )
        response_y_lines = self._integrate_green_lines(
            green_y, rhs_y_lines, y_axis, self.config.integration_rule
        )
        response_x_local, response_y_local = self._compute_local_green_response_fields(
            response_x_lines, response_y_lines
        )
        magnitude_x, magnitude_y = self._compute_smoothed_response_magnitudes(
            response_x_local, response_y_local
        )

        rhs_common = rhs_x_lines[:, 1:-1]
        weight_grid = magnitude_y / (magnitude_x + magnitude_y)
        phi_baseline_grid = weight_grid * rhs_common
        psi_baseline_grid = (1.0 - weight_grid) * rhs_common

        phi_baseline_lines = self._pad_interior_lines(phi_baseline_grid, x_axis.numel())
        psi_baseline_lines = self._pad_interior_lines(
            psi_baseline_grid.transpose(-1, -2), y_axis.numel()
        )

        return StructuredBaselineBundle(
            npz_path=self.config.npz_path,
            x_axis=x_axis,
            y_axis=y_axis,
            x_line_positions=x_line_positions,
            y_line_positions=y_line_positions,
            rhs_x_lines=rhs_x_lines,
            rhs_y_lines=rhs_y_lines,
            flux_x_lines=flux_x_lines,
            flux_y_lines=flux_y_lines,
            a_x_lines=a_x_lines,
            a_y_lines=a_y_lines,
            response_x_lines=response_x_lines,
            response_y_lines=response_y_lines,
            response_x_local=response_x_local,
            response_y_local=response_y_local,
            magnitude_x=magnitude_x,
            magnitude_y=magnitude_y,
            weight_grid=weight_grid,
            phi_baseline_grid=phi_baseline_grid,
            psi_baseline_grid=psi_baseline_grid,
            phi_baseline_lines=phi_baseline_lines,
            psi_baseline_lines=psi_baseline_lines,
            integration_rule=self.config.integration_rule,
        )


class StructuredBaselinePlotMixin:
    @staticmethod
    def _font() -> dict[str, object]:
        return {"family": "Times New Roman", "size": 18}

    def _save_figure(self, fig: go.Figure, base_path: Path) -> None:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(base_path.with_suffix(".html")), include_plotlyjs="cdn")
        try:
            fig.write_image(str(base_path.with_suffix(".png")))
            fig.write_image(str(base_path.with_suffix(".pdf")))
        except Exception:
            self.logger.info(
                "Static export skipped for %s (requires kaleido + Chrome); HTML saved.",
                base_path.name,
            )

    def _build_line_figure(
        self,
        axis_name: str,
        axis_coords: torch.Tensor,
        line_positions: torch.Tensor,
        rhs_lines: torch.Tensor,
        flux_lines: torch.Tensor,
        baseline_lines: torch.Tensor,
        correction_lines: torch.Tensor,
    ) -> go.Figure:
        n_lines = rhs_lines.shape[0]
        n_cols = 1 if n_lines == 1 else 2
        n_rows = ceil(n_lines / n_cols)
        subplot_titles = [
            f"{axis_name}-line {idx} @ {float(line_positions[idx]):.3f}"
            for idx in range(n_lines)
        ]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            shared_yaxes=False,
        )
        axis_coords_int = axis_coords[1:-1]
        rhs_lines_int = rhs_lines[:, 1:-1]
        flux_lines_int = flux_lines[:, 1:-1]
        baseline_lines_int = baseline_lines[:, 1:-1]
        coords_np = axis_coords_int.cpu().numpy()
        for idx in range(n_lines):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            fig.add_trace(
                go.Scatter(
                    x=coords_np,
                    y=rhs_lines_int[idx].cpu().numpy(),
                    mode="lines",
                    name="rhs",
                    line=dict(color="#1f77b4", width=2),
                    showlegend=idx == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=coords_np,
                    y=flux_lines_int[idx].cpu().numpy(),
                    mode="lines",
                    name="exact flux-div",
                    line=dict(color="#d62728", width=2),
                    showlegend=idx == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=coords_np,
                    y=baseline_lines_int[idx].cpu().numpy(),
                    mode="lines",
                    name="structured baseline",
                    line=dict(color="#2ca02c", width=2, dash="dash"),
                    showlegend=idx == 0,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=coords_np,
                    y=correction_lines[idx].cpu().numpy(),
                    mode="lines",
                    name="correction",
                    line=dict(color="#9467bd", width=2, dash="dot"),
                    showlegend=idx == 0,
                ),
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text=f"{axis_name}", row=row, col=col)
            fig.update_yaxes(title_text="value", row=row, col=col)

        fig.update_layout(
            template="plotly_white",
            width=1200 if n_cols > 1 else 900,
            height=max(420 * n_rows, 500),
            title=(
                f"{axis_name.upper()}-axial lines: interior rhs, exact flux-divergence, "
                "structured baseline, and correction"
            ),
            font=self._font(),
            legend=dict(orientation="h"),
        )
        return fig

    def _build_heatmap_figure(
        self,
        title: str,
        grid: torch.Tensor,
        x_positions: torch.Tensor,
        y_positions: torch.Tensor,
    ) -> go.Figure:
        return go.Figure(
            data=go.Heatmap(
                z=grid.cpu().numpy(),
                x=x_positions.cpu().numpy(),
                y=y_positions.cpu().numpy(),
                colorscale="Viridis",
                colorbar=dict(exponentformat="power", showexponent="all"),
            ),
            layout=go.Layout(
                template="plotly_white",
                width=900,
                height=800,
                title=title,
                font=self._font(),
                xaxis_title="x-line position",
                yaxis_title="y-line position",
            ),
        )


class StructuredBaselineVisualizer(
    StructuredBaselineLoggingMixin,
    StructuredBaselineDatasetMixin,
    StructuredBaselineComputationMixin,
    StructuredBaselinePlotMixin,
):
    def __init__(self, config: BaselineVisualizationConfig) -> None:
        self.config = config
        super().__init__(work_dir=config.outdir)

    def run(self) -> None:
        bundle = self._build_bundle()
        self.logger.info(
            "Loaded %s with %s x-lines and %s y-lines",
            bundle.npz_path.name,
            bundle.rhs_x_lines.shape[0],
            bundle.rhs_y_lines.shape[0],
        )
        self.logger.info(
            "Using local pointwise smoothed inverse structured baseline weighting with exact Green responses, fixed delta=%s, and TrainCLI.a_fun coefficients.",
            self.BASELINE_EPS,
        )
        correction_x_lines = bundle.flux_x_lines[:, 1:-1] - bundle.phi_baseline_lines[:, 1:-1]
        correction_y_lines = bundle.flux_y_lines[:, 1:-1] - bundle.psi_baseline_lines[:, 1:-1]

        x_fig = self._build_line_figure(
            axis_name="x",
            axis_coords=bundle.x_axis,
            line_positions=bundle.x_line_positions,
            rhs_lines=bundle.rhs_x_lines,
            flux_lines=bundle.flux_x_lines,
            baseline_lines=bundle.phi_baseline_lines,
            correction_lines=correction_x_lines,
        )
        self._save_figure(x_fig, self.config.outdir / "x_axial_lines")

        y_fig = self._build_line_figure(
            axis_name="y",
            axis_coords=bundle.y_axis,
            line_positions=bundle.y_line_positions,
            rhs_lines=bundle.rhs_y_lines,
            flux_lines=bundle.flux_y_lines,
            baseline_lines=bundle.psi_baseline_lines,
            correction_lines=correction_y_lines,
        )
        self._save_figure(y_fig, self.config.outdir / "y_axial_lines")

        weight_fig = self._build_heatmap_figure(
            title="Structured baseline weight w",
            grid=bundle.weight_grid,
            x_positions=bundle.y_line_positions,
            y_positions=bundle.x_line_positions,
        )
        self._save_figure(weight_fig, self.config.outdir / "structured_baseline_weight")

        phi_fig = self._build_heatmap_figure(
            title="Structured baseline phi_str on the interior grid",
            grid=bundle.phi_baseline_grid,
            x_positions=bundle.y_line_positions,
            y_positions=bundle.x_line_positions,
        )
        self._save_figure(phi_fig, self.config.outdir / "structured_baseline_phi")

        psi_fig = self._build_heatmap_figure(
            title="Structured baseline psi_str on the interior grid",
            grid=bundle.psi_baseline_grid,
            x_positions=bundle.y_line_positions,
            y_positions=bundle.x_line_positions,
        )
        self._save_figure(psi_fig, self.config.outdir / "structured_baseline_psi")
        self.logger.info("Saved structured baseline visualizations to %s", self.config.outdir)


class StructuredBaselineVisualizationCLI:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Visualize CouplingNet's local pointwise smoothed inverse structured "
                "baseline on axial lines using ExactGreenFunction and the current "
                "coefficient in cli/train.py."
            )
        )
        parser.add_argument("--npz-path", type=Path, required=True, help="Input .npz file.")
        parser.add_argument(
            "--outdir",
            type=Path,
            default=Path("structured_baseline_plots"),
            help="Directory to save Plotly outputs.",
        )
        parser.add_argument(
            "--step-size",
            type=float,
            default=0.0625,
            help="Axial line spacing used to sample the .npz file.",
        )
        parser.add_argument(
            "--n-points-per-line",
            type=int,
            default=None,
            help="Number of sample points on each axial line.",
        )
        parser.add_argument(
            "--integration-rule",
            choices=("simpson", "trapezoid"),
            default="simpson",
            help="Quadrature rule used for exact Green responses.",
        )
        self.parser = parser

    def run(self) -> None:
        args = self.parser.parse_args()
        config = BaselineVisualizationConfig(
            npz_path=args.npz_path,
            outdir=args.outdir,
            step_size=args.step_size,
            n_points_per_line=args.n_points_per_line,
            integration_rule=args.integration_rule,
        )
        visualizer = StructuredBaselineVisualizer(config)
        visualizer.run()


if __name__ == "__main__":
    StructuredBaselineVisualizationCLI().run()
