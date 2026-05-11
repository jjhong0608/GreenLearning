from __future__ import annotations

from pathlib import Path
from typing import Any, List
from concurrent.futures import ProcessPoolExecutor

import torch
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objs as go
from torch.nn.functional import pad

from greenonet.numerics import IntegrationRule, integrate, line_operator_fd
from greenonet.logging_mixin import LoggingMixin
from greenonet.coupling_data import coupling_collate_fn, CouplingDataset


def _render_heatmap_task(task: dict[str, Any]) -> None:
    z = task["z"]
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            zmax=task.get("zmax") if task.get("zmax") is not None else None,
            zmin=task.get("zmin") if task.get("zmin") is not None else None,
            colorscale=task.get("colorscale", "Viridis"),
            colorbar=dict(exponentformat="power", showexponent="all"),
        ),
        layout=go.Layout(
            template="plotly_white",
            width=900,
            height=900,
            title=task["title"],
            font=task["font"],
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False, scaleanchor="x", scaleratio=1),
        ),
    )
    base_path = Path(task["base_path"])
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        print("Static export skipped (requires kaleido + Chrome); HTML saved instead.")


class CouplingEvaluator(LoggingMixin):
    """Evaluate CouplingNet on test data with per-sample metrics and plots."""

    def __init__(
        self,
        model: torch.nn.Module,
        green_kernel: torch.Tensor,
        device: torch.device,
        work_dir: Path | str,
        integration_rule: IntegrationRule = "simpson",
    ) -> None:
        self.model = model.to(device)
        self.green_kernel = green_kernel.to(device)
        self.device = device
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.integration_rule = integration_rule
        super().__init__(logger_name="CouplingEvaluator", work_dir=self.work_dir)

    def _integrate(
        self, green: torch.Tensor, values: torch.Tensor, x_axis: torch.Tensor
    ) -> torch.Tensor:
        weighted = values.unsqueeze(-2) * green.unsqueeze(0)  # (B,n,m,m)
        return integrate(
            weighted, x=x_axis, dim=-1, rule=self.integration_rule
        )  # (B,n,m)

    def _relative_l2_integral(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        x_axis: torch.Tensor,
        eps: float = 1e-12,
    ) -> float:
        num = integrate(
            (pred - target).pow(2),
            x=x_axis,
            dim=-1,
            rule=self.integration_rule,
        )
        num = integrate(num, x=x_axis, dim=-1, rule=self.integration_rule).mean()
        den = integrate(target.pow(2), x=x_axis, dim=-1, rule=self.integration_rule)
        den = (
            integrate(den, x=x_axis, dim=-1, rule=self.integration_rule)
            .mean()
            .clamp_min(eps)
        )
        return float(torch.sqrt(num / den).item())

    @staticmethod
    def _grid_from_axial(x_data: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        """Merge axial data into a 2D grid with zero boundary. x_data/y_data: (n_lines, m)."""
        n_lines, m = x_data.shape
        grid = torch.zeros((m, m), dtype=x_data.dtype, device=x_data.device)
        # fill interior rows from x-lines
        start_row = 1
        grid[start_row : start_row + n_lines, :] = x_data
        # fill interior cols from y-lines (transpose to match)
        start_col = 1
        patch = y_data.transpose(-1, -2)
        grid[:, start_col : start_col + patch.shape[1]] = torch.where(
            grid[:, start_col : start_col + patch.shape[1]] == 0,
            patch,
            0.5 * (grid[:, start_col : start_col + patch.shape[1]] + patch),
        )
        return grid

    @staticmethod
    def _heatmap(
        fig_title: str,
        grid: torch.Tensor,
        font: dict[str, object],
        zmax: float | None = None,
        zmin: float | None = None,
    ) -> go.Figure:
        return go.Figure(
            data=go.Heatmap(
                z=grid.cpu().numpy(),
                zmax=zmax if zmax else None,
                zmin=zmin if zmin else None,
                colorscale="Viridis",
                colorbar=dict(
                    # Set the exponent format to "power"
                    exponentformat="power",
                    # Optionally, you can force the display of the exponent
                    # even for smaller numbers using 'all' or 'first'
                    showexponent="all",
                ),
            ),
            layout=go.Layout(
                template="plotly_white",
                width=900,
                height=900,
                title=fig_title,
                font=font,
                # xaxis_title="x",
                # yaxis_title="y",
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(
                    visible=False, showgrid=False, scaleanchor="x", scaleratio=1
                ),
            ),
        )

    @staticmethod
    def _save_fig(fig: go.Figure, base_path: Path) -> None:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(str(base_path.with_suffix(".png")))
            fig.write_image(str(base_path.with_suffix(".pdf")))
        except Exception:
            print(
                "Static export skipped (requires kaleido + Chrome); HTML saved instead."
            )
        # fig.write_html(str(base_path.with_suffix(".html")))

    def _null_space_solution_diagnostics(
        self,
        pred_flux_x_lines: torch.Tensor,
        pred_flux_y_lines: torch.Tensor,
        flux_x_lines: torch.Tensor,
        flux_y_lines: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q_x_lines = pred_flux_x_lines - flux_x_lines
        q_y_lines = flux_y_lines - pred_flux_y_lines

        u_q_x_raw = self._integrate(
            self.green_kernel[0],
            q_x_lines.unsqueeze(0).to(self.device),
            x_axis.to(self.device),
        )
        u_q_y_raw = self._integrate(
            self.green_kernel[1],
            q_y_lines.unsqueeze(0).to(self.device),
            y_axis.to(self.device),
        )

        u_q_x = pad(u_q_x_raw[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)[
            0
        ]
        u_q_y = pad(u_q_y_raw[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)[
            0
        ].transpose(-1, -2)
        u_q_residual = u_q_x + u_q_y

        return {
            "u_q_x": u_q_x.detach().cpu(),
            "u_q_y": u_q_y.detach().cpu(),
            "u_q_residual": u_q_residual.detach().cpu(),
        }

    def _closure_residual_diagnostics(
        self,
        flux_x_lines: torch.Tensor,
        flux_y_lines: torch.Tensor,
        a_x_lines: torch.Tensor,
        a_y_lines: torch.Tensor,
        b_x_lines: torch.Tensor,
        b_y_lines: torch.Tensor,
        c_x_lines: torch.Tensor,
        c_y_lines: torch.Tensor,
        x_axis: torch.Tensor,
        y_axis: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        flux_x_lines = flux_x_lines.to(self.device)
        flux_y_lines = flux_y_lines.to(self.device)
        a_x_lines = a_x_lines.to(self.device)
        a_y_lines = a_y_lines.to(self.device)
        b_x_lines = b_x_lines.to(self.device)
        b_y_lines = b_y_lines.to(self.device)
        c_x_lines = c_x_lines.to(self.device)
        c_y_lines = c_y_lines.to(self.device)
        x_axis = x_axis.to(self.device)
        y_axis = y_axis.to(self.device)

        u_phi_exact_raw = self._integrate(
            self.green_kernel[0], flux_x_lines.unsqueeze(0), x_axis
        )
        u_psi_exact_raw = self._integrate(
            self.green_kernel[1], flux_y_lines.unsqueeze(0), y_axis
        )

        u_phi_exact = pad(
            u_phi_exact_raw[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0
        )
        u_psi_exact = pad(
            u_psi_exact_raw[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0
        )

        phi_closure = line_operator_fd(
            u_lines=u_phi_exact[:, 1:-1, :],
            a_lines=a_x_lines.unsqueeze(0),
            b_lines=b_x_lines.unsqueeze(0),
            c_lines=c_x_lines.unsqueeze(0),
            axis_coords=x_axis,
        )
        psi_closure = line_operator_fd(
            u_lines=u_psi_exact[:, 1:-1, :],
            a_lines=a_y_lines.unsqueeze(0),
            b_lines=b_y_lines.unsqueeze(0),
            c_lines=c_y_lines.unsqueeze(0),
            axis_coords=y_axis,
        )

        phi_residual = phi_closure - flux_x_lines.unsqueeze(0)[..., 1:-1]
        psi_residual = psi_closure - flux_y_lines.unsqueeze(0)[..., 1:-1]
        return {
            "phi_residual": phi_residual[0].detach().cpu(),
            "psi_residual": psi_residual[0].transpose(-1, -2).detach().cpu(),
        }

    def _evaluate_batch(
        self,
        coords: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
        sol: torch.Tensor,
        flux: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, _, n_lines, m_points = rhs_raw.shape
        x_axis = coords[0, 0, :, 0].to(self.device)
        y_axis = coords[1, 0, :, 1].to(self.device)

        rhs_raw = rhs_raw.to(self.device)
        rhs_tilde = rhs_tilde.to(self.device)
        rhs_norm = rhs_norm.to(self.device)
        sol = sol.to(self.device)
        flux = flux.to(self.device)
        a_vals = a_vals.to(self.device)
        b_vals = b_vals.to(self.device)
        c_vals = c_vals.to(self.device)

        pred_flux = self.model(
            coords=coords.to(self.device),
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw.to(self.device),
            rhs_tilde=rhs_tilde.to(self.device),
            rhs_norm=rhs_norm.to(self.device),
        )  # (B,2,n,m)

        # pred_flux = flux.clone().to(self.device)

        gx = self.green_kernel[0]  # (n,m,m)
        gy = self.green_kernel[1]

        phi_x = pred_flux[:, 0]
        psi_y = pred_flux[:, 1]

        phi_x[..., 0] = 0.0
        phi_x[..., -1] = 0.0
        psi_y[..., 0] = 0.0
        psi_y[..., -1] = 0.0
        u_phi_x = self._integrate(gx, phi_x, x_axis)
        u_psi_y = self._integrate(gy, psi_y, y_axis)

        u_phi_x = pad(u_phi_x[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)
        u_psi_y = pad(u_psi_y[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)

        pred_flux_target = torch.stack((phi_x, psi_y), dim=1)
        pred_flux_target = pad(
            pred_flux_target[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0
        )
        flux = pad(flux[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)
        sol = pad(sol[..., 1:-1], pad=(1, 1, 1, 1), mode="constant", value=0.0)

        tensors = {
            "pred_flux": pred_flux_target.detach().cpu(),
            "flux": flux.cpu(),
            "pred_sol_x": u_phi_x.detach().cpu(),
            "pred_sol_y": u_psi_y.detach().cpu(),
            "sol": sol.cpu(),
        }
        return tensors

    def evaluate(
        self,
        dataset: Dataset[tuple[torch.Tensor, ...]],
        dataset_name: str = "test",
        batch_size: int = 1,
        plot_workers: int = 20,
    ) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=coupling_collate_fn,
            )
            rows: List[dict[str, str | float]] = []
            font = {"family": "Times New Roman", "size": 24}
            executor: ProcessPoolExecutor | None = None
            if plot_workers > 1:
                executor = ProcessPoolExecutor(max_workers=plot_workers)
            try:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(loader):
                        (
                            coords,
                            rhs_raw,
                            rhs_tilde,
                            rhs_norm,
                            sol,
                            flux,
                            a_vals,
                            b_vals,
                            c_vals,
                            ap,
                        ) = batch
                        tensors = self._evaluate_batch(
                            coords,
                            rhs_raw,
                            rhs_tilde,
                            rhs_norm,
                            sol,
                            flux,
                            a_vals,
                            b_vals,
                            c_vals,
                        )
                        batch_size_actual = rhs_raw.shape[0]
                        for i in range(batch_size_actual):
                            global_idx = batch_idx * batch_size + i
                            file_stem = ""
                            if isinstance(
                                dataset, CouplingDataset
                            ) and global_idx < len(dataset.files):
                                file_stem = dataset.files[global_idx].stem
                            total = (
                                len(dataset) if hasattr(dataset, "__len__") else None
                            )
                            if total is not None:
                                self.logger.info(
                                    "Evaluating sample %s/%s (%s)",
                                    global_idx + 1,
                                    total,
                                    file_stem or f"sample_{global_idx}",
                                )
                            else:
                                self.logger.info(
                                    "Evaluating sample %s (%s)",
                                    global_idx + 1,
                                    file_stem or f"sample_{global_idx}",
                                )

                            pred_flux = tensors["pred_flux"][i : i + 1]
                            flux_i = tensors["flux"][i : i + 1]
                            pred_sol = torch.stack(
                                (
                                    tensors["pred_sol_x"][i],
                                    tensors["pred_sol_y"][i],
                                ),
                                dim=0,
                            ).unsqueeze(0)
                            sol_i = tensors["sol"][i : i + 1]
                            x_axis = coords[0, 0, :, 0].to(self.device)
                            y_axis = coords[1, 0, :, 1].to(self.device)

                            metrics = {
                                "rel_flux": self._relative_l2_integral(
                                    pred_flux.to(self.device),
                                    flux_i.to(self.device),
                                    x_axis,
                                ),
                                "rel_sol": self._relative_l2_integral(
                                    pred_sol.to(self.device),
                                    sol_i.to(self.device),
                                    x_axis,
                                ),
                            }
                            rows.append({"file": file_stem, **metrics})

                            # Build grids for plotting
                            sol_x = sol[i, 0]
                            sol_y = sol[i, 1]
                            pred_sol_x = tensors["pred_sol_x"][i]
                            pred_sol_y = tensors["pred_sol_y"][i]
                            sol_grid = (
                                sol_x[..., 1:-1] + sol_y[..., 1:-1].transpose(-1, -2)
                            ) / 2
                            sol_grid = pad(
                                sol_grid,
                                pad=(1, 1, 1, 1),
                                mode="constant",
                                value=0.0,
                            )
                            sol_grid_x = pad(
                                sol_x[..., 1:-1],
                                pad=(1, 1, 1, 1),
                                mode="constant",
                                value=0.0,
                            )
                            sol_grid_y = pad(
                                sol_y[..., 1:-1],
                                pad=(1, 1, 1, 1),
                                mode="constant",
                                value=0.0,
                            ).transpose(-1, -2)
                            pred_sol_grid = (
                                pred_sol_x + pred_sol_y.transpose(-1, -2)
                            ) / 2
                            pred_sol_grid_x = pred_sol_x
                            pred_sol_grid_y = pred_sol_y.transpose(-1, -2)
                            err_sol_grid = (pred_sol_grid - sol_grid).abs()
                            err_sol_grid_x = (pred_sol_grid_x - sol_grid_x).abs()
                            err_sol_grid_y = (pred_sol_grid_y - sol_grid_y).abs()

                            flux_x = flux[i, 0]
                            flux_y = flux[i, 1]
                            pred_flux_x = tensors["pred_flux"][i, 0]
                            pred_flux_y = tensors["pred_flux"][i, 1]
                            pred_flux_x_lines = pred_flux_x[1:-1, :]
                            pred_flux_y_lines = pred_flux_y[1:-1, :]
                            flux_x_grid = flux_x[..., 1:-1]
                            pred_flux_x_grid = pred_flux_x[1:-1, 1:-1]
                            flux_y_grid = flux_y[..., 1:-1].T
                            pred_flux_y_grid = pred_flux_y[1:-1, 1:-1].T
                            err_flux_x_grid = (pred_flux_x_grid - flux_x_grid).abs()
                            err_flux_y_grid = (pred_flux_y_grid - flux_y_grid).abs()

                            err_flux_sum_grid = (
                                pred_flux_x_grid
                                - flux_x_grid
                                + pred_flux_y_grid
                                - flux_y_grid
                            )
                            err_flux_sub_grid = (
                                pred_flux_x_grid
                                - flux_x_grid
                                - pred_flux_y_grid
                                + flux_y_grid
                            )
                            null_diag = self._null_space_solution_diagnostics(
                                pred_flux_x_lines=pred_flux_x_lines,
                                pred_flux_y_lines=pred_flux_y_lines,
                                flux_x_lines=flux_x,
                                flux_y_lines=flux_y,
                                x_axis=x_axis,
                                y_axis=y_axis,
                            )
                            null_sol_x_grid = null_diag["u_q_x"]
                            null_sol_y_grid = null_diag["u_q_y"]
                            null_sol_residual_grid = null_diag["u_q_residual"]
                            null_abs_max: float | None = max(
                                null_sol_x_grid.abs().max().item(),
                                null_sol_y_grid.abs().max().item(),
                                null_sol_residual_grid.abs().max().item(),
                            )
                            if null_abs_max == 0.0:
                                null_abs_max = None

                            closure_diag = self._closure_residual_diagnostics(
                                flux_x_lines=flux_x,
                                flux_y_lines=flux_y,
                                a_x_lines=a_vals[i, 0],
                                a_y_lines=a_vals[i, 1],
                                b_x_lines=b_vals[i, 0],
                                b_y_lines=b_vals[i, 1],
                                c_x_lines=c_vals[i, 0],
                                c_y_lines=c_vals[i, 1],
                                x_axis=x_axis,
                                y_axis=y_axis,
                            )
                            closure_phi_grid = closure_diag["phi_residual"]
                            closure_psi_grid = closure_diag["psi_residual"]
                            closure_abs_max: float | None = max(
                                closure_phi_grid.abs().max().item(),
                                closure_psi_grid.abs().max().item(),
                            )
                            if closure_abs_max == 0.0:
                                closure_abs_max = None

                            u_max = sol_grid.max().item()
                            u_min = sol_grid.min().item()
                            phi_max = flux_x_grid.max().item()
                            phi_min = flux_x_grid.min().item()
                            psi_max = flux_y_grid.max().item()
                            psi_min = flux_y_grid.min().item()

                            out_dir = self.work_dir / dataset_name
                            stem = file_stem or f"sample_{global_idx}"
                            plot_tasks: List[dict[str, Any]] = [
                                {
                                    "title": "Exact Solution",
                                    "z": sol_grid.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_exact"),
                                    "font": font,
                                },
                                {
                                    "title": "Predicted Solution",
                                    "z": pred_sol_grid.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_pred"),
                                    "font": font,
                                },
                                {
                                    "title": "Solution Error",
                                    "z": err_sol_grid.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_sol_error"),
                                    "font": font,
                                },
                                {
                                    "title": "Exact Solution along x",
                                    "z": sol_grid_x.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_exact_x"),
                                    "font": font,
                                },
                                {
                                    "title": "Predicted Solution along x",
                                    "z": pred_sol_grid_x.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_pred_x"),
                                    "font": font,
                                },
                                {
                                    "title": "Solution Error along x",
                                    "z": err_sol_grid_x.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_sol_error_x"),
                                    "font": font,
                                },
                                {
                                    "title": "Exact Solution along y",
                                    "z": sol_grid_y.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_exact_y"),
                                    "font": font,
                                },
                                {
                                    "title": "Predicted Solution along y",
                                    "z": pred_sol_grid_y.cpu().numpy(),
                                    "zmax": u_max,
                                    "zmin": u_min,
                                    "base_path": str(out_dir / f"{stem}_sol_pred_y"),
                                    "font": font,
                                },
                                {
                                    "title": "Solution Error along y",
                                    "z": err_sol_grid_y.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_sol_error_y"),
                                    "font": font,
                                },
                                {
                                    "title": "Exact x-Flux-Div",
                                    "z": flux_x_grid.cpu().numpy(),
                                    "zmax": phi_max,
                                    "zmin": phi_min,
                                    "base_path": str(out_dir / f"{stem}_flux_x_exact"),
                                    "font": font,
                                },
                                {
                                    "title": "Predicted x-Flux-Div",
                                    "z": pred_flux_x_grid.cpu().numpy(),
                                    "zmax": phi_max,
                                    "zmin": phi_min,
                                    "base_path": str(out_dir / f"{stem}_flux_x_pred"),
                                    "font": font,
                                },
                                {
                                    "title": "x-Flux-Div Error",
                                    "z": err_flux_x_grid.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_flux_x_error"),
                                    "font": font,
                                },
                                {
                                    "title": "Exact y-Flux-Div",
                                    "z": flux_y_grid.cpu().numpy(),
                                    "zmax": psi_max,
                                    "zmin": psi_min,
                                    "base_path": str(out_dir / f"{stem}_flux_y_exact"),
                                    "font": font,
                                },
                                {
                                    "title": "Predicted y-Flux-Div",
                                    "z": pred_flux_y_grid.cpu().numpy(),
                                    "zmax": psi_max,
                                    "zmin": psi_min,
                                    "base_path": str(out_dir / f"{stem}_flux_y_pred"),
                                    "font": font,
                                },
                                {
                                    "title": "y-Flux-Div Error",
                                    "z": err_flux_y_grid.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_flux_y_error"),
                                    "font": font,
                                },
                                {
                                    "title": "Flux-Div Sum",
                                    "z": err_flux_sum_grid.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_flux_sum"),
                                    "font": font,
                                },
                                {
                                    "title": "Flux-Div Sub",
                                    "z": err_flux_sub_grid.cpu().numpy(),
                                    "zmax": None,
                                    "zmin": None,
                                    "base_path": str(out_dir / f"{stem}_flux_sub"),
                                    "font": font,
                                },
                                {
                                    "title": "Null-space Solution Contribution along x",
                                    "z": null_sol_x_grid.cpu().numpy(),
                                    "zmax": null_abs_max,
                                    "zmin": (
                                        -null_abs_max
                                        if null_abs_max is not None
                                        else None
                                    ),
                                    "base_path": str(out_dir / f"{stem}_null_sol_x"),
                                    "font": font,
                                    "colorscale": "RdBu",
                                },
                                {
                                    "title": "Null-space Solution Contribution along y",
                                    "z": null_sol_y_grid.cpu().numpy(),
                                    "zmax": null_abs_max,
                                    "zmin": (
                                        -null_abs_max
                                        if null_abs_max is not None
                                        else None
                                    ),
                                    "base_path": str(out_dir / f"{stem}_null_sol_y"),
                                    "font": font,
                                    "colorscale": "RdBu",
                                },
                                {
                                    "title": "Null-space Solution Residual",
                                    "z": null_sol_residual_grid.cpu().numpy(),
                                    "zmax": null_abs_max,
                                    "zmin": (
                                        -null_abs_max
                                        if null_abs_max is not None
                                        else None
                                    ),
                                    "base_path": str(
                                        out_dir / f"{stem}_null_sol_residual"
                                    ),
                                    "font": font,
                                    "colorscale": "RdBu",
                                },
                                {
                                    "title": "Closure Residual along x",
                                    "z": closure_phi_grid.cpu().numpy(),
                                    "zmax": closure_abs_max,
                                    "zmin": (
                                        -closure_abs_max
                                        if closure_abs_max is not None
                                        else None
                                    ),
                                    "base_path": str(
                                        out_dir / f"{stem}_closure_phi_residual"
                                    ),
                                    "font": font,
                                    "colorscale": "RdBu",
                                },
                                {
                                    "title": "Closure Residual along y",
                                    "z": closure_psi_grid.cpu().numpy(),
                                    "zmax": closure_abs_max,
                                    "zmin": (
                                        -closure_abs_max
                                        if closure_abs_max is not None
                                        else None
                                    ),
                                    "base_path": str(
                                        out_dir / f"{stem}_closure_psi_residual"
                                    ),
                                    "font": font,
                                    "colorscale": "RdBu",
                                },
                            ]
                            if executor is not None:
                                futures = [
                                    executor.submit(_render_heatmap_task, task)
                                    for task in plot_tasks
                                ]
                                for fut in futures:
                                    fut.result()
                            else:
                                for task in plot_tasks:
                                    _render_heatmap_task(task)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

            # Save metrics CSV
            import csv

            csv_path = self.work_dir / dataset_name / "metrics.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=["file", "rel_flux", "rel_sol"])
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        finally:
            self.model.train(was_training)
