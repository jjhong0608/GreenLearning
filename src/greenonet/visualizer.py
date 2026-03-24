from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable, Optional

import plotly.graph_objects as go
import torch


class LossVisualizer:
    """Plotly-based training loss visualization."""

    @staticmethod
    def save_loss_curve(
        losses: Iterable[float],
        output_path: Path,
        logger: Optional[logging.Logger] = None,
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                y=list(losses),
                mode="lines+markers",
                name="training_loss",
                line=dict(color="#1f77b4"),
            )
        )
        fig.update_layout(
            title="Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_white",
        )
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        if logger is not None:
            logger.info(f"Saved loss curve to {output_path}")
        return fig


class GreenVisualizer:
    """Plot the learned Green's function on a randomly selected axial line."""

    @staticmethod
    def save_green_heatmap(
        model: torch.nn.Module,
        trunk_grid: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        output_path: Path,
        logger: Optional[logging.Logger] = None,
    ) -> go.Figure:
        """
        Choose a random axial line from the batch and plot its Green predictions as a heatmap.

        Args:
            model: trained GreenONetModel
            trunk_grid: (m, m, 2) shared grid used for the trunk
            a_vals, ap_vals, b_vals, c_vals: (B, 2, n_lines, m) or (2, n_lines, m)
        """
        model.eval()
        with torch.no_grad():
            predictions = model(
                trunk_grid=trunk_grid,
                a_vals=a_vals,
                ap_vals=ap_vals,
                b_vals=b_vals,
                c_vals=c_vals,
            )  # (2, n, m, m)

        axes, n_lines, m, _ = predictions.shape
        axis_idx = random.randrange(axes)
        line_idx = random.randrange(n_lines)

        green_slice = predictions[axis_idx, line_idx]  # (m, m)
        fig = go.Figure(
            data=go.Heatmap(z=green_slice.cpu().numpy(), colorscale="Viridis")
        )
        fig.update_layout(
            title=(f"Green function heatmap (axis={axis_idx}, line={line_idx})"),
            xaxis_title="ξ index",
            yaxis_title="x index",
            template="plotly_white",
        )
        fig.write_html(str(output_path), include_plotlyjs="cdn")
        if logger is not None:
            logger.info(
                "Saved Green heatmap to %s (axis=%s, line=%s)",
                output_path,
                axis_idx,
                line_idx,
            )
        return fig
