from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

from greenonet.coefficients import CoefficientFunctions


SourceInterpolation = Literal["linear"]


def gauss_legendre_nodes_weights(
    order: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not isinstance(order, int) or isinstance(order, bool):
        raise TypeError("Gauss-Legendre order must be an integer.")
    if order <= 0:
        raise ValueError("Gauss-Legendre order must be positive.")
    nodes_np, weights_np = np.polynomial.legendre.leggauss(order)
    nodes = torch.as_tensor(nodes_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    return nodes, weights


def split_gauss_legendre_nodes(
    t_grid: Tensor,
    order: int,
) -> tuple[Tensor, Tensor]:
    """Return split Gauss-Legendre source nodes/weights for each target ``t``."""

    if t_grid.dim() != 1:
        raise ValueError("t_grid must be one-dimensional.")
    nodes, weights = gauss_legendre_nodes_weights(
        order,
        dtype=t_grid.dtype,
        device=t_grid.device,
    )
    t = t_grid.unsqueeze(-1)
    left_nodes = 0.5 * t * (nodes + 1.0)
    left_weights = 0.5 * t * weights
    right_nodes = 0.5 * ((1.0 - t) * nodes + (1.0 + t))
    right_weights = 0.5 * (1.0 - t) * weights
    return (
        torch.cat((left_nodes, right_nodes), dim=-1),
        torch.cat((left_weights, right_weights), dim=-1),
    )


def build_split_pair_coords(t_grid: Tensor, eta_nodes: Tensor) -> Tensor:
    if eta_nodes.dim() != 2:
        raise ValueError("eta_nodes must have shape (M, Q).")
    if t_grid.dim() != 1:
        raise ValueError("t_grid must be one-dimensional.")
    if eta_nodes.shape[0] != t_grid.numel():
        raise ValueError("eta_nodes first dimension must match t_grid length.")
    t_nodes = t_grid[:, None].expand_as(eta_nodes)
    return torch.stack((t_nodes, eta_nodes), dim=-1)


def interpolate_source_on_unit_grid(
    *,
    unit_grid: Tensor,
    source: Tensor,
    query_points: Tensor,
    method: SourceInterpolation = "linear",
) -> Tensor:
    """Interpolate source values from a unit grid to arbitrary query points."""

    if method != "linear":
        raise ValueError("Only linear source interpolation is supported.")
    if unit_grid.dim() != 1:
        raise ValueError("unit_grid must be one-dimensional.")
    if unit_grid.numel() < 2:
        raise ValueError("Need at least two source grid points for interpolation.")
    if source.shape[-1] != unit_grid.numel():
        raise ValueError("source last dimension must match unit_grid length.")

    grid = unit_grid.to(device=source.device, dtype=source.dtype).contiguous()
    query = query_points.to(device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
    idx_right = torch.searchsorted(grid, query).clamp(1, grid.numel() - 1)
    idx_left = idx_right - 1

    x0 = grid[idx_left]
    x1 = grid[idx_right]
    denom = (x1 - x0).clamp_min(torch.finfo(source.dtype).eps)
    weight = ((query - x0) / denom).clamp(0.0, 1.0)

    flat_source = source.reshape(-1, grid.numel())
    flat_left = idx_left.reshape(1, -1).expand(flat_source.shape[0], -1)
    flat_right = idx_right.reshape(1, -1).expand(flat_source.shape[0], -1)
    left_values = flat_source.gather(1, flat_left)
    right_values = flat_source.gather(1, flat_right)
    flat_interp = left_values + (right_values - left_values) * weight.reshape(1, -1)
    return flat_interp.reshape(*source.shape[:-1], *query.shape)


def split_gauss_legendre_weighted_sum(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape[-2:] != weights.shape:
        raise ValueError("values final dimensions must match weights shape.")
    return (values * weights).sum(dim=-1)


def reconstruct_split_gauss_legendre(
    *,
    kernel_nodes: Tensor,
    source: Tensor,
    source_grid: Tensor,
    target_grid: Tensor,
    order: int,
    source_interpolation: SourceInterpolation = "linear",
) -> Tensor:
    """Reconstruct ``u(t)`` from kernel values at split Gaussian source nodes."""

    eta_nodes, weights = split_gauss_legendre_nodes(target_grid, order)
    if kernel_nodes.shape[-2:] != eta_nodes.shape:
        raise ValueError("kernel_nodes must have final shape (M, 2 * order).")
    source_nodes = interpolate_source_on_unit_grid(
        unit_grid=source_grid,
        source=source,
        query_points=eta_nodes,
        method=source_interpolation,
    )
    return split_gauss_legendre_weighted_sum(
        source_nodes * kernel_nodes.unsqueeze(0),
        weights,
    )


def evaluate_unit_line_coefficients(
    coeffs: CoefficientFunctions,
    *,
    axis_id: Tensor,
    left: Tensor,
    fixed: Tensor,
    length: Tensor,
    t_nodes: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate unit-scaled ``a``, ``a'`` and ``b`` at line-local positions."""

    if t_nodes.dim() < 1:
        raise ValueError("t_nodes must have at least one dimension.")
    axis = axis_id.to(device=t_nodes.device)
    left = left.to(device=t_nodes.device, dtype=t_nodes.dtype)
    fixed = fixed.to(device=t_nodes.device, dtype=t_nodes.dtype)
    length = length.to(device=t_nodes.device, dtype=t_nodes.dtype)
    view_shape = (axis.numel(),) + (1,) * t_nodes.dim()

    s = left.reshape(view_shape) + length.reshape(view_shape) * t_nodes.unsqueeze(0)
    fixed_nodes = fixed.reshape(view_shape).expand_as(s)
    is_x = axis.reshape(view_shape) == 0
    x = torch.where(is_x, s, fixed_nodes)
    y = torch.where(is_x, fixed_nodes, s)

    a_eval = coeffs.a_fun(x, y).to(device=t_nodes.device, dtype=t_nodes.dtype)
    apx = coeffs.apx_fun(x, y).to(device=t_nodes.device, dtype=t_nodes.dtype)
    apy = coeffs.apy_fun(x, y).to(device=t_nodes.device, dtype=t_nodes.dtype)
    bx = coeffs.bx_fun(x, y).to(device=t_nodes.device, dtype=t_nodes.dtype)
    by = coeffs.by_fun(x, y).to(device=t_nodes.device, dtype=t_nodes.dtype)
    length_scale = length.reshape(view_shape)
    ap_eval = length_scale * torch.where(is_x, apx, apy)
    b_eval = length_scale * torch.where(is_x, bx, by)
    return a_eval, ap_eval, b_eval
