from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from greenonet.greens import EllipticGreenFunction
from greenonet.numerics import IntegrationRule, integrate


def gauss_legendre_nodes_weights(
    order: int,
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
    x_axis: Tensor,
    order: int,
) -> tuple[Tensor, Tensor]:
    if x_axis.dim() != 1:
        raise ValueError("x_axis must be one-dimensional.")
    nodes, weights = gauss_legendre_nodes_weights(
        order=order,
        dtype=x_axis.dtype,
        device=x_axis.device,
    )
    x = x_axis.unsqueeze(-1)

    left_nodes = 0.5 * x * (nodes + 1.0)
    left_weights = 0.5 * x * weights
    right_nodes = 0.5 * ((1.0 - x) * nodes + (1.0 + x))
    right_weights = 0.5 * (1.0 - x) * weights

    return (
        torch.cat((left_nodes, right_nodes), dim=-1),
        torch.cat((left_weights, right_weights), dim=-1),
    )


def linear_interpolate_line_values(
    x_grid: Tensor,
    values: Tensor,
    query_points: Tensor,
) -> Tensor:
    if x_grid.dim() != 1:
        raise ValueError("x_grid must be one-dimensional.")
    if values.shape[-1] != x_grid.numel():
        raise ValueError("values last dimension must match x_grid length.")
    if x_grid.numel() < 2:
        raise ValueError("Need at least two grid points for interpolation.")

    x_grid = x_grid.to(device=values.device, dtype=values.dtype).contiguous()
    query = query_points.to(device=values.device, dtype=values.dtype).contiguous()
    idx_right = torch.searchsorted(x_grid, query).clamp(1, x_grid.numel() - 1)
    idx_left = idx_right - 1

    x0 = x_grid[idx_left]
    x1 = x_grid[idx_right]
    denom = (x1 - x0).clamp_min(torch.finfo(values.dtype).eps)
    weight = ((query - x0) / denom).clamp(0.0, 1.0)

    flat_values = values.reshape(-1, x_grid.numel())
    flat_left = idx_left.reshape(1, -1).expand(flat_values.shape[0], -1)
    flat_right = idx_right.reshape(1, -1).expand(flat_values.shape[0], -1)
    left_values = flat_values.gather(1, flat_left)
    right_values = flat_values.gather(1, flat_right)
    flat_interp = left_values + (right_values - left_values) * weight.reshape(1, -1)
    return flat_interp.reshape(*values.shape[:-1], *query.shape)


def natural_cubic_interpolate_line_values(
    x_grid: Tensor,
    values: Tensor,
    query_points: Tensor,
) -> Tensor:
    if x_grid.dim() != 1:
        raise ValueError("x_grid must be one-dimensional.")
    if values.shape[-1] != x_grid.numel():
        raise ValueError("values last dimension must match x_grid length.")
    if x_grid.numel() < 2:
        raise ValueError("Need at least two grid points for interpolation.")
    if x_grid.numel() == 2:
        return linear_interpolate_line_values(
            x_grid=x_grid,
            values=values,
            query_points=query_points,
        )

    x_grid = x_grid.to(device=values.device, dtype=values.dtype).contiguous()
    query = query_points.to(device=values.device, dtype=values.dtype).contiguous()
    n_points = x_grid.numel()
    h = x_grid[1:] - x_grid[:-1]
    if torch.any(h <= 0):
        raise ValueError("x_grid must be strictly increasing.")

    flat_values = values.reshape(-1, n_points)
    n_inner = n_points - 2
    matrix = torch.zeros(
        (n_inner, n_inner),
        device=values.device,
        dtype=values.dtype,
    )
    diag = 2.0 * (h[:-1] + h[1:])
    matrix.diagonal().copy_(diag)
    if n_inner > 1:
        matrix.diagonal(offset=1).copy_(h[1:-1])
        matrix.diagonal(offset=-1).copy_(h[1:-1])

    slopes = (flat_values[:, 1:] - flat_values[:, :-1]) / h.reshape(1, -1)
    rhs = 6.0 * (slopes[:, 1:] - slopes[:, :-1])
    inner_second = torch.linalg.solve(matrix, rhs.T).T
    second_derivatives = torch.zeros_like(flat_values)
    second_derivatives[:, 1:-1] = inner_second

    idx_right = torch.searchsorted(x_grid, query).clamp(1, n_points - 1)
    idx_left = idx_right - 1
    interval_h = h[idx_left]
    x_left = x_grid[idx_left]
    x_right = x_grid[idx_right]
    left_weight = (x_right - query) / interval_h
    right_weight = (query - x_left) / interval_h

    flat_left = idx_left.reshape(1, -1).expand(flat_values.shape[0], -1)
    flat_right = idx_right.reshape(1, -1).expand(flat_values.shape[0], -1)
    y_left = flat_values.gather(1, flat_left)
    y_right = flat_values.gather(1, flat_right)
    m_left = second_derivatives.gather(1, flat_left)
    m_right = second_derivatives.gather(1, flat_right)

    interval_h_flat = interval_h.reshape(1, -1)
    left_weight_flat = left_weight.reshape(1, -1)
    right_weight_flat = right_weight.reshape(1, -1)
    flat_interp = (
        m_left * (left_weight_flat.pow(3) - left_weight_flat) * interval_h_flat.pow(2)
        / 6.0
        + m_right
        * (right_weight_flat.pow(3) - right_weight_flat)
        * interval_h_flat.pow(2)
        / 6.0
        + y_left * left_weight_flat
        + y_right * right_weight_flat
    )
    return flat_interp.reshape(*values.shape[:-1], *query.shape)


def interpolate_line_values(
    x_grid: Tensor,
    values: Tensor,
    query_points: Tensor,
    method: Literal["linear", "cubic"],
) -> Tensor:
    if method == "linear":
        return linear_interpolate_line_values(
            x_grid=x_grid,
            values=values,
            query_points=query_points,
        )
    if method == "cubic":
        return natural_cubic_interpolate_line_values(
            x_grid=x_grid,
            values=values,
            query_points=query_points,
        )
    raise ValueError(f"Unsupported source interpolation method: {method}.")


def split_gauss_legendre_weighted_sum(values: Tensor, weights: Tensor) -> Tensor:
    if values.shape[-2:] != weights.shape:
        raise ValueError("values final dimensions must match weights shape.")
    return (values * weights).sum(dim=-1)


def _model_evaluate_pairs(
    model: torch.nn.Module,
    pair_coords: Tensor,
    a_vals: Tensor,
    ap_vals: Tensor,
    b_vals: Tensor,
    c_vals: Tensor,
    x_indices: Tensor,
) -> Tensor:
    method = getattr(model, "evaluate_pairs", None)
    if method is None:
        original = getattr(model, "_orig_mod", None)
        method = getattr(original, "evaluate_pairs", None)
    if method is None:
        raise TypeError("Green model must provide evaluate_pairs(...).")
    return cast(
        Tensor,
        method(
            pair_coords=pair_coords,
            a_vals=a_vals,
            ap_vals=ap_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            x_indices=x_indices,
        ),
    )


def reconstruct_with_split_gauss_legendre(
    model: torch.nn.Module,
    source: Tensor,
    coords: Tensor,
    a_vals: Tensor,
    ap_vals: Tensor,
    b_vals: Tensor,
    c_vals: Tensor,
    order: int,
    source_interpolation: Literal["linear", "cubic"] = "linear",
    source_grid: Tensor | None = None,
) -> Tensor:
    x_axis = coords[0, 0, :, 0].to(device=source.device, dtype=source.dtype)
    source_x_axis = (
        x_axis
        if source_grid is None
        else source_grid.to(device=source.device, dtype=source.dtype)
    )
    xi_nodes, weights = split_gauss_legendre_nodes(x_axis=x_axis, order=order)
    pair_coords = torch.stack(
        (
            x_axis[:, None].expand_as(xi_nodes),
            xi_nodes,
        ),
        dim=-1,
    )
    x_indices = torch.arange(x_axis.numel(), device=source.device)
    kernel_nodes = _model_evaluate_pairs(
        model=model,
        pair_coords=pair_coords,
        a_vals=a_vals,
        ap_vals=ap_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        x_indices=x_indices,
    )
    source_nodes = interpolate_line_values(
        x_grid=source_x_axis,
        values=source,
        query_points=xi_nodes,
        method=source_interpolation,
    )
    return split_gauss_legendre_weighted_sum(
        source_nodes * kernel_nodes.unsqueeze(0),
        weights,
    )


def source_interpolation_energy_rel_error(
    source: Tensor,
    x_axis: Tensor,
    order: int,
    integration_rule: IntegrationRule,
    source_interpolation: Literal["linear", "cubic"] = "linear",
    source_grid: Tensor | None = None,
) -> Tensor:
    x_axis = x_axis.to(device=source.device, dtype=source.dtype)
    source_x_axis = (
        x_axis
        if source_grid is None
        else source_grid.to(device=source.device, dtype=source.dtype)
    )
    xi_nodes, weights = split_gauss_legendre_nodes(x_axis=x_axis, order=order)
    source_nodes = interpolate_line_values(
        x_grid=source_x_axis,
        values=source,
        query_points=xi_nodes,
        method=source_interpolation,
    )
    split_energy_per_x = split_gauss_legendre_weighted_sum(
        source_nodes.pow(2),
        weights,
    )
    split_energy = split_energy_per_x.mean(dim=-1)
    grid_energy = integrate(
        source.pow(2),
        x=source_x_axis,
        dim=-1,
        rule=integration_rule,
    ).clamp_min(1.0e-12)
    return ((split_energy - grid_energy).abs() / grid_energy).mean()


def _validate_poisson_coefficients(a_vals: Tensor, b_vals: Tensor, c_vals: Tensor) -> None:
    a_used = a_vals[0] if a_vals.dim() == 4 else a_vals
    b_used = b_vals[0] if b_vals.dim() == 4 else b_vals
    c_used = c_vals[0] if c_vals.dim() == 4 else c_vals
    ones = torch.ones_like(a_used)
    zeros_b = torch.zeros_like(b_used)
    zeros_c = torch.zeros_like(c_used)
    if not torch.allclose(a_used, ones, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError(
            "split_gauss_legendre rel_green currently supports Poisson "
            "constant unit diffusion only."
        )
    if not torch.allclose(b_used, zeros_b, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError(
            "split_gauss_legendre rel_green currently requires zero convection."
        )
    if not torch.allclose(c_used, zeros_c, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError(
            "split_gauss_legendre rel_green currently requires zero reaction."
        )


def poisson_rel_green_by_line_split_gauss_legendre(
    model: torch.nn.Module,
    coords: Tensor,
    a_vals: Tensor,
    b_vals: Tensor,
    c_vals: Tensor,
    order: int,
    outer_rule: IntegrationRule,
) -> Tensor:
    _validate_poisson_coefficients(a_vals=a_vals, b_vals=b_vals, c_vals=c_vals)
    x_axis = coords[0, 0, :, 0].to(device=a_vals.device, dtype=a_vals.dtype)
    xi_nodes, weights = split_gauss_legendre_nodes(x_axis=x_axis, order=order)
    pair_coords = torch.stack(
        (
            x_axis[:, None].expand_as(xi_nodes),
            xi_nodes,
        ),
        dim=-1,
    )
    x_indices = torch.arange(x_axis.numel(), device=a_vals.device)
    ap_vals = torch.zeros_like(a_vals)
    pred = _model_evaluate_pairs(
        model=model,
        pair_coords=pair_coords,
        a_vals=a_vals,
        ap_vals=ap_vals,
        b_vals=b_vals,
        c_vals=c_vals,
        x_indices=x_indices,
    )
    exact = cast(Tensor, EllipticGreenFunction()(pair_coords)).squeeze(-1)
    exact = exact.unsqueeze(0).unsqueeze(0).expand_as(pred)

    num_inner = split_gauss_legendre_weighted_sum((pred - exact).pow(2), weights)
    den_inner = split_gauss_legendre_weighted_sum(exact.pow(2), weights)
    num = integrate(num_inner, x=x_axis, dim=-1, rule=outer_rule)
    den = integrate(den_inner, x=x_axis, dim=-1, rule=outer_rule).clamp_min(1.0e-12)
    return torch.sqrt(num / den).unsqueeze(0)


def green_quadrature_summary(config: Any) -> dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "rule": str(config.rule),
        "order": int(config.order),
        "source_interpolation": str(config.source_interpolation),
        "apply_to_loss": bool(config.apply_to_loss),
        "apply_to_rel_green": bool(config.apply_to_rel_green),
        "log_source_interpolation_diagnostic": bool(
            config.log_source_interpolation_diagnostic
        ),
        "source_sampling": {
            "enabled": bool(config.source_sampling.enabled),
            "factor": int(config.source_sampling.factor),
        },
    }
