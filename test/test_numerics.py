import pytest
import torch

from greenonet.green_quadrature import (
    gauss_legendre_nodes_weights,
    natural_cubic_interpolate_line_values,
    linear_interpolate_line_values,
    reconstruct_with_split_gauss_legendre,
    split_gauss_legendre_nodes,
    split_gauss_legendre_weighted_sum,
)
from greenonet.numerics import (
    integrate,
    line_first_derivative_fd,
    line_operator_fd,
    simpson,
    trapezoid,
)


def test_simpson_quadratic_uniform_dx() -> None:
    x = torch.linspace(0.0, 2.0, steps=5)
    y = x**2
    result = simpson(y, x=x, dim=0)
    assert result.item() == pytest.approx(8.0 / 3.0, rel=1e-6)


def test_simpson_even_samples_raises() -> None:
    y = torch.linspace(0.0, 1.0, steps=4)
    with pytest.raises(ValueError):
        simpson(y, x=None, dim=0)


def test_simpson_x_length_mismatch_raises() -> None:
    y = torch.linspace(0.0, 1.0, steps=5)
    x = torch.linspace(0.0, 1.0, steps=3)
    with pytest.raises(ValueError):
        simpson(y, x=x, dim=0)


def test_line_operator_fd_uses_negative_diffusion_sign() -> None:
    x = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    u = (x * (1.0 - x)).reshape(1, 1, -1)
    a = torch.ones_like(u)
    b = torch.zeros_like(u)
    c = torch.zeros_like(u)

    result = line_operator_fd(u, a, b, c, x)
    expected = torch.full((1, 1, 3), 2.0, dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_line_first_derivative_fd_matches_centered_difference() -> None:
    x = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    u = (x**2).reshape(1, 1, -1)

    result = line_first_derivative_fd(u, x)
    expected = torch.tensor([[[0.5, 1.0, 1.5]]], dtype=torch.float64)
    assert torch.allclose(result, expected, atol=1e-12, rtol=1e-12)


def test_trapezoid_quadratic_uniform_dx() -> None:
    x = torch.linspace(0.0, 2.0, steps=5)
    y = x**2
    result = trapezoid(y, x=x, dim=0)
    assert result.item() == pytest.approx(2.75, rel=1e-6)


def test_integrate_dispatches_trapezoid_rule() -> None:
    x = torch.linspace(0.0, 2.0, steps=5)
    y = x**2
    result = integrate(y, x=x, dim=0, rule="trapezoid")
    assert result.item() == pytest.approx(2.75, rel=1e-6)


def test_integrate_invalid_rule_raises() -> None:
    x = torch.linspace(0.0, 1.0, steps=5)
    y = x**2
    with pytest.raises(ValueError):
        integrate(y, x=x, dim=0, rule="unknown")


def test_gauss_legendre_integrates_low_order_polynomial() -> None:
    nodes, weights = gauss_legendre_nodes_weights(
        order=3,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    result = (weights * nodes.pow(4)).sum()

    assert result.item() == pytest.approx(2.0 / 5.0, rel=1e-12)


def test_split_gauss_legendre_weights_cover_unit_interval() -> None:
    x_axis = torch.tensor([0.0, 0.25, 0.5, 1.0], dtype=torch.float64)

    _nodes, weights = split_gauss_legendre_nodes(x_axis=x_axis, order=4)

    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(x_axis))
    assert torch.isfinite(weights).all()


def test_split_gauss_legendre_weighted_sum_handles_constant() -> None:
    x_axis = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    nodes, weights = split_gauss_legendre_nodes(x_axis=x_axis, order=4)
    values = torch.ones_like(nodes)

    result = split_gauss_legendre_weighted_sum(values, weights)

    assert torch.allclose(result, torch.ones_like(x_axis), atol=1e-12)


def test_reconstruct_with_split_gauss_legendre_uses_source_grid() -> None:
    class UnitKernelModel(torch.nn.Module):
        def evaluate_pairs(
            self,
            pair_coords: torch.Tensor,
            a_vals: torch.Tensor,
            ap_vals: torch.Tensor,
            b_vals: torch.Tensor,
            c_vals: torch.Tensor,
            x_indices: torch.Tensor,
        ) -> torch.Tensor:
            del a_vals, ap_vals, b_vals, c_vals, x_indices
            return torch.ones(
                (2, 1, pair_coords.shape[0], pair_coords.shape[1]),
                dtype=pair_coords.dtype,
                device=pair_coords.device,
            )

    x_axis = torch.linspace(0.0, 1.0, steps=3, dtype=torch.float64)
    coords_x = torch.stack(
        (
            x_axis,
            torch.full_like(x_axis, 0.5),
        ),
        dim=-1,
    ).reshape(1, 3, 2)
    coords_y = torch.stack(
        (
            torch.full_like(x_axis, 0.5),
            x_axis,
        ),
        dim=-1,
    ).reshape(1, 3, 2)
    coords = torch.stack((coords_x, coords_y), dim=0)
    source_grid = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    source = source_grid.reshape(1, 1, 1, -1).expand(1, 2, 1, -1).clone()
    coeffs = torch.ones((2, 1, 3), dtype=torch.float64)
    zeros = torch.zeros_like(coeffs)

    reconstruction = reconstruct_with_split_gauss_legendre(
        model=UnitKernelModel(),
        source=source,
        coords=coords,
        a_vals=coeffs,
        ap_vals=zeros,
        b_vals=zeros,
        c_vals=zeros,
        order=4,
        source_grid=source_grid,
    )

    expected = torch.full((1, 2, 1, 3), 0.5, dtype=torch.float64)
    torch.testing.assert_close(reconstruction, expected)


def test_linear_interpolate_line_values_matches_linear_function() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    values = (2.0 * x_grid + 1.0).reshape(1, 1, 1, -1)
    query = torch.tensor([[0.125, 0.375], [0.625, 0.875]], dtype=torch.float64)

    result = linear_interpolate_line_values(x_grid, values, query)

    assert result.shape == (1, 1, 1, 2, 2)
    assert torch.allclose(result[0, 0, 0], 2.0 * query + 1.0, atol=1e-12)


def test_natural_cubic_interpolate_line_values_matches_linear_function() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=6, dtype=torch.float64)
    values = (3.0 * x_grid - 0.5).reshape(1, 1, 1, -1)
    query = torch.tensor([[0.0, 0.2, 0.55], [0.8, 0.95, 1.0]], dtype=torch.float64)

    result = natural_cubic_interpolate_line_values(x_grid, values, query)

    assert result.shape == (1, 1, 1, 2, 3)
    assert torch.allclose(result[0, 0, 0], 3.0 * query - 0.5, atol=1e-12)


def test_natural_cubic_interpolate_line_values_matches_boundary_values() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=7, dtype=torch.float64)
    values = torch.sin(torch.pi * x_grid).reshape(1, 1, -1)
    query = torch.tensor([0.0, 1.0], dtype=torch.float64)

    result = natural_cubic_interpolate_line_values(x_grid, values, query)

    assert torch.allclose(result[0, 0], torch.tensor([0.0, 0.0], dtype=torch.float64), atol=1e-12)


def test_natural_cubic_interpolate_line_values_two_points_matches_linear() -> None:
    x_grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    values = torch.tensor([[[2.0, 4.0]]], dtype=torch.float64)
    query = torch.tensor([[0.25, 0.75]], dtype=torch.float64)

    cubic = natural_cubic_interpolate_line_values(x_grid, values, query)
    linear = linear_interpolate_line_values(x_grid, values, query)

    assert torch.allclose(cubic, linear, atol=1e-12)


def test_natural_cubic_interpolate_line_values_keeps_batch_and_query_shape() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    base = torch.stack((torch.sin(torch.pi * x_grid), torch.cos(torch.pi * x_grid)))
    values = base.reshape(1, 2, -1)
    query = torch.linspace(0.0, 1.0, steps=12, dtype=torch.float64).reshape(3, 4)

    result = natural_cubic_interpolate_line_values(x_grid, values, query)

    assert result.shape == (1, 2, 3, 4)
    assert torch.isfinite(result).all()


def test_natural_cubic_interpolate_line_values_improves_smooth_sine_midpoints() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    values = torch.sin(torch.pi * x_grid).reshape(1, 1, -1)
    query = (x_grid[:-1] + x_grid[1:]) / 2.0
    exact = torch.sin(torch.pi * query)

    linear = linear_interpolate_line_values(x_grid, values, query)[0, 0]
    cubic = natural_cubic_interpolate_line_values(x_grid, values, query)[0, 0]

    linear_error = (linear - exact).abs().mean()
    cubic_error = (cubic - exact).abs().mean()
    assert cubic_error < linear_error


def test_natural_cubic_interpolate_line_values_invalid_shape_raises() -> None:
    x_grid = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    values = torch.zeros(1, 4, dtype=torch.float64)
    query = torch.tensor([0.5], dtype=torch.float64)

    with pytest.raises(ValueError):
        natural_cubic_interpolate_line_values(x_grid, values, query)
