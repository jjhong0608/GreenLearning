import pytest
import torch

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
