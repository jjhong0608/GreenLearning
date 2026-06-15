import pytest
import torch

from greenonet.greens import (
    ExactGreenFunction,
    exact_green_kernel_from_coefficients,
    select_green_reference_policy,
)
from greenonet.trainer import Trainer


def _coords(x: torch.Tensor) -> torch.Tensor:
    coords = torch.zeros((2, 1, x.numel(), 2), dtype=x.dtype, device=x.device)
    coords[0, 0, :, 0] = x
    coords[0, 0, :, 1] = 0.25
    coords[1, 0, :, 0] = 0.75
    coords[1, 0, :, 1] = x
    return coords


def test_convection_diffusion_matches_diffusion_when_convection_zero() -> None:
    x = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    a = 1.0 + 0.25 * torch.sin(torch.pi * x)
    b = torch.zeros_like(a)

    gf = ExactGreenFunction(x, a=a)

    assert torch.allclose(gf.convection_diffusion(b), gf.forward(), atol=1e-12)


def test_convection_diffusion_matches_poisson_for_constant_diffusion() -> None:
    x = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    a = torch.ones_like(x)
    b = torch.zeros_like(x)

    gf = ExactGreenFunction(x, a=a)

    assert torch.allclose(gf.convection_diffusion(b), gf.poisson(), atol=1e-12)


def test_convection_diffusion_supports_batched_coefficients() -> None:
    x = torch.linspace(0.0, 1.0, steps=7, dtype=torch.float64)
    a = torch.stack((torch.ones_like(x), 1.0 + 0.2 * x))
    b = torch.stack((0.5 * torch.ones_like(x), -0.25 * torch.ones_like(x)))

    result = ExactGreenFunction(x, a=a).convection_diffusion(b)

    assert result.shape == (2, 7, 7)
    assert result.dtype == torch.float64
    assert torch.isfinite(result).all()


def test_convection_diffusion_boundary_values_are_zero() -> None:
    x = torch.linspace(0.0, 1.0, steps=11, dtype=torch.float64)
    a = 1.0 + 0.1 * x
    b = 0.75 * torch.cos(torch.pi * x)

    result = ExactGreenFunction(x, a=a).convection_diffusion(b)
    expected = torch.zeros_like(x)

    assert torch.allclose(result[0, :], expected, atol=1e-12)
    assert torch.allclose(result[-1, :], expected, atol=1e-12)
    assert torch.allclose(result[:, 0], expected, atol=1e-12)
    assert torch.allclose(result[:, -1], expected, atol=1e-12)


def test_convection_diffusion_shape_mismatch_raises() -> None:
    x = torch.linspace(0.0, 1.0, steps=5, dtype=torch.float64)
    a = torch.ones_like(x)
    b = torch.ones(4, dtype=torch.float64)

    gf = ExactGreenFunction(x, a=a)
    with pytest.raises(ValueError, match="same shape"):
        gf.convection_diffusion(b)


def test_convection_diffusion_orientation_matches_constant_coefficient_formula() -> None:
    x = torch.linspace(0.0, 1.0, steps=9, dtype=torch.float64)
    beta = torch.tensor(1.7, dtype=torch.float64)
    a = torch.ones_like(x)
    b = beta.expand_as(x)

    result = ExactGreenFunction(x, a=a).convection_diffusion(b)

    x_eval = x.unsqueeze(1)
    xi_src = x.unsqueeze(0)
    total = torch.exp(beta) - 1.0
    common = beta * torch.exp(beta * xi_src) * total
    left = (torch.exp(beta * x_eval) - 1.0) * (
        torch.exp(beta) - torch.exp(beta * xi_src)
    ) / common
    right = (torch.exp(beta * xi_src) - 1.0) * (
        torch.exp(beta) - torch.exp(beta * x_eval)
    ) / common
    expected = torch.where(x_eval < xi_src, left, right)

    assert not torch.allclose(result, result.T)
    assert torch.allclose(result, expected, atol=2e-3, rtol=2e-3)


def test_select_green_reference_policy_distinguishes_supported_cases() -> None:
    zeros = torch.zeros((1, 2, 1, 5), dtype=torch.float64)
    nonzero_b = zeros.clone()
    nonzero_b[:, 0] = 1.0
    nonzero_c = zeros.clone()
    nonzero_c[:, 1] = 1.0

    diffusion = select_green_reference_policy(zeros, zeros)
    convection = select_green_reference_policy(nonzero_b, zeros)
    reaction = select_green_reference_policy(zeros, nonzero_c)

    assert diffusion.valid is True
    assert diffusion.reference == "diffusion"
    assert convection.valid is True
    assert convection.reference == "convection_diffusion"
    assert reaction.valid is False
    assert reaction.reference is None
    assert reaction.skip_reason is not None
    assert "reaction" in reaction.skip_reason


def test_exact_green_kernel_from_coefficients_uses_axis_local_convection() -> None:
    x = torch.linspace(0.0, 1.0, steps=7, dtype=torch.float64)
    coords = _coords(x)
    a_vals = torch.ones((2, 1, x.numel()), dtype=torch.float64)
    b_vals = torch.zeros_like(a_vals)
    b_vals[0, 0] = 0.5
    b_vals[1, 0] = -0.25

    result = exact_green_kernel_from_coefficients(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        reference="convection_diffusion",
    )

    assert torch.allclose(
        result[0, 0],
        ExactGreenFunction(x, a_vals[0, 0]).convection_diffusion(b_vals[0, 0]),
    )
    assert torch.allclose(
        result[1, 0],
        ExactGreenFunction(x, a_vals[1, 0]).convection_diffusion(b_vals[1, 0]),
    )


def test_trainer_rel_green_uses_convection_diffusion_reference() -> None:
    x = torch.linspace(0.0, 1.0, steps=7, dtype=torch.float64)
    coords = _coords(x)
    a_vals = torch.ones((1, 2, 1, x.numel()), dtype=torch.float64)
    b_vals = torch.ones_like(a_vals) * 0.4
    c_vals = torch.zeros_like(a_vals)
    exact = exact_green_kernel_from_coefficients(
        coords=coords,
        a_vals=a_vals,
        b_vals=b_vals,
        reference="convection_diffusion",
    )

    rel_by_line = Trainer._green_kernel_rel_by_line(
        prediction=exact,
        coords=coords,
        a_val=a_vals,
        ap_val=torch.zeros_like(a_vals),
        b_val=b_vals,
        c_val=c_vals,
        integration_rule="trapezoid",
    )
    rel_scalar = Trainer._green_kernel_error(
        prediction=exact,
        coords=coords,
        a_val=a_vals,
        ap_val=torch.zeros_like(a_vals),
        b_val=b_vals,
        c_val=c_vals,
        integration_rule="trapezoid",
    )

    assert torch.allclose(rel_by_line, torch.zeros_like(rel_by_line))
    assert rel_scalar.item() == pytest.approx(0.0, abs=1e-12)


def test_trainer_rel_green_returns_nan_for_reaction_reference_gap() -> None:
    x = torch.linspace(0.0, 1.0, steps=7, dtype=torch.float64)
    coords = _coords(x)
    a_vals = torch.ones((1, 2, 1, x.numel()), dtype=torch.float64)
    b_vals = torch.zeros_like(a_vals)
    c_vals = torch.ones_like(a_vals)
    prediction = torch.zeros((1, 2, 1, x.numel(), x.numel()), dtype=torch.float64)

    rel_by_line = Trainer._green_kernel_rel_by_line(
        prediction=prediction,
        coords=coords,
        a_val=a_vals,
        ap_val=torch.zeros_like(a_vals),
        b_val=b_vals,
        c_val=c_vals,
        integration_rule="trapezoid",
    )
    rel_scalar = Trainer._green_kernel_error(
        prediction=prediction,
        coords=coords,
        a_val=a_vals,
        ap_val=torch.zeros_like(a_vals),
        b_val=b_vals,
        c_val=c_vals,
        integration_rule="trapezoid",
    )

    assert torch.isnan(rel_by_line).all()
    assert torch.isnan(rel_scalar)
