from __future__ import annotations

from typing import Literal

import torch

IntegrationRule = Literal["simpson", "trapezoid"]


def uniform_spacing(axis_coords: torch.Tensor) -> torch.Tensor:
    if axis_coords.numel() < 2:
        raise ValueError("Need at least two coordinates to compute spacing.")
    diffs = axis_coords[1:] - axis_coords[:-1]
    step = diffs[0]
    if not torch.allclose(diffs, step.expand_as(diffs), rtol=1e-6, atol=1e-12):
        raise ValueError("Finite-difference operator requires a uniform grid.")
    return step


def line_first_derivative_fd(
    u_lines: torch.Tensor,
    axis_coords: torch.Tensor,
) -> torch.Tensor:
    """Centered first derivative on the interior nodes of a uniform line grid."""
    step = uniform_spacing(axis_coords)
    return (u_lines[..., 2:] - u_lines[..., :-2]) / (2.0 * step)


def line_operator_fd(
    u_lines: torch.Tensor,
    a_lines: torch.Tensor,
    b_lines: torch.Tensor,
    c_lines: torch.Tensor,
    axis_coords: torch.Tensor,
) -> torch.Tensor:
    """Conservative second-order stencil for -d_s(a d_s u) + b d_s u + c u."""
    step = uniform_spacing(axis_coords)
    face_a = 0.5 * (a_lines[..., 1:] + a_lines[..., :-1])
    face_du = (u_lines[..., 1:] - u_lines[..., :-1]) / step
    face_flux = face_a * face_du
    diffusion = (face_flux[..., 1:] - face_flux[..., :-1]) / step

    centered_du = line_first_derivative_fd(u_lines, axis_coords)
    advection = b_lines[..., 1:-1] * centered_du
    reaction = c_lines[..., 1:-1] * u_lines[..., 1:-1]
    return -diffusion + advection + reaction


def simpson(
    y: torch.Tensor, x: torch.Tensor | None = None, dim: int = -1
) -> torch.Tensor:
    """Simpson's rule along a single axis (requires odd samples)."""
    if x is not None:
        if x.dim() != 1:
            raise ValueError(f"x must be 1-D, got {x.dim()}D")
        if x.numel() != y.size(dim):
            raise ValueError(
                f"x length ({x.numel()}) must match y size along dim ({y.size(dim)})"
            )
        dx = (x[-1] - x[0]) / (x.numel() - 1)
    else:
        dx = torch.tensor(1.0, dtype=y.dtype, device=y.device)

    n = y.size(dim)
    if n % 2 == 0:
        raise ValueError(f"Simpson's rule requires an odd number of samples, got {n}")

    weights = torch.ones(n, dtype=y.dtype, device=y.device)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2

    return (weights * y).sum(dim=dim) * dx / 3.0


def trapezoid(
    y: torch.Tensor, x: torch.Tensor | None = None, dim: int = -1
) -> torch.Tensor:
    """Composite trapezoid rule along a single axis."""
    n = y.size(dim)
    if n < 2:
        raise ValueError(f"Trapezoid rule requires at least two samples, got {n}")

    moved = y.movedim(dim, -1)
    if x is not None:
        if x.dim() != 1:
            raise ValueError(f"x must be 1-D, got {x.dim()}D")
        if x.numel() != moved.size(-1):
            raise ValueError(
                f"x length ({x.numel()}) must match y size along dim ({moved.size(-1)})"
            )
        dx = x[1:] - x[:-1]
    else:
        dx = torch.ones(
            moved.size(-1) - 1,
            dtype=moved.dtype,
            device=moved.device,
        )

    segment = 0.5 * (moved[..., 1:] + moved[..., :-1]) * dx
    return segment.sum(dim=-1)


def integrate(
    y: torch.Tensor,
    x: torch.Tensor | None = None,
    dim: int = -1,
    rule: IntegrationRule = "simpson",
) -> torch.Tensor:
    if rule == "simpson":
        return simpson(y=y, x=x, dim=dim)
    if rule == "trapezoid":
        return trapezoid(y=y, x=x, dim=dim)
    raise ValueError(f"Unsupported integration rule: {rule}")
