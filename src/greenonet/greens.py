from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn, where, cumulative_trapezoid
from torch.nn.functional import pad

from greenonet.numerics import IntegrationRule, integrate

GreenReferenceKind = Literal["diffusion", "convection_diffusion"]


@dataclass(frozen=True)
class GreenReferencePolicy:
    valid: bool
    reference: GreenReferenceKind | None
    skip_reason: str | None


def select_green_reference_policy(
    b_vals: Tensor,
    c_vals: Tensor,
    zero_tol: float = 1.0e-12,
) -> GreenReferencePolicy:
    c_max = float(c_vals.detach().abs().max().item())
    if c_max > zero_tol:
        return GreenReferencePolicy(
            valid=False,
            reference=None,
            skip_reason=(
                "Skipped because sampled reaction c_vals are nonzero; "
                "reaction Green reference is not implemented."
            ),
        )

    b_max = float(b_vals.detach().abs().max().item())
    reference: GreenReferenceKind = (
        "diffusion" if b_max <= zero_tol else "convection_diffusion"
    )
    return GreenReferencePolicy(valid=True, reference=reference, skip_reason=None)


def exact_green_kernel_from_coefficients(
    coords: Tensor,
    a_vals: Tensor,
    b_vals: Tensor,
    reference: GreenReferenceKind,
) -> Tensor:
    if coords.dim() != 4:
        raise ValueError("coords must have shape (axes, lines, points, 2)")
    if a_vals.shape != b_vals.shape:
        raise ValueError("a_vals and b_vals must have the same shape")
    if a_vals.dim() not in (3, 4):
        raise ValueError("a_vals must have shape (axes, lines, points) or batched")

    single_batch = a_vals.dim() == 3
    if single_batch:
        a_vals = a_vals.unsqueeze(0)
        b_vals = b_vals.unsqueeze(0)

    batch_size, axes, n_lines, m_points = a_vals.shape
    if coords.shape[0] < axes or coords.shape[1] < n_lines:
        raise ValueError("coords does not cover the requested axes and lines")
    if coords.shape[2] != m_points:
        raise ValueError("coords and coefficient tensors must share point count")

    exact = torch.zeros(
        (batch_size, axes, n_lines, m_points, m_points),
        device=a_vals.device,
        dtype=a_vals.dtype,
    )
    for batch_idx in range(batch_size):
        for axis in range(axes):
            for line_idx in range(n_lines):
                x_axis = coords[axis, line_idx, :, axis].to(
                    device=a_vals.device,
                    dtype=a_vals.dtype,
                )
                gf = ExactGreenFunction(x_axis, a=a_vals[batch_idx, axis, line_idx])
                if reference == "diffusion":
                    exact[batch_idx, axis, line_idx] = gf()
                elif reference == "convection_diffusion":
                    exact[batch_idx, axis, line_idx] = gf.convection_diffusion(
                        b_vals[batch_idx, axis, line_idx]
                    )
                else:
                    raise ValueError(f"Unknown Green reference: {reference}")

    return exact.squeeze(0) if single_batch else exact


def exact_green_kernel_from_unit_coefficients(
    x_axis: Tensor,
    a_vals: Tensor,
    b_vals: Tensor,
    reference: GreenReferenceKind,
) -> Tensor:
    """Build flat unit-interval exact kernels for shape ``(..., intervals, M)``."""

    if x_axis.dim() != 1:
        raise ValueError("x_axis must be one-dimensional.")
    if a_vals.shape != b_vals.shape:
        raise ValueError("a_vals and b_vals must have the same shape.")
    if a_vals.dim() not in (2, 3):
        raise ValueError(
            "a_vals must have shape (intervals, M) or (batch, intervals, M)."
        )
    if a_vals.shape[-1] != x_axis.numel():
        raise ValueError("Coefficient tensors must share x_axis point count.")

    single_batch = a_vals.dim() == 2
    if single_batch:
        a_vals = a_vals.unsqueeze(0)
        b_vals = b_vals.unsqueeze(0)

    batch_size, num_intervals, m_points = a_vals.shape
    exact = torch.zeros(
        (batch_size, num_intervals, m_points, m_points),
        device=a_vals.device,
        dtype=a_vals.dtype,
    )
    x_axis = x_axis.to(device=a_vals.device, dtype=a_vals.dtype)
    for batch_idx in range(batch_size):
        for interval_idx in range(num_intervals):
            gf = ExactGreenFunction(x_axis, a=a_vals[batch_idx, interval_idx])
            if reference == "diffusion":
                exact[batch_idx, interval_idx] = gf()
            elif reference == "convection_diffusion":
                exact[batch_idx, interval_idx] = gf.convection_diffusion(
                    b_vals[batch_idx, interval_idx]
                )
            else:
                raise ValueError(f"Unknown Green reference: {reference}")
    return exact.squeeze(0) if single_batch else exact


class EllipticGreenFunction(nn.Module):
    """Lightweight analytic surrogate for Poisson's Green function on the unit square."""

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        lt = x * (1 - y)
        rt = y * (1 - x)
        return where(x < y, lt, rt)


class IntegrationEllipticGreenFunction(nn.Module):
    """Antiderivative-like term to mimic integral effects in the original model."""

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        lt = 0.5 * x**2 * (1 - y)
        rt = 0.5 * y * (2 * x - x**2 - y)
        return where(x < y, lt, rt)


class ExactGreenFunction:
    def __init__(self, x: Tensor, a: Tensor) -> None:
        self.x = x
        self.a = a
        self.g = self._green_function(x, a).squeeze(0)

    def _green_function(self, x: Tensor, a: Tensor) -> Tensor:
        single_function = a.dim() == 1
        if single_function:
            a = a.unsqueeze(0)

        e = 1.0 / a
        p = cumulative_trapezoid(e, x, dim=-1)
        e_flip = torch.flip(e, dims=(-1,))
        x_flip = torch.flip(x, dims=(-1,))
        q = -torch.flip(cumulative_trapezoid(e_flip, x_flip, dim=-1), dims=(-1,))
        r = q[..., :1].unsqueeze(-1)

        p = pad(p, (1, 0))
        q = pad(q, (0, 1))

        mask = x.unsqueeze(0).unsqueeze(-1) < x.unsqueeze(0).unsqueeze(-2)
        g_left = p.unsqueeze(-1) * q.unsqueeze(-2) / r
        g_right = p.unsqueeze(-2) * q.unsqueeze(-1) / r
        g = torch.where(mask, g_left, g_right)
        return g.squeeze(0) if single_function else g

    def forward(self) -> Tensor:
        return self._green_function(self.x, self.a)

    def convection_diffusion(self, b: Tensor) -> Tensor:
        return self._convection_diffusion_green_function(self.x, self.a, b)

    def _convection_diffusion_green_function(
        self,
        x: Tensor,
        a: Tensor,
        b: Tensor,
    ) -> Tensor:
        if x.dim() != 1:
            raise ValueError("x must be a one-dimensional tensor")
        if x.shape[0] < 2:
            raise ValueError("x must contain at least two grid points")
        if a.dim() == 0:
            raise ValueError("a must have at least one dimension")
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if a.shape[-1] != x.shape[0]:
            raise ValueError("a and b must have the same final dimension as x")
        if a.device != x.device or b.device != x.device:
            raise ValueError("x, a, and b must be on the same device")

        single_function = a.dim() == 1
        if single_function:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)

        h_integral = cumulative_trapezoid(b / a, x, dim=-1)
        h_integral = pad(h_integral, (1, 0))

        weighted_integrand = torch.exp(h_integral) / a
        weighted_integral = cumulative_trapezoid(weighted_integrand, x, dim=-1)
        weighted_integral = pad(weighted_integral, (1, 0))

        weighted_total = weighted_integral[..., -1:]
        source_factor = torch.exp(-h_integral) / weighted_total

        mask = x.unsqueeze(0).unsqueeze(-1) < x.unsqueeze(0).unsqueeze(-2)
        g_left = (
            weighted_integral.unsqueeze(-1)
            * (weighted_total - weighted_integral).unsqueeze(-2)
            * source_factor.unsqueeze(-2)
        )
        g_right = (
            weighted_integral.unsqueeze(-2)
            * (weighted_total - weighted_integral).unsqueeze(-1)
            * source_factor.unsqueeze(-2)
        )
        g = torch.where(mask, g_left, g_right)
        return g.squeeze(0) if single_function else g

    def poisson(self) -> Tensor:
        xi = self.x.unsqueeze(0)
        x = self.x.unsqueeze(1)
        g_left = x * (1 - xi)
        g_right = xi * (1 - x)
        return torch.where(x < xi, g_left, g_right)

    def __call__(self) -> Tensor:
        return self.forward()

    def error(self, g: Tensor, integration_rule: IntegrationRule = "simpson") -> float:
        g = g.reshape(self.g.shape)
        output = (self.g - g) ** 2
        output = integrate(x=self.x, y=output, dim=-1, rule=integration_rule)
        output = integrate(x=self.x, y=output, dim=-1, rule=integration_rule)
        ex = integrate(x=self.x, y=self.g**2, dim=-1, rule=integration_rule)
        ex = integrate(x=self.x, y=ex, dim=-1, rule=integration_rule)
        output = output / ex
        if output.dim() > 0:
            output = output.mean()
        return float(output.item())
