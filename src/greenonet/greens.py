from __future__ import annotations

import torch
from torch import Tensor, nn, where, cumulative_trapezoid
from torch.nn.functional import pad

from greenonet.numerics import IntegrationRule, integrate


class EllipticGreenFunction(nn.Module):  # type: ignore[misc]
    """Lightweight analytic surrogate for Poisson's Green function on the unit square."""

    def forward(self, coords: Tensor) -> Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        lt = x * (1 - y)
        rt = y * (1 - x)
        return where(x < y, lt, rt)


class IntegrationEllipticGreenFunction(nn.Module):  # type: ignore[misc]
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
