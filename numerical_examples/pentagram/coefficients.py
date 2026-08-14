from __future__ import annotations

import torch
from torch import Tensor


OUTER_RADIUS = 0.5
CONVECTION_AMPLITUDE = 0.5
DIFFUSION_VARIATION = 0.5
REACTION_VARIATION = 0.5


def _scaled_coordinates(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    return torch.pi * x / OUTER_RADIUS, torch.pi * y / OUTER_RADIUS


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    scaled_x, scaled_y = _scaled_coordinates(x, y)
    return 1.0 + DIFFUSION_VARIATION * torch.sin(scaled_x) * torch.sin(scaled_y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    scaled_x, scaled_y = _scaled_coordinates(x, y)
    return (
        DIFFUSION_VARIATION
        * torch.pi
        / OUTER_RADIUS
        * torch.cos(scaled_x)
        * torch.sin(scaled_y)
    )


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    scaled_x, scaled_y = _scaled_coordinates(x, y)
    return (
        DIFFUSION_VARIATION
        * torch.pi
        / OUTER_RADIUS
        * torch.sin(scaled_x)
        * torch.cos(scaled_y)
    )


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    return -CONVECTION_AMPLITUDE * y / OUTER_RADIUS


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    return CONVECTION_AMPLITUDE * x / OUTER_RADIUS


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    scaled_x, scaled_y = _scaled_coordinates(x, y)
    return 1.0 + REACTION_VARIATION * torch.cos(scaled_x) * torch.cos(scaled_y)
