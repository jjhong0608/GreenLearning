from __future__ import annotations

import torch
from torch import Tensor


INNER_RADIUS = 0.2
OUTER_RADIUS = 0.5
CONVECTION_AMPLITUDE = 0.5

_INNER_RADIUS_SQUARED = INNER_RADIUS**2
_OUTER_RADIUS_SQUARED = OUTER_RADIUS**2
_RADIUS_SQUARED_SPAN = _OUTER_RADIUS_SQUARED - _INNER_RADIUS_SQUARED


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)


def _annulus_envelope(x: Tensor, y: Tensor) -> Tensor:
    radius_squared = x.square() + y.square()
    return (
        4.0
        * (radius_squared - _INNER_RADIUS_SQUARED)
        * (_OUTER_RADIUS_SQUARED - radius_squared)
        / (_RADIUS_SQUARED_SPAN**2)
    )


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    envelope = _annulus_envelope(x, y)
    return -CONVECTION_AMPLITUDE * envelope * y / OUTER_RADIUS


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    envelope = _annulus_envelope(x, y)
    return CONVECTION_AMPLITUDE * envelope * x / OUTER_RADIUS


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return 0.5 * (1 + 0.5 * torch.cos(2 * torch.pi * x) * torch.cos(2 * torch.pi * y))
