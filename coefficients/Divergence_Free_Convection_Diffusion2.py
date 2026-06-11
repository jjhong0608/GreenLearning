from __future__ import annotations

import torch
from torch import Tensor


CONVECTION_AMPLITUDE = 1.0


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(2 * torch.pi * y)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(2 * torch.pi * y)


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    return (
        CONVECTION_AMPLITUDE
        * torch.pi
        * torch.sin(torch.pi * x)
        * torch.cos(torch.pi * y)
    )


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    return (
        -CONVECTION_AMPLITUDE
        * torch.pi
        * torch.cos(torch.pi * x)
        * torch.sin(torch.pi * y)
    )


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)
