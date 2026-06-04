from __future__ import annotations

import torch
from torch import Tensor


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return 2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * y)


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)
