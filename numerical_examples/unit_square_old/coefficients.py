from __future__ import annotations

import torch
from torch import Tensor


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.ones_like(x)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)
