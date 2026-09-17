from __future__ import annotations

import torch
from torch import Tensor


# Geometry metadata for the paper experiment; the coefficient itself is evaluated
# directly in physical coordinates.
DISK_RADIUS = 0.5


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    """Asymmetric smooth diffusion on the radius-0.5 disk."""
    return 1.0 + 0.5 * torch.sin(2.0 * torch.pi * x) * torch.sin(4.0 * torch.pi * y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.cos(2.0 * torch.pi * x) * torch.sin(4.0 * torch.pi * y)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return (
        2.0 * torch.pi * torch.sin(2.0 * torch.pi * x) * torch.cos(4.0 * torch.pi * y)
    )


def bx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def by_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)
