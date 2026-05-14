from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, cast

import torch
from torch import Tensor


CoefficientFunction = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class CoefficientFunctions:
    """Coefficient functions used consistently by Green and Coupling pipelines."""

    a_fun: CoefficientFunction
    apx_fun: CoefficientFunction
    apy_fun: CoefficientFunction
    b_fun: CoefficientFunction
    c_fun: CoefficientFunction


def a_fun(x: Tensor, y: Tensor) -> Tensor:
    return 1 + 0.5 * torch.sin(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)


def apx_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.pi * torch.cos(2 * torch.pi * x) * torch.sin(4 * torch.pi * y)


def apy_fun(x: Tensor, y: Tensor) -> Tensor:
    return 2 * torch.pi * torch.sin(2 * torch.pi * x) * torch.cos(4 * torch.pi * y)


def b_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def c_fun(x: Tensor, y: Tensor) -> Tensor:
    return torch.zeros_like(x)


def default_coefficient_functions() -> CoefficientFunctions:
    return CoefficientFunctions(
        a_fun=a_fun,
        apx_fun=apx_fun,
        apy_fun=apy_fun,
        b_fun=b_fun,
        c_fun=c_fun,
    )


def _load_module_from_path(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Coefficient functions file does not exist: {path}")
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    module_name = f"greenonet_user_coefficients_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import coefficient functions from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_coefficient_function(module: ModuleType, name: str) -> CoefficientFunction:
    value = getattr(module, name, None)
    if value is None:
        raise ValueError(f"Coefficient functions file is missing callable '{name}'.")
    if not callable(value):
        raise TypeError(f"Coefficient functions entry '{name}' must be callable.")
    return cast(CoefficientFunction, value)


def load_coefficient_functions(path: Path | None) -> CoefficientFunctions:
    if path is None:
        return default_coefficient_functions()
    module = _load_module_from_path(path)
    return CoefficientFunctions(
        a_fun=_get_coefficient_function(module, "a_fun"),
        apx_fun=_get_coefficient_function(module, "apx_fun"),
        apy_fun=_get_coefficient_function(module, "apy_fun"),
        b_fun=_get_coefficient_function(module, "b_fun"),
        c_fun=_get_coefficient_function(module, "c_fun"),
    )
