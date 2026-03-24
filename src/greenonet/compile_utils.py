from __future__ import annotations

import logging
from typing import cast

import torch
from torch import nn

from greenonet.config import CompileConfig


def unwrap_compiled_model(model: nn.Module) -> nn.Module:
    compiled_origin = getattr(model, "_orig_mod", None)
    if isinstance(compiled_origin, nn.Module):
        return compiled_origin
    return model


def model_state_dict_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    return cast(dict[str, torch.Tensor], unwrap_compiled_model(model).state_dict())


def maybe_compile_model(
    model: nn.Module,
    compile_cfg: CompileConfig,
    logger: logging.Logger | None = None,
    model_name: str = "model",
) -> nn.Module:
    if not compile_cfg.enabled:
        return model
    if isinstance(getattr(model, "_orig_mod", None), nn.Module):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
    if logger is not None:
        logger.info("Compiling %s with torch.compile", model_name)
    return cast(nn.Module, torch.compile(model))
