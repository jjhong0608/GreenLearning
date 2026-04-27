from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, cast

import torch

from greenonet.compile_utils import model_state_dict_for_save
from greenonet.config import CouplerConfig, CouplingModelConfig, ModelConfig


def save_state_dict_safetensors(
    state_dict: dict[str, torch.Tensor],
    path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Save a state dict to safetensors, fallback to torch.save."""
    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(path))
        if logger:
            logger.info("Saved model weights to %s", path)
    except Exception as exc:  # pragma: no cover - fallback
        fallback = path.with_suffix(".pt")
        torch.save(state_dict, fallback)
        if logger:
            logger.warning(
                "Failed to save with safetensors (%s); fell back to torch.save at %s",
                exc,
                fallback,
            )


def _normalize_config_payload(payload: Any) -> Any:
    if isinstance(payload, torch.dtype):
        if hasattr(payload, "name"):
            return payload.name
        return str(payload).replace("torch.", "")
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, tuple):
        return [_normalize_config_payload(item) for item in payload]
    if isinstance(payload, list):
        return [_normalize_config_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {key: _normalize_config_payload(value) for key, value in payload.items()}
    return payload


def _serialize_config(config: ModelConfig | CouplingModelConfig) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config must be a dataclass instance")
    payload = asdict(config)
    return cast(dict[str, Any], _normalize_config_payload(payload))


def _parse_dtype(value: str) -> torch.dtype:
    name = value.replace("torch.", "")
    return getattr(torch, name)


def _deserialize_config(
    payload: dict[str, Any], config_cls: type[ModelConfig] | type[CouplingModelConfig]
) -> ModelConfig | CouplingModelConfig:
    data = dict(payload)
    if "dtype" in data and isinstance(data["dtype"], str):
        data["dtype"] = _parse_dtype(data["dtype"])
    if config_cls is CouplingModelConfig:
        coupler_raw = data.get("coupler")
        if isinstance(coupler_raw, dict):
            data["coupler"] = CouplerConfig(**coupler_raw)
    allowed_keys = {field.name for field in fields(config_cls)}
    filtered = {key: value for key, value in data.items() if key in allowed_keys}
    return config_cls(**filtered)


def _model_type_from_config(config: ModelConfig | CouplingModelConfig) -> str:
    if isinstance(config, ModelConfig):
        return "green"
    if isinstance(config, CouplingModelConfig):
        return "coupling"
    raise TypeError(f"Unsupported config type: {type(config)}")


def save_model_with_config(
    model: torch.nn.Module,
    config: ModelConfig | CouplingModelConfig,
    path: Path,
    logger: logging.Logger | None = None,
) -> None:
    """Save model weights plus JSON-encoded config metadata."""
    model_type = _model_type_from_config(config)
    payload = _serialize_config(config)
    metadata = {
        "model_type": model_type,
        "model_config": json.dumps(payload),
    }
    try:
        from safetensors.torch import save_file

        save_file(model_state_dict_for_save(model), str(path), metadata=metadata)
        if logger:
            logger.info("Saved model+config to %s", path)
    except Exception as exc:  # pragma: no cover - fallback
        fallback = path.with_suffix(".pt")
        torch.save(
            {
                "state_dict": model_state_dict_for_save(model),
                **metadata,
                "model_config": payload,
            },
            fallback,
        )
        if logger:
            logger.warning(
                "Failed to save with safetensors (%s); fell back to torch.save at %s",
                exc,
                fallback,
            )


def load_state_dict_auto(
    model: torch.nn.Module, path: Path, map_location: str | None = "cpu"
) -> None:
    """Load weights from safetensors if possible, otherwise torch.load."""
    try:
        from safetensors.torch import load_file

        state = load_file(str(path), device=map_location or "cpu")
        model.load_state_dict(state)
        return
    except Exception:
        state = torch.load(str(path), map_location=map_location)
        model.load_state_dict(state)


def load_model_with_config(
    path: Path, map_location: str | None = "cpu"
) -> tuple[torch.nn.Module, ModelConfig | CouplingModelConfig]:
    """Load a model + config from a safetensors file with embedded metadata."""
    metadata: dict[str, Any] | None = None
    state: dict[str, torch.Tensor] | None = None
    try:
        from safetensors import safe_open

        with safe_open(
            str(path), framework="pt", device=map_location or "cpu"
        ) as handle:
            metadata = handle.metadata()
            state = {key: handle.get_tensor(key) for key in handle.keys()}
    except Exception:
        metadata = None
        state = None

    if metadata and state is not None:
        model_type = metadata.get("model_type")
        config_json = metadata.get("model_config")
        if not model_type or not config_json:
            raise ValueError(
                "Missing model_type or model_config metadata in checkpoint."
            )
        config_payload = json.loads(config_json)
    else:
        loaded = torch.load(str(path), map_location=map_location)
        if not isinstance(loaded, dict):
            raise ValueError("Unsupported checkpoint format for config loading.")
        state = loaded.get("state_dict")
        model_type = loaded.get("model_type")
        config_payload = loaded.get("model_config")
        if state is None or model_type is None or config_payload is None:
            raise ValueError("Checkpoint does not contain model config metadata.")

    if model_type == "green":
        green_config = cast(
            ModelConfig, _deserialize_config(config_payload, ModelConfig)
        )
        from greenonet.model import GreenONetModel

        model = GreenONetModel(green_config)
        model.load_state_dict(state)
        return model, green_config
    if model_type == "coupling":
        coupling_config = cast(
            CouplingModelConfig,
            _deserialize_config(config_payload, CouplingModelConfig),
        )
        from greenonet.coupling_model import CouplingNet

        model = CouplingNet(coupling_config)
        model.load_state_dict(state)
        return model, coupling_config
    raise ValueError(f"Unknown model_type '{model_type}' in checkpoint metadata.")
