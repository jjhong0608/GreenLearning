from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
import torch


MAX_TRAINING_SEED = 2**32 - 1
TrainingStage = Literal["green", "coupling"]

_STAGE_NAMESPACES: dict[TrainingStage, tuple[str, ...]] = {
    "green": (
        "data_train",
        "data_valid",
        "model",
        "runtime",
        "loader_train",
        "loader_lbfgs",
    ),
    "coupling": (
        "model",
        "runtime",
        "loader_train",
    ),
}


def validate_training_seed(seed: int | None, *, field_name: str) -> None:
    """Validate an optional uint32 base seed without accepting booleans."""

    if seed is None:
        return
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"{field_name} must be an integer or null.")
    if not 0 <= seed <= MAX_TRAINING_SEED:
        raise ValueError(f"{field_name} must be in [0, {MAX_TRAINING_SEED}].")


def derive_training_subseed(
    base_seed: int,
    *,
    stage: TrainingStage,
    namespace: str,
) -> int:
    """Derive a stable uint32 seed for one stage-local RNG namespace."""

    validate_training_seed(base_seed, field_name="base_seed")
    if namespace not in _STAGE_NAMESPACES[stage]:
        supported = ", ".join(_STAGE_NAMESPACES[stage])
        raise ValueError(
            f"Unsupported {stage} seed namespace {namespace!r}; expected one of: "
            f"{supported}."
        )
    payload = f"greenonet-training-seed-v1:{stage}:{base_seed}:{namespace}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _configure_deterministic_algorithms(
    *,
    enabled: bool,
    device: torch.device,
) -> None:
    if not isinstance(enabled, bool):
        raise TypeError("deterministic_algorithms must be a boolean.")

    if enabled and device.type == "cuda":
        required = ":4096:8"
        configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if configured is None:
            is_cuda_initialized = cast(Callable[[], bool], torch.cuda.is_initialized)
            if is_cuda_initialized():
                raise RuntimeError(
                    "Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG to "
                    "be set before CUDA initialization. Restart the process and set "
                    "CUBLAS_WORKSPACE_CONFIG=:4096:8."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
        elif configured not in {":4096:8", ":16:8"}:
            raise ValueError(
                "Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG to be "
                f"':4096:8' or ':16:8', got {configured!r}."
            )

    torch.use_deterministic_algorithms(enabled, warn_only=False)
    torch.backends.cudnn.deterministic = enabled
    if enabled:
        torch.backends.cudnn.benchmark = False


def seed_global_generators(seed: int) -> None:
    """Apply one resolved seed to Python, NumPy, CPU Torch and all CUDA RNGs."""

    validate_training_seed(seed, field_name="seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python and NumPy from the worker seed assigned by DataLoader."""

    worker_seed = torch.initial_seed() % (MAX_TRAINING_SEED + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class TrainingSeedContext:
    """Namespaced RNG and deterministic-execution contract for one stage."""

    stage: TrainingStage
    base_seed: int
    deterministic_algorithms: bool
    device: str | torch.device

    def __post_init__(self) -> None:
        validate_training_seed(
            self.base_seed,
            field_name=f"{self.stage}_training.seed",
        )
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError("deterministic_algorithms must be a boolean.")
        object.__setattr__(self, "device", torch.device(self.device))

    @property
    def namespaces(self) -> tuple[str, ...]:
        return _STAGE_NAMESPACES[self.stage]

    def seed_for(self, namespace: str) -> int:
        return derive_training_subseed(
            self.base_seed,
            stage=self.stage,
            namespace=namespace,
        )

    def configure_process(self) -> None:
        _configure_deterministic_algorithms(
            enabled=self.deterministic_algorithms,
            device=torch.device(self.device),
        )

    def apply(self, namespace: str) -> int:
        resolved = self.seed_for(namespace)
        seed_global_generators(resolved)
        return resolved

    def make_generator(self, namespace: str) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed_for(namespace))
        return generator

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "stage": self.stage,
            "base_seed": self.base_seed,
            "deterministic_algorithms": self.deterministic_algorithms,
            "device": str(self.device),
            "subseeds": {
                namespace: self.seed_for(namespace) for namespace in self.namespaces
            },
            "hash_derivation": "sha256_uint32",
            "checkpoint_rng_state": "not_saved",
        }

    def log(self, logger: logging.Logger) -> None:
        resolved = self.as_dict()["subseeds"]
        assert isinstance(resolved, dict)
        seed_text = ",".join(
            f"{namespace}={resolved[namespace]}" for namespace in self.namespaces
        )
        logger.info(
            "training reproducibility stage=%s base_seed=%d "
            "deterministic_algorithms=%s device=%s subseeds=%s "
            "checkpoint_rng_state=not_saved",
            self.stage,
            self.base_seed,
            self.deterministic_algorithms,
            self.device,
            seed_text,
        )
