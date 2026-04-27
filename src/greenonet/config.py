from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import torch

from greenonet.numerics import IntegrationRule


@dataclass
class DatasetConfig:
    """Sampling settings for synthetic Poisson data."""

    step_size: float = 0.25
    n_points_per_line: int | None = None
    sampler_mode: Literal["forward", "backward"] = "forward"
    validation_sampler_mode: Literal["forward", "backward"] | None = None
    samples_per_line: int = 2
    validation_samples_per_line: int = 0
    scale_length: float | tuple[float, float] = 0.1
    validation_scale_length: float | tuple[float, float] | None = None
    deterministic: bool = True
    use_operator_learning: bool = True
    training_path: Optional[Path] = None
    validation_path: Optional[Path] = None
    test_path: Optional[Path] = None
    dtype: torch.dtype = torch.float64


@dataclass
class ModelConfig:
    """Neural network architecture settings."""

    input_dim: int = 2
    hidden_dim: int = 64
    depth: int = 4
    activation: Literal["tanh", "relu", "gelu", "rational"] = "tanh"
    use_bias: bool = True
    dropout: float = 0.0
    use_green: bool = True
    branch_input_dim: int = 4
    use_fourier: bool = False
    fourier_dim: int = 16
    fourier_scale: float = 1.0
    fourier_include_input: bool = False
    dtype: torch.dtype = torch.float64


@dataclass
class CouplerConfig:
    """Optional local 2D coupler inserted before CouplingNet balance projection."""

    enabled: bool = False
    type: Literal["five_stencil_stencil_mlp"] = "five_stencil_stencil_mlp"
    hidden_channels: int = 64
    depth: int = 2
    activation: Literal["tanh", "relu", "gelu", "rational"] = "gelu"
    dropout: float = 0.0
    residual_scale_init: float = 0.05
    padding: Literal["replicate", "zero"] = "replicate"
    eps: float = 1.0e-12


@dataclass
class CouplingModelConfig:
    """Architecture settings for CouplingNet."""

    branch_input_dim: int = 4  # number of samples per line
    trunk_input_dim: int = 2  # full (x, y) coordinates
    hidden_dim: int = 64
    depth: int = 4
    activation: Literal["tanh", "relu", "gelu", "rational"] = "tanh"
    use_bias: bool = True
    dropout: float = 0.0
    dtype: torch.dtype = torch.float64
    coupler: CouplerConfig = field(default_factory=CouplerConfig)


@dataclass
class CompileConfig:
    """Optional torch.compile settings for model execution."""

    enabled: bool = False


@dataclass
class CouplingLossTermConfig:
    """Single CouplingNet loss toggle and weight."""

    enabled: bool = True
    weight: float = 1.0


@dataclass
class CouplingLossesConfig:
    """Nested CouplingNet loss settings."""

    l2_consistency: CouplingLossTermConfig = field(
        default_factory=CouplingLossTermConfig
    )
    energy_consistency: CouplingLossTermConfig = field(
        default_factory=CouplingLossTermConfig
    )
    cross_consistency: CouplingLossTermConfig = field(
        default_factory=CouplingLossTermConfig
    )


@dataclass
class CouplingPeriodicCheckpointConfig:
    """Periodic checkpoint settings for CouplingNet Adam training."""

    enabled: bool = False
    every_epochs: int = 0


@dataclass
class CouplingBestRelSolCheckpointConfig:
    """Best validation rel_sol checkpoint settings for Adam training."""

    enabled: bool = False


@dataclass
class CouplingTrainingConfig:
    """Training settings for CouplingNet."""

    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 4
    log_interval: int = 1
    device: str = "cpu"
    losses: CouplingLossesConfig = field(default_factory=CouplingLossesConfig)
    use_lr_schedule: bool = False
    warmup_epochs: int = 0
    min_lr: float = 1e-6
    integration_rule: IntegrationRule = "simpson"
    compile: CompileConfig = field(default_factory=CompileConfig)
    periodic_checkpoint: CouplingPeriodicCheckpointConfig = field(
        default_factory=CouplingPeriodicCheckpointConfig
    )
    best_rel_sol_checkpoint: CouplingBestRelSolCheckpointConfig = field(
        default_factory=CouplingBestRelSolCheckpointConfig
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    log_interval: int = 1
    device: str = "cpu"
    compute_validation_rel_sol: bool = False
    integration_rule: IntegrationRule = "simpson"
    compile: CompileConfig = field(default_factory=CompileConfig)
    lbfgs_max_iter: int = 0
    lbfgs_history_size: int = 10
    lbfgs_lr: float = 1.0
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_epochs: int = 1


@dataclass
class PipelineConfig:
    """Control flags for training pipelines."""

    run_green: bool = True
    run_coupling: bool = False
    green_pretrained_path: Optional[Path] = None
    coupling_pretrained_path: Optional[Path] = None
