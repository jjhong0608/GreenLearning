from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

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
    coefficient_functions_path: Optional[Path] = None
    dtype: torch.dtype = torch.float64


@dataclass
class TerminalConfig:
    """Terminal rendering settings for Rich console logs."""

    width: int | None = None

    def __post_init__(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("terminal.width must be positive or null.")


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
class SourceStencilLiftConfig:
    """Optional input-side learned source lift for CouplingNet."""

    enabled: bool = False
    encoder_type: Literal["linear", "mlp", "MLP"] = "mlp"
    coefficient_normalization: Literal["rms", "tanh"] = "rms"
    coefficient_tanh_beta: float = 1.0
    hidden_dim: int = 32
    depth: int = 2
    activation: Literal["tanh", "relu", "gelu", "rational"] = "gelu"
    use_bias: bool = True
    dropout: float = 0.0
    use_g_normalization: bool = True
    eps: float = 1.0e-12


@dataclass
class GreenResponseFeatureConfig:
    """Optional axial Green response feature for CouplingNet source branch."""

    enabled: bool = False


@dataclass
class CouplingCoefficientTermsConfig:
    """Coefficient terms used by the standard CouplingNet branch path."""

    diffusion: bool = True
    convection: bool = False
    reaction: bool = False


@dataclass
class CouplingTrunkPositionalEncodingConfig:
    """Optional deterministic features for CouplingNet trunk coordinates."""

    enabled: bool = False
    mode: Literal["fourier", "boundary_algebraic"] = "fourier"
    num_frequencies: int = 4
    max_frequency: float = 8.0
    include_input: bool = True


@dataclass
class Axis1DTrunkConfig:
    """Shared 1D trunk with boundary-aware transverse branch settings."""

    enabled: bool = False
    boundary_aware_modes: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("axis_1d_trunk.enabled must be a boolean.")
        if not isinstance(self.boundary_aware_modes, int) or isinstance(
            self.boundary_aware_modes,
            bool,
        ):
            raise TypeError("axis_1d_trunk.boundary_aware_modes must be an integer.")
        if self.boundary_aware_modes <= 0:
            raise ValueError("axis_1d_trunk.boundary_aware_modes must be positive.")

    @classmethod
    def from_raw(
        cls,
        raw: Axis1DTrunkConfig | dict[str, Any] | None,
    ) -> Axis1DTrunkConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "boundary_aware_modes"})
            if unknown:
                raise TypeError(
                    "axis_1d_trunk has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("axis_1d_trunk must be an object.")


@dataclass
class BalanceProjectionConfig:
    """CouplingNet output balance projection settings."""

    enabled: bool = True
    mode: Literal["symmetric", "smooth_mask"] = "symmetric"
    mask: Literal["quadratic", "sin"] = "quadratic"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("balance_projection.enabled must be a boolean.")
        if self.mode not in {"symmetric", "smooth_mask"}:
            raise ValueError(
                "balance_projection.mode must be 'symmetric' or 'smooth_mask'."
            )
        if self.mask not in {"quadratic", "sin"}:
            raise ValueError("balance_projection.mask must be 'quadratic' or 'sin'.")

    @classmethod
    def from_raw(
        cls,
        raw: BalanceProjectionConfig | str | dict[str, Any] | None,
    ) -> BalanceProjectionConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            return cls(
                enabled=True,
                mode=cast(Literal["symmetric", "smooth_mask"], raw),
            )
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "mode", "mask"})
            if unknown:
                raise TypeError(
                    "balance_projection has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            enabled = data.get("enabled", True)
            mode = data.get("mode", "symmetric")
            mask = data.get("mask", "quadratic")
            if not isinstance(enabled, bool):
                raise TypeError("balance_projection.enabled must be a boolean.")
            if not isinstance(mode, str):
                raise TypeError("balance_projection.mode must be a string.")
            if not isinstance(mask, str):
                raise TypeError("balance_projection.mask must be a string.")
            return cls(
                enabled=enabled,
                mode=cast(Literal["symmetric", "smooth_mask"], mode),
                mask=cast(Literal["quadratic", "sin"], mask),
            )
        raise TypeError("balance_projection must be a string or an object.")


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
    balance_projection: (
        BalanceProjectionConfig | Literal["symmetric", "smooth_mask"] | dict[str, Any]
    ) = field(default_factory=BalanceProjectionConfig)
    smooth_mask_normalize: bool = True
    smooth_mask_eps: float = 1.0e-12
    smooth_mask_power: float = 1.0
    smooth_mask_diff_power: float = 1.0
    smooth_mask_diff_power_trainable: bool = False
    smooth_mask_diff_power_min: float = 0.25
    smooth_mask_diff_power_max: float = 2.0
    source_stencil_lift: SourceStencilLiftConfig = field(
        default_factory=SourceStencilLiftConfig
    )
    coefficient_terms: CouplingCoefficientTermsConfig = field(
        default_factory=CouplingCoefficientTermsConfig
    )
    green_response_feature: GreenResponseFeatureConfig = field(
        default_factory=GreenResponseFeatureConfig
    )
    trunk_positional_encoding: CouplingTrunkPositionalEncodingConfig = field(
        default_factory=CouplingTrunkPositionalEncodingConfig
    )
    axis_1d_trunk: Axis1DTrunkConfig | dict[str, Any] = field(
        default_factory=Axis1DTrunkConfig
    )

    def __post_init__(self) -> None:
        self.balance_projection = BalanceProjectionConfig.from_raw(
            self.balance_projection
        )
        self.axis_1d_trunk = Axis1DTrunkConfig.from_raw(self.axis_1d_trunk)


@dataclass
class CompileConfig:
    """Optional torch.compile settings for model execution."""

    enabled: bool = False


@dataclass
class CouplingLossTermConfig:
    """Single CouplingNet loss toggle and weight."""

    enabled: bool = True
    weight: float = 1.0


def _disabled_loss_term_config() -> CouplingLossTermConfig:
    return CouplingLossTermConfig(enabled=False, weight=1.0)


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
    balance_loss: CouplingLossTermConfig = field(
        default_factory=_disabled_loss_term_config
    )
    symmetric_boundary_loss: CouplingLossTermConfig = field(
        default_factory=_disabled_loss_term_config
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
    source_stencil_lift_learning_rate: float | None = None
    weight_decay: float = 0.0
    source_stencil_lift_weight_decay: float | None = None
    gradient_clip_max_norm: float | None = 1.0
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
