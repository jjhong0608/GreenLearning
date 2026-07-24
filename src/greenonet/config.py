from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

import torch

from greenonet.numerics import IntegrationRule


@dataclass
class DatasetConfig:
    """Sampling settings for synthetic Poisson data."""

    geometry_mode: Literal["unit_square", "complex"] = "unit_square"
    geometry_path: Optional[Path] = None
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
class CouplingBranchFusionConfig:
    """Branch feature fusion mode for CouplingNet."""

    mode: Literal["product", "product_fuser"] = "product"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("branch_fusion.mode must be a string.")
        if self.mode not in {"product", "product_fuser"}:
            raise ValueError("branch_fusion.mode must be 'product' or 'product_fuser'.")

    @classmethod
    def from_raw(
        cls,
        raw: CouplingBranchFusionConfig | dict[str, Any] | None,
    ) -> CouplingBranchFusionConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"mode"})
            if unknown:
                raise TypeError(
                    f"branch_fusion has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("branch_fusion must be an object.")


@dataclass
class CouplingTrunkPositionalEncodingConfig:
    """Optional deterministic features for CouplingNet trunk coordinates."""

    enabled: bool = False
    mode: Literal["fourier", "boundary_algebraic"] = "fourier"
    num_frequencies: int = 4
    max_frequency: float = 8.0
    include_input: bool = True


@dataclass
class TransverseTrunkConfig:
    """Optional pointwise cross-axis trunk settings for complex geometry."""

    enabled: bool = False
    fusion: Literal["product", "product_fuser"] = "product"
    length_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("axis_1d_trunk.transverse_trunk.enabled must be a boolean.")
        if self.fusion not in {"product", "product_fuser"}:
            raise ValueError(
                "axis_1d_trunk.transverse_trunk.fusion must be "
                "'product' or 'product_fuser'."
            )
        if not isinstance(self.length_context, bool):
            raise TypeError(
                "axis_1d_trunk.transverse_trunk.length_context must be a boolean."
            )

    @classmethod
    def from_raw(
        cls,
        raw: TransverseTrunkConfig | dict[str, Any] | None,
    ) -> TransverseTrunkConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "fusion", "length_context"})
            if unknown:
                raise TypeError(
                    "axis_1d_trunk.transverse_trunk has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("axis_1d_trunk.transverse_trunk must be an object.")


@dataclass
class Axis1DTrunkConfig:
    """Shared 1D trunk with boundary-aware transverse branch settings."""

    enabled: bool = False
    boundary_aware_modes: int = 4
    num_frequencies: int = 4
    max_frequency: float = 8.0
    transverse_trunk: TransverseTrunkConfig = field(
        default_factory=TransverseTrunkConfig
    )

    def __post_init__(self) -> None:
        self.transverse_trunk = TransverseTrunkConfig.from_raw(self.transverse_trunk)
        if not isinstance(self.enabled, bool):
            raise TypeError("axis_1d_trunk.enabled must be a boolean.")
        if not isinstance(self.boundary_aware_modes, int) or isinstance(
            self.boundary_aware_modes,
            bool,
        ):
            raise TypeError("axis_1d_trunk.boundary_aware_modes must be an integer.")
        if self.boundary_aware_modes <= 0:
            raise ValueError("axis_1d_trunk.boundary_aware_modes must be positive.")
        if not isinstance(self.num_frequencies, int) or isinstance(
            self.num_frequencies,
            bool,
        ):
            raise TypeError("axis_1d_trunk.num_frequencies must be an integer.")
        if self.num_frequencies <= 0:
            raise ValueError("axis_1d_trunk.num_frequencies must be positive.")
        if not isinstance(self.max_frequency, (int, float)) or isinstance(
            self.max_frequency,
            bool,
        ):
            raise TypeError("axis_1d_trunk.max_frequency must be numeric.")
        if self.max_frequency <= 0.0:
            raise ValueError("axis_1d_trunk.max_frequency must be positive.")

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
            unknown = sorted(
                set(data)
                - {
                    "enabled",
                    "boundary_aware_modes",
                    "num_frequencies",
                    "max_frequency",
                    "transverse_trunk",
                }
            )
            if unknown:
                raise TypeError(
                    f"axis_1d_trunk has unknown keys: {', '.join(unknown)}."
                )
            if "transverse_trunk" in data:
                data["transverse_trunk"] = TransverseTrunkConfig.from_raw(
                    data["transverse_trunk"]
                )
            return cls(**data)
        raise TypeError("axis_1d_trunk must be an object.")


@dataclass
class BalanceProjectionConfig:
    """CouplingNet output balance projection settings."""

    enabled: bool = True
    mode: Literal[
        "symmetric",
        "smooth_mask",
        "response_space",
        "physical_symmetric",
    ] = "symmetric"
    mask: Literal["quadratic", "sin"] = "quadratic"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("balance_projection.enabled must be a boolean.")
        mode = str(self.mode)
        if mode in {"geometry_weighted", "response_preconditioned"}:
            raise ValueError(
                f"balance_projection.mode='{mode}' has been removed from the "
                "complex output-contract path. Retrain ComplexCouplingNet "
                "with mode='physical_symmetric'."
            )
        if mode not in {
            "symmetric",
            "smooth_mask",
            "response_space",
            "physical_symmetric",
        }:
            raise ValueError(
                "balance_projection.mode must be 'symmetric', 'smooth_mask', "
                "'response_space', or 'physical_symmetric'."
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
                mode=cast(
                    Literal[
                        "symmetric",
                        "smooth_mask",
                        "response_space",
                        "physical_symmetric",
                    ],
                    raw,
                ),
            )
        if isinstance(raw, dict):
            data = dict(raw)
            retired_keys = {
                "geometry_weighted_rule",
                "geometry_weighted_lambda",
            }.intersection(data)
            if retired_keys:
                raise ValueError(
                    "Retired balance_projection fields have been removed: "
                    f"{', '.join(sorted(retired_keys))}."
                )
            unknown = sorted(
                set(data)
                - {
                    "enabled",
                    "mode",
                    "mask",
                }
            )
            if unknown:
                raise TypeError(
                    f"balance_projection has unknown keys: {', '.join(unknown)}."
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
                mode=cast(
                    Literal[
                        "symmetric",
                        "smooth_mask",
                        "response_space",
                        "physical_symmetric",
                    ],
                    mode,
                ),
                mask=cast(Literal["quadratic", "sin"], mask),
            )
        raise TypeError("balance_projection must be a string or an object.")


@dataclass
class ComplexPreProjectionFusionConfig:
    """Optional physical difference correction before complex projection."""

    enabled: bool = False
    nonlinear_hidden_dim: int = 16
    nonlinear_depth: int = 1
    gate_initial_value: float = 0.05
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("pre_projection_fusion.enabled must be a boolean.")
        if (
            isinstance(self.nonlinear_hidden_dim, bool)
            or not isinstance(self.nonlinear_hidden_dim, int)
            or self.nonlinear_hidden_dim < 1
        ):
            raise ValueError(
                "pre_projection_fusion.nonlinear_hidden_dim must be a positive integer."
            )
        if (
            isinstance(self.nonlinear_depth, bool)
            or not isinstance(self.nonlinear_depth, int)
            or self.nonlinear_depth < 1
        ):
            raise ValueError(
                "pre_projection_fusion.nonlinear_depth must be a positive integer."
            )
        gate = float(self.gate_initial_value)
        if not math.isfinite(gate) or not 0.0 < gate < 1.0:
            raise ValueError(
                "pre_projection_fusion.gate_initial_value must be finite and "
                "strictly between 0 and 1."
            )
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("pre_projection_fusion.eps must be finite and positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexPreProjectionFusionConfig | dict[str, Any] | None,
    ) -> ComplexPreProjectionFusionConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(
                set(data)
                - {
                    "enabled",
                    "nonlinear_hidden_dim",
                    "nonlinear_depth",
                    "gate_initial_value",
                    "eps",
                }
            )
            if unknown:
                raise TypeError(
                    f"pre_projection_fusion has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("pre_projection_fusion must be an object.")


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
        BalanceProjectionConfig
        | Literal[
            "symmetric",
            "smooth_mask",
            "response_space",
            "physical_symmetric",
        ]
        | dict[str, Any]
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
    branch_fusion: CouplingBranchFusionConfig | dict[str, Any] = field(
        default_factory=CouplingBranchFusionConfig
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
    pre_projection_fusion: ComplexPreProjectionFusionConfig | dict[str, Any] = field(
        default_factory=ComplexPreProjectionFusionConfig
    )

    def __post_init__(self) -> None:
        self.balance_projection = BalanceProjectionConfig.from_raw(
            self.balance_projection
        )
        self.branch_fusion = CouplingBranchFusionConfig.from_raw(self.branch_fusion)
        self.axis_1d_trunk = Axis1DTrunkConfig.from_raw(self.axis_1d_trunk)
        self.pre_projection_fusion = ComplexPreProjectionFusionConfig.from_raw(
            self.pre_projection_fusion
        )


@dataclass
class CompileConfig:
    """Optional torch.compile settings for model execution."""

    enabled: bool = False
    backend: str | None = None
    mode: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("compile.enabled must be a boolean.")
        for field_name, value in (("backend", self.backend), ("mode", self.mode)):
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"compile.{field_name} must be a string or null.")


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
class CouplingBestEnergyCheckpointConfig:
    """Best validation reference-free energy checkpoint settings."""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("best_energy_checkpoint.enabled must be a boolean.")

    @classmethod
    def from_raw(
        cls,
        raw: CouplingBestEnergyCheckpointConfig | dict[str, Any] | None,
    ) -> CouplingBestEnergyCheckpointConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled"})
            if unknown:
                raise TypeError(
                    f"best_energy_checkpoint has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("best_energy_checkpoint must be an object.")


@dataclass
class CouplingBestPhysicsCheckpointConfig:
    """Best validation reference-free total physics checkpoint settings."""

    enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("best_physics_checkpoint.enabled must be a boolean.")

    @classmethod
    def from_raw(
        cls,
        raw: CouplingBestPhysicsCheckpointConfig | dict[str, Any] | None,
    ) -> CouplingBestPhysicsCheckpointConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled"})
            if unknown:
                raise TypeError(
                    f"best_physics_checkpoint has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("best_physics_checkpoint must be an object.")


@dataclass
class ComplexRelativeSplitConsistencyConfig:
    """Source-normalized complex split energy and value consistency."""

    enabled: bool = False
    weight: float = 1.0
    mass_weight: float = 1.0
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("relative_split_consistency.enabled must be a boolean.")
        for field_name, value in (
            ("weight", self.weight),
            ("mass_weight", self.mass_weight),
            ("eps", self.eps),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"relative_split_consistency.{field_name} must be numeric."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"relative_split_consistency.{field_name} must be finite."
                )
        if self.weight < 0.0:
            raise ValueError("relative_split_consistency.weight must be non-negative.")
        if self.mass_weight < 0.0:
            raise ValueError(
                "relative_split_consistency.mass_weight must be non-negative."
            )
        if self.eps <= 0.0:
            raise ValueError("relative_split_consistency.eps must be positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexRelativeSplitConsistencyConfig | dict[str, Any] | None,
    ) -> ComplexRelativeSplitConsistencyConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "weight", "mass_weight", "eps"})
            if unknown:
                raise TypeError(
                    "relative_split_consistency has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("relative_split_consistency must be an object.")


@dataclass
class ComplexWeakOperatorClosureConfig:
    """Reference-free directional weak operator closure settings."""

    enabled: bool = False
    weight: float = 1.0
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("weak_operator_closure.enabled must be a boolean.")
        for field_name, value in (("weight", self.weight), ("eps", self.eps)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"weak_operator_closure.{field_name} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"weak_operator_closure.{field_name} must be finite.")
        if self.weight < 0.0:
            raise ValueError("weak_operator_closure.weight must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("weak_operator_closure.eps must be positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexWeakOperatorClosureConfig | dict[str, Any] | None,
    ) -> ComplexWeakOperatorClosureConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "weight", "eps"})
            if unknown:
                raise TypeError(
                    f"weak_operator_closure has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("weak_operator_closure must be an object.")


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
    best_energy_checkpoint: CouplingBestEnergyCheckpointConfig | dict[str, Any] = field(
        default_factory=CouplingBestEnergyCheckpointConfig
    )
    best_physics_checkpoint: CouplingBestPhysicsCheckpointConfig | dict[str, Any] = (
        field(default_factory=CouplingBestPhysicsCheckpointConfig)
    )
    relative_split_consistency: (
        ComplexRelativeSplitConsistencyConfig | dict[str, Any]
    ) = field(default_factory=ComplexRelativeSplitConsistencyConfig)
    weak_operator_closure: ComplexWeakOperatorClosureConfig | dict[str, Any] = field(
        default_factory=ComplexWeakOperatorClosureConfig
    )

    def __post_init__(self) -> None:
        self.best_energy_checkpoint = CouplingBestEnergyCheckpointConfig.from_raw(
            self.best_energy_checkpoint
        )
        self.best_physics_checkpoint = CouplingBestPhysicsCheckpointConfig.from_raw(
            self.best_physics_checkpoint
        )
        self.relative_split_consistency = (
            ComplexRelativeSplitConsistencyConfig.from_raw(
                self.relative_split_consistency
            )
        )
        self.weak_operator_closure = ComplexWeakOperatorClosureConfig.from_raw(
            self.weak_operator_closure
        )


def validate_unit_square_coupling_training_config(
    config: CouplingTrainingConfig,
) -> None:
    """Reject complex-only training options on the unit-square path."""

    if ComplexRelativeSplitConsistencyConfig.from_raw(
        config.relative_split_consistency
    ).enabled:
        raise ValueError(
            "coupling_training.relative_split_consistency is available only for "
            "ComplexCouplingTrainer."
        )
    if ComplexWeakOperatorClosureConfig.from_raw(config.weak_operator_closure).enabled:
        raise ValueError(
            "coupling_training.weak_operator_closure is available only for "
            "ComplexCouplingTrainer."
        )
    if CouplingBestPhysicsCheckpointConfig.from_raw(
        config.best_physics_checkpoint
    ).enabled:
        raise ValueError(
            "coupling_training.best_physics_checkpoint is available only for "
            "ComplexCouplingTrainer."
        )


def reject_retired_coupling_training_options(raw: dict[str, Any]) -> None:
    """Reject removed training options before dataclass construction."""

    if "length_jump_balance" in raw:
        raise TypeError(
            "coupling_training.length_jump_balance has been removed. "
            "Complex CouplingNet now always uses the full-domain canonical "
            "bulk-plus-boundary energy; remove this config block."
        )


@dataclass
class GreenQuadratureConfig:
    """GreenNet-only reconstruction quadrature settings."""

    enabled: bool = False
    rule: Literal["split_gauss_legendre"] = "split_gauss_legendre"
    order: int = 4
    source_sampling_factor: int = 1
    source_interpolation: Literal["linear", "cubic"] = "linear"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("green_quadrature.enabled must be a boolean.")
        if self.rule != "split_gauss_legendre":
            raise ValueError("green_quadrature.rule must be 'split_gauss_legendre'.")
        if not isinstance(self.order, int) or isinstance(self.order, bool):
            raise TypeError("green_quadrature.order must be an integer.")
        if self.order <= 0:
            raise ValueError("green_quadrature.order must be positive.")
        if not isinstance(self.source_sampling_factor, int) or isinstance(
            self.source_sampling_factor,
            bool,
        ):
            raise TypeError(
                "green_quadrature.source_sampling_factor must be an integer."
            )
        if self.source_sampling_factor < 1:
            raise ValueError(
                "green_quadrature.source_sampling_factor must be positive."
            )
        if self.source_interpolation not in {"linear", "cubic"}:
            raise ValueError(
                "green_quadrature.source_interpolation must be 'linear' or 'cubic'."
            )

    @classmethod
    def from_raw(
        cls,
        raw: GreenQuadratureConfig | dict[str, Any] | None,
    ) -> GreenQuadratureConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(
                set(data)
                - {
                    "enabled",
                    "rule",
                    "order",
                    "source_sampling_factor",
                    "source_interpolation",
                }
            )
            if unknown:
                raise TypeError(
                    f"green_quadrature has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("green_quadrature must be an object.")


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
    green_quadrature: GreenQuadratureConfig = field(
        default_factory=GreenQuadratureConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)
    lbfgs_max_iter: int = 0
    lbfgs_history_size: int = 10
    lbfgs_lr: float = 1.0
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_epochs: int = 1

    def __post_init__(self) -> None:
        self.green_quadrature = GreenQuadratureConfig.from_raw(self.green_quadrature)


@dataclass
class PipelineConfig:
    """Control flags for training pipelines."""

    run_green: bool = True
    run_coupling: bool = False
    green_pretrained_path: Optional[Path] = None
    coupling_pretrained_path: Optional[Path] = None
