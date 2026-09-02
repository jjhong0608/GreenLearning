from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

import torch

from greenonet.numerics import IntegrationRule


def _validate_training_step_fields(
    *,
    field_prefix: str,
    warmup_epochs: object,
    warmup_steps: object,
    validation_every_steps: object,
) -> None:
    if not isinstance(warmup_epochs, int) or isinstance(warmup_epochs, bool):
        raise TypeError(f"{field_prefix}.warmup_epochs must be an integer.")
    if warmup_epochs < 0:
        raise ValueError(f"{field_prefix}.warmup_epochs must be non-negative.")
    if warmup_steps is not None:
        if not isinstance(warmup_steps, int) or isinstance(warmup_steps, bool):
            raise TypeError(f"{field_prefix}.warmup_steps must be an integer or null.")
        if warmup_steps < 0:
            raise ValueError(f"{field_prefix}.warmup_steps must be non-negative.")
        if warmup_epochs > 0:
            raise ValueError(
                f"{field_prefix}.warmup_steps and a positive warmup_epochs "
                "cannot be configured together."
            )
    if validation_every_steps is not None:
        if not isinstance(validation_every_steps, int) or isinstance(
            validation_every_steps, bool
        ):
            raise TypeError(
                f"{field_prefix}.validation_every_steps must be an integer or null."
            )
        if validation_every_steps <= 0:
            raise ValueError(f"{field_prefix}.validation_every_steps must be positive.")


@dataclass(frozen=True)
class IndexedGpSourceConfig:
    """Fixed index-seeded GP source settings for complex CouplingNet."""

    num_train: int
    num_valid: int
    seed: int = 0
    lengthscale: float = 0.2
    amplitude: float = 1.0
    mean: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value, minimum in (
            ("num_train", self.num_train, 1),
            ("num_valid", self.num_valid, 0),
            ("seed", self.seed, 0),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"indexed_gp.{field_name} must be an integer.")
            if value < minimum:
                raise ValueError(f"indexed_gp.{field_name} must be >= {minimum}.")
        for field_name, numeric_value in (
            ("lengthscale", self.lengthscale),
            ("amplitude", self.amplitude),
            ("mean", self.mean),
        ):
            if not isinstance(numeric_value, (int, float)) or isinstance(
                numeric_value, bool
            ):
                raise TypeError(f"indexed_gp.{field_name} must be numeric.")
            if not math.isfinite(float(numeric_value)):
                raise ValueError(f"indexed_gp.{field_name} must be finite.")
        if self.lengthscale <= 0.0:
            raise ValueError("indexed_gp.lengthscale must be positive.")
        if self.amplitude < 0.0:
            raise ValueError("indexed_gp.amplitude must be non-negative.")

    @classmethod
    def from_raw(
        cls,
        raw: IndexedGpSourceConfig | dict[str, Any] | None,
    ) -> IndexedGpSourceConfig | None:
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("dataset.coupling_source.indexed_gp must be an object.")
        data = dict(raw)
        allowed = {
            "num_train",
            "num_valid",
            "seed",
            "lengthscale",
            "amplitude",
            "mean",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise TypeError(
                "dataset.coupling_source.indexed_gp has unknown keys: "
                f"{', '.join(unknown)}."
            )
        return cls(**data)


@dataclass(frozen=True)
class ComplexCouplingSourceConfig:
    """Source backend used by complex CouplingNet train/validation splits."""

    mode: Literal["npz", "indexed_gp"] = "npz"
    indexed_gp: IndexedGpSourceConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("dataset.coupling_source.mode must be a string.")
        if self.mode not in {"npz", "indexed_gp"}:
            raise ValueError(
                "dataset.coupling_source.mode must be 'npz' or 'indexed_gp'."
            )
        indexed_gp = IndexedGpSourceConfig.from_raw(self.indexed_gp)
        object.__setattr__(self, "indexed_gp", indexed_gp)
        if self.mode == "npz" and indexed_gp is not None:
            raise ValueError(
                "dataset.coupling_source.indexed_gp is unused when mode='npz'."
            )
        if self.mode == "indexed_gp" and indexed_gp is None:
            raise ValueError(
                "dataset.coupling_source.indexed_gp is required when mode='indexed_gp'."
            )

    @classmethod
    def from_raw(
        cls,
        raw: ComplexCouplingSourceConfig | dict[str, Any] | None,
    ) -> ComplexCouplingSourceConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("dataset.coupling_source must be an object.")
        data = dict(raw)
        unknown = sorted(set(data) - {"mode", "indexed_gp"})
        if unknown:
            raise TypeError(
                f"dataset.coupling_source has unknown keys: {', '.join(unknown)}."
            )
        return cls(**data)


@dataclass(frozen=True)
class ComplexReferenceDiagnosticsConfig:
    """Reference metric policy for complex train/validation splits."""

    training: bool = True
    validation: bool = True

    def __post_init__(self) -> None:
        for field_name, value in (
            ("training", self.training),
            ("validation", self.validation),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"dataset.reference_diagnostics.{field_name} must be a boolean."
                )

    @classmethod
    def from_raw(
        cls,
        raw: ComplexReferenceDiagnosticsConfig | dict[str, Any] | None,
    ) -> ComplexReferenceDiagnosticsConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("dataset.reference_diagnostics must be an object.")
        data = dict(raw)
        unknown = sorted(set(data) - {"training", "validation"})
        if unknown:
            raise TypeError(
                f"dataset.reference_diagnostics has unknown keys: {', '.join(unknown)}."
            )
        return cls(**data)


@dataclass
class DatasetConfig:
    """Sampling and stored-data settings."""

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
    coupling_source: ComplexCouplingSourceConfig = field(
        default_factory=ComplexCouplingSourceConfig
    )
    reference_diagnostics: ComplexReferenceDiagnosticsConfig = field(
        default_factory=ComplexReferenceDiagnosticsConfig
    )
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.geometry_mode not in {"unit_square", "complex"}:
            raise ValueError(
                "dataset.geometry_mode must be 'unit_square' or 'complex'."
            )
        for field_name in (
            "geometry_path",
            "training_path",
            "validation_path",
            "test_path",
            "coefficient_functions_path",
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Path):
                setattr(self, field_name, Path(value))
        self.scale_length = self._parse_scale(self.scale_length, "scale_length")
        if self.validation_scale_length is not None:
            self.validation_scale_length = self._parse_scale(
                self.validation_scale_length,
                "validation_scale_length",
            )
        if isinstance(self.dtype, str):
            self.dtype = self._parse_dtype(self.dtype)
        if self.dtype not in {torch.float32, torch.float64}:
            raise ValueError("dataset.dtype must be float32 or float64.")
        self.coupling_source = ComplexCouplingSourceConfig.from_raw(
            self.coupling_source
        )
        self.reference_diagnostics = ComplexReferenceDiagnosticsConfig.from_raw(
            self.reference_diagnostics
        )

    @classmethod
    def from_raw(
        cls,
        raw: DatasetConfig | dict[str, Any] | None,
    ) -> DatasetConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("dataset must be an object.")
        data = dict(raw)
        data.pop("domain", None)
        return cls(**data)

    @staticmethod
    def _parse_dtype(value: str) -> torch.dtype:
        if value == "float32":
            return torch.float32
        if value == "float64":
            return torch.float64
        raise ValueError("dataset.dtype must be 'float32' or 'float64'.")

    @staticmethod
    def _parse_scale(
        value: float | tuple[float, float] | list[float],
        field_name: str,
    ) -> float | tuple[float, float]:
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"dataset.{field_name} list must have two values.")
            return (float(value[0]), float(value[1]))
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"dataset.{field_name} tuple must have two values.")
            return (float(value[0]), float(value[1]))
        return float(value)


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

    mode: Literal["product", "product_fuser", "concat_fuser"] = "product"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str):
            raise TypeError("branch_fusion.mode must be a string.")
        if self.mode not in {"product", "product_fuser", "concat_fuser"}:
            raise ValueError(
                "branch_fusion.mode must be 'product', 'product_fuser', or "
                "'concat_fuser'."
            )

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
    fusion: Literal["product", "product_fuser", "concat_fuser"] = "product"
    length_context: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("axis_1d_trunk.transverse_trunk.enabled must be a boolean.")
        if self.fusion not in {"product", "product_fuser", "concat_fuser"}:
            raise ValueError(
                "axis_1d_trunk.transverse_trunk.fusion must be "
                "'product', 'product_fuser', or 'concat_fuser'."
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
class FixedLineTransverseBranchConfig:
    """Optional fixed-line transverse branch settings for complex geometry."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "axis_1d_trunk.fixed_line_transverse_branch.enabled must be a boolean."
            )

    @classmethod
    def from_raw(
        cls,
        raw: FixedLineTransverseBranchConfig | dict[str, Any] | None,
    ) -> FixedLineTransverseBranchConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled"})
            if unknown:
                raise TypeError(
                    "axis_1d_trunk.fixed_line_transverse_branch has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("axis_1d_trunk.fixed_line_transverse_branch must be an object.")


@dataclass
class Axis1DTrunkConfig:
    """Shared 1D trunk with boundary-aware transverse branch settings."""

    enabled: bool = False
    boundary_aware_modes: int = 4
    num_frequencies: int = 4
    max_frequency: float = 8.0
    fixed_line_transverse_branch: FixedLineTransverseBranchConfig = field(
        default_factory=FixedLineTransverseBranchConfig
    )
    transverse_trunk: TransverseTrunkConfig = field(
        default_factory=TransverseTrunkConfig
    )

    def __post_init__(self) -> None:
        self.fixed_line_transverse_branch = FixedLineTransverseBranchConfig.from_raw(
            self.fixed_line_transverse_branch
        )
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
                    "fixed_line_transverse_branch",
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
            if "fixed_line_transverse_branch" in data:
                data["fixed_line_transverse_branch"] = (
                    FixedLineTransverseBranchConfig.from_raw(
                        data["fixed_line_transverse_branch"]
                    )
                )
            return cls(**data)
        raise TypeError("axis_1d_trunk must be an object.")


@dataclass
class ColumnDiagonalGreenResponseProjectionConfig:
    """Column-diagonal Green-response projection settings."""

    gain_squared_eps: float = 1.0e-12
    gain_exponent: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.gain_squared_eps, (int, float)) or isinstance(
            self.gain_squared_eps,
            bool,
        ):
            raise TypeError(
                "balance_projection.column_diagonal_green_response."
                "gain_squared_eps must be numeric."
            )
        if (
            not math.isfinite(float(self.gain_squared_eps))
            or float(self.gain_squared_eps) <= 0.0
        ):
            raise ValueError(
                "balance_projection.column_diagonal_green_response."
                "gain_squared_eps must be finite and positive."
            )
        if not isinstance(self.gain_exponent, (int, float)) or isinstance(
            self.gain_exponent,
            bool,
        ):
            raise TypeError(
                "balance_projection.column_diagonal_green_response."
                "gain_exponent must be numeric."
            )
        if (
            not math.isfinite(float(self.gain_exponent))
            or float(self.gain_exponent) < 0.0
            or float(self.gain_exponent) > 1.0
        ):
            raise ValueError(
                "balance_projection.column_diagonal_green_response."
                "gain_exponent must be finite and in [0, 1]."
            )

    @classmethod
    def from_raw(
        cls,
        raw: ColumnDiagonalGreenResponseProjectionConfig | dict[str, Any] | None,
    ) -> ColumnDiagonalGreenResponseProjectionConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"gain_squared_eps", "gain_exponent"})
            if unknown:
                raise TypeError(
                    "balance_projection.column_diagonal_green_response has unknown "
                    f"keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError(
            "balance_projection.column_diagonal_green_response must be an object."
        )


@dataclass
class GeometryKSelectionConfig:
    """Geometry-only tangent-subspace dimension selection settings."""

    enabled: bool = False
    global_reach_threshold: float = 0.99
    pointwise_tail_reach_threshold: float = 0.99

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "geometry_k_selection.enabled must be a boolean."
            )
        for name, value in (
            ("global_reach_threshold", self.global_reach_threshold),
            (
                "pointwise_tail_reach_threshold",
                self.pointwise_tail_reach_threshold,
            ),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    "balance_projection.symmetric_tangent_green_response."
                    f"geometry_k_selection.{name} must be numeric."
                )
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(
                    "balance_projection.symmetric_tangent_green_response."
                    f"geometry_k_selection.{name} must be finite and in (0, 1]."
                )

    @classmethod
    def from_raw(
        cls,
        raw: GeometryKSelectionConfig | dict[str, Any] | None,
    ) -> GeometryKSelectionConfig:
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
                    "global_reach_threshold",
                    "pointwise_tail_reach_threshold",
                }
            )
            if unknown:
                raise TypeError(
                    "balance_projection.symmetric_tangent_green_response."
                    "geometry_k_selection has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError(
            "balance_projection.symmetric_tangent_green_response."
            "geometry_k_selection must be an object."
        )


@dataclass
class SymmetricTangentGreenResponseProjectionConfig:
    """Matrix-free tangent Green-response correction settings."""

    subspace_dimension: int = 1
    max_subspace_dimension: int = 8
    geometry_k_selection: GeometryKSelectionConfig | dict[str, Any] = field(
        default_factory=GeometryKSelectionConfig
    )
    eta: float = 0.01
    eta_cap_enabled: bool = True
    eta_strategy: Literal["fixed", "closed_loop_exact_line_search"] = "fixed"
    line_search_relative_eps: float = 1.0e-12
    relative_lambda: float = 0.01
    denominator_relative_eps: float = 1.0e-12
    preconditioner_variant: Literal[
        "separable",
        "exact_diagonal",
        "absolute_cross_axis",
        "normalized_quadratic_cross_axis",
    ] = "separable"
    cross_axis_relative_eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if isinstance(self.subspace_dimension, bool) or not isinstance(
            self.subspace_dimension, int
        ):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "subspace_dimension must be an integer."
            )
        if self.subspace_dimension < 1:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "subspace_dimension must be positive."
            )
        if isinstance(self.max_subspace_dimension, bool) or not isinstance(
            self.max_subspace_dimension, int
        ):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "max_subspace_dimension must be an integer."
            )
        if self.max_subspace_dimension < 1:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "max_subspace_dimension must be positive."
            )
        if self.subspace_dimension > self.max_subspace_dimension:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "subspace_dimension must not exceed max_subspace_dimension."
            )
        self.geometry_k_selection = GeometryKSelectionConfig.from_raw(
            self.geometry_k_selection
        )
        self._validate_nonnegative("eta", self.eta)
        if not isinstance(self.eta_cap_enabled, bool):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "eta_cap_enabled must be a boolean."
            )
        if not isinstance(self.eta_strategy, str):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "eta_strategy must be a string."
            )
        if self.eta_strategy not in {"fixed", "closed_loop_exact_line_search"}:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "eta_strategy must be 'fixed' or "
                "'closed_loop_exact_line_search'."
            )
        if (
            self.subspace_dimension >= 2
            and self.eta_strategy != "closed_loop_exact_line_search"
        ):
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "subspace_dimension>=2 requires "
                "eta_strategy='closed_loop_exact_line_search'."
            )
        if (
            not self.eta_cap_enabled
            and self.eta_strategy != "closed_loop_exact_line_search"
        ):
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "eta_cap_enabled=false requires "
                "eta_strategy='closed_loop_exact_line_search'."
            )
        self._validate_positive(
            "line_search_relative_eps",
            self.line_search_relative_eps,
        )
        self._validate_nonnegative("relative_lambda", self.relative_lambda)
        self._validate_positive(
            "denominator_relative_eps",
            self.denominator_relative_eps,
        )
        if not isinstance(self.preconditioner_variant, str):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                "preconditioner_variant must be a string."
            )
        allowed = {
            "separable",
            "exact_diagonal",
            "absolute_cross_axis",
            "normalized_quadratic_cross_axis",
        }
        if self.preconditioner_variant not in allowed:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                "preconditioner_variant must be 'separable', 'exact_diagonal', "
                "'absolute_cross_axis', or 'normalized_quadratic_cross_axis'."
            )
        self._validate_positive(
            "cross_axis_relative_eps",
            self.cross_axis_relative_eps,
        )

    @staticmethod
    def _validate_nonnegative(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                f"{name} must be numeric."
            )
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                f"{name} must be finite and non-negative."
            )

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                "balance_projection.symmetric_tangent_green_response."
                f"{name} must be numeric."
            )
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(
                "balance_projection.symmetric_tangent_green_response."
                f"{name} must be finite and positive."
            )

    @classmethod
    def from_raw(
        cls,
        raw: SymmetricTangentGreenResponseProjectionConfig | dict[str, Any] | None,
    ) -> SymmetricTangentGreenResponseProjectionConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(
                set(data)
                - {
                    "subspace_dimension",
                    "max_subspace_dimension",
                    "geometry_k_selection",
                    "eta",
                    "eta_cap_enabled",
                    "eta_strategy",
                    "line_search_relative_eps",
                    "relative_lambda",
                    "denominator_relative_eps",
                    "preconditioner_variant",
                    "cross_axis_relative_eps",
                }
            )
            if unknown:
                raise TypeError(
                    "balance_projection.symmetric_tangent_green_response has "
                    f"unknown keys: {', '.join(unknown)}."
                )
            geometry_k_selection = GeometryKSelectionConfig.from_raw(
                data.pop("geometry_k_selection", None)
            )
            return cls(geometry_k_selection=geometry_k_selection, **data)
        raise TypeError(
            "balance_projection.symmetric_tangent_green_response must be an object."
        )


@dataclass
class BalanceProjectionConfig:
    """CouplingNet output balance projection settings."""

    enabled: bool = True
    mode: Literal[
        "symmetric",
        "smooth_mask",
        "response_space",
        "physical_symmetric",
        "column_diagonal_green_response",
        "symmetric_tangent_green_response",
    ] = "symmetric"
    mask: Literal["quadratic", "sin"] = "quadratic"
    column_diagonal_green_response: (
        ColumnDiagonalGreenResponseProjectionConfig | dict[str, Any]
    ) = field(default_factory=ColumnDiagonalGreenResponseProjectionConfig)
    symmetric_tangent_green_response: (
        SymmetricTangentGreenResponseProjectionConfig | dict[str, Any]
    ) = field(default_factory=SymmetricTangentGreenResponseProjectionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("balance_projection.enabled must be a boolean.")
        self.column_diagonal_green_response = (
            ColumnDiagonalGreenResponseProjectionConfig.from_raw(
                self.column_diagonal_green_response
            )
        )
        self.symmetric_tangent_green_response = (
            SymmetricTangentGreenResponseProjectionConfig.from_raw(
                self.symmetric_tangent_green_response
            )
        )
        mode = str(self.mode)
        if mode in {"geometry_weighted", "response_preconditioned"}:
            raise ValueError(
                f"balance_projection.mode='{mode}' has been removed from the "
                "complex output-contract path. Retrain ComplexCouplingNet "
                "with mode='physical_symmetric'."
            )
        if mode in {"row_norm", "green_response_row_norm", "row_norm_green_response"}:
            raise ValueError(
                "Row-norm Green-response projection is not supported. Use "
                "mode='column_diagonal_green_response', which measures source-column "
                "response cost."
            )
        if mode not in {
            "symmetric",
            "smooth_mask",
            "response_space",
            "physical_symmetric",
            "column_diagonal_green_response",
            "symmetric_tangent_green_response",
        }:
            raise ValueError(
                "balance_projection.mode must be 'symmetric', 'smooth_mask', "
                "'response_space', 'physical_symmetric', or "
                "'column_diagonal_green_response', or "
                "'symmetric_tangent_green_response'."
            )
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            self.symmetric_tangent_green_response
        )
        geometry_k_selection = GeometryKSelectionConfig.from_raw(
            tangent.geometry_k_selection
        )
        if geometry_k_selection.enabled and mode != "symmetric_tangent_green_response":
            raise ValueError(
                "geometry_k_selection.enabled=true requires "
                "balance_projection.mode='symmetric_tangent_green_response'."
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
                        "column_diagonal_green_response",
                        "symmetric_tangent_green_response",
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
                    "column_diagonal_green_response",
                    "symmetric_tangent_green_response",
                }
            )
            if unknown:
                raise TypeError(
                    f"balance_projection has unknown keys: {', '.join(unknown)}."
                )
            enabled = data.get("enabled", True)
            mode = data.get("mode", "symmetric")
            mask = data.get("mask", "quadratic")
            column_diagonal = data.get("column_diagonal_green_response")
            symmetric_tangent = data.get("symmetric_tangent_green_response")
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
                        "column_diagonal_green_response",
                        "symmetric_tangent_green_response",
                    ],
                    mode,
                ),
                mask=cast(Literal["quadratic", "sin"], mask),
                column_diagonal_green_response=(
                    ColumnDiagonalGreenResponseProjectionConfig.from_raw(
                        column_diagonal
                    )
                ),
                symmetric_tangent_green_response=(
                    SymmetricTangentGreenResponseProjectionConfig.from_raw(
                        symmetric_tangent
                    )
                ),
            )
        raise TypeError("balance_projection must be a string or an object.")


@dataclass
class ComplexPreProjectionFusionConfig:
    """Optional physical difference fusion before complex projection."""

    enabled: bool = False
    mode: Literal["residual", "absolute"] = "residual"
    hidden_dim: int = 16
    depth: int = 1
    eps: float = 1.0e-12
    final_layer_init_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("pre_projection_fusion.enabled must be a boolean.")
        if not isinstance(self.mode, str):
            raise TypeError("pre_projection_fusion.mode must be a string.")
        if self.mode not in {"residual", "absolute"}:
            raise ValueError(
                "pre_projection_fusion.mode must be 'residual' or 'absolute'."
            )
        if (
            isinstance(self.hidden_dim, bool)
            or not isinstance(self.hidden_dim, int)
            or self.hidden_dim < 1
        ):
            raise ValueError(
                "pre_projection_fusion.hidden_dim must be a positive integer."
            )
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth < 1
        ):
            raise ValueError("pre_projection_fusion.depth must be a positive integer.")
        if isinstance(self.eps, bool):
            raise TypeError("pre_projection_fusion.eps must be numeric.")
        eps = float(self.eps)
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("pre_projection_fusion.eps must be finite and positive.")
        if isinstance(self.final_layer_init_scale, bool) or not isinstance(
            self.final_layer_init_scale, (int, float)
        ):
            raise TypeError(
                "pre_projection_fusion.final_layer_init_scale must be numeric."
            )
        final_layer_init_scale = float(self.final_layer_init_scale)
        if (
            not math.isfinite(final_layer_init_scale)
            or final_layer_init_scale < 0.0
            or final_layer_init_scale > 1.0
        ):
            raise ValueError(
                "pre_projection_fusion.final_layer_init_scale must be finite "
                "and in [0, 1]."
            )

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
                    "mode",
                    "hidden_dim",
                    "depth",
                    "eps",
                    "final_layer_init_scale",
                }
            )
            if unknown:
                raise TypeError(
                    f"pre_projection_fusion has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("pre_projection_fusion must be an object.")


@dataclass
class ComplexCrossAxisReconstructionConfig:
    """Optional final-solution blend for complex CouplingNet inference."""

    enabled: bool = False
    mode: Literal["local_weak_residual_reliability"] = "local_weak_residual_reliability"
    gamma: float = 0.5
    smoothing_steps: int = 2
    smoothing_relaxation: float = 0.5
    relative_floor: float = 0.1
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        prefix = "cross_axis_reconstruction"
        if not isinstance(self.enabled, bool):
            raise TypeError(f"{prefix}.enabled must be a boolean.")
        if not isinstance(self.mode, str):
            raise TypeError(f"{prefix}.mode must be a string.")
        if self.mode != "local_weak_residual_reliability":
            raise ValueError(
                f"{prefix}.mode must be 'local_weak_residual_reliability'. "
                "Geometry-only and mismatch-detected reconstruction modes are not "
                "available in production inference."
            )
        if isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)):
            raise TypeError(f"{prefix}.gamma must be numeric.")
        if not math.isfinite(float(self.gamma)) or not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError(f"{prefix}.gamma must be finite and in [0, 1].")
        if (
            isinstance(self.smoothing_steps, bool)
            or not isinstance(self.smoothing_steps, int)
            or self.smoothing_steps < 0
        ):
            raise ValueError(
                f"{prefix}.smoothing_steps must be a non-negative integer."
            )
        if isinstance(self.smoothing_relaxation, bool) or not isinstance(
            self.smoothing_relaxation,
            (int, float),
        ):
            raise TypeError(f"{prefix}.smoothing_relaxation must be numeric.")
        if not math.isfinite(float(self.smoothing_relaxation)) or not (
            0.0 < float(self.smoothing_relaxation) <= 1.0
        ):
            raise ValueError(
                f"{prefix}.smoothing_relaxation must be finite and in (0, 1]."
            )
        if isinstance(self.relative_floor, bool) or not isinstance(
            self.relative_floor,
            (int, float),
        ):
            raise TypeError(f"{prefix}.relative_floor must be numeric.")
        if (
            not math.isfinite(float(self.relative_floor))
            or float(self.relative_floor) < 0.0
        ):
            raise ValueError(
                f"{prefix}.relative_floor must be finite and non-negative."
            )
        if isinstance(self.eps, bool) or not isinstance(self.eps, (int, float)):
            raise TypeError(f"{prefix}.eps must be numeric.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError(f"{prefix}.eps must be finite and positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexCrossAxisReconstructionConfig | dict[str, Any] | None,
    ) -> ComplexCrossAxisReconstructionConfig:
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
                    "mode",
                    "gamma",
                    "smoothing_steps",
                    "smoothing_relaxation",
                    "relative_floor",
                    "eps",
                }
            )
            if unknown:
                raise TypeError(
                    f"cross_axis_reconstruction has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("cross_axis_reconstruction must be an object.")


@dataclass
class CouplingGeometryBranchConfig:
    """Optional segment-geometry branch settings for complex CouplingNet."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("geometry_branch.enabled must be a boolean.")

    @classmethod
    def from_raw(
        cls,
        raw: CouplingGeometryBranchConfig | dict[str, Any] | None,
    ) -> CouplingGeometryBranchConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled"})
            if unknown:
                raise TypeError(
                    f"geometry_branch has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("geometry_branch must be an object.")


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
            "column_diagonal_green_response",
            "symmetric_tangent_green_response",
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
    geometry_branch: CouplingGeometryBranchConfig | dict[str, Any] = field(
        default_factory=CouplingGeometryBranchConfig
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
    cross_axis_reconstruction: ComplexCrossAxisReconstructionConfig | dict[str, Any] = (
        field(default_factory=ComplexCrossAxisReconstructionConfig)
    )

    def __post_init__(self) -> None:
        self.balance_projection = BalanceProjectionConfig.from_raw(
            self.balance_projection
        )
        self.branch_fusion = CouplingBranchFusionConfig.from_raw(self.branch_fusion)
        self.geometry_branch = CouplingGeometryBranchConfig.from_raw(
            self.geometry_branch
        )
        self.axis_1d_trunk = Axis1DTrunkConfig.from_raw(self.axis_1d_trunk)
        self.pre_projection_fusion = ComplexPreProjectionFusionConfig.from_raw(
            self.pre_projection_fusion
        )
        self.cross_axis_reconstruction = ComplexCrossAxisReconstructionConfig.from_raw(
            self.cross_axis_reconstruction
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
class ComplexCanonicalEnergyConfig:
    """Complex canonical-energy optimization weights."""

    boundary_weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_weight, (int, float)) or isinstance(
            self.boundary_weight,
            bool,
        ):
            raise TypeError("canonical_energy.boundary_weight must be numeric.")
        if not math.isfinite(float(self.boundary_weight)):
            raise ValueError("canonical_energy.boundary_weight must be finite.")
        if self.boundary_weight < 0.0:
            raise ValueError("canonical_energy.boundary_weight must be non-negative.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexCanonicalEnergyConfig | dict[str, Any] | None,
    ) -> ComplexCanonicalEnergyConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"boundary_weight"})
            if unknown:
                raise TypeError(
                    f"canonical_energy has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("canonical_energy must be an object.")


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
class ComplexPostLineSearchStationarityConfig:
    """Normalized stationarity regularization after tangent exact line search."""

    enabled: bool = False
    weight: float = 1.0
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("post_line_search_stationarity.enabled must be a boolean.")
        for field_name, value in (("weight", self.weight), ("eps", self.eps)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"post_line_search_stationarity.{field_name} must be numeric."
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"post_line_search_stationarity.{field_name} must be finite."
                )
        if self.weight < 0.0:
            raise ValueError(
                "post_line_search_stationarity.weight must be non-negative."
            )
        if self.eps <= 0.0:
            raise ValueError("post_line_search_stationarity.eps must be positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexPostLineSearchStationarityConfig | dict[str, Any] | None,
    ) -> ComplexPostLineSearchStationarityConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "weight", "eps"})
            if unknown:
                raise TypeError(
                    "post_line_search_stationarity has unknown keys: "
                    f"{', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("post_line_search_stationarity must be an object.")


@dataclass
class ComplexResponseTrustConfig:
    """Source-normalized trust objective for the applied tangent response."""

    enabled: bool = False
    weight: float = 1.0
    trust_weight: float = 0.01
    eps: float = 1.0e-12

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("response_trust.enabled must be a boolean.")
        for field_name, value in (
            ("weight", self.weight),
            ("trust_weight", self.trust_weight),
            ("eps", self.eps),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"response_trust.{field_name} must be numeric.")
            if not math.isfinite(float(value)):
                raise ValueError(f"response_trust.{field_name} must be finite.")
        if self.weight < 0.0:
            raise ValueError("response_trust.weight must be non-negative.")
        if self.trust_weight < 0.0:
            raise ValueError("response_trust.trust_weight must be non-negative.")
        if self.eps <= 0.0:
            raise ValueError("response_trust.eps must be positive.")

    @classmethod
    def from_raw(
        cls,
        raw: ComplexResponseTrustConfig | dict[str, Any] | None,
    ) -> ComplexResponseTrustConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            data = dict(raw)
            unknown = sorted(set(data) - {"enabled", "weight", "trust_weight", "eps"})
            if unknown:
                raise TypeError(
                    f"response_trust has unknown keys: {', '.join(unknown)}."
                )
            return cls(**data)
        raise TypeError("response_trust must be an object.")


@dataclass
class SoapOptimizerConfig:
    """SOAP-specific optimizer settings shared by supported training paths."""

    shampoo_beta: float = -1.0
    precondition_frequency: int = 10
    max_precondition_dim: int = 1024
    merge_dims: bool = False
    precondition_1d: bool = False
    normalize_grads: bool = False
    correct_bias: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.shampoo_beta, (int, float)) or isinstance(
            self.shampoo_beta, bool
        ):
            raise TypeError("optimizer.soap.shampoo_beta must be numeric.")
        shampoo_beta = float(self.shampoo_beta)
        if not math.isfinite(shampoo_beta):
            raise ValueError("optimizer.soap.shampoo_beta must be finite.")
        if shampoo_beta != -1.0 and not 0.0 <= shampoo_beta < 1.0:
            raise ValueError("optimizer.soap.shampoo_beta must be -1 or in [0, 1).")
        self.shampoo_beta = shampoo_beta

        for field_name, value in (
            ("precondition_frequency", self.precondition_frequency),
            ("max_precondition_dim", self.max_precondition_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"optimizer.soap.{field_name} must be an integer.")
            if value < 1:
                raise ValueError(f"optimizer.soap.{field_name} must be positive.")

        for field_name, value in (
            ("merge_dims", self.merge_dims),
            ("precondition_1d", self.precondition_1d),
            ("normalize_grads", self.normalize_grads),
            ("correct_bias", self.correct_bias),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"optimizer.soap.{field_name} must be a boolean.")

    @classmethod
    def from_raw(
        cls,
        raw: SoapOptimizerConfig | dict[str, Any] | None,
    ) -> SoapOptimizerConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("optimizer.soap must be an object.")
        data = dict(raw)
        unknown = sorted(
            set(data)
            - {
                "shampoo_beta",
                "precondition_frequency",
                "max_precondition_dim",
                "merge_dims",
                "precondition_1d",
                "normalize_grads",
                "correct_bias",
            }
        )
        if unknown:
            raise TypeError(f"optimizer.soap has unknown keys: {', '.join(unknown)}.")
        return cls(**data)


@dataclass
class CouplingOptimizerConfig:
    """Optimizer selection shared by CouplingNet training configs."""

    name: Literal["adamw", "soap"] = "adamw"
    betas: tuple[float, float] | list[float] = (0.9, 0.999)
    eps: float = 1.0e-8
    profile_step_time: bool = False
    soap: SoapOptimizerConfig | dict[str, Any] = field(
        default_factory=SoapOptimizerConfig
    )

    def __post_init__(self) -> None:
        if self.name not in {"adamw", "soap"}:
            raise ValueError("optimizer.name must be 'adamw' or 'soap'.")
        if (
            not isinstance(self.betas, (tuple, list))
            or len(self.betas) != 2
            or any(
                not isinstance(beta, (int, float)) or isinstance(beta, bool)
                for beta in self.betas
            )
        ):
            raise TypeError("optimizer.betas must contain two numeric values.")
        betas = (float(self.betas[0]), float(self.betas[1]))
        if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("optimizer.betas must be finite and in [0, 1).")
        self.betas = betas
        if not isinstance(self.eps, (int, float)) or isinstance(self.eps, bool):
            raise TypeError("optimizer.eps must be numeric.")
        self.eps = float(self.eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("optimizer.eps must be finite and positive.")
        if not isinstance(self.profile_step_time, bool):
            raise TypeError("optimizer.profile_step_time must be a boolean.")
        self.soap = SoapOptimizerConfig.from_raw(self.soap)

    @classmethod
    def from_raw(
        cls,
        raw: CouplingOptimizerConfig | dict[str, Any] | None,
    ) -> CouplingOptimizerConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("coupling_training.optimizer must be an object.")
        data = dict(raw)
        unknown = sorted(
            set(data) - {"name", "betas", "eps", "profile_step_time", "soap"}
        )
        if unknown:
            raise TypeError(
                f"coupling_training.optimizer has unknown keys: {', '.join(unknown)}."
            )
        return cls(**data)


@dataclass
class TangentContextCheckpointConfig:
    """Persistence policy for the frozen complex tangent response context."""

    enabled: bool = False
    path: Path | None = None
    load_policy: Literal["never", "if_available", "required"] = "if_available"
    save_after_build: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("tangent_context_checkpoint.enabled must be a boolean.")
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if self.path is not None and not isinstance(self.path, Path):
            raise TypeError("tangent_context_checkpoint.path must be a path or null.")
        if not isinstance(self.load_policy, str):
            raise TypeError("tangent_context_checkpoint.load_policy must be a string.")
        if self.load_policy not in {"never", "if_available", "required"}:
            raise ValueError(
                "tangent_context_checkpoint.load_policy must be 'never', "
                "'if_available', or 'required'."
            )
        if not isinstance(self.save_after_build, bool):
            raise TypeError(
                "tangent_context_checkpoint.save_after_build must be a boolean."
            )
        if not self.enabled:
            if self.path is not None:
                raise ValueError(
                    "tangent_context_checkpoint.path requires enabled=true."
                )
            if self.load_policy != "if_available":
                raise ValueError(
                    "A non-default tangent_context_checkpoint.load_policy requires "
                    "enabled=true."
                )
            if not self.save_after_build:
                raise ValueError(
                    "tangent_context_checkpoint.save_after_build=false requires "
                    "enabled=true."
                )
        if self.enabled and self.load_policy == "required" and self.save_after_build:
            raise ValueError(
                "tangent_context_checkpoint.load_policy='required' is load-only "
                "and requires save_after_build=false."
            )

    @classmethod
    def from_raw(
        cls,
        raw: TangentContextCheckpointConfig | dict[str, Any] | None,
    ) -> TangentContextCheckpointConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("tangent_context_checkpoint must be an object.")
        data = dict(raw)
        unknown = sorted(
            set(data) - {"enabled", "path", "load_policy", "save_after_build"}
        )
        if unknown:
            raise TypeError(
                "tangent_context_checkpoint has unknown keys: "
                + ", ".join(unknown)
                + "."
            )
        return cls(**data)


@dataclass
class GreenOptimizerConfig:
    """Optimizer selection shared by unit-square and complex GreenNet."""

    name: Literal["adamw", "soap"] = "adamw"
    betas: tuple[float, float] | list[float] = (0.9, 0.999)
    eps: float = 1.0e-8
    profile_step_time: bool = False
    soap: SoapOptimizerConfig | dict[str, Any] = field(
        default_factory=SoapOptimizerConfig
    )

    def __post_init__(self) -> None:
        name = str(self.name)
        if name == "adam":
            raise ValueError("GreenNet Adam has been removed; use adamw.")
        if name not in {"adamw", "soap"}:
            raise ValueError("training.optimizer.name must be 'adamw' or 'soap'.")
        if (
            not isinstance(self.betas, (tuple, list))
            or len(self.betas) != 2
            or any(
                not isinstance(beta, (int, float)) or isinstance(beta, bool)
                for beta in self.betas
            )
        ):
            raise TypeError("training.optimizer.betas must contain two numeric values.")
        betas = (float(self.betas[0]), float(self.betas[1]))
        if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("training.optimizer.betas must be finite and in [0, 1).")
        self.betas = betas
        if not isinstance(self.eps, (int, float)) or isinstance(self.eps, bool):
            raise TypeError("training.optimizer.eps must be numeric.")
        self.eps = float(self.eps)
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("training.optimizer.eps must be finite and positive.")
        if not isinstance(self.profile_step_time, bool):
            raise TypeError("training.optimizer.profile_step_time must be a boolean.")
        self.soap = SoapOptimizerConfig.from_raw(self.soap)

    @classmethod
    def from_raw(
        cls,
        raw: GreenOptimizerConfig | dict[str, Any] | None,
    ) -> GreenOptimizerConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("training.optimizer must be an object.")
        data = dict(raw)
        unknown = sorted(
            set(data) - {"name", "betas", "eps", "profile_step_time", "soap"}
        )
        if unknown:
            raise TypeError(
                f"training.optimizer has unknown keys: {', '.join(unknown)}."
            )
        return cls(**data)


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
    seed: int | None = None
    deterministic_algorithms: bool = False
    losses: CouplingLossesConfig = field(default_factory=CouplingLossesConfig)
    use_lr_schedule: bool = False
    warmup_epochs: int = 0
    warmup_steps: int | None = None
    validation_every_steps: int | None = None
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
    canonical_energy: ComplexCanonicalEnergyConfig | dict[str, Any] = field(
        default_factory=ComplexCanonicalEnergyConfig
    )
    relative_split_consistency: (
        ComplexRelativeSplitConsistencyConfig | dict[str, Any]
    ) = field(default_factory=ComplexRelativeSplitConsistencyConfig)
    weak_operator_closure: ComplexWeakOperatorClosureConfig | dict[str, Any] = field(
        default_factory=ComplexWeakOperatorClosureConfig
    )
    post_line_search_stationarity: (
        ComplexPostLineSearchStationarityConfig | dict[str, Any]
    ) = field(default_factory=ComplexPostLineSearchStationarityConfig)
    response_trust: ComplexResponseTrustConfig | dict[str, Any] = field(
        default_factory=ComplexResponseTrustConfig
    )
    optimizer: CouplingOptimizerConfig | dict[str, Any] = field(
        default_factory=CouplingOptimizerConfig
    )
    tangent_context_checkpoint: TangentContextCheckpointConfig | dict[str, Any] = field(
        default_factory=TangentContextCheckpointConfig
    )

    def __post_init__(self) -> None:
        from greenonet.reproducibility import validate_training_seed

        validate_training_seed(self.seed, field_name="coupling_training.seed")
        _validate_training_step_fields(
            field_prefix="coupling_training",
            warmup_epochs=self.warmup_epochs,
            warmup_steps=self.warmup_steps,
            validation_every_steps=self.validation_every_steps,
        )
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError(
                "coupling_training.deterministic_algorithms must be a boolean."
            )
        self.best_energy_checkpoint = CouplingBestEnergyCheckpointConfig.from_raw(
            self.best_energy_checkpoint
        )
        self.best_physics_checkpoint = CouplingBestPhysicsCheckpointConfig.from_raw(
            self.best_physics_checkpoint
        )
        self.canonical_energy = ComplexCanonicalEnergyConfig.from_raw(
            self.canonical_energy
        )
        self.relative_split_consistency = (
            ComplexRelativeSplitConsistencyConfig.from_raw(
                self.relative_split_consistency
            )
        )
        self.weak_operator_closure = ComplexWeakOperatorClosureConfig.from_raw(
            self.weak_operator_closure
        )
        self.post_line_search_stationarity = (
            ComplexPostLineSearchStationarityConfig.from_raw(
                self.post_line_search_stationarity
            )
        )
        self.response_trust = ComplexResponseTrustConfig.from_raw(self.response_trust)
        self.optimizer = CouplingOptimizerConfig.from_raw(self.optimizer)
        self.tangent_context_checkpoint = TangentContextCheckpointConfig.from_raw(
            self.tangent_context_checkpoint
        )


def validate_unit_square_coupling_training_config(
    config: CouplingTrainingConfig,
) -> None:
    """Reject complex-only training options on the unit-square path."""

    if (
        ComplexCanonicalEnergyConfig.from_raw(config.canonical_energy)
        != ComplexCanonicalEnergyConfig()
    ):
        raise ValueError(
            "coupling_training.canonical_energy is available only for "
            "ComplexCouplingTrainer."
        )

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
    if ComplexPostLineSearchStationarityConfig.from_raw(
        config.post_line_search_stationarity
    ).enabled:
        raise ValueError(
            "coupling_training.post_line_search_stationarity is available only "
            "for ComplexCouplingTrainer."
        )
    if ComplexResponseTrustConfig.from_raw(config.response_trust).enabled:
        raise ValueError(
            "coupling_training.response_trust is available only for "
            "ComplexCouplingTrainer."
        )
    if CouplingBestPhysicsCheckpointConfig.from_raw(
        config.best_physics_checkpoint
    ).enabled:
        raise ValueError(
            "coupling_training.best_physics_checkpoint is available only for "
            "ComplexCouplingTrainer."
        )
    optimizer = CouplingOptimizerConfig.from_raw(config.optimizer)
    if optimizer.name == "soap":
        raise ValueError(
            "coupling_training.optimizer.name='soap' is available only for "
            "ComplexCouplingTrainer."
        )
    if optimizer != CouplingOptimizerConfig():
        raise ValueError(
            "Custom coupling_training.optimizer settings are available only for "
            "ComplexCouplingTrainer; omit the optimizer block for unit-square "
            "AdamW training."
        )
    if TangentContextCheckpointConfig.from_raw(
        config.tangent_context_checkpoint
    ).enabled:
        raise ValueError(
            "coupling_training.tangent_context_checkpoint is available only for "
            "ComplexCouplingTrainer."
        )


def validate_complex_tangent_context_checkpoint_config(
    *,
    training: CouplingTrainingConfig,
    balance_projection: BalanceProjectionConfig | dict[str, Any] | str,
) -> TangentContextCheckpointConfig:
    """Validate tangent context persistence against the projection mode."""

    checkpoint = TangentContextCheckpointConfig.from_raw(
        training.tangent_context_checkpoint
    )
    if not checkpoint.enabled:
        return checkpoint
    projection = BalanceProjectionConfig.from_raw(balance_projection)
    if not projection.enabled or projection.mode != "symmetric_tangent_green_response":
        raise ValueError(
            "coupling_training.tangent_context_checkpoint.enabled=true requires "
            "coupling_model.balance_projection.mode="
            "'symmetric_tangent_green_response'."
        )
    return checkpoint


def validate_complex_post_line_search_stationarity_config(
    *,
    training: CouplingTrainingConfig,
    balance_projection: BalanceProjectionConfig | dict[str, Any] | str,
) -> ComplexPostLineSearchStationarityConfig:
    """Validate the objective against the complex tangent projection contract."""

    stationarity = ComplexPostLineSearchStationarityConfig.from_raw(
        training.post_line_search_stationarity
    )
    if not stationarity.enabled:
        return stationarity
    projection = BalanceProjectionConfig.from_raw(balance_projection)
    if not projection.enabled or projection.mode != "symmetric_tangent_green_response":
        raise ValueError(
            "coupling_training.post_line_search_stationarity.enabled=true requires "
            "coupling_model.balance_projection.mode="
            "'symmetric_tangent_green_response'."
        )
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )
    if tangent.eta_strategy != "closed_loop_exact_line_search":
        raise ValueError(
            "coupling_training.post_line_search_stationarity.enabled=true requires "
            "balance_projection.symmetric_tangent_green_response.eta_strategy="
            "'closed_loop_exact_line_search'."
        )
    return stationarity


def validate_complex_response_trust_config(
    *,
    training: CouplingTrainingConfig,
    balance_projection: BalanceProjectionConfig | dict[str, Any] | str,
) -> ComplexResponseTrustConfig:
    """Validate response trust against the applied tangent projection contract."""

    response_trust = ComplexResponseTrustConfig.from_raw(training.response_trust)
    if not response_trust.enabled:
        return response_trust
    projection = BalanceProjectionConfig.from_raw(balance_projection)
    if not projection.enabled or projection.mode != "symmetric_tangent_green_response":
        raise ValueError(
            "coupling_training.response_trust.enabled=true requires "
            "coupling_model.balance_projection.mode="
            "'symmetric_tangent_green_response'."
        )
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )
    if tangent.eta_strategy != "closed_loop_exact_line_search":
        raise ValueError(
            "coupling_training.response_trust.enabled=true requires "
            "balance_projection.symmetric_tangent_green_response.eta_strategy="
            "'closed_loop_exact_line_search'."
        )
    return response_trust


def validate_complex_coupling_source_config(
    dataset: DatasetConfig,
    training: CouplingTrainingConfig,
) -> None:
    """Validate source backend paths, diagnostics, and validation availability."""

    source = ComplexCouplingSourceConfig.from_raw(dataset.coupling_source)
    diagnostics = ComplexReferenceDiagnosticsConfig.from_raw(
        dataset.reference_diagnostics
    )
    if dataset.geometry_mode != "complex":
        if source != ComplexCouplingSourceConfig():
            raise ValueError(
                "dataset.coupling_source options are available only for complex "
                "geometry CouplingNet training."
            )
        if diagnostics != ComplexReferenceDiagnosticsConfig():
            raise ValueError(
                "dataset.reference_diagnostics options are available only for "
                "complex geometry CouplingNet training."
            )
        return

    best_energy = CouplingBestEnergyCheckpointConfig.from_raw(
        training.best_energy_checkpoint
    ).enabled
    best_physics = CouplingBestPhysicsCheckpointConfig.from_raw(
        training.best_physics_checkpoint
    ).enabled

    if source.mode == "npz":
        if dataset.training_path is None:
            raise ValueError(
                "dataset.training_path is required when "
                "dataset.coupling_source.mode='npz'."
            )
        if (best_energy or best_physics) and dataset.validation_path is None:
            raise ValueError(
                "A validation source is required when best_energy_checkpoint or "
                "best_physics_checkpoint is enabled."
            )
        return

    if dataset.training_path is not None:
        raise ValueError(
            "dataset.training_path is unused when "
            "dataset.coupling_source.mode='indexed_gp'."
        )
    if dataset.validation_path is not None:
        raise ValueError(
            "dataset.validation_path is unused when "
            "dataset.coupling_source.mode='indexed_gp'."
        )
    if diagnostics.training or diagnostics.validation:
        raise ValueError(
            "dataset.reference_diagnostics.training and validation must both be "
            "false when dataset.coupling_source.mode='indexed_gp'."
        )
    indexed = cast(IndexedGpSourceConfig, source.indexed_gp)
    if (best_energy or best_physics) and indexed.num_valid == 0:
        raise ValueError(
            "indexed_gp.num_valid must be positive when best_energy_checkpoint "
            "or best_physics_checkpoint is enabled."
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
    weight_decay: float = 0.0
    epochs: int = 10
    batch_size: int = 32
    log_interval: int = 1
    device: str = "cpu"
    seed: int | None = None
    deterministic_algorithms: bool = False
    compute_validation_rel_sol: bool = False
    use_lr_schedule: bool = False
    warmup_epochs: int = 0
    warmup_steps: int | None = None
    validation_every_steps: int | None = None
    min_lr: float = 1e-6
    optimizer: GreenOptimizerConfig | dict[str, Any] = field(
        default_factory=GreenOptimizerConfig
    )
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
        from greenonet.reproducibility import validate_training_seed

        validate_training_seed(self.seed, field_name="training.seed")
        _validate_training_step_fields(
            field_prefix="training",
            warmup_epochs=self.warmup_epochs,
            warmup_steps=self.warmup_steps,
            validation_every_steps=self.validation_every_steps,
        )
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError("training.deterministic_algorithms must be a boolean.")
        self.green_quadrature = GreenQuadratureConfig.from_raw(self.green_quadrature)
        self.optimizer = GreenOptimizerConfig.from_raw(self.optimizer)


@dataclass
class PipelineConfig:
    """Control flags for training pipelines."""

    run_green: bool = True
    run_coupling: bool = False
    green_pretrained_path: Optional[Path] = None
    coupling_pretrained_path: Optional[Path] = None


@dataclass(frozen=True)
class CouplingArtifactsConfig:
    """Optional post-training export settings for complex CouplingNet."""

    enabled: bool = False
    checkpoint: Literal["best_energy"] = "best_energy"
    device: str | None = None
    theme: str = "plotly_white"
    selected_samples: tuple[int, ...] | None = None
    save_generated_data: bool = True
    plot_workers: int = 1
    coefficient_vector_max_points: int = 400
    show_domain_boundary: bool = True
    visualization_mesh: Path | None = None
    directional_color_quantile: float = 0.99

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("coupling_artifacts.enabled must be a boolean.")
        if not isinstance(self.checkpoint, str):
            raise TypeError("coupling_artifacts.checkpoint must be a string.")
        if self.checkpoint != "best_energy":
            raise ValueError("coupling_artifacts.checkpoint must be 'best_energy'.")
        if self.device is not None:
            if not isinstance(self.device, str):
                raise TypeError("coupling_artifacts.device must be a string or null.")
            if not self.device.strip():
                raise ValueError("coupling_artifacts.device cannot be empty.")
            device = self.device.strip()
            try:
                torch.device(device)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(
                    "coupling_artifacts.device must be a valid torch device."
                ) from exc
            object.__setattr__(self, "device", device)
        if not isinstance(self.theme, str):
            raise TypeError("coupling_artifacts.theme must be a string.")
        if not self.theme.strip():
            raise ValueError("coupling_artifacts.theme cannot be empty.")
        object.__setattr__(self, "theme", self.theme.strip())

        selected = self.selected_samples
        if selected is not None:
            if not isinstance(selected, (tuple, list)):
                raise TypeError(
                    "coupling_artifacts.selected_samples must be an array or null."
                )
            parsed_selected: list[int] = []
            for value in selected:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(
                        "coupling_artifacts.selected_samples must contain integers."
                    )
                if value < 0:
                    raise ValueError(
                        "coupling_artifacts.selected_samples must be non-negative."
                    )
                parsed_selected.append(value)
            object.__setattr__(self, "selected_samples", tuple(parsed_selected))

        for field_name, value in (
            ("save_generated_data", self.save_generated_data),
            ("show_domain_boundary", self.show_domain_boundary),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"coupling_artifacts.{field_name} must be a boolean.")
        for field_name, value in (
            ("plot_workers", self.plot_workers),
            ("coefficient_vector_max_points", self.coefficient_vector_max_points),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"coupling_artifacts.{field_name} must be an integer.")
            if value <= 0:
                raise ValueError(f"coupling_artifacts.{field_name} must be positive.")

        visualization_mesh = self.visualization_mesh
        if isinstance(visualization_mesh, str):
            visualization_mesh = Path(visualization_mesh)
            object.__setattr__(self, "visualization_mesh", visualization_mesh)
        if visualization_mesh is not None and not isinstance(visualization_mesh, Path):
            raise TypeError(
                "coupling_artifacts.visualization_mesh must be a path or null."
            )

        quantile = self.directional_color_quantile
        if not isinstance(quantile, (int, float)) or isinstance(quantile, bool):
            raise TypeError(
                "coupling_artifacts.directional_color_quantile must be numeric."
            )
        quantile = float(quantile)
        if not math.isfinite(quantile) or quantile <= 0.5 or quantile > 1.0:
            raise ValueError(
                "coupling_artifacts.directional_color_quantile must be finite "
                "and in (0.5, 1.0]."
            )
        object.__setattr__(self, "directional_color_quantile", quantile)

    @classmethod
    def from_raw(
        cls,
        raw: CouplingArtifactsConfig | dict[str, Any] | None,
    ) -> CouplingArtifactsConfig:
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, dict):
            raise TypeError("coupling_artifacts must be an object.")
        data = dict(raw)
        allowed = {
            "enabled",
            "checkpoint",
            "device",
            "theme",
            "selected_samples",
            "save_generated_data",
            "plot_workers",
            "coefficient_vector_max_points",
            "show_domain_boundary",
            "visualization_mesh",
            "directional_color_quantile",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise TypeError(
                f"coupling_artifacts has unknown keys: {', '.join(unknown)}."
            )
        return cls(**data)

    def to_raw(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "checkpoint": self.checkpoint,
            "device": self.device,
            "theme": self.theme,
            "selected_samples": (
                None if self.selected_samples is None else list(self.selected_samples)
            ),
            "save_generated_data": self.save_generated_data,
            "plot_workers": self.plot_workers,
            "coefficient_vector_max_points": self.coefficient_vector_max_points,
            "show_domain_boundary": self.show_domain_boundary,
            "visualization_mesh": (
                None
                if self.visualization_mesh is None
                else str(self.visualization_mesh)
            ),
            "directional_color_quantile": self.directional_color_quantile,
        }


def validate_active_training_seeds(
    *,
    training: TrainingConfig,
    coupling_training: CouplingTrainingConfig,
    pipeline: PipelineConfig,
) -> None:
    """Require explicit seeds only for stages started by the training CLI."""

    if pipeline.run_green and training.seed is None:
        raise ValueError("training.seed is required when pipeline.run_green=true.")
    if pipeline.run_coupling and coupling_training.seed is None:
        raise ValueError(
            "coupling_training.seed is required when pipeline.run_coupling=true."
        )


def validate_coupling_artifacts_config(
    *,
    artifacts: CouplingArtifactsConfig,
    dataset: DatasetConfig,
    coupling_training: CouplingTrainingConfig,
    pipeline: PipelineConfig,
) -> None:
    """Validate the optional post-training complex artifact lifecycle."""

    if not artifacts.enabled:
        return
    if not pipeline.run_coupling:
        raise ValueError(
            "coupling_artifacts.enabled=true requires pipeline.run_coupling=true."
        )
    if dataset.geometry_mode != "complex":
        raise ValueError(
            "coupling_artifacts.enabled=true is available only for complex "
            "CouplingNet training."
        )
    if dataset.geometry_path is None:
        raise ValueError(
            "dataset.geometry_path is required when coupling_artifacts is enabled."
        )
    if not Path(dataset.geometry_path).is_file():
        raise FileNotFoundError(dataset.geometry_path)
    if not CouplingBestEnergyCheckpointConfig.from_raw(
        coupling_training.best_energy_checkpoint
    ).enabled:
        raise ValueError(
            "coupling_artifacts.enabled=true requires "
            "coupling_training.best_energy_checkpoint.enabled=true."
        )
    validate_complex_coupling_source_config(dataset, coupling_training)
    source = ComplexCouplingSourceConfig.from_raw(dataset.coupling_source)
    if source.mode == "npz":
        if dataset.validation_path is None:
            raise RuntimeError(
                "Artifact validation unexpectedly resolved no validation path."
            )
        validation_path = Path(dataset.validation_path)
        if not validation_path.is_dir():
            raise FileNotFoundError(validation_path)
        if not any(validation_path.glob("*.npz")):
            raise FileNotFoundError(
                f"No validation npz files found in {validation_path}"
            )
    if dataset.test_path is None:
        raise ValueError(
            "dataset.test_path is required when coupling_artifacts is enabled."
        )
    if not Path(dataset.test_path).is_dir():
        raise FileNotFoundError(dataset.test_path)
    if not pipeline.run_green:
        if pipeline.green_pretrained_path is None:
            raise ValueError(
                "pipeline.green_pretrained_path is required when "
                "coupling_artifacts is enabled and pipeline.run_green=false."
            )
        if not Path(pipeline.green_pretrained_path).is_file():
            raise FileNotFoundError(pipeline.green_pretrained_path)
    if artifacts.visualization_mesh is not None and not (
        artifacts.visualization_mesh.is_file()
    ):
        raise FileNotFoundError(artifacts.visualization_mesh)
