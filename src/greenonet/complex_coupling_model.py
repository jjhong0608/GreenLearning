from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, cast

import torch
from torch import nn

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_pre_projection_fusion import (
    ComplexPreProjectionFusion,
    ComplexPreProjectionFusionResult,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexPreProjectionFusionConfig,
    CouplingBranchFusionConfig,
    CouplingModelConfig,
)
from greenonet.coupling_model import ActivationFactoryMixin, MLP


class ComplexCouplingNet(nn.Module, ActivationFactoryMixin):
    """Source-conditioned response predictor for complex geometries."""

    OUTPUT_CONTRACT_VERSION = 6
    PRE_PROJECTION_FUSION_MODE_KEY = "pre_projection_fusion._fusion_mode_code"

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        self.config = config
        self._validate_complex_config(config)
        self.branch_input_dim = int(config.branch_input_dim)
        self.hidden_dim = int(config.hidden_dim)
        self._output_contract_version: torch.Tensor
        self.register_buffer(
            "_output_contract_version",
            torch.tensor(self.OUTPUT_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        if self.branch_input_dim < 2:
            raise ValueError("coupling_model.branch_input_dim must be at least 2.")

        coefficient_terms = config.coefficient_terms
        active_coefficient_terms = []
        if coefficient_terms.diffusion:
            active_coefficient_terms.append("diffusion")
        if coefficient_terms.convection:
            active_coefficient_terms.extend(
                ("convection_primary", "convection_transverse")
            )
        if coefficient_terms.reaction:
            active_coefficient_terms.append("reaction")
        self.active_coefficient_terms = tuple(active_coefficient_terms)
        self.coefficient_branch_channels = len(self.active_coefficient_terms)
        self.coefficient_branch_input_dim = (
            self.coefficient_branch_channels * self.branch_input_dim
        )

        axis_cfg = Axis1DTrunkConfig.from_raw(config.axis_1d_trunk)
        self.transverse_num_frequencies = int(axis_cfg.num_frequencies)
        self.transverse_max_frequency = float(axis_cfg.max_frequency)
        self.transverse_feature_dim = 2 * self.transverse_num_frequencies
        frequencies = torch.logspace(
            start=0.0,
            end=math.log2(self.transverse_max_frequency),
            steps=self.transverse_num_frequencies,
            base=2.0,
            dtype=config.dtype,
        )
        self.transverse_frequencies: torch.Tensor
        self.register_buffer(
            "transverse_frequencies",
            frequencies,
            persistent=False,
        )

        self.geometry_feature_dim = 6
        self.branch_source = MLP(
            input_dim=self.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.branch_coefficient: MLP | None
        if self.coefficient_branch_input_dim > 0:
            self.branch_coefficient = MLP(
                input_dim=self.coefficient_branch_input_dim,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                activation=config.activation,
                use_bias=config.use_bias,
                dropout=config.dropout,
                last_activation=False,
            )
        else:
            self.branch_coefficient = None
        self.branch_transverse = MLP(
            input_dim=self.transverse_feature_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.branch_geometry = MLP(
            input_dim=self.geometry_feature_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.trunk = MLP(
            input_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=True,
        )
        transverse_trunk = axis_cfg.transverse_trunk
        self.transverse_trunk_enabled = bool(transverse_trunk.enabled)
        self.transverse_length_context_enabled = bool(transverse_trunk.length_context)
        self.transverse_trunk_fusion_mode = transverse_trunk.fusion
        self.trunk_transverse: MLP | None
        self.trunk_fuser: nn.Linear | None
        self.trunk_fuser_activation: nn.Module
        self.trunk_fuser_dropout: nn.Module
        if self.transverse_trunk_enabled:
            self.trunk_transverse = MLP(
                input_dim=4,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                activation=config.activation,
                use_bias=config.use_bias,
                dropout=config.dropout,
                last_activation=True,
            )
            if self.transverse_trunk_fusion_mode == "product_fuser":
                self.trunk_fuser = nn.Linear(
                    3 * config.hidden_dim,
                    config.hidden_dim,
                    bias=config.use_bias,
                )
                self.trunk_fuser_activation = self.build_activation(config.activation)
                self.trunk_fuser_dropout = (
                    nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
                )
            else:
                self.trunk_fuser = None
                self.trunk_fuser_activation = nn.Identity()
                self.trunk_fuser_dropout = nn.Identity()
        else:
            self.trunk_transverse = None
            self.trunk_fuser = None
            self.trunk_fuser_activation = nn.Identity()
            self.trunk_fuser_dropout = nn.Identity()

        branch_fusion = CouplingBranchFusionConfig.from_raw(config.branch_fusion)
        self.branch_fusion_mode = branch_fusion.mode
        branch_component_count = 3 + (1 if self.branch_coefficient is not None else 0)
        self.branch_fuser: nn.Linear | None
        self.branch_fuser_activation: nn.Module
        self.branch_fuser_dropout: nn.Module
        if self.branch_fusion_mode == "product_fuser":
            self.branch_fuser = nn.Linear(
                (branch_component_count + 1) * config.hidden_dim,
                config.hidden_dim,
                bias=config.use_bias,
            )
            self.branch_fuser_activation = self.build_activation(config.activation)
            self.branch_fuser_dropout = (
                nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
            )
        else:
            self.branch_fuser = None
            self.branch_fuser_activation = nn.Identity()
            self.branch_fuser_dropout = nn.Identity()

        pre_projection_fusion = ComplexPreProjectionFusionConfig.from_raw(
            config.pre_projection_fusion
        )
        self.pre_projection_fusion_enabled = bool(pre_projection_fusion.enabled)
        self.pre_projection_fusion: ComplexPreProjectionFusion | None
        if self.pre_projection_fusion_enabled:
            self.pre_projection_fusion = ComplexPreProjectionFusion(
                pre_projection_fusion,
                activation=config.activation,
                use_bias=config.use_bias,
                dtype=config.dtype,
            )
        else:
            self.pre_projection_fusion = None

    @staticmethod
    def _validate_complex_config(config: CouplingModelConfig) -> None:
        ComplexCrossAxisReconstructionConfig.from_raw(config.cross_axis_reconstruction)
        if config.trunk_positional_encoding.enabled:
            raise ValueError(
                "ComplexCouplingNet uses local 1D trunk coordinates; "
                "trunk_positional_encoding.enabled must be false. Use "
                "axis_1d_trunk.num_frequencies/max_frequency for fixed-line "
                "transverse branch encoding or axis_1d_trunk.transverse_trunk "
                "for pointwise transverse trunk coordinates."
            )
        balance_projection = BalanceProjectionConfig.from_raw(config.balance_projection)
        if not balance_projection.enabled:
            raise ValueError(
                "ComplexCouplingNet requires balance_projection.enabled=true."
            )
        if balance_projection.mode not in {
            "physical_symmetric",
            "column_diagonal_green_response",
            "symmetric_tangent_green_response",
        }:
            raise ValueError(
                "ComplexCouplingNet output-contract version 6 requires "
                "balance_projection.mode='physical_symmetric' or "
                "'column_diagonal_green_response' or "
                "'symmetric_tangent_green_response'. Response-space and earlier "
                "complex checkpoints require retraining."
            )
        axis_cfg = Axis1DTrunkConfig.from_raw(config.axis_1d_trunk)
        if not axis_cfg.enabled:
            raise ValueError(
                "ComplexCouplingNet output-contract version 6 requires "
                "axis_1d_trunk.enabled=true."
            )
        if not axis_cfg.transverse_trunk.enabled:
            raise ValueError(
                "ComplexCouplingNet output-contract version 6 requires "
                "axis_1d_trunk.transverse_trunk.enabled=true."
            )
        if not axis_cfg.transverse_trunk.length_context:
            raise ValueError(
                "ComplexCouplingNet output-contract version 6 requires "
                "axis_1d_trunk.transverse_trunk.length_context=true."
            )
        if config.source_stencil_lift.enabled:
            raise ValueError("ComplexCouplingNet does not support source_stencil_lift.")
        if config.green_response_feature.enabled:
            raise ValueError(
                "ComplexCouplingNet does not support green_response_feature."
            )

    def prepare_checkpoint_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reject checkpoints that do not use the physical-projection v6 contract."""

        prepared = dict(state_dict)
        key = "_output_contract_version"
        if key not in prepared:
            raise ValueError(
                "Legacy complex CouplingNet checkpoint has no output contract "
                "version and cannot be loaded into physical-symmetric output "
                "contract version 6. Retrain the CouplingNet; GreenNet checkpoints remain "
                "compatible."
            )

        version_tensor = prepared[key]
        if version_tensor.numel() != 1:
            raise ValueError(
                "Invalid complex CouplingNet output contract version tensor: "
                "expected exactly one value."
            )
        version = int(version_tensor.detach().cpu().item())
        if version == self.OUTPUT_CONTRACT_VERSION:
            self._prepare_pre_projection_fusion_checkpoint(prepared)
            return prepared
        raise ValueError(
            "Incompatible complex CouplingNet output contract version "
            f"{version}; expected {self.OUTPUT_CONTRACT_VERSION}. Raw-unit, "
            "physical-raw, and earlier response CouplingNet checkpoints require "
            "retraining; "
            "GreenNet checkpoints remain compatible."
        )

    def validate_checkpoint_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Validate the physical-symmetric output contract marker."""

        key = "_output_contract_version"
        if key not in state_dict:
            raise ValueError(
                "Complex CouplingNet checkpoint has no output contract version. "
                "Physical-symmetric output contract version 6 is required."
            )
        version_tensor = state_dict[key]
        if version_tensor.numel() != 1:
            raise ValueError(
                "Invalid complex CouplingNet output contract version tensor."
            )
        version = int(version_tensor.detach().cpu().item())
        if version != self.OUTPUT_CONTRACT_VERSION:
            raise ValueError(
                "Incompatible complex CouplingNet output contract version "
                f"{version}; expected {self.OUTPUT_CONTRACT_VERSION}."
            )
        self._validate_pre_projection_fusion_checkpoint(state_dict)

    def _prepare_pre_projection_fusion_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """Upgrade unmarked v6 single-residual fuser checkpoints safely."""

        if self.pre_projection_fusion is None:
            return
        key = self.PRE_PROJECTION_FUSION_MODE_KEY
        if key in state_dict:
            self._validate_pre_projection_fusion_checkpoint(state_dict)
            return
        if self.pre_projection_fusion.mode != "residual":
            raise ValueError(
                "The complex CouplingNet checkpoint has no pre-projection "
                "fusion mode marker and is treated as a legacy residual-mode "
                "checkpoint. It cannot be loaded with "
                "pre_projection_fusion.mode='absolute'. Train the absolute "
                "mode from scratch."
            )
        state_dict[key] = (
            self.pre_projection_fusion._fusion_mode_code.detach().cpu().clone()
        )

    def _validate_pre_projection_fusion_checkpoint(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Reject checkpoints whose fuser semantics differ from the config."""

        if self.pre_projection_fusion is None:
            return
        key = self.PRE_PROJECTION_FUSION_MODE_KEY
        if key not in state_dict:
            raise ValueError(
                "Complex CouplingNet checkpoint has no pre-projection fusion "
                "mode marker after checkpoint preparation."
            )
        marker = state_dict[key]
        if marker.numel() != 1:
            raise ValueError(
                "Invalid pre-projection fusion mode marker: expected exactly one value."
            )
        checkpoint_code = int(marker.detach().cpu().item())
        expected_code = int(
            self.pre_projection_fusion._fusion_mode_code.detach().cpu().item()
        )
        if checkpoint_code != expected_code:
            raise ValueError(
                "Incompatible pre-projection fusion mode checkpoint: "
                f"checkpoint code {checkpoint_code}, expected {expected_code} "
                f"for mode '{self.pre_projection_fusion.mode}'. Residual and "
                "absolute fuser checkpoints cannot be cross-loaded."
            )

    def forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        x_source_branch: torch.Tensor,
        y_source_branch: torch.Tensor,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
        x_coefficient_branch: torch.Tensor,
        y_coefficient_branch: torch.Tensor,
        rhs_phys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw reference-response proposals shaped ``(B, 2, P)``."""

        base_response = self._base_response(
            geometry=geometry,
            x_source_branch=x_source_branch,
            y_source_branch=y_source_branch,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
            x_coefficient_branch=x_coefficient_branch,
            y_coefficient_branch=y_coefficient_branch,
        )
        if not self.pre_projection_fusion_enabled:
            return base_response
        if self.pre_projection_fusion is None:
            raise RuntimeError(
                "pre_projection_fusion must be initialized when enabled."
            )
        if rhs_phys is None:
            raise ValueError(
                "rhs_phys is required when pre_projection_fusion.enabled=true."
            )
        return cast(
            torch.Tensor,
            self.pre_projection_fusion(
                base_response=base_response,
                rhs_phys=rhs_phys,
                geometry=geometry,
                x_source_amplitude=x_source_amplitude,
                y_source_amplitude=y_source_amplitude,
            ),
        )

    def forward_with_fusion_diagnostics(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        x_source_branch: torch.Tensor,
        y_source_branch: torch.Tensor,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
        x_coefficient_branch: torch.Tensor,
        y_coefficient_branch: torch.Tensor,
        rhs_phys: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ComplexPreProjectionFusionResult | None]:
        """Return projection input plus optional fusion audit tensors."""

        base_response = self._base_response(
            geometry=geometry,
            x_source_branch=x_source_branch,
            y_source_branch=y_source_branch,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
            x_coefficient_branch=x_coefficient_branch,
            y_coefficient_branch=y_coefficient_branch,
        )
        if not self.pre_projection_fusion_enabled:
            return base_response, None
        if self.pre_projection_fusion is None:
            raise RuntimeError(
                "pre_projection_fusion must be initialized when enabled."
            )
        if rhs_phys is None:
            raise ValueError(
                "rhs_phys is required when pre_projection_fusion.enabled=true."
            )
        diagnostics = self.pre_projection_fusion.forward_with_diagnostics(
            base_response=base_response,
            rhs_phys=rhs_phys,
            geometry=geometry,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
        )
        return diagnostics.fused_response, diagnostics

    def _base_response(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        x_source_branch: torch.Tensor,
        y_source_branch: torch.Tensor,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
        x_coefficient_branch: torch.Tensor,
        y_coefficient_branch: torch.Tensor,
    ) -> torch.Tensor:
        phi_response = self._axis_forward(
            geometry=geometry,
            source_branch=x_source_branch,
            source_amplitude=x_source_amplitude,
            coefficient_branch=x_coefficient_branch,
            axis="x",
        )
        psi_response = self._axis_forward(
            geometry=geometry,
            source_branch=y_source_branch,
            source_amplitude=y_source_amplitude,
            coefficient_branch=y_coefficient_branch,
            axis="y",
        )
        return torch.stack((phi_response, psi_response), dim=1)

    def _axis_forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        source_branch: torch.Tensor,
        source_amplitude: torch.Tensor,
        coefficient_branch: torch.Tensor,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        if source_branch.dim() != 3 or source_branch.shape[-1] != self.branch_input_dim:
            raise ValueError(
                "complex source branch tensor must have shape (B, S, branch_input_dim)."
            )
        bsz, segment_count, _samples = source_branch.shape
        if axis == "x":
            expected_segments = geometry.num_x_segments
            segment_id = geometry.x_segment_id
        else:
            expected_segments = geometry.num_y_segments
            segment_id = geometry.y_segment_id
        if segment_count != expected_segments:
            raise ValueError(
                f"{axis}_source_branch segment count {segment_count} does not match "
                f"geometry count {expected_segments}."
            )
        if source_amplitude.shape != (bsz, segment_count):
            raise ValueError(
                f"{axis}_source_amplitude must have shape {(bsz, segment_count)}."
            )
        expected_coeff_shape = (
            bsz,
            segment_count,
            self.coefficient_branch_channels,
            self.branch_input_dim,
        )
        if coefficient_branch.shape != expected_coeff_shape:
            raise ValueError(
                f"{axis}_coefficient_branch must have shape {expected_coeff_shape}."
            )

        source_features = self._source_features(source_branch)
        branch_components = [source_features]
        if self.branch_coefficient is not None:
            branch_components.append(self._coefficient_features(coefficient_branch))
        transverse_features = self.branch_transverse(
            self._transverse_features(geometry, axis).to(source_branch.device)
        )
        geometry_features = self.branch_geometry(
            self._geometry_features(geometry, axis).to(source_branch.device)
        )
        branch_components.append(transverse_features.unsqueeze(0).expand(bsz, -1, -1))
        branch_components.append(geometry_features.unsqueeze(0).expand(bsz, -1, -1))
        segment_features = self._fuse_branch_components(branch_components)

        gathered_segment = segment_features[:, segment_id]
        primary_t, _transverse_t = self._trunk_coordinates(geometry, axis)
        trunk_features = self._pointwise_trunk_features(
            geometry=geometry,
            axis=axis,
            primary_t=primary_t,
            device=source_branch.device,
        )
        trunk_features = trunk_features.unsqueeze(0).expand(bsz, -1, -1)
        output_tilde = (gathered_segment * trunk_features).sum(dim=-1)
        response_scale = self._primary_length_squared(geometry, axis).to(
            device=source_branch.device,
            dtype=source_branch.dtype,
        )
        return (
            output_tilde * source_amplitude[:, segment_id] * response_scale.unsqueeze(0)
        )

    @staticmethod
    def _trunk_coordinates(
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if axis == "x":
            return geometry.x_local_t, geometry.y_local_t
        return geometry.y_local_t, geometry.x_local_t

    def _pointwise_trunk_features(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
        primary_t: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        primary_features = cast(
            torch.Tensor,
            self.trunk(primary_t.to(device).unsqueeze(-1)),
        )
        if not self.transverse_trunk_enabled:
            return primary_features
        if self.trunk_transverse is None:
            raise RuntimeError(
                "trunk_transverse must be initialized when "
                "axis_1d_trunk.transverse_trunk.enabled=true."
            )
        transverse_features = cast(
            torch.Tensor,
            self.trunk_transverse(
                self.transverse_length_context_features(geometry, axis).to(device)
            ),
        )
        product_features = primary_features * transverse_features
        if self.transverse_trunk_fusion_mode == "product":
            return product_features
        if self.transverse_trunk_fusion_mode == "product_fuser":
            if self.trunk_fuser is None:
                raise RuntimeError(
                    "trunk_fuser must be initialized when "
                    "axis_1d_trunk.transverse_trunk.fusion='product_fuser'."
                )
            fused = self.trunk_fuser(
                torch.cat(
                    (primary_features, transverse_features, product_features),
                    dim=-1,
                )
            )
            fused = cast(torch.Tensor, self.trunk_fuser_activation(fused))
            return cast(torch.Tensor, self.trunk_fuser_dropout(fused))
        raise ValueError(
            "Unsupported axis_1d_trunk.transverse_trunk.fusion: "
            f"{self.transverse_trunk_fusion_mode}"
        )

    @staticmethod
    def _primary_length_squared(
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        lengths = (
            geometry.x_lengths_for_valid_points()
            if axis == "x"
            else geometry.y_lengths_for_valid_points()
        )
        if torch.any(lengths <= 0.0):
            raise ValueError("Complex geometry segment lengths must be positive.")
        return lengths.square()

    def transverse_length_context_features(
        self,
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        """Return pointwise cross-axis response geometry features."""

        if not self.transverse_length_context_enabled:
            raise RuntimeError(
                "transverse length context is disabled for this ComplexCouplingNet."
            )
        if axis == "x":
            transverse_t = geometry.y_local_t
            parallel_length = geometry.x_lengths_for_valid_points()
            perpendicular_length = geometry.y_lengths_for_valid_points()
        else:
            transverse_t = geometry.x_local_t
            parallel_length = geometry.y_lengths_for_valid_points()
            perpendicular_length = geometry.x_lengths_for_valid_points()

        x_extent = geometry.y_transverse_max - geometry.y_transverse_min
        y_extent = geometry.x_transverse_max - geometry.x_transverse_min
        reference_length = torch.maximum(x_extent, y_extent)
        if reference_length <= 0.0:
            raise ValueError("Complex geometry reference extent must be positive.")
        if torch.any(parallel_length <= 0.0) or torch.any(perpendicular_length <= 0.0):
            raise ValueError("Complex geometry segment lengths must be positive.")

        sigma_parallel = parallel_length.square()
        sigma_perpendicular = perpendicular_length.square()
        response_sum = sigma_parallel + sigma_perpendicular
        kappa = 4.0 * sigma_parallel * sigma_perpendicular / response_sum.square()
        return torch.stack(
            (
                transverse_t,
                torch.log(perpendicular_length / reference_length),
                torch.log(parallel_length / perpendicular_length),
                kappa,
            ),
            dim=-1,
        )

    def _source_features(self, source_branch: torch.Tensor) -> torch.Tensor:
        bsz, segment_count, _samples = source_branch.shape
        flat = source_branch.reshape(bsz * segment_count, self.branch_input_dim)
        encoded = cast(torch.Tensor, self.branch_source(flat))
        return encoded.reshape(bsz, segment_count, self.hidden_dim)

    def _coefficient_features(self, coefficient_branch: torch.Tensor) -> torch.Tensor:
        if self.branch_coefficient is None:
            raise RuntimeError("branch_coefficient is not initialized.")
        bsz, segment_count, _channels, _samples = coefficient_branch.shape
        flat = coefficient_branch.reshape(
            bsz * segment_count,
            self.coefficient_branch_input_dim,
        )
        encoded = cast(torch.Tensor, self.branch_coefficient(flat))
        return encoded.reshape(bsz, segment_count, self.hidden_dim)

    def _geometry_features(
        self,
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        if axis == "x":
            left = geometry.x_segment_left
            right = geometry.x_segment_right
            length = geometry.x_segment_length
        else:
            left = geometry.y_segment_bottom
            right = geometry.y_segment_top
            length = geometry.y_segment_length
        mid = 0.5 * (left + right)
        return torch.cat(
            (
                left.unsqueeze(-1),
                right.unsqueeze(-1),
                mid.unsqueeze(-1),
                length.unsqueeze(-1),
                length.pow(2).unsqueeze(-1),
                (1.0 / length).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _transverse_features(
        self,
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        if axis == "x":
            r_hat = geometry.x_transverse_normalized()
        else:
            r_hat = geometry.y_transverse_normalized()
        frequencies = self.transverse_frequencies.to(
            device=r_hat.device, dtype=r_hat.dtype
        )
        phase = (
            2.0
            * math.pi
            * torch.unsqueeze(r_hat, -1)
            * torch.unsqueeze(
                frequencies,
                0,
            )
        )
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)

    def _fuse_branch_components(self, components: list[torch.Tensor]) -> torch.Tensor:
        if not components:
            raise ValueError("At least one branch component is required.")
        product_feature = components[0]
        for component in components[1:]:
            product_feature = product_feature * component
        if self.branch_fusion_mode == "product":
            return product_feature
        if self.branch_fusion_mode == "product_fuser":
            if self.branch_fuser is None:
                raise RuntimeError(
                    "branch_fuser must be initialized when "
                    "branch_fusion.mode='product_fuser'."
                )
            fused = self.branch_fuser(torch.cat(components + [product_feature], dim=-1))
            fused = cast(torch.Tensor, self.branch_fuser_activation(fused))
            return cast(torch.Tensor, self.branch_fuser_dropout(fused))
        raise ValueError(f"Unsupported branch_fusion mode: {self.branch_fusion_mode}")
