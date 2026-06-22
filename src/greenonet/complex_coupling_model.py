from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import nn

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingBranchFusionConfig,
    CouplingModelConfig,
)
from greenonet.coupling_model import ActivationFactoryMixin, MLP


class ComplexCouplingNet(nn.Module, ActivationFactoryMixin):
    """Source-conditioned CouplingNet for precomputed complex geometries."""

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        self.config = config
        self._validate_complex_config(config)
        self.branch_input_dim = int(config.branch_input_dim)
        self.hidden_dim = int(config.hidden_dim)
        if self.branch_input_dim < 2:
            raise ValueError("coupling_model.branch_input_dim must be at least 2.")

        coefficient_terms = config.coefficient_terms
        active_coefficient_terms = []
        if coefficient_terms.diffusion:
            active_coefficient_terms.append("diffusion")
        if coefficient_terms.convection:
            active_coefficient_terms.append("convection")
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

    @staticmethod
    def _validate_complex_config(config: CouplingModelConfig) -> None:
        if config.trunk_positional_encoding.enabled:
            raise ValueError(
                "ComplexCouplingNet uses local 1D trunk coordinates; "
                "trunk_positional_encoding.enabled must be false. Use "
                "axis_1d_trunk.num_frequencies/max_frequency for normalized "
                "transverse encoding."
            )
        balance_projection = BalanceProjectionConfig.from_raw(config.balance_projection)
        if not balance_projection.enabled:
            raise ValueError(
                "ComplexCouplingNet requires balance_projection.enabled=true."
            )
        if balance_projection.mode != "symmetric":
            raise ValueError(
                "ComplexCouplingNet supports only symmetric balance projection."
            )
        if config.source_stencil_lift.enabled:
            raise ValueError("ComplexCouplingNet does not support source_stencil_lift.")
        if config.green_response_feature.enabled:
            raise ValueError(
                "ComplexCouplingNet does not support green_response_feature."
            )

    def forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        x_source_branch: torch.Tensor,
        y_source_branch: torch.Tensor,
        x_source_norm: torch.Tensor,
        y_source_norm: torch.Tensor,
        x_coefficient_branch: torch.Tensor,
        y_coefficient_branch: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw unit outputs shaped ``(B, 2, P)``."""

        phi = self._axis_forward(
            geometry=geometry,
            source_branch=x_source_branch,
            source_norm=x_source_norm,
            coefficient_branch=x_coefficient_branch,
            axis="x",
        )
        psi = self._axis_forward(
            geometry=geometry,
            source_branch=y_source_branch,
            source_norm=y_source_norm,
            coefficient_branch=y_coefficient_branch,
            axis="y",
        )
        return torch.stack((phi, psi), dim=1)

    def _axis_forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        source_branch: torch.Tensor,
        source_norm: torch.Tensor,
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
            local_t = geometry.x_local_t
        else:
            expected_segments = geometry.num_y_segments
            segment_id = geometry.y_segment_id
            local_t = geometry.y_local_t
        if segment_count != expected_segments:
            raise ValueError(
                f"{axis}_source_branch segment count {segment_count} does not match "
                f"geometry count {expected_segments}."
            )
        if source_norm.shape != (bsz, segment_count):
            raise ValueError(
                f"{axis}_source_norm must have shape {(bsz, segment_count)}."
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
        trunk_features = self.trunk(local_t.to(source_branch.device).unsqueeze(-1))
        trunk_features = trunk_features.unsqueeze(0).expand(bsz, -1, -1)
        output_tilde = (gathered_segment * trunk_features).sum(dim=-1)
        return cast(torch.Tensor, output_tilde * source_norm[:, segment_id])

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
