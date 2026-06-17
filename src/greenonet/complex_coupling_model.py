from __future__ import annotations

import math
from typing import Literal, cast

import torch
from torch import nn

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import CouplingModelConfig
from greenonet.coupling_model import ActivationFactoryMixin, MLP


class ComplexCouplingNet(nn.Module, ActivationFactoryMixin):
    """Segment-wise CouplingNet for precomputed complex geometries."""

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        self.config = config
        self.branch_input_dim = int(config.branch_input_dim)
        self.hidden_dim = int(config.hidden_dim)
        if self.branch_input_dim < 2:
            raise ValueError("coupling_model.branch_input_dim must be at least 2.")

        pe_dim = 2 * int(config.trunk_positional_encoding.num_frequencies)
        self.geometry_feature_dim = pe_dim + 6
        self.function_branch = MLP(
            input_dim=3 * self.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.geometry_branch = MLP(
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
        self.product_fuser = nn.Sequential(
            nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=config.use_bias),
            self.build_activation(config.activation),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, 1, bias=config.use_bias),
        )

    def forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        x_branch: torch.Tensor,
        y_branch: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw unit outputs shaped ``(B, 2, P)``."""

        phi = self._axis_forward(
            geometry=geometry,
            branch=x_branch,
            axis="x",
        )
        psi = self._axis_forward(
            geometry=geometry,
            branch=y_branch,
            axis="y",
        )
        return torch.stack((phi, psi), dim=1)

    def _axis_forward(
        self,
        *,
        geometry: ComplexGeometryMetadata,
        branch: torch.Tensor,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        if branch.dim() != 4 or branch.shape[2:] != (3, self.branch_input_dim):
            raise ValueError(
                "complex branch tensor must have shape (B, S, 3, branch_input_dim)."
            )
        bsz, segment_count, _channels, _samples = branch.shape
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
                f"{axis}_branch segment count {segment_count} does not match "
                f"geometry count {expected_segments}."
            )

        function_features = self._function_features(branch)
        geometry_features = self._geometry_features(geometry, axis).to(branch.device)
        geometry_encoded = self.geometry_branch(geometry_features)
        segment_features = function_features * geometry_encoded.unsqueeze(0)
        gathered_segment = segment_features[:, segment_id]
        trunk_features = self.trunk(local_t.to(branch.device).unsqueeze(-1))
        trunk_features = trunk_features.unsqueeze(0).expand(bsz, -1, -1)
        fused = torch.cat(
            (
                gathered_segment,
                trunk_features,
                gathered_segment * trunk_features,
                gathered_segment * geometry_encoded[segment_id].unsqueeze(0),
            ),
            dim=-1,
        )
        return cast(torch.Tensor, self.product_fuser(fused)).squeeze(-1)

    def _function_features(self, branch: torch.Tensor) -> torch.Tensor:
        bsz, segment_count, _channels, _samples = branch.shape
        flat = branch.reshape(bsz * segment_count, 3 * self.branch_input_dim)
        encoded = cast(torch.Tensor, self.function_branch(flat))
        return encoded.reshape(bsz, segment_count, self.hidden_dim)

    def _geometry_features(
        self,
        geometry: ComplexGeometryMetadata,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        if axis == "x":
            r = geometry.x_segment_y
            left = geometry.x_segment_left
            right = geometry.x_segment_right
            length = geometry.x_segment_length
        else:
            r = geometry.y_segment_x
            left = geometry.y_segment_bottom
            right = geometry.y_segment_top
            length = geometry.y_segment_length
        mid = 0.5 * (left + right)
        pe = self._positional_encode_transverse(r)
        return torch.cat(
            (
                pe,
                left.unsqueeze(-1),
                right.unsqueeze(-1),
                mid.unsqueeze(-1),
                length.unsqueeze(-1),
                length.pow(2).unsqueeze(-1),
                (1.0 / length).unsqueeze(-1),
            ),
            dim=-1,
        )

    def _positional_encode_transverse(self, r: torch.Tensor) -> torch.Tensor:
        cfg = self.config.trunk_positional_encoding
        num = int(cfg.num_frequencies)
        max_frequency = float(cfg.max_frequency)
        if num <= 0:
            raise ValueError("trunk_positional_encoding.num_frequencies must be > 0.")
        if max_frequency <= 0.0:
            raise ValueError("trunk_positional_encoding.max_frequency must be > 0.")
        if num == 1:
            frequencies = torch.ones(1, dtype=r.dtype, device=r.device)
        else:
            frequencies = torch.logspace(
                0.0,
                math.log2(max_frequency),
                steps=num,
                base=2.0,
                dtype=r.dtype,
                device=r.device,
            )
        phase = math.pi * r.unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat((phase.sin(), phase.cos()), dim=-1)
