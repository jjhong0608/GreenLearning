from __future__ import annotations

import math
from typing import List, cast

import torch
from torch import nn

from greenonet.activations import RationalActivation
from greenonet.config import (
    CouplingGatedAttentionFusionConfig,
    CouplingLineEncoderConfig,
    CouplingModelConfig,
    CouplingTrunkFourierConfig,
)


class ActivationFactoryMixin:
    """Build activation modules from string names."""

    @staticmethod
    def build_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "rational":
            return RationalActivation()
        raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        activation: str,
        use_bias: bool,
        dropout: float,
        last_activation: bool = False,
        output_dim: int | None = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
            layers.append(self.build_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        final_dim = hidden_dim if output_dim is None else output_dim
        layers.append(nn.Linear(in_dim, final_dim, bias=use_bias))
        if last_activation:
            layers.append(self.build_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class LineGeometryMixin:
    """Normalize line/geometry inputs for the CNN encoders."""

    @staticmethod
    def _as_line_channel(name: str, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 2:
            return value.unsqueeze(1)
        if value.dim() == 3:
            return value
        raise ValueError(f"{name} must be 2D or 3D, got shape {tuple(value.shape)}.")

    @classmethod
    def _expand_geometry(
        cls,
        name: str,
        value: torch.Tensor,
        batch_size: int,
        m_points: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if value.dim() == 1:
            value = value.view(1, 1, -1)
        elif value.dim() == 2:
            value = value.unsqueeze(1)
        elif value.dim() != 3:
            raise ValueError(f"{name} must be 1D, 2D, or 3D, got {tuple(value.shape)}.")
        if value.shape[-1] != m_points:
            raise ValueError(
                f"{name} must have {m_points} points along the line axis, got {value.shape[-1]}."
            )
        value = value.to(dtype=dtype, device=device)
        if value.shape[0] == 1 and batch_size != 1:
            value = value.expand(batch_size, -1, -1)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must be 1 or {batch_size}, got {value.shape[0]}."
            )
        if value.shape[1] != 1:
            raise ValueError(f"{name} must have exactly one channel, got {value.shape[1]}.")
        return value

    @staticmethod
    def _default_geometry(
        batch_size: int,
        m_points: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.linspace(0.0, 1.0, steps=m_points, dtype=dtype, device=device)
        pos = pos.view(1, 1, m_points).expand(batch_size, -1, -1)
        db = torch.minimum(pos, 1.0 - pos)
        return pos, db


class CouplingLineEncoder(nn.Module, ActivationFactoryMixin, LineGeometryMixin):  # type: ignore[misc]
    """CNN-based encoder for one axial line plus geometry channels."""

    def __init__(
        self,
        config: CouplingLineEncoderConfig,
        output_dim: int,
        fallback_hidden_dim: int,
        fallback_depth: int,
        fallback_activation: str,
        fallback_use_bias: bool,
        fallback_dropout: float,
    ) -> None:
        super().__init__()
        self.config = config
        if config.type != "cnn1d":
            raise ValueError(f"Unsupported line encoder type: {config.type}")
        if config.pooling != "meanmax":
            raise ValueError(f"Unsupported line encoder pooling mode: {config.pooling}")
        if config.kernel_size % 2 == 0:
            raise ValueError("line_encoder.kernel_size must be odd to preserve line length.")
        if len(config.conv_channels) == 0:
            raise ValueError("line_encoder.conv_channels must be non-empty.")
        if len(config.conv_channels) != len(config.dilations):
            raise ValueError(
                "line_encoder.conv_channels and line_encoder.dilations must have matching lengths."
            )
        expected_channels = 1 + int(config.include_position) + int(
            config.include_boundary_distance
        )
        if config.in_channels != expected_channels:
            raise ValueError(
                "line_encoder.in_channels must match the enabled geometry channels "
                f"(expected {expected_channels}, got {config.in_channels})."
            )

        activation_name = config.activation or fallback_activation
        layers: List[nn.Module] = []
        in_channels = config.in_channels
        for out_channels, dilation in zip(config.conv_channels, config.dilations):
            padding = dilation * (config.kernel_size - 1) // 2
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    dilation=dilation,
                    padding=padding,
                    bias=fallback_use_bias,
                )
            )
            layers.append(self.build_activation(activation_name))
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*layers)

        head_cfg = config.mlp_head
        head_hidden_dim = (
            fallback_hidden_dim if head_cfg.hidden_dim is None else head_cfg.hidden_dim
        )
        head_depth = fallback_depth if head_cfg.depth is None else head_cfg.depth
        head_activation = (
            fallback_activation if head_cfg.activation is None else head_cfg.activation
        )
        head_use_bias = (
            fallback_use_bias if head_cfg.use_bias is None else head_cfg.use_bias
        )
        head_dropout = (
            fallback_dropout if head_cfg.dropout is None else head_cfg.dropout
        )
        pooled_dim = 2 * config.conv_channels[-1]
        self.mlp_head = MLP(
            input_dim=pooled_dim,
            hidden_dim=head_hidden_dim,
            depth=head_depth,
            activation=head_activation,
            use_bias=head_use_bias,
            dropout=head_dropout,
            output_dim=output_dim,
        )

    def forward(
        self,
        line: torch.Tensor,
        pos: torch.Tensor | None = None,
        db: torch.Tensor | None = None,
    ) -> torch.Tensor:
        line_channel = self._as_line_channel("line", line)
        batch_size, _, m_points = line_channel.shape
        if pos is None or db is None:
            pos_channel, db_channel = self._default_geometry(
                batch_size=batch_size,
                m_points=m_points,
                dtype=line_channel.dtype,
                device=line_channel.device,
            )
        else:
            pos_channel = self._expand_geometry(
                "pos",
                pos,
                batch_size=batch_size,
                m_points=m_points,
                dtype=line_channel.dtype,
                device=line_channel.device,
            )
            db_channel = self._expand_geometry(
                "db",
                db,
                batch_size=batch_size,
                m_points=m_points,
                dtype=line_channel.dtype,
                device=line_channel.device,
            )

        channels = [line_channel]
        if self.config.include_position:
            channels.append(pos_channel)
        if self.config.include_boundary_distance:
            channels.append(db_channel)
        encoder_input = torch.cat(channels, dim=1)
        if encoder_input.shape[1] != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} encoder channels, got {encoder_input.shape[1]}."
            )
        features = cast(torch.Tensor, self.conv_stack(encoder_input))
        pooled = torch.cat((features.mean(dim=-1), features.amax(dim=-1)), dim=-1)
        return self.mlp_head(pooled)


class CouplingFixedFourierFeatures(nn.Module):  # type: ignore[misc]
    """Fixed-frequency Fourier embedding for trunk coordinates only."""

    def __init__(
        self,
        input_dim: int,
        config: CouplingTrunkFourierConfig,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("trunk_input_dim must be positive.")
        if len(config.frequencies) == 0:
            raise ValueError(
                "coupling_model.trunk_fourier.frequencies must be non-empty when enabled."
            )
        for freq in config.frequencies:
            if not math.isfinite(freq) or freq <= 0.0:
                raise ValueError(
                    "coupling_model.trunk_fourier.frequencies must contain positive finite values."
                )
        self.input_dim = input_dim
        self.include_input = include_input
        self.register_buffer(
            "frequencies",
            torch.tensor(config.frequencies, dtype=torch.float64),
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        base_dim = self.input_dim if self.include_input else 0
        return base_dim + 2 * self.input_dim * int(self.frequencies.numel())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected trunk coordinates with trailing dim {self.input_dim}, got {coords.shape[-1]}."
            )
        frequencies = self.frequencies.to(dtype=coords.dtype, device=coords.device)
        embedded: List[torch.Tensor] = [coords] if self.include_input else []
        for freq in frequencies:
            scaled = coords * freq
            embedded.append(torch.sin(scaled))
            embedded.append(torch.cos(scaled))
        return torch.cat(embedded, dim=-1)


class CouplingGatedAttentionFusion(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """Single-head self-attention plus gated pooling over branch tokens."""

    GATE_EPS = 1e-12

    def __init__(
        self,
        hidden_dim: int,
        config: CouplingGatedAttentionFusionConfig,
        fallback_activation: str,
        fallback_use_bias: bool,
        fallback_dropout: float,
    ) -> None:
        super().__init__()
        activation_name = config.activation or fallback_activation
        use_bias = fallback_use_bias if config.use_bias is None else config.use_bias
        dropout = fallback_dropout if config.dropout is None else config.dropout

        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.gate_proj = nn.Linear(hidden_dim, 1, bias=use_bias)
        self.pool_proj = nn.Linear(hidden_dim, 1, bias=use_bias)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_bias)
        self.output_activation = self.build_activation(activation_name)
        self.output_dropout: nn.Module
        if dropout > 0:
            self.output_dropout = nn.Dropout(dropout)
        else:
            self.output_dropout = nn.Identity()
        self.scale = float(hidden_dim) ** -0.5

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(tokens)
        key = self.key_proj(tokens)
        value = self.value_proj(tokens)

        attn_logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, value)
        updated = tokens + attended

        gate = torch.sigmoid(self.gate_proj(updated)).squeeze(-1)
        pool_logits = self.pool_proj(updated).squeeze(-1)
        pool_weights = torch.softmax(pool_logits, dim=-1)
        combined_weights = gate * pool_weights
        combined_weights = combined_weights / combined_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.GATE_EPS)

        pooled = (updated * combined_weights.unsqueeze(-1)).sum(dim=1)
        fused = self.output_proj(pooled)
        fused = cast(torch.Tensor, self.output_activation(fused))
        return cast(torch.Tensor, self.output_dropout(fused))


class CouplingNet(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """MIONet-style model with CNN line encoders and one trunk over coordinates."""

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        self.config = config
        self.branch_a = CouplingLineEncoder(
            config=config.line_encoder,
            output_dim=config.hidden_dim,
            fallback_hidden_dim=config.hidden_dim,
            fallback_depth=config.depth,
            fallback_activation=config.activation,
            fallback_use_bias=config.use_bias,
            fallback_dropout=config.dropout,
        )
        self.branch_ap = CouplingLineEncoder(
            config=config.line_encoder,
            output_dim=config.hidden_dim,
            fallback_hidden_dim=config.hidden_dim,
            fallback_depth=config.depth,
            fallback_activation=config.activation,
            fallback_use_bias=config.use_bias,
            fallback_dropout=config.dropout,
        )
        self.branch_b = CouplingLineEncoder(
            config=config.line_encoder,
            output_dim=config.hidden_dim,
            fallback_hidden_dim=config.hidden_dim,
            fallback_depth=config.depth,
            fallback_activation=config.activation,
            fallback_use_bias=config.use_bias,
            fallback_dropout=config.dropout,
        )
        self.branch_c = CouplingLineEncoder(
            config=config.line_encoder,
            output_dim=config.hidden_dim,
            fallback_hidden_dim=config.hidden_dim,
            fallback_depth=config.depth,
            fallback_activation=config.activation,
            fallback_use_bias=config.use_bias,
            fallback_dropout=config.dropout,
        )
        self.branch_rhs = CouplingLineEncoder(
            config=config.line_encoder,
            output_dim=config.hidden_dim,
            fallback_hidden_dim=config.hidden_dim,
            fallback_depth=config.depth,
            fallback_activation=config.activation,
            fallback_use_bias=config.use_bias,
            fallback_dropout=config.dropout,
        )
        if config.fusion.type not in {"linear", "gated_attention"}:
            raise ValueError(f"Unsupported coupling_model.fusion.type: {config.fusion.type}")
        self.branch_fusion_type = config.fusion.type
        self.branch_fuser: nn.Linear | None = None
        self.branch_activation: nn.Module | None = None
        self.branch_dropout: nn.Module | None = None
        self.branch_gated_attention: CouplingGatedAttentionFusion | None = None
        if self.branch_fusion_type == "linear":
            self.branch_fuser = nn.Linear(
                5 * config.hidden_dim,
                config.hidden_dim,
                bias=config.use_bias,
            )
            self.branch_activation = self.build_activation(config.activation)
            if config.dropout > 0:
                self.branch_dropout = nn.Dropout(config.dropout)
            else:
                self.branch_dropout = nn.Identity()
        else:
            self.branch_gated_attention = CouplingGatedAttentionFusion(
                hidden_dim=config.hidden_dim,
                config=config.fusion.gated_attention,
                fallback_activation=config.activation,
                fallback_use_bias=config.use_bias,
                fallback_dropout=config.dropout,
            )
        self.trunk_fourier: CouplingFixedFourierFeatures | None
        trunk_input_dim = config.trunk_input_dim
        if config.trunk_fourier.enabled:
            self.trunk_fourier = CouplingFixedFourierFeatures(
                input_dim=config.trunk_input_dim,
                config=config.trunk_fourier,
                include_input=True,
            )
            trunk_input_dim = self.trunk_fourier.output_dim
        else:
            self.trunk_fourier = None
        self.trunk = MLP(
            input_dim=trunk_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=True,
        )

    def _fuse_branch_tokens(self, branch_tokens: torch.Tensor) -> torch.Tensor:
        if self.branch_fusion_type == "linear":
            if (
                self.branch_fuser is None
                or self.branch_activation is None
                or self.branch_dropout is None
            ):
                raise RuntimeError("Linear branch fusion is not initialized.")
            branch_feat = self.branch_fuser(branch_tokens.reshape(branch_tokens.shape[0], -1))
            branch_feat = cast(torch.Tensor, self.branch_activation(branch_feat))
            return cast(torch.Tensor, self.branch_dropout(branch_feat))
        if self.branch_gated_attention is None:
            raise RuntimeError("Gated-attention branch fusion is not initialized.")
        return self.branch_gated_attention(branch_tokens)

    def _apply_balance_projection(
        self, flux_int: torch.Tensor, rhs_raw: torch.Tensor
    ) -> torch.Tensor:
        """Apply interior balance projection with a fixed 1/2 residual split."""
        rhs_x_int = rhs_raw[:, 0, :, 1:-1]
        phi = flux_int[:, 0]
        psi_t = flux_int[:, 1].transpose(-1, -2)
        res = rhs_x_int - (phi + psi_t)

        phi = phi + 0.5 * res
        psi_t = psi_t + 0.5 * res
        projected = flux_int.clone()
        projected[:, 0] = phi
        projected[:, 1] = psi_t.transpose(-1, -2)
        return projected

    @staticmethod
    def _line_geometry_from_coords(
        coords: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        line_pos = torch.stack((coords[0, :, :, 0], coords[1, :, :, 1]), dim=0)
        axis, n_lines, m_points = line_pos.shape
        line_pos = line_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        line_pos = line_pos.reshape(batch_size * axis * n_lines, m_points)
        db = torch.minimum(line_pos, 1.0 - line_pos)
        return line_pos, db

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords: (2, n_lines, m_points, 2) (interior trunk uses coords[:, :, 1:-1])
            a_vals/ap_vals/b_vals/c_vals: (B, 2, n_lines, m_points)
            rhs_raw: (B, 2, n_lines, m_points) raw source
            rhs_tilde: (B, 2, n_lines, m_points) normalized source
            rhs_norm: (B, 2, n_lines) line-wise L2 norms
        Returns:
            projected_flux: (B, 2, n_lines, m_points) axial flux-divergence per
                axis after the cross-axis balance projection with zero-padded
                boundaries.
        """
        if (
            coords.dim() != 4
            or a_vals.dim() != 4
            or ap_vals.dim() != 4
            or b_vals.dim() != 4
            or c_vals.dim() != 4
            or rhs_raw.dim() != 4
            or rhs_tilde.dim() != 4
            or rhs_norm.dim() != 3
        ):
            raise ValueError(
                "coords must be 4D; a/ap/b/c/rhs_raw/rhs_tilde (B,2,n,m); rhs_norm (B,2,n)"
            )
        b, axis, n_lines, m_points = a_vals.shape
        if m_points != self.config.branch_input_dim:
            raise ValueError(
                f"Expected branch_input_dim == m_points (got {self.config.branch_input_dim} vs {m_points})."
            )
        if n_lines + 2 != m_points:
            raise ValueError(
                f"Expected n_lines + 2 == m_points (got n_lines={n_lines}, m_points={m_points})."
            )

        pos_flat, db_flat = self._line_geometry_from_coords(coords, b)
        branch_a_in = a_vals.reshape(b * axis * n_lines, m_points)
        branch_ap_in = ap_vals.reshape(b * axis * n_lines, m_points)
        branch_b_in = b_vals.reshape(b * axis * n_lines, m_points)
        branch_c_in = c_vals.reshape(b * axis * n_lines, m_points)
        branch_rhs_in = rhs_tilde.reshape(b * axis * n_lines, m_points)
        branch_tokens = torch.stack(
            (
                self.branch_a(branch_a_in, pos_flat, db_flat),
                self.branch_ap(branch_ap_in, pos_flat, db_flat),
                self.branch_b(branch_b_in, pos_flat, db_flat),
                self.branch_c(branch_c_in, pos_flat, db_flat),
                self.branch_rhs(branch_rhs_in, pos_flat, db_flat),
            ),
            dim=1,
        )
        branch_feat = self._fuse_branch_tokens(branch_tokens)
        branch_feat = branch_feat.unsqueeze(-1)

        coords_int = coords[:, :, 1:-1, :]
        trunk_flat = coords_int.reshape(axis * n_lines * (m_points - 2), -1)
        if trunk_flat.shape[-1] != self.config.trunk_input_dim:
            raise ValueError(
                "coords trunk dimension must match coupling_model.trunk_input_dim "
                f"(expected {self.config.trunk_input_dim}, got {trunk_flat.shape[-1]})."
            )
        if self.trunk_fourier is not None:
            trunk_flat = self.trunk_fourier(trunk_flat)
        trunk_out = self.trunk(trunk_flat).reshape(axis * n_lines, m_points - 2, -1)

        trunk_expanded = trunk_out.unsqueeze(0).expand(b, -1, -1, -1)
        trunk_expanded = trunk_expanded.reshape(b * axis * n_lines, m_points - 2, -1)
        combined = torch.bmm(trunk_expanded, branch_feat)
        flux_tilde = combined.reshape(b, axis, n_lines, m_points - 2)

        norm_exp = rhs_norm.unsqueeze(-1)
        raw_int = flux_tilde * norm_exp
        projected_int = self._apply_balance_projection(raw_int, rhs_raw)

        projected_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=raw_int.dtype,
            device=raw_int.device,
        )
        projected_flux[:, :, :, 1:-1] = projected_int
        return projected_flux
