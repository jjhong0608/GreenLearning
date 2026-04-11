from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import torch
from torch import nn

from greenonet.activations import RationalActivation
from greenonet.config import CouplingModelConfig, CouplingQHeadConfig


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


class BranchCombinationMixin:
    """Share the current MIONet branch-combination rule across heads."""

    @staticmethod
    def combine_branch_latents(*latents: torch.Tensor) -> torch.Tensor:
        if not latents:
            raise ValueError("at least one branch latent is required")
        combined = latents[0]
        for latent in latents[1:]:
            combined = combined * latent
        return combined


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


@dataclass
class CouplingForwardOutputs:
    raw_flux: torch.Tensor
    projected_flux: torch.Tensor


class CouplingQHead(nn.Module, BranchCombinationMixin):  # type: ignore[misc]
    """Shared-trunk two-branch MIONet head for the auxiliary q field."""

    def __init__(
        self,
        q_config: CouplingQHeadConfig,
        model_config: CouplingModelConfig,
        trunk: nn.Module,
    ) -> None:
        super().__init__()
        if not q_config.share_trunk:
            raise ValueError("coupling_model.q_head.share_trunk must be true.")
        if q_config.fusion != "add_transpose":
            raise ValueError(
                "coupling_model.q_head.fusion must be 'add_transpose'."
            )
        latent_dim = (
            model_config.hidden_dim if q_config.latent_dim is None else q_config.latent_dim
        )
        if latent_dim != model_config.hidden_dim:
            raise ValueError(
                "coupling_model.q_head.latent_dim must match coupling_model.hidden_dim "
                "when the trunk is shared."
            )

        s_hidden_dim = (
            model_config.hidden_dim
            if q_config.s_branch_hidden_dim is None
            else q_config.s_branch_hidden_dim
        )
        s_depth = model_config.depth if q_config.s_branch_depth is None else q_config.s_branch_depth
        m_hidden_dim = (
            model_config.hidden_dim
            if q_config.m_branch_hidden_dim is None
            else q_config.m_branch_hidden_dim
        )
        m_depth = model_config.depth if q_config.m_branch_depth is None else q_config.m_branch_depth

        self.branch_s = MLP(
            input_dim=model_config.branch_input_dim,
            hidden_dim=s_hidden_dim,
            depth=s_depth,
            activation=model_config.activation,
            use_bias=model_config.use_bias,
            dropout=model_config.dropout,
            last_activation=False,
            output_dim=latent_dim,
        )
        self.branch_m = MLP(
            input_dim=model_config.branch_input_dim,
            hidden_dim=m_hidden_dim,
            depth=m_depth,
            activation=model_config.activation,
            use_bias=model_config.use_bias,
            dropout=model_config.dropout,
            last_activation=False,
            output_dim=latent_dim,
        )
        object.__setattr__(self, "_shared_trunk", trunk)

    @property
    def trunk(self) -> nn.Module:
        return cast(nn.Module, getattr(self, "_shared_trunk"))

    def forward(
        self,
        coords_view: torch.Tensor,
        s_view: torch.Tensor,
        m_view: torch.Tensor,
    ) -> torch.Tensor:
        if coords_view.dim() != 3 or coords_view.shape[-1] != 2:
            raise ValueError("coords_view must have shape (n_lines, m_points, 2).")
        if s_view.shape != m_view.shape or s_view.dim() != 3:
            raise ValueError("s_view and m_view must share shape (B, n_lines, m_points).")
        batch_size, n_lines, m_points = s_view.shape
        if coords_view.shape[:2] != (n_lines, m_points):
            raise ValueError("coords_view must align with the q-head line layout.")

        branch_s_in = s_view.reshape(batch_size * n_lines, m_points)
        branch_m_in = m_view.reshape(batch_size * n_lines, m_points)
        branch_s_out = self.branch_s(branch_s_in)
        branch_m_out = self.branch_m(branch_m_in)
        branch_feat = self.combine_branch_latents(branch_s_out, branch_m_out).unsqueeze(-1)

        trunk_flat = coords_view.reshape(n_lines * m_points, -1)
        trunk_out = self.trunk(trunk_flat).reshape(n_lines, m_points, -1)
        trunk_expanded = trunk_out.unsqueeze(0).expand(batch_size, -1, -1, -1)
        trunk_expanded = trunk_expanded.reshape(batch_size * n_lines, m_points, -1)
        q_view = torch.bmm(trunk_expanded, branch_feat).reshape(batch_size, n_lines, m_points)

        q_view = q_view.clone()
        q_view[:, 0, :] = 0.0
        q_view[:, -1, :] = 0.0
        q_view[..., 0] = 0.0
        q_view[..., -1] = 0.0
        return q_view


class CouplingNet(nn.Module, ActivationFactoryMixin, BranchCombinationMixin):  # type: ignore[misc]
    """MIONet-style model: four branches (a, b, c, rhs) and one trunk (coords).

    The network ingests the full stack of axial lines per batch (both x- and y-lines)
    and predicts axial flux-divergences per axis on interior points, then zero-pads
    the boundaries.
    """

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        self.config = config
        self.branch_a = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.branch_b = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.branch_c = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.branch_rhs = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.trunk = MLP(
            input_dim=config.trunk_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=True,
        )
        self.q_head = (
            CouplingQHead(config.q_head, config, self.trunk)
            if config.q_head.enabled
            else None
        )

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

    def _forward_projected_and_raw(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> CouplingForwardOutputs:
        """
        Args:
            coords: (2, n_lines, m_points, 2) (interior trunk uses coords[:, :, 1:-1])
            a_vals: (B, 2, n_lines, m_points)
            b_vals: (B, 2, n_lines, m_points)
            c_vals: (B, 2, n_lines, m_points)
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
            or b_vals.dim() != 4
            or c_vals.dim() != 4
            or rhs_raw.dim() != 4
            or rhs_tilde.dim() != 4
            or rhs_norm.dim() != 3
        ):
            raise ValueError(
                "coords must be 4D; a/b/c/rhs_raw/rhs_tilde (B,2,n,m); rhs_norm (B,2,n)"
            )
        b, axis, n_lines, m_points = a_vals.shape
        if n_lines + 2 != m_points:
            raise ValueError(
                f"Expected n_lines + 2 == m_points (got n_lines={n_lines}, m_points={m_points})."
            )

        branch_a_in = a_vals.reshape(b * axis * n_lines, m_points)
        branch_r_in = rhs_tilde.reshape(b * axis * n_lines, m_points)
        branch_a_out = self.branch_a(branch_a_in)
        branch_r_out = self.branch_rhs(branch_r_in)
        branch_feat = self.combine_branch_latents(branch_a_out, branch_r_out)
        branch_feat = branch_feat.unsqueeze(-1)

        coords_int = coords[:, :, 1:-1, :]
        trunk_flat = coords_int.reshape(axis * n_lines * (m_points - 2), -1)
        trunk_out = self.trunk(trunk_flat).reshape(axis * n_lines, m_points - 2, -1)

        trunk_expanded = trunk_out.unsqueeze(0).expand(b, -1, -1, -1)
        trunk_expanded = trunk_expanded.reshape(b * axis * n_lines, m_points - 2, -1)
        combined = torch.bmm(trunk_expanded, branch_feat)
        flux_tilde = combined.reshape(b, axis, n_lines, m_points - 2)

        norm_exp = rhs_norm.unsqueeze(-1)
        raw_int = flux_tilde * norm_exp
        projected_int = self._apply_balance_projection(raw_int, rhs_raw)

        raw_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=raw_int.dtype,
            device=raw_int.device,
        )
        projected_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=raw_int.dtype,
            device=raw_int.device,
        )
        raw_flux[:, :, :, 1:-1] = raw_int
        projected_flux[:, :, :, 1:-1] = projected_int
        return CouplingForwardOutputs(raw_flux=raw_flux, projected_flux=projected_flux)

    def forward_with_aux(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> CouplingForwardOutputs:
        return self._forward_projected_and_raw(
            coords=coords,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            rhs_tilde=rhs_tilde,
            rhs_norm=rhs_norm,
        )

    def forward_q_head(
        self,
        coords_view: torch.Tensor,
        s_view: torch.Tensor,
        m_view: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_head is None:
            raise ValueError("q_head is disabled in coupling_model.q_head.enabled.")
        return self.q_head(coords_view=coords_view, s_view=s_view, m_view=m_view)

    def forward(
        self,
        coords: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        rhs_tilde: torch.Tensor,
        rhs_norm: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self._forward_projected_and_raw(
            coords=coords,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            rhs_tilde=rhs_tilde,
            rhs_norm=rhs_norm,
        )
        return outputs.projected_flux
