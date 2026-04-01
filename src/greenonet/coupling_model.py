from __future__ import annotations

from typing import List, cast, Optional

import torch
from torch import nn

from greenonet.activations import RationalActivation
from greenonet.config import CouplingModelConfig
from greenonet.numerics import IntegrationRule, integrate


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
        layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
        if last_activation:
            layers.append(self.build_activation(activation))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class CouplingNet(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """Axial branch/trunk model with structured baseline plus learned correction.

    The network ingests the full stack of axial lines per batch (both x- and y-lines),
    builds a structured baseline from Green-response L2 norms, predicts interior
    correction terms, and applies the existing balance projection to the assembled
    flux-divergence fields.
    """

    BASELINE_EPS = 1e-12

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
        self.branch_fuser = nn.Linear(
            4 * config.hidden_dim,
            config.hidden_dim,
            bias=config.use_bias,
        )
        self.branch_fuser_activation = self.build_activation(config.activation)
        self.branch_fuser_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
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
        self.green_kernel: Optional[torch.Tensor] = None
        self.integration_rule: IntegrationRule = "simpson"

    def set_structured_baseline_context(
        self,
        green_kernel: torch.Tensor,
        integration_rule: IntegrationRule = "simpson",
    ) -> None:
        param = next(self.parameters())
        self.green_kernel = green_kernel.to(device=param.device, dtype=param.dtype)
        self.integration_rule = integration_rule

    def _require_structured_context(self) -> torch.Tensor:
        if self.green_kernel is None:
            raise ValueError(
                "Structured baseline context is not set. Call "
                "CouplingNet.set_structured_baseline_context(...) before forward()."
            )
        return self.green_kernel

    def _integrate_green_lines(
        self,
        green: torch.Tensor,
        values: torch.Tensor,
        axis_coords: torch.Tensor,
    ) -> torch.Tensor:
        weighted = values.unsqueeze(-2) * green.unsqueeze(0)
        return integrate(
            weighted,
            x=axis_coords,
            dim=-1,
            rule=self.integration_rule,
        )

    def _compute_green_response_norms(
        self,
        coords: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        green_kernel = self._require_structured_context()
        _, _, n_lines, m_points = rhs_raw.shape
        if green_kernel.shape != (2, n_lines, m_points, m_points):
            raise ValueError(
                "green_kernel shape must match the current CouplingNet sample geometry: "
                f"expected (2, {n_lines}, {m_points}, {m_points}), got {tuple(green_kernel.shape)}."
            )
        x_axis = coords[0, 0, :, 0].to(rhs_raw.device)
        y_axis = coords[1, 0, :, 1].to(rhs_raw.device)

        response_x = self._integrate_green_lines(green_kernel[0], rhs_raw[:, 0], x_axis)
        response_y = self._integrate_green_lines(green_kernel[1], rhs_raw[:, 1], y_axis)

        rx = integrate(
            response_x.pow(2),
            x=x_axis,
            dim=-1,
            rule=self.integration_rule,
        ).sqrt()
        ry = integrate(
            response_y.pow(2),
            x=y_axis,
            dim=-1,
            rule=self.integration_rule,
        ).sqrt()
        return rx, ry

    def _build_structured_baseline(
        self,
        coords: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the structured interior baseline
            phi_str = w * f,  psi_str = (1 - w) * f
        on the common interior intersection grid, then pack it back into the
        current x-line / y-line tensor layout. The split uses the opposite-side
        Green-response weighting rule, so the x-flux baseline is weighted by the
        y-line response norm and vice versa.
        """
        rx, ry = self._compute_green_response_norms(coords, rhs_raw)
        rhs_common = rhs_raw[:, 0, :, 1:-1]
        weights = ry.unsqueeze(-2) / (
            rx.unsqueeze(-1) + ry.unsqueeze(-2) + self.BASELINE_EPS
        )
        phi_str = weights * rhs_common
        psi_str = (1.0 - weights) * rhs_common
        return torch.stack((phi_str, psi_str.transpose(-1, -2)), dim=1)

    def _fuse_branch_features(
        self,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_tilde: torch.Tensor,
    ) -> torch.Tensor:
        bsz, axis, n_lines, m_points = a_vals.shape
        flat_size = bsz * axis * n_lines
        branch_parts = [
            self.branch_a(a_vals.reshape(flat_size, m_points)),
            self.branch_b(b_vals.reshape(flat_size, m_points)),
            self.branch_c(c_vals.reshape(flat_size, m_points)),
            self.branch_rhs(rhs_tilde.reshape(flat_size, m_points)),
        ]
        fused = self.branch_fuser(torch.cat(branch_parts, dim=-1))
        fused = self.branch_fuser_activation(fused)
        return cast(torch.Tensor, self.branch_fuser_dropout(fused))

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
                axis after assembling structured baseline plus learned correction
                and applying the cross-axis balance projection with zero-padded
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

        structured_int = self._build_structured_baseline(coords, rhs_raw)

        branch_feat = self._fuse_branch_features(a_vals, b_vals, c_vals, rhs_tilde)
        branch_feat = branch_feat.unsqueeze(-1)

        coords_int = coords[:, :, 1:-1, :]
        trunk_flat = coords_int.reshape(axis * n_lines * (m_points - 2), -1)
        trunk_out = self.trunk(trunk_flat).reshape(axis * n_lines, m_points - 2, -1)

        trunk_expanded = trunk_out.unsqueeze(0).expand(b, -1, -1, -1)
        trunk_expanded = trunk_expanded.reshape(b * axis * n_lines, m_points - 2, -1)
        combined = torch.bmm(trunk_expanded, branch_feat)
        delta_tilde = combined.reshape(b, axis, n_lines, m_points - 2)

        norm_exp = rhs_norm.unsqueeze(-1)
        delta_int = delta_tilde * norm_exp
        raw_int = structured_int + delta_int
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
