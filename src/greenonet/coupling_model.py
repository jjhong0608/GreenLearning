from __future__ import annotations

from typing import List, cast

import torch
from torch import nn
import torch.nn.functional as F

from greenonet.activations import RationalActivation
from greenonet.config import CouplerConfig, CouplingModelConfig


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


class FiveStencilStencilMLPCoupler(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """Explicit five-stencil gather and pointwise MLP null-space coupler."""

    point_features: int = 9

    def __init__(self, config: CouplerConfig) -> None:
        super().__init__()
        if config.type != "five_stencil_stencil_mlp":
            raise ValueError(f"Unsupported coupler type: {config.type}")
        if config.hidden_channels <= 0:
            raise ValueError("coupler.hidden_channels must be positive.")
        if config.depth <= 0:
            raise ValueError("coupler.depth must be positive.")
        if config.padding not in {"replicate", "zero"}:
            raise ValueError("coupler.padding must be 'replicate' or 'zero'.")

        self.padding = config.padding
        self.eps = float(config.eps)
        in_channels = 5 * self.point_features

        layers: List[nn.Module] = []
        ch = in_channels
        for _ in range(config.depth):
            layers.append(
                nn.Conv2d(
                    in_channels=ch,
                    out_channels=config.hidden_channels,
                    kernel_size=1,
                    bias=True,
                )
            )
            layers.append(self.build_activation(config.activation))
            if config.dropout > 0.0:
                layers.append(nn.Dropout2d(config.dropout))
            ch = config.hidden_channels

        final = nn.Conv2d(
            in_channels=ch,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)

        self.local_mlp = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(config.residual_scale_init))
        )

    def _pad(self, q: torch.Tensor) -> torch.Tensor:
        if self.padding == "replicate":
            return F.pad(q, (1, 1, 1, 1), mode="replicate")
        if self.padding == "zero":
            return F.pad(q, (1, 1, 1, 1), mode="constant", value=0.0)
        raise RuntimeError(f"Unexpected padding mode: {self.padding}")

    def _gather_5_stencil(self, q: torch.Tensor) -> torch.Tensor:
        """Gather center, i+1, i-1, j+1, j-1 features without corners."""
        q_pad = self._pad(q)
        center = q_pad[:, :, 1:-1, 1:-1]
        plus_i = q_pad[:, :, 2:, 1:-1]
        minus_i = q_pad[:, :, :-2, 1:-1]
        plus_j = q_pad[:, :, 1:-1, 2:]
        minus_j = q_pad[:, :, 1:-1, :-2]
        return torch.cat([center, plus_i, minus_i, plus_j, minus_j], dim=1)

    @staticmethod
    def _canonicalize_flux(raw_int: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = raw_int[:, 0]
        psi = raw_int[:, 1].transpose(-1, -2)
        return phi, psi

    @staticmethod
    def _decanonicalize_flux(
        phi: torch.Tensor, psi: torch.Tensor, template: torch.Tensor
    ) -> torch.Tensor:
        out = template.clone()
        out[:, 0] = phi
        out[:, 1] = psi.transpose(-1, -2)
        return out

    def _validate_inputs(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
    ) -> None:
        if raw_int.dim() != 4:
            raise ValueError("raw_int must have shape (B, 2, N, N).")
        bsz, axis, n_i, n_j = raw_int.shape
        if axis != 2:
            raise ValueError(f"Expected raw_int axis dimension 2, got {axis}.")
        if n_i != n_j:
            raise ValueError(
                f"Expected square raw_int interior grid, got {n_i} x {n_j}."
            )

        expected_full = (bsz, 2, n_i, n_i + 2)
        for name, value in (
            ("a_vals", a_vals),
            ("b_vals", b_vals),
            ("c_vals", c_vals),
            ("rhs_raw", rhs_raw),
        ):
            if tuple(value.shape) != expected_full:
                raise ValueError(
                    f"{name} must have shape {expected_full}, got {tuple(value.shape)}."
                )

    def _build_canonical_point_features(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        auxiliary_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_inputs(raw_int, a_vals, b_vals, c_vals, rhs_raw)

        phi0, psi0 = self._canonicalize_flux(raw_int)
        f = rhs_raw[:, 0, :, 1:-1]

        ax = a_vals[:, 0, :, 1:-1]
        ay = a_vals[:, 1, :, 1:-1].transpose(-1, -2)
        bx = b_vals[:, 0, :, 1:-1]
        by = b_vals[:, 1, :, 1:-1].transpose(-1, -2)
        cx = c_vals[:, 0, :, 1:-1]
        cy = c_vals[:, 1, :, 1:-1].transpose(-1, -2)

        scale = torch.sqrt(torch.mean(f * f, dim=(-1, -2), keepdim=True) + self.eps)
        diff_field = 0.5 * (phi0 - psi0)
        if auxiliary_residual is None:
            balance_residual = f - (phi0 + psi0)
        else:
            if tuple(auxiliary_residual.shape) != tuple(f.shape):
                raise ValueError(
                    "auxiliary_residual must have shape matching canonical source "
                    f"{tuple(f.shape)}, got {tuple(auxiliary_residual.shape)}."
                )
            balance_residual = auxiliary_residual
        diff_hat = diff_field / scale
        res_hat = balance_residual / scale
        f_hat = f / scale

        q = torch.stack(
            [diff_hat, res_hat, f_hat, ax, ay, bx, by, cx, cy],
            dim=1,
        )
        return q, scale, phi0, psi0

    def forward(
        self,
        raw_int: torch.Tensor,
        a_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
        rhs_raw: torch.Tensor,
        auxiliary_residual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q, scale, phi0, psi0 = self._build_canonical_point_features(
            raw_int=raw_int,
            a_vals=a_vals,
            b_vals=b_vals,
            c_vals=c_vals,
            rhs_raw=rhs_raw,
            auxiliary_residual=auxiliary_residual,
        )
        q5 = self._gather_5_stencil(q)
        delta_hat = cast(torch.Tensor, self.local_mlp(q5)).squeeze(1)
        delta = self.residual_scale * scale * delta_hat

        phi1 = phi0 + delta
        psi1 = psi0 - delta
        return self._decanonicalize_flux(phi1, psi1, raw_int)


class CouplingNet(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """MIONet-style model: four branches (a, b, c, rhs) and one trunk (coords).

    The network ingests the full stack of axial lines per batch (both x- and y-lines)
    and predicts axial flux-divergences per axis on interior points, then zero-pads
    the boundaries.
    """

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
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
        if config.coupler.enabled:
            self.coupler: FiveStencilStencilMLPCoupler | None = (
                FiveStencilStencilMLPCoupler(config.coupler)
            )
        else:
            self.coupler = None

    def _compute_balance_residual(
        self, flux_int: torch.Tensor, rhs_raw: torch.Tensor
    ) -> torch.Tensor:
        """Compute canonical balance residual f - (phi + psi)."""
        rhs_x_int = rhs_raw[:, 0, :, 1:-1]
        phi = flux_int[:, 0]
        psi_t = flux_int[:, 1].transpose(-1, -2)
        return rhs_x_int - (phi + psi_t)

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
        return_intermediates: bool = False,
        detach_coupler_input: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
            output_flux: (B, 2, n_lines, m_points) axial flux-divergence per
                axis after projection/coupling with zero-padded
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
        branch_feat = branch_a_out * branch_r_out
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
        pre_projection_residual = self._compute_balance_residual(
            flux_int=raw_int,
            rhs_raw=rhs_raw,
        )
        projected_int = self._apply_balance_projection(raw_int, rhs_raw)
        projected_int_before_coupler = projected_int
        if self.coupler is not None:
            coupler_input = projected_int
            coupler_residual = pre_projection_residual
            if detach_coupler_input:
                coupler_input = coupler_input.detach()
                coupler_residual = coupler_residual.detach()
            coupled_int = self.coupler(
                raw_int=coupler_input,
                a_vals=a_vals,
                b_vals=b_vals,
                c_vals=c_vals,
                rhs_raw=rhs_raw,
                auxiliary_residual=coupler_residual,
            )
        else:
            coupled_int = projected_int

        output_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=coupled_int.dtype,
            device=coupled_int.device,
        )
        output_flux[:, :, :, 1:-1] = coupled_int
        if return_intermediates:
            return output_flux, {
                "raw_int": raw_int,
                "pre_projection_residual": pre_projection_residual,
                "projected_int": projected_int_before_coupler,
                "coupled_int": coupled_int,
            }
        return output_flux
