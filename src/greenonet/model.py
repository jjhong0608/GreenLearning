from __future__ import annotations

from typing import List, Optional, cast

import torch
from torch import nn

from greenonet.activations import RationalActivation
from greenonet.config import ModelConfig
from greenonet.greens import EllipticGreenFunction, IntegrationEllipticGreenFunction


class ActivationFactoryMixin:
    """Builds activation modules from string names."""

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
    """Generic MLP used for branch and trunk networks."""

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


class FourierFeatures(nn.Module):  # type: ignore[misc]
    """Fourier feature mapping for coordinate inputs."""

    def __init__(
        self,
        input_dim: int,
        num_frequencies: int,
        scale: float,
        include_input: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.include_input = include_input
        b_matrix = torch.randn(input_dim, num_frequencies, dtype=dtype) * scale
        self.b_matrix: torch.Tensor
        self.register_buffer("b_matrix", b_matrix)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * coords @ self.b_matrix
        embedded = torch.cat([proj.sin(), proj.cos()], dim=-1)
        if self.include_input:
            return torch.cat([coords, embedded], dim=-1)
        return embedded


class StructuredGreenKernelMixin:
    """Helpers for the structured branch/trunk Green kernel parameterization."""

    DIAGONAL_SMOOTH_EPS = 1e-12

    def _structured_trunk_features(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != 2:
            raise ValueError("GreenONetModel expects trunk coordinates shaped as (x, xi).")

        x = coords[..., 0:1]
        xi = coords[..., 1:2]
        delta = x - xi
        smooth_abs_delta = torch.sqrt(delta.pow(2) + self.DIAGONAL_SMOOTH_EPS)
        return torch.cat(
            [
                x,
                xi,
                x * xi,
                x.pow(2),
                xi.pow(2),
                delta,
                delta.pow(2),
                smooth_abs_delta,
            ],
            dim=-1,
        )

    @staticmethod
    def _broadcast_x_side(samples: torch.Tensor) -> torch.Tensor:
        """
        Broadcast line coefficients over the xi-axis so values depend on x only.

        Inputs:
            samples: (2, n_lines, m_points) or (B, 2, n_lines, m_points)
        Returns:
            tensor shaped (..., m_points, m_points) where the first grid axis is x
        """
        if samples.dim() == 4:
            _, axis, n_lines, m_points = samples.shape
            return samples.unsqueeze(-1).expand(-1, axis, n_lines, m_points, m_points)
        if samples.dim() == 3:
            axis, n_lines, m_points = samples.shape
            return samples.unsqueeze(-1).expand(axis, n_lines, m_points, m_points)
        raise ValueError(f"Samples must be 3D or 4D, got {samples.dim()}D")


class GreenONetModel(nn.Module, ActivationFactoryMixin, StructuredGreenKernelMixin):  # type: ignore[misc]
    """DeepONet-style branch/trunk model with analytic Green wrapping."""

    EPS = 1e-12
    STRUCTURED_TRUNK_DIM = 8

    def __init__(self, config: ModelConfig) -> None:
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
        self.branch_ap = MLP(
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
        self.branch_fuser = nn.Linear(
            4 * config.hidden_dim,
            config.hidden_dim,
            bias=config.use_bias,
        )
        self.branch_fuser_activation = self.build_activation(config.activation)
        self.branch_fuser_dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.fourier: Optional[FourierFeatures] = None
        trunk_input_dim = self.STRUCTURED_TRUNK_DIM
        if config.use_fourier:
            self.fourier = FourierFeatures(
                input_dim=config.input_dim,
                num_frequencies=config.fourier_dim,
                scale=config.fourier_scale,
                include_input=config.fourier_include_input,
                dtype=config.dtype,
            )
            trunk_input_dim += 2 * config.fourier_dim + (
                config.input_dim if config.fourier_include_input else 0
            )
        self.trunk = MLP(
            input_dim=trunk_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=True,
        )
        self.output_bias = nn.Parameter(torch.zeros(1, dtype=config.dtype))
        self.use_green = config.use_green
        if self.use_green:
            self.green = EllipticGreenFunction()
            self.igreen = IntegrationEllipticGreenFunction()

    def _envelope(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        y = coords[..., 1:2]
        return x * y * (1 - y)

    def _remain(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        return 1 - x

    def _bias_term(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[..., 0:1]
        return torch.full_like(x, -0.5)

    def _build_trunk_inputs(self, coords: torch.Tensor) -> torch.Tensor:
        trunk_inputs = [self._structured_trunk_features(coords)]
        if self.fourier is not None:
            trunk_inputs.append(self.fourier(coords))
        return torch.cat(trunk_inputs, dim=-1)

    def _fuse_branch_features(
        self,
        a_used: torch.Tensor,
        ap_used: torch.Tensor,
        b_used: torch.Tensor,
        c_used: torch.Tensor,
    ) -> torch.Tensor:
        n_axis, n_lines, m_points = a_used.shape
        branch_a_flat = a_used.reshape(n_axis * n_lines, m_points)
        branch_ap_flat = ap_used.reshape(n_axis * n_lines, m_points)
        branch_b_flat = b_used.reshape(n_axis * n_lines, m_points)
        branch_c_flat = c_used.reshape(n_axis * n_lines, m_points)

        branch_parts = [
            cast(torch.Tensor, self.branch_a(branch_a_flat)),
            cast(torch.Tensor, self.branch_ap(branch_ap_flat)),
            cast(torch.Tensor, self.branch_b(branch_b_flat)),
            cast(torch.Tensor, self.branch_c(branch_c_flat)),
        ]
        fused = self.branch_fuser(torch.cat(branch_parts, dim=-1))
        fused = cast(torch.Tensor, self.branch_fuser_activation(fused))
        fused = cast(torch.Tensor, self.branch_fuser_dropout(fused))
        return fused

    def _apply_analytic_green_wrap(
        self,
        trunk_grid: torch.Tensor,
        learned_output: torch.Tensor,
        a_used: torch.Tensor,
        ap_used: torch.Tensor,
        b_used: torch.Tensor,
    ) -> torch.Tensor:
        ap_used = torch.where(ap_used.abs() < 1e-15, torch.zeros_like(ap_used), ap_used)

        a_x = self._broadcast_x_side(a_used)
        ap_x = self._broadcast_x_side(ap_used)
        b_x = self._broadcast_x_side(b_used)

        envelope = self._envelope(trunk_grid).squeeze(-1)
        remain = self._remain(trunk_grid).squeeze(-1)
        bias_term = self._bias_term(trunk_grid).squeeze(-1)
        green_term = cast(torch.Tensor, self.green(trunk_grid)).squeeze(-1)
        igreen_term = cast(torch.Tensor, self.igreen(trunk_grid)).squeeze(-1)

        envelope_b = envelope.unsqueeze(0).unsqueeze(0).expand_as(learned_output)
        remain_b = remain.unsqueeze(0).unsqueeze(0).expand_as(learned_output)
        bias_term_b = bias_term.unsqueeze(0).unsqueeze(0).expand_as(learned_output)
        green_b = green_term.unsqueeze(0).unsqueeze(0).expand_as(learned_output)
        igreen_b = igreen_term.unsqueeze(0).unsqueeze(0).expand_as(learned_output)

        # GreenNet follows the conservative one-dimensional operator
        #   -d_x(a(x) d_x u) + b(x) d_x u + c(x) u = f.
        # Under this sign convention the analytic wrapping coefficients are
        #   A(x, xi) = 1 / a(x),
        #   B(x, xi) = (a'(x) + b(x)) / a(x)^2,
        # so both coefficients depend on the evaluation-side x values only.
        a_recip = 1.0 / (a_x + self.EPS)
        b_coeff = (ap_x + b_x) * a_recip.pow(2)

        return (
            envelope_b * remain_b * learned_output
            + b_coeff * (igreen_b + envelope_b * bias_term_b)
            + a_recip * green_b
        )

    def forward(
        self,
        trunk_grid: torch.Tensor,
        a_vals: torch.Tensor,
        ap_vals: torch.Tensor,
        b_vals: torch.Tensor,
        c_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batched forward without flattening.

        Inputs:
            trunk_grid: (m_points, m_points, 2) meshgrid over x and xi
            a_vals, ap_vals, b_vals, c_vals: (B, 2, n_lines, m_points) or (2, n_lines, m_points)
        Returns:
            output: (2, n_lines, m_points, m_points) — Green depends on coefficients, not per-sample batch.
        """
        m_points = trunk_grid.shape[0]
        if a_vals.dim() == 4:
            _, n_axis, n_lines, _ = a_vals.shape
            a_used = a_vals[0]
            ap_used = ap_vals[0]
            b_used = b_vals[0]
            c_used = c_vals[0]
        elif a_vals.dim() == 3:
            n_axis, n_lines, _ = a_vals.shape
            a_used = a_vals
            ap_used = ap_vals
            b_used = b_vals
            c_used = c_vals
        else:
            raise ValueError("a_vals must be 3D or 4D")

        trunk_flat = trunk_grid.reshape(m_points * m_points, 2)
        trunk_flat = self._build_trunk_inputs(trunk_flat)
        trunk_out = cast(torch.Tensor, self.trunk(trunk_flat))  # (m*m, hidden)

        branch_feat = self._fuse_branch_features(a_used, ap_used, b_used, c_used)
        core = branch_feat @ trunk_out.T  # (2*n, m*m)
        output = core.view(n_axis, n_lines, m_points, m_points) + self.output_bias

        if not self.use_green:
            return output

        return self._apply_analytic_green_wrap(
            trunk_grid=trunk_grid,
            learned_output=output,
            a_used=a_used,
            ap_used=ap_used,
            b_used=b_used,
        )
