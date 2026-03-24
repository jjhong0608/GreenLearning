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


class GreenONetModel(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """DeepONet-style branch/trunk model with analytic Green wrapping."""

    # EPS = 1e-8
    EPS = 0.0

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
        # self.branch_b = MLP(
        #     input_dim=config.branch_input_dim,
        #     hidden_dim=config.hidden_dim,
        #     depth=config.depth,
        #     activation=config.activation,
        #     use_bias=config.use_bias,
        #     dropout=config.dropout,
        #     last_activation=False,
        # )
        self.branch_c = MLP(
            input_dim=config.branch_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        self.fourier: Optional[FourierFeatures] = None
        trunk_input_dim = config.input_dim
        if config.use_fourier:
            self.fourier = FourierFeatures(
                input_dim=config.input_dim,
                num_frequencies=config.fourier_dim,
                scale=config.fourier_scale,
                include_input=config.fourier_include_input,
                dtype=config.dtype,
            )
            trunk_input_dim = 2 * config.fourier_dim + (
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

    @staticmethod
    def _refactor_x0(samples: torch.Tensor) -> torch.Tensor:
        """
        Broadcast along x0 axis to shape (..., 2, n_lines, m_points, m_points).
        Accepts either 3D (2, n, m) or 4D (B, 2, n, m).
        """
        if samples.dim() == 4:
            _, axis, n_lines, m = samples.shape
            return samples.unsqueeze(-2).expand(-1, axis, n_lines, m, m)
        if samples.dim() == 3:
            axis, n_lines, m = samples.shape
            return samples.unsqueeze(-2).expand(axis, n_lines, m, m)
        raise ValueError(f"Samples must be 3D or 4D, got {samples.dim()}D")

    @staticmethod
    def _refactor_x1(samples: torch.Tensor) -> torch.Tensor:
        """
        Broadcast along x1 axis to shape (..., 2, n_lines, m_points, m_points).
        Accepts either 3D (2, n, m) or 4D (B, 2, n, m).
        """
        if samples.dim() == 4:
            _, axis, n_lines, m = samples.shape
            return samples.unsqueeze(-1).expand(-1, axis, n_lines, m, m)
        if samples.dim() == 3:
            axis, n_lines, m = samples.shape
            return samples.unsqueeze(-1).expand(axis, n_lines, m, m)
        raise ValueError(f"Samples must be 3D or 4D, got {samples.dim()}D")

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
        elif a_vals.dim() == 3:
            n_axis, n_lines, _ = a_vals.shape
            a_used = a_vals
            ap_used = ap_vals
            b_used = b_vals
        else:
            raise ValueError("a_vals must be 3D or 4D")

        trunk_flat = trunk_grid.reshape(m_points * m_points, 2)
        if self.fourier is not None:
            trunk_flat = self.fourier(trunk_flat)
        trunk_out = cast(torch.Tensor, self.trunk(trunk_flat))  # (m*m, hidden)

        branch_a_flat = a_used.reshape(n_axis * n_lines, m_points)
        # branch_b_flat = b_used.reshape(n_axis * n_lines, m_points)
        # branch_c_flat = c_used.reshape(n_axis * n_lines, m_points)
        branch_a_out = cast(torch.Tensor, self.branch_a(branch_a_flat))  # (2*n, hidden)
        # branch_b_out = cast(torch.Tensor, self.branch_b(branch_b_flat))  # (2*n, hidden)
        # branch_c_out = cast(torch.Tensor, self.branch_c(branch_c_flat))  # (2*n, hidden)
        # branch_feat = branch_a_out * branch_b_out * branch_c_out
        # branch_feat = branch_a_out * branch_c_out
        branch_feat = branch_a_out
        core = branch_feat @ trunk_out.T  # (2*n, m*m)
        output = core.view(n_axis, n_lines, m_points, m_points)

        if not self.use_green:
            return output

        ap_used = torch.where(ap_used < 1e-15, torch.zeros_like(ap_used), ap_used)
        # ap_used = ap_used.abs()

        a0_val_t = self._refactor_x0(a_used)  # (2,n,m,m)
        a1_val_t = self._refactor_x1(a_used)  # (2,n,m,m)
        ap1_val_t = self._refactor_x1(ap_used)  # (2,n,m,m)
        b1_val_t = self._refactor_x1(b_used)  # (2,n,m,m)

        envelope = self._envelope(trunk_grid).squeeze(-1)  # (m, m)
        remain = self._remain(trunk_grid).squeeze(-1)  # (m, m)
        bias_term = self._bias_term(trunk_grid).squeeze(-1)  # (m, m)
        green_term = cast(torch.Tensor, self.green(trunk_grid)).squeeze(-1)  # (m, m)
        igreen_term = cast(torch.Tensor, self.igreen(trunk_grid)).squeeze(-1)  # (m, m)

        envelope_b = envelope.unsqueeze(0).unsqueeze(0).expand(n_axis, n_lines, -1, -1)
        remain_b = remain.unsqueeze(0).unsqueeze(0).expand(n_axis, n_lines, -1, -1)
        bias_term_b = (
            bias_term.unsqueeze(0).unsqueeze(0).expand(n_axis, n_lines, -1, -1)
        )
        green_b = green_term.unsqueeze(0).unsqueeze(0).expand(n_axis, n_lines, -1, -1)
        igreen_b = igreen_term.unsqueeze(0).unsqueeze(0).expand(n_axis, n_lines, -1, -1)

        inv_a0 = 1.0 / (a0_val_t + self.EPS)
        coeff_common = (ap1_val_t + b1_val_t) / (a1_val_t + self.EPS)

        full_fn = (
            envelope_b * remain_b * output
            + (igreen_b + envelope_b * bias_term_b) * coeff_common * inv_a0
            + inv_a0 * green_b
        )
        return full_fn
