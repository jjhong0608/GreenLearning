from __future__ import annotations

from typing import List, cast

import torch
from torch import nn

from greenonet.activations import RationalActivation
from greenonet.config import CouplingModelConfig, SourceStencilLiftConfig


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


class FiveStencilSourceLift(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """Input-side learned scalar branch lift from source/coefficient stencils."""

    stencil_positions: int = 5
    stencil_features: int = stencil_positions

    def __init__(self, config: SourceStencilLiftConfig) -> None:
        super().__init__()
        if config.hidden_dim <= 0:
            raise ValueError("source_stencil_lift.hidden_dim must be positive.")
        if config.depth <= 0:
            raise ValueError("source_stencil_lift.depth must be positive.")
        if config.dropout < 0.0:
            raise ValueError("source_stencil_lift.dropout must be non-negative.")
        if config.eps <= 0.0:
            raise ValueError("source_stencil_lift.eps must be positive.")
        encoder_type = config.encoder_type.lower()
        if encoder_type not in {"linear", "mlp"}:
            raise ValueError(
                "source_stencil_lift.encoder_type must be 'linear' or 'mlp'."
            )
        coefficient_normalization = config.coefficient_normalization.lower()
        if coefficient_normalization not in {"rms", "tanh"}:
            raise ValueError(
                "source_stencil_lift.coefficient_normalization must be 'rms' or 'tanh'."
            )
        if config.coefficient_tanh_beta <= 0.0:
            raise ValueError(
                "source_stencil_lift.coefficient_tanh_beta must be positive."
            )

        self.encoder_type = encoder_type
        self.coefficient_normalization = coefficient_normalization
        self.coefficient_tanh_beta = float(config.coefficient_tanh_beta)
        self.use_g_normalization = bool(config.use_g_normalization)
        self.eps = float(config.eps)

        self.source_encoder = self._build_pointwise_encoder(
            input_dim=self.stencil_features,
            config=config,
        )
        self.coefficient_encoder = self._build_pointwise_encoder(
            input_dim=self.stencil_features,
            config=config,
        )

    def _build_pointwise_encoder(
        self,
        input_dim: int,
        config: SourceStencilLiftConfig,
    ) -> nn.Sequential:
        if self.encoder_type == "linear":
            layers: List[nn.Module] = [nn.Linear(input_dim, 1, bias=config.use_bias)]
        else:
            layers = []
            in_dim = input_dim
            for _ in range(config.depth):
                layers.append(
                    nn.Linear(in_dim, config.hidden_dim, bias=config.use_bias)
                )
                layers.append(self.build_activation(config.activation))
                if config.dropout > 0.0:
                    layers.append(nn.Dropout(config.dropout))
                in_dim = config.hidden_dim
            layers.append(nn.Linear(in_dim, 1, bias=config.use_bias))
        return nn.Sequential(*layers)

    @staticmethod
    def _validate_axis_field(
        values: torch.Tensor, field_name: str
    ) -> tuple[int, int, int]:
        if values.dim() != 4:
            raise ValueError(f"{field_name} must have shape (B, 2, N, N + 2).")
        bsz, axis, n_lines, m_points = values.shape
        if axis != 2:
            raise ValueError(f"Expected {field_name} axis dimension 2, got {axis}.")
        if n_lines + 2 != m_points:
            raise ValueError(
                f"{field_name} must have n_lines + 2 == m_points "
                f"(got n_lines={n_lines}, m_points={m_points})."
            )
        return bsz, n_lines, m_points

    def _validate_rhs_raw(self, rhs_raw: torch.Tensor) -> tuple[int, int, int]:
        return self._validate_axis_field(rhs_raw, "rhs_raw")

    def _build_canonical_full_field(
        self, values: torch.Tensor, field_name: str
    ) -> torch.Tensor:
        """Reconstruct a canonical full grid with zero corners."""
        bsz, _n_lines, m_points = self._validate_axis_field(values, field_name)
        full = torch.zeros(
            (bsz, m_points, m_points),
            dtype=values.dtype,
            device=values.device,
        )
        full[:, 1:-1, :] = values[:, 0]
        full[:, 0, 1:-1] = values[:, 1, :, 0]
        full[:, -1, 1:-1] = values[:, 1, :, -1]
        return full

    def _build_canonical_full_source(self, rhs_raw: torch.Tensor) -> torch.Tensor:
        """Reconstruct canonical full source grid with zero corners."""
        return self._build_canonical_full_field(rhs_raw, "rhs_raw")

    @staticmethod
    def _gather_5_stencil(full_source: torch.Tensor) -> torch.Tensor:
        """Gather center, east, west, north, south for all interior points."""
        center = full_source[:, 1:-1, 1:-1]
        east = full_source[:, 1:-1, 2:]
        west = full_source[:, 1:-1, :-2]
        north = full_source[:, 2:, 1:-1]
        south = full_source[:, :-2, 1:-1]
        return torch.stack((center, east, west, north, south), dim=-1)

    def _build_source_stencil_features(
        self, rhs_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_source = self._build_canonical_full_source(rhs_raw)
        f_int = full_source[:, 1:-1, 1:-1]
        scale = torch.sqrt(
            torch.mean(f_int * f_int, dim=(-1, -2), keepdim=True) + self.eps
        )
        normalized_source = full_source / scale
        return self._gather_5_stencil(normalized_source), f_int / scale

    def _build_coefficient_stencil_features(
        self,
        a_vals: torch.Tensor,
        expected_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        a_shape = self._validate_axis_field(a_vals, "a_vals")
        if a_shape != expected_shape:
            raise ValueError(
                "a_vals must match rhs_raw shape metadata "
                f"(got rhs={expected_shape}, a={a_shape})."
            )
        full_a = self._build_canonical_full_field(a_vals, "a_vals")
        return self._gather_5_stencil(full_a)

    def _build_stencil_features(
        self, rhs_raw: torch.Tensor, a_vals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_shape = self._validate_rhs_raw(rhs_raw)
        source_stencil, f_hat = self._build_source_stencil_features(rhs_raw)
        coefficient_stencil = self._build_coefficient_stencil_features(
            a_vals,
            expected_shape=source_shape,
        )
        return torch.cat((source_stencil, coefficient_stencil), dim=-1), f_hat

    @staticmethod
    def _encode_stencil(encoder: nn.Module, stencil: torch.Tensor) -> torch.Tensor:
        bsz, n_lines, n_inner, _features = stencil.shape
        return cast(
            torch.Tensor,
            encoder(stencil.reshape(bsz * n_lines * n_inner, -1)),
        ).reshape(bsz, n_lines, n_inner)

    def _normalize_lifted_scalar(self, value: torch.Tensor) -> torch.Tensor:
        if not self.use_g_normalization:
            return value
        scale = torch.sqrt(
            torch.mean(value * value, dim=(-1, -2), keepdim=True) + self.eps
        )
        return value / scale

    def _normalize_coefficient_lift(self, value: torch.Tensor) -> torch.Tensor:
        if self.coefficient_normalization == "tanh":
            return self.coefficient_tanh_beta * torch.tanh(value)
        return self._normalize_lifted_scalar(value)

    @staticmethod
    def _to_axis_lift(value: torch.Tensor) -> torch.Tensor:
        bsz, n_lines, n_inner = value.shape
        lifted = torch.empty(
            (bsz, 2, n_lines, n_inner),
            dtype=value.dtype,
            device=value.device,
        )
        lifted[:, 0] = value
        lifted[:, 1] = value.transpose(-1, -2)
        return lifted

    def _encoded_components(
        self, rhs_raw: torch.Tensor, a_vals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_shape = self._validate_rhs_raw(rhs_raw)
        source_stencil, f_hat = self._build_source_stencil_features(rhs_raw)
        coefficient_stencil = self._build_coefficient_stencil_features(
            a_vals,
            expected_shape=source_shape,
        )
        g_source_raw = self._encode_stencil(self.source_encoder, source_stencil)
        g_coefficient_raw = self._encode_stencil(
            self.coefficient_encoder,
            coefficient_stencil,
        )
        g_source = self._normalize_lifted_scalar(g_source_raw)
        g_coefficient = self._normalize_coefficient_lift(g_coefficient_raw)
        return g_source, g_coefficient, f_hat

    def forward_components(
        self, rhs_raw: torch.Tensor, a_vals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_source, g_coefficient, _f_hat = self._encoded_components(rhs_raw, a_vals)
        return self._to_axis_lift(g_source), self._to_axis_lift(g_coefficient)

    def lift_with_diagnostics(
        self, rhs_raw: torch.Tensor, a_vals: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        g_source, g_coefficient, f_hat = self._encoded_components(rhs_raw, a_vals)
        g = g_source * g_coefficient
        lifted = self._to_axis_lift(g)

        bsz = g.shape[0]
        g_flat = g.reshape(bsz, -1)
        f_flat = f_hat.reshape(bsz, -1)
        inner = torch.sum(g_flat * f_flat, dim=-1)
        g_norm = torch.linalg.vector_norm(g_flat, dim=-1)
        f_norm = torch.linalg.vector_norm(f_flat, dim=-1)
        corr = inner / (g_norm * f_norm + self.eps)
        rel_diff = torch.linalg.vector_norm(g_flat - f_flat, dim=-1) / (
            f_norm + self.eps
        )
        g_rms = torch.sqrt(torch.mean(g * g, dim=(-1, -2)))
        diagnostics = {
            "source_lift_corr_g_f": corr.mean(),
            "source_lift_rel_diff_g_f": rel_diff.mean(),
            "source_lift_g_rms": g_rms.mean(),
        }
        return lifted, diagnostics

    def forward(self, rhs_raw: torch.Tensor, a_vals: torch.Tensor) -> torch.Tensor:
        lifted, _diagnostics = self.lift_with_diagnostics(rhs_raw, a_vals)
        return lifted


class CouplingNet(nn.Module, ActivationFactoryMixin):  # type: ignore[misc]
    """MIONet-style model: four branches (a, b, c, rhs) and one trunk (coords).

    The network ingests the full stack of axial lines per batch (both x- and y-lines)
    and predicts axial flux-divergences per axis on interior points, then zero-pads
    the boundaries.
    """

    def __init__(self, config: CouplingModelConfig) -> None:
        super().__init__()
        torch.set_default_dtype(config.dtype)
        if config.balance_projection not in {"symmetric", "smooth_mask"}:
            raise ValueError(
                f"Unsupported balance_projection: {config.balance_projection}"
            )
        if config.smooth_mask_eps <= 0.0:
            raise ValueError("smooth_mask_eps must be positive.")
        self.balance_projection = config.balance_projection
        self.smooth_mask_normalize = bool(config.smooth_mask_normalize)
        self.smooth_mask_eps = float(config.smooth_mask_eps)
        if config.source_stencil_lift.enabled and config.branch_input_dim <= 2:
            raise ValueError(
                "source_stencil_lift requires branch_input_dim > 2 because "
                "the lifted source branch uses the interior length "
                "branch_input_dim - 2."
            )
        self.branch_input_dim = config.branch_input_dim
        self.branch_rhs_input_dim = (
            config.branch_input_dim - 2
            if config.source_stencil_lift.enabled
            else config.branch_input_dim
        )
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
            input_dim=self.branch_rhs_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=False,
        )
        if config.source_stencil_lift.enabled:
            self.branch_coefficient: MLP | None = MLP(
                input_dim=self.branch_rhs_input_dim,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                activation=config.activation,
                use_bias=config.use_bias,
                dropout=config.dropout,
                last_activation=False,
            )
        else:
            self.branch_coefficient = None
        self.trunk = MLP(
            input_dim=config.trunk_input_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            activation=config.activation,
            use_bias=config.use_bias,
            dropout=config.dropout,
            last_activation=True,
        )
        if config.source_stencil_lift.enabled:
            self.source_stencil_lift: FiveStencilSourceLift | None = (
                FiveStencilSourceLift(config.source_stencil_lift)
            )
        else:
            self.source_stencil_lift = None

    def _apply_symmetric_balance_projection(
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

    def _smooth_mask_components(
        self,
        coords: torch.Tensor,
        flux_int: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if coords.dim() != 4:
            raise ValueError("coords must have shape (2, N, N + 2, 2).")
        if flux_int.dim() != 4:
            raise ValueError("flux_int must have shape (B, 2, N, N).")
        _bsz, _axis, n_lines, n_inner = flux_int.shape

        x_axis = coords[0, 0, :, 0].to(
            device=flux_int.device,
            dtype=flux_int.dtype,
        )
        y_lines = coords[0, :, 0, 1].to(
            device=flux_int.device,
            dtype=flux_int.dtype,
        )
        x_inner = x_axis[1:-1]
        y_inner = y_lines
        if x_inner.numel() != n_inner or y_inner.numel() != n_lines:
            raise ValueError(
                "coords interior dimensions must match flux_int common grid "
                f"({n_lines}, {n_inner}); got x_inner={x_inner.numel()}, "
                f"y_inner={y_inner.numel()}."
            )

        m_phi = y_inner * (1.0 - y_inner)
        m_psi = x_inner * (1.0 - x_inner)
        if self.smooth_mask_normalize:
            m_phi = 4.0 * m_phi
            m_psi = 4.0 * m_psi
        return m_phi.view(1, -1, 1), m_psi.view(1, 1, -1)

    def _apply_smooth_mask_balance_projection(
        self,
        flux_int: torch.Tensor,
        rhs_raw: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Apply smooth-mask balance projection on the interior common grid."""
        phi0 = flux_int[:, 0]
        psi0 = flux_int[:, 1].transpose(-1, -2)
        f = rhs_raw[:, 0, :, 1:-1]

        m_phi, m_psi = self._smooth_mask_components(coords, flux_int)
        denom = (m_phi + m_psi).clamp_min(self.smooth_mask_eps)
        w_phi = m_phi / denom
        w_psi = m_psi / denom
        alpha = (m_phi * m_psi) / denom

        diff = phi0 - psi0
        phi = w_phi * f + alpha * diff
        psi_t = w_psi * f - alpha * diff

        projected = flux_int.clone()
        projected[:, 0] = phi
        projected[:, 1] = psi_t.transpose(-1, -2)
        return projected

    def _apply_balance_projection(
        self,
        flux_int: torch.Tensor,
        rhs_raw: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if self.balance_projection == "symmetric":
            return self._apply_symmetric_balance_projection(flux_int, rhs_raw)
        if self.balance_projection == "smooth_mask":
            return self._apply_smooth_mask_balance_projection(
                flux_int=flux_int,
                rhs_raw=rhs_raw,
                coords=coords,
            )
        raise ValueError(f"Unsupported balance_projection: {self.balance_projection}")

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
            output_flux: (B, 2, n_lines, m_points) axial flux-divergence per
                axis after projection with zero-padded boundaries.
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

        if self.source_stencil_lift is None:
            branch_a_in = a_vals.reshape(b * axis * n_lines, m_points)
            branch_rhs_source = rhs_tilde
            branch_r_in = branch_rhs_source.reshape(
                b * axis * n_lines, self.branch_rhs_input_dim
            )
            branch_a_out = self.branch_a(branch_a_in)
            branch_r_out = self.branch_rhs(branch_r_in)
            branch_feat = branch_a_out * branch_r_out
        else:
            if self.branch_coefficient is None:
                raise RuntimeError(
                    "branch_coefficient must be initialized when source lift is enabled."
                )
            lifted_source, lifted_coefficient = (
                self.source_stencil_lift.forward_components(rhs_raw, a_vals)
            )
            branch_r_in = lifted_source.reshape(
                b * axis * n_lines, self.branch_rhs_input_dim
            )
            branch_coefficient_in = lifted_coefficient.reshape(
                b * axis * n_lines, self.branch_rhs_input_dim
            )
            branch_r_out = self.branch_rhs(branch_r_in)
            branch_coefficient_out = self.branch_coefficient(branch_coefficient_in)
            branch_feat = branch_r_out * branch_coefficient_out
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
        projected_int = self._apply_balance_projection(raw_int, rhs_raw, coords)

        output_flux = torch.zeros(
            b,
            axis,
            n_lines,
            m_points,
            dtype=projected_int.dtype,
            device=projected_int.device,
        )
        output_flux[:, :, :, 1:-1] = projected_int
        return output_flux

    def source_lift_diagnostics(
        self, rhs_raw: torch.Tensor, a_vals: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.source_stencil_lift is None:
            return {}
        _lifted, diagnostics = self.source_stencil_lift.lift_with_diagnostics(
            rhs_raw,
            a_vals,
        )
        return diagnostics
