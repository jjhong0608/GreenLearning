from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import ComplexPreProjectionFusionConfig
from greenonet.coupling_model import ActivationFactoryMixin


FUSION_ARCHITECTURE = "single_nonlinear_fusion_mlp"
FINAL_LAYER_INITIALIZATION = "scaled_torch_linear_default"


def pre_projection_fusion_formula(mode: str) -> str:
    """Return the reader-facing physical difference formula for a fusion mode."""

    if mode == "residual":
        return "d_fused=d_base+A_safe*h_theta([d_base/A_safe,rhs/A_safe])"
    if mode == "absolute":
        return "d_fused=A_safe*h_theta([d_base/A_safe,rhs/A_safe])"
    raise ValueError(f"Unsupported pre-projection fusion mode: {mode}.")


@dataclass(frozen=True)
class ComplexPreProjectionFusionResult:
    """Audit tensors for the optional physical difference fusion."""

    mode: str
    base_response: torch.Tensor
    base_physical: torch.Tensor
    fused_response: torch.Tensor
    fused_physical: torch.Tensor
    base_difference: torch.Tensor
    normalized_difference: torch.Tensor
    normalized_rhs: torch.Tensor
    normalized_network_output: torch.Tensor
    physical_network_output: torch.Tensor
    normalized_residual: torch.Tensor
    physical_residual: torch.Tensor
    fused_difference: torch.Tensor
    difference_delta: torch.Tensor
    source_scale: torch.Tensor
    safe_source_scale: torch.Tensor
    pre_projection_balance_residual: torch.Tensor


class ComplexPreProjectionFusion(nn.Module, ActivationFactoryMixin):
    """Fuse a normalized nonlinear output into the physical split difference."""

    INPUT_FEATURE_NAMES = (
        "normalized_difference",
        "normalized_rhs",
    )
    MODE_CODES = {
        "residual": 0,
        "absolute": 1,
    }

    def __init__(
        self,
        config: ComplexPreProjectionFusionConfig,
        *,
        activation: str,
        use_bias: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.config = ComplexPreProjectionFusionConfig.from_raw(config)
        self.eps = float(self.config.eps)
        self.mode = self.config.mode
        self._fusion_mode_code: torch.Tensor
        self.register_buffer(
            "_fusion_mode_code",
            torch.tensor(self.MODE_CODES[self.mode], dtype=torch.int64),
            persistent=True,
        )

        layers: list[nn.Module] = []
        input_dim = len(self.INPUT_FEATURE_NAMES)
        for _ in range(self.config.depth):
            layers.append(
                nn.Linear(
                    input_dim,
                    self.config.hidden_dim,
                    bias=use_bias,
                    dtype=dtype,
                )
            )
            layers.append(self.build_activation(activation))
            input_dim = self.config.hidden_dim
        layers.append(nn.Linear(input_dim, 1, bias=use_bias, dtype=dtype))
        self.residual_mlp = nn.Sequential(*layers)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        final_layer = cast(nn.Linear, self.residual_mlp[-1])
        scale = float(self.config.final_layer_init_scale)
        with torch.no_grad():
            final_layer.weight.mul_(scale)
            if final_layer.bias is not None:
                final_layer.bias.mul_(scale)

    def forward(
        self,
        *,
        base_response: torch.Tensor,
        rhs_phys: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the fused response for the compiled training path."""

        return self._forward_tensors(
            base_response=base_response,
            rhs_phys=rhs_phys,
            geometry=geometry,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
        )[2]

    def forward_with_diagnostics(
        self,
        *,
        base_response: torch.Tensor,
        rhs_phys: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
    ) -> ComplexPreProjectionFusionResult:
        """Return fused response and interpretable residual components."""

        (
            computed_base_response,
            base_physical,
            fused_response,
            fused_physical,
            base_difference,
            normalized_difference,
            normalized_rhs,
            normalized_network_output,
            physical_network_output,
            normalized_residual,
            physical_residual,
            fused_difference,
            difference_delta,
            source_scale,
            safe_source_scale,
            pre_projection_balance_residual,
        ) = self._forward_tensors(
            base_response=base_response,
            rhs_phys=rhs_phys,
            geometry=geometry,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
        )
        return ComplexPreProjectionFusionResult(
            mode=self.mode,
            base_response=computed_base_response,
            base_physical=base_physical,
            fused_response=fused_response,
            fused_physical=fused_physical,
            base_difference=base_difference,
            normalized_difference=normalized_difference,
            normalized_rhs=normalized_rhs,
            normalized_network_output=normalized_network_output,
            physical_network_output=physical_network_output,
            normalized_residual=normalized_residual,
            physical_residual=physical_residual,
            fused_difference=fused_difference,
            difference_delta=difference_delta,
            source_scale=source_scale,
            safe_source_scale=safe_source_scale,
            pre_projection_balance_residual=pre_projection_balance_residual,
        )

    def _forward_tensors(
        self,
        *,
        base_response: torch.Tensor,
        rhs_phys: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        self._validate_inputs(
            base_response=base_response,
            rhs_phys=rhs_phys,
            geometry=geometry,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
        )
        device = base_response.device
        dtype = base_response.dtype
        x_length = geometry.x_lengths_for_valid_points().to(device=device, dtype=dtype)
        y_length = geometry.y_lengths_for_valid_points().to(device=device, dtype=dtype)
        sigma_x = x_length.square().unsqueeze(0)
        sigma_y = y_length.square().unsqueeze(0)
        base_physical = torch.stack(
            (
                base_response[:, 0] / sigma_x,
                base_response[:, 1] / sigma_y,
            ),
            dim=1,
        )
        base_difference = base_physical[:, 0] - base_physical[:, 1]

        x_amplitude_valid = x_source_amplitude[
            :, geometry.x_segment_id.to(device=device)
        ].to(dtype=dtype)
        y_amplitude_valid = y_source_amplitude[
            :, geometry.y_segment_id.to(device=device)
        ].to(dtype=dtype)
        source_scale = torch.sqrt(
            0.5 * (x_amplitude_valid.square() + y_amplitude_valid.square())
        )
        safe_source_scale = source_scale.clamp_min(self.eps)
        normalized_difference = base_difference / safe_source_scale
        normalized_rhs = rhs_phys / safe_source_scale
        residual_input = torch.stack(
            (normalized_difference, normalized_rhs),
            dim=-1,
        )
        normalized_network_output = self.residual_mlp(residual_input).squeeze(-1)
        physical_network_output = safe_source_scale * normalized_network_output
        physical_network_output = torch.where(
            source_scale > 0.0,
            physical_network_output,
            torch.zeros_like(physical_network_output),
        )
        if self.mode == "residual":
            fused_difference = base_difference + physical_network_output
        else:
            fused_difference = physical_network_output
        difference_delta = fused_difference - base_difference
        physical_residual = difference_delta
        normalized_residual = torch.where(
            source_scale > 0.0,
            difference_delta / safe_source_scale,
            torch.zeros_like(difference_delta),
        )

        fused_physical = torch.stack(
            (
                0.5 * (rhs_phys + fused_difference),
                0.5 * (rhs_phys - fused_difference),
            ),
            dim=1,
        )
        pre_projection_balance_residual = fused_physical.sum(dim=1) - rhs_phys
        fused_response = torch.stack(
            (
                sigma_x * fused_physical[:, 0],
                sigma_y * fused_physical[:, 1],
            ),
            dim=1,
        )
        return (
            base_response,
            base_physical,
            fused_response,
            fused_physical,
            base_difference,
            normalized_difference,
            normalized_rhs,
            normalized_network_output,
            physical_network_output,
            normalized_residual,
            physical_residual,
            fused_difference,
            difference_delta,
            source_scale,
            safe_source_scale,
            pre_projection_balance_residual,
        )

    @staticmethod
    def _validate_inputs(
        *,
        base_response: torch.Tensor,
        rhs_phys: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        x_source_amplitude: torch.Tensor,
        y_source_amplitude: torch.Tensor,
    ) -> None:
        if base_response.dim() != 3 or base_response.shape[1] != 2:
            raise ValueError("base_response must have shape (B, 2, P).")
        batch_size, _axes, point_count = base_response.shape
        if point_count != geometry.num_points:
            raise ValueError("base_response point count does not match geometry.")
        if rhs_phys.shape != (batch_size, point_count):
            raise ValueError("rhs_phys must have shape (B, P).")
        if x_source_amplitude.shape != (batch_size, geometry.num_x_segments):
            raise ValueError("x_source_amplitude must have shape (B, num_x_segments).")
        if y_source_amplitude.shape != (batch_size, geometry.num_y_segments):
            raise ValueError("y_source_amplitude must have shape (B, num_y_segments).")
        x_length = geometry.x_lengths_for_valid_points()
        y_length = geometry.y_lengths_for_valid_points()
        if torch.any(x_length <= 0.0) or torch.any(y_length <= 0.0):
            raise ValueError("Complex geometry segment lengths must be positive.")
