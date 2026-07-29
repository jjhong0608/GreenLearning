from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import ComplexPreProjectionFusionConfig
from greenonet.coupling_model import ActivationFactoryMixin


@dataclass(frozen=True)
class ComplexPreProjectionFusionResult:
    """Audit tensors for the optional physical difference fusion."""

    mode: str
    combination: str
    base_response: torch.Tensor
    base_physical: torch.Tensor
    fused_response: torch.Tensor
    fused_physical: torch.Tensor
    base_difference: torch.Tensor
    fused_difference: torch.Tensor
    linear_component: torch.Tensor
    nonlinear_component: torch.Tensor
    combined_component: torch.Tensor
    source_scale: torch.Tensor
    gate: torch.Tensor

    @property
    def linear_correction(self) -> torch.Tensor:
        """Legacy alias used by residual-correction diagnostics."""

        return self.linear_component

    @property
    def nonlinear_correction(self) -> torch.Tensor:
        """Legacy alias used by residual-correction diagnostics."""

        return self.nonlinear_component

    @property
    def blended_correction(self) -> torch.Tensor:
        """Legacy alias used by residual-correction diagnostics."""

        return self.combined_component


class ComplexPreProjectionFusion(nn.Module, ActivationFactoryMixin):
    """Fuse linear/nonlinear components of the physical split difference."""

    NONLINEAR_FEATURE_NAMES = (
        "normalized_difference",
        "normalized_rhs",
        "x_local_t",
        "y_local_t",
        "log_x_length_over_reference",
        "log_y_length_over_reference",
        "log_x_over_y_length",
        "length_balance_kappa",
    )

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
        self.linear_correction = nn.Linear(2, 1, bias=False, dtype=dtype)

        layers: list[nn.Module] = []
        input_dim = len(self.NONLINEAR_FEATURE_NAMES)
        for _ in range(self.config.nonlinear_depth):
            layers.append(
                nn.Linear(
                    input_dim,
                    self.config.nonlinear_hidden_dim,
                    bias=use_bias,
                    dtype=dtype,
                )
            )
            layers.append(self.build_activation(activation))
            input_dim = self.config.nonlinear_hidden_dim
        layers.append(nn.Linear(input_dim, 1, bias=use_bias, dtype=dtype))
        self.nonlinear_correction = nn.Sequential(*layers)

        gate = float(self.config.gate_initial_value)
        gate_logit = math.log(gate / (1.0 - gate))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=dtype))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        final_layer = cast(nn.Linear, self.nonlinear_correction[-1])
        if self.config.mode == "residual_correction":
            nn.init.zeros_(self.linear_correction.weight)
            nn.init.zeros_(final_layer.weight)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
            return

        with torch.no_grad():
            self.linear_correction.weight.copy_(
                self.linear_correction.weight.new_tensor([[1.0, 0.0]])
            )
            scale = float(self.config.nonlinear_final_init_scale)
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
        """Return fused response and interpretable correction components."""

        (
            computed_base_response,
            base_physical,
            fused_response,
            fused_physical,
            base_difference,
            fused_difference,
            linear_component,
            nonlinear_component,
            combined_component,
            source_scale,
            gate,
        ) = self._forward_tensors(
            base_response=base_response,
            rhs_phys=rhs_phys,
            geometry=geometry,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
        )
        return ComplexPreProjectionFusionResult(
            mode=self.config.mode,
            combination=self.config.combination,
            base_response=computed_base_response,
            base_physical=base_physical,
            fused_response=fused_response,
            fused_physical=fused_physical,
            base_difference=base_difference,
            fused_difference=fused_difference,
            linear_component=linear_component,
            nonlinear_component=nonlinear_component,
            combined_component=combined_component,
            source_scale=source_scale,
            gate=gate,
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
        safe_scale = source_scale.clamp_min(self.eps)
        normalized_difference = base_difference / safe_scale
        normalized_rhs = rhs_phys / safe_scale

        linear_input = torch.stack(
            (normalized_difference, normalized_rhs),
            dim=-1,
        )
        linear_normalized = self.linear_correction(linear_input).squeeze(-1)

        geometry_features = self._geometry_features(
            geometry,
            device=device,
            dtype=dtype,
        )
        expanded_geometry = geometry_features.unsqueeze(0).expand(
            base_response.shape[0],
            -1,
            -1,
        )
        nonlinear_input = torch.cat((linear_input, expanded_geometry), dim=-1)
        nonlinear_normalized = self.nonlinear_correction(nonlinear_input).squeeze(-1)

        linear_component = source_scale * linear_normalized
        nonlinear_component = source_scale * nonlinear_normalized
        gate = torch.sigmoid(self.gate_logit)
        combined_component, fused_difference = self._combine_components(
            base_difference=base_difference,
            linear_component=linear_component,
            nonlinear_component=nonlinear_component,
            gate=gate,
        )
        fused_physical = self._physical_pair_from_difference(
            base_physical=base_physical,
            fused_difference=fused_difference,
            combined_component=combined_component,
        )
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
            fused_difference,
            linear_component,
            nonlinear_component,
            combined_component,
            source_scale,
            gate,
        )

    def _combine_components(
        self,
        *,
        base_difference: torch.Tensor,
        linear_component: torch.Tensor,
        nonlinear_component: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.mode == "residual_correction":
            blended_correction = (
                1.0 - gate
            ) * linear_component + gate * nonlinear_component
            return blended_correction, base_difference + blended_correction

        if self.config.combination == "linear_plus_nonlinear":
            absolute_difference = linear_component + gate * nonlinear_component
        else:
            absolute_difference = (
                1.0 - gate
            ) * linear_component + gate * nonlinear_component
        return absolute_difference, absolute_difference

    def _physical_pair_from_difference(
        self,
        *,
        base_physical: torch.Tensor,
        fused_difference: torch.Tensor,
        combined_component: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.mode == "residual_correction":
            return torch.stack(
                (
                    base_physical[:, 0] + 0.5 * combined_component,
                    base_physical[:, 1] - 0.5 * combined_component,
                ),
                dim=1,
            )

        base_common = base_physical[:, 0] + base_physical[:, 1]
        return torch.stack(
            (
                0.5 * (base_common + fused_difference),
                0.5 * (base_common - fused_difference),
            ),
            dim=1,
        )

    def _geometry_features(
        self,
        geometry: ComplexGeometryMetadata,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x_length = geometry.x_lengths_for_valid_points().to(device=device, dtype=dtype)
        y_length = geometry.y_lengths_for_valid_points().to(device=device, dtype=dtype)
        x_extent = (geometry.y_transverse_max - geometry.y_transverse_min).to(
            device=device, dtype=dtype
        )
        y_extent = (geometry.x_transverse_max - geometry.x_transverse_min).to(
            device=device, dtype=dtype
        )
        reference_length = torch.maximum(x_extent, y_extent)
        sigma_x = x_length.square()
        sigma_y = y_length.square()
        denominator = (sigma_x + sigma_y).square()
        kappa = 4.0 * sigma_x * sigma_y / denominator
        return torch.stack(
            (
                geometry.x_local_t.to(device=device, dtype=dtype),
                geometry.y_local_t.to(device=device, dtype=dtype),
                torch.log(x_length / reference_length),
                torch.log(y_length / reference_length),
                torch.log(x_length / y_length),
                kappa,
            ),
            dim=-1,
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
