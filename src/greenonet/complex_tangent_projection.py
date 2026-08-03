from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from greenonet.complex_axial_response_operator import (
    FrozenAxialResponseOperatorBuilder,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig


@dataclass(frozen=True)
class SymmetricTangentGreenResponseContext:
    """Frozen response operator and fixed Jacobi tangent preconditioner."""

    response_operator: FrozenBidirectionalResponseOperator
    gamma_x_squared: torch.Tensor
    gamma_y_squared: torch.Tensor
    preconditioner_base: torch.Tensor
    gain_scale: torch.Tensor
    denominator: torch.Tensor
    point_mass: torch.Tensor
    eta: float
    relative_lambda: float
    denominator_relative_eps: float

    @classmethod
    def from_response_operator(
        cls,
        *,
        response_operator: FrozenBidirectionalResponseOperator,
        point_mass: torch.Tensor | float,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> SymmetricTangentGreenResponseContext:
        resolved = SymmetricTangentGreenResponseProjectionConfig.from_raw(config)
        mass = torch.as_tensor(
            point_mass,
            dtype=response_operator.x.dtype,
            device=response_operator.x.device,
        )
        if mass.numel() != 1 or not torch.isfinite(mass):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((mass <= 0.0).item()):
            raise ValueError("point_mass must be positive.")
        gamma_x_squared, gamma_y_squared = response_operator.column_gain_squared(
            point_mass=mass
        )
        preconditioner_base = gamma_x_squared + gamma_y_squared
        gain_scale = preconditioner_base.mean()
        if not torch.isfinite(gain_scale) or bool((gain_scale <= 0.0).item()):
            raise ValueError(
                "The tangent Green-response preconditioner gain scale must be "
                "finite and positive."
            )
        denominator = (
            preconditioner_base
            + (
                float(resolved.relative_lambda)
                + float(resolved.denominator_relative_eps)
            )
            * gain_scale
        )
        if not torch.all(torch.isfinite(denominator)) or torch.any(denominator <= 0.0):
            raise ValueError(
                "The tangent Green-response denominator must be finite and positive."
            )
        return cls(
            response_operator=response_operator,
            gamma_x_squared=gamma_x_squared.detach(),
            gamma_y_squared=gamma_y_squared.detach(),
            preconditioner_base=preconditioner_base.detach(),
            gain_scale=gain_scale.detach(),
            denominator=denominator.detach(),
            point_mass=mass.detach(),
            eta=float(resolved.eta),
            relative_lambda=float(resolved.relative_lambda),
            denominator_relative_eps=float(resolved.denominator_relative_eps),
        )

    @property
    def num_points(self) -> int:
        return self.response_operator.point_count

    def validate_for(self, reference: torch.Tensor) -> None:
        if reference.shape[-1] != self.num_points:
            raise ValueError(
                "Tangent Green-response point count does not match projection input."
            )
        if reference.device != self.denominator.device:
            raise ValueError(
                "Tangent Green-response context and projection input must share a "
                "device."
            )
        if reference.dtype != self.denominator.dtype:
            raise ValueError(
                "Tangent Green-response context and projection input must share a "
                "dtype."
            )

    def tangent_gradient(self, mismatch: torch.Tensor) -> torch.Tensor:
        self.validate_for(mismatch)
        return self.response_operator.tangent_gradient(
            mismatch,
            point_mass=self.point_mass,
        )

    def tangent_delta(self, gradient: torch.Tensor) -> torch.Tensor:
        self.validate_for(gradient)
        if self.eta == 0.0:
            return torch.zeros_like(gradient)
        return -self.eta * gradient / self.denominator.unsqueeze(0)

    def statistics(self) -> dict[str, float | int | bool | str]:
        x_stats = self.response_operator.x.statistics()
        y_stats = self.response_operator.y.statistics()
        return {
            "eta": self.eta,
            "relative_lambda": self.relative_lambda,
            "denominator_relative_eps": self.denominator_relative_eps,
            "gain_scale": float(self.gain_scale.item()),
            "gamma_x_squared_min": float(self.gamma_x_squared.min().item()),
            "gamma_x_squared_max": float(self.gamma_x_squared.max().item()),
            "gamma_y_squared_min": float(self.gamma_y_squared.min().item()),
            "gamma_y_squared_max": float(self.gamma_y_squared.max().item()),
            "preconditioner_base_min": float(self.preconditioner_base.min().item()),
            "preconditioner_base_max": float(self.preconditioner_base.max().item()),
            "denominator_min": float(self.denominator.min().item()),
            "denominator_max": float(self.denominator.max().item()),
            "x_segment_block_count": int(x_stats["segment_block_count"]),
            "y_segment_block_count": int(y_stats["segment_block_count"]),
            "x_local_matrix_entry_count": int(x_stats["local_matrix_entry_count"]),
            "y_local_matrix_entry_count": int(y_stats["local_matrix_entry_count"]),
            "global_matrix_materialized": False,
            "full_gram_solve": False,
            "row_norm_used": False,
        }


class SymmetricTangentGreenResponseContextBuilder:
    """Build one frozen tangent response context from the production Green kernel."""

    def __init__(
        self,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.config = SymmetricTangentGreenResponseProjectionConfig.from_raw(config)

    @torch.no_grad()
    def build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> SymmetricTangentGreenResponseContext:
        response_operator = FrozenAxialResponseOperatorBuilder.build(
            green_model=green_model,
            geometry=geometry,
            x_green_branch=x_green_branch,
            y_green_branch=y_green_branch,
        )
        point_mass = (geometry.hx * geometry.hy).to(
            device=x_green_branch.device,
            dtype=x_green_branch.dtype,
        )
        return SymmetricTangentGreenResponseContext.from_response_operator(
            response_operator=response_operator,
            point_mass=point_mass,
            config=self.config,
        )


class SymmetricTangentGreenResponseContextCache:
    """Lazily build and reuse one tangent response context per runtime."""

    def __init__(
        self,
        config: SymmetricTangentGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.builder = SymmetricTangentGreenResponseContextBuilder(config)
        self.context: SymmetricTangentGreenResponseContext | None = None
        self.build_count = 0
        self.build_seconds = 0.0

    def get_or_build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> SymmetricTangentGreenResponseContext:
        if self.context is None:
            start = time.perf_counter()
            self.context = self.builder.build(
                green_model=green_model,
                geometry=geometry,
                x_green_branch=x_green_branch,
                y_green_branch=y_green_branch,
            )
            self.build_seconds = time.perf_counter() - start
            self.build_count += 1
        self.context.validate_for(x_green_branch.new_empty(geometry.num_points))
        return self.context
