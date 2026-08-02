from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_weak_closure import (
    ComplexDirectionalWeakContext,
    assemble_directional_weak_residuals,
)
from greenonet.config import ComplexCrossAxisReconstructionConfig


@dataclass(frozen=True)
class LocalWeakResidualReliabilityContext:
    """Sample-independent axial weak operator and valid-point graph data."""

    weak: ComplexDirectionalWeakContext
    edges: torch.Tensor
    degree: torch.Tensor
    connected: torch.Tensor

    @classmethod
    def build(
        cls,
        geometry: ComplexGeometryMetadata,
        weak_context: ComplexDirectionalWeakContext,
    ) -> LocalWeakResidualReliabilityContext:
        if weak_context.num_points != geometry.num_points:
            raise ValueError(
                "Weak context point count does not match cross-axis geometry."
            )
        edges = torch.cat((geometry.x_edges, geometry.y_edges), dim=0)
        if edges.dim() != 2 or edges.shape[1] != 2:
            raise ValueError("Cross-axis graph edges must have shape (E, 2).")
        if edges.numel() and (
            int(edges.min().item()) < 0
            or int(edges.max().item()) >= geometry.num_points
        ):
            raise ValueError("Cross-axis graph edges contain invalid point indices.")
        degree = weak_context.point_area.new_zeros((geometry.num_points,))
        if edges.numel():
            ones = degree.new_ones((edges.shape[0],))
            degree.index_add_(0, edges[:, 0], ones)
            degree.index_add_(0, edges[:, 1], ones)
        return cls(
            weak=weak_context,
            edges=edges,
            degree=degree,
            connected=degree > 0,
        )

    def to(
        self,
        device: torch.device | str,
    ) -> LocalWeakResidualReliabilityContext:
        return type(self)(
            weak=self.weak.to(device),
            edges=self.edges.to(device),
            degree=self.degree.to(device),
            connected=self.connected.to(device),
        )


@dataclass(frozen=True)
class LocalWeakResidualReliabilityFields:
    """Candidate full-PDE weak defects and their reliability partition."""

    phi_x_residual: torch.Tensor
    phi_y_residual: torch.Tensor
    phi_full_residual: torch.Tensor
    psi_x_residual: torch.Tensor
    psi_y_residual: torch.Tensor
    psi_full_residual: torch.Tensor
    nodal_mass: torch.Tensor
    phi_indicator_raw: torch.Tensor
    psi_indicator_raw: torch.Tensor
    phi_indicator: torch.Tensor
    psi_indicator: torch.Tensor
    sample_floor: torch.Tensor
    theta: torch.Tensor
    w_phi: torch.Tensor
    w_psi: torch.Tensor
    support_mask: torch.Tensor


@dataclass(frozen=True)
class ComplexCrossAxisReconstructionResult:
    """Official complex solution prediction and equal-mean audit baseline."""

    mode: Literal["equal_mean", "local_weak_residual_reliability"]
    u_pred_valid: torch.Tensor
    u_equal_mean_valid: torch.Tensor
    reliability: LocalWeakResidualReliabilityFields | None


class ComplexCrossAxisReconstructor:
    """Build the optional reference-free final directional-solution blend."""

    def __init__(
        self,
        config: ComplexCrossAxisReconstructionConfig | dict[str, Any] | None,
    ) -> None:
        self.config = ComplexCrossAxisReconstructionConfig.from_raw(config)
        self._context: LocalWeakResidualReliabilityContext | None = None
        self._context_build_count = 0

    @property
    def context(self) -> LocalWeakResidualReliabilityContext | None:
        return self._context

    @property
    def context_build_count(self) -> int:
        return self._context_build_count

    def _context_for(
        self,
        geometry: ComplexGeometryMetadata,
        weak_context: ComplexDirectionalWeakContext,
    ) -> LocalWeakResidualReliabilityContext:
        if self._context is None:
            self._context = LocalWeakResidualReliabilityContext.build(
                geometry,
                weak_context,
            )
            self._context_build_count += 1
        elif self._context.weak.num_points != geometry.num_points:
            raise ValueError(
                "Cross-axis reconstructor cannot reuse a context with a different "
                "geometry point count."
            )
        return self._context

    @torch.no_grad()
    def reconstruct(
        self,
        *,
        u_phi_valid: torch.Tensor,
        u_psi_valid: torch.Tensor,
        projected_physical: torch.Tensor,
        geometry: ComplexGeometryMetadata,
        weak_context: ComplexDirectionalWeakContext,
    ) -> ComplexCrossAxisReconstructionResult:
        self._validate_candidate_inputs(
            u_phi_valid,
            u_psi_valid,
            projected_physical,
            geometry.num_points,
        )
        equal_mean = 0.5 * (u_phi_valid + u_psi_valid)
        if not self.config.enabled:
            return ComplexCrossAxisReconstructionResult(
                mode="equal_mean",
                u_pred_valid=equal_mean,
                u_equal_mean_valid=equal_mean,
                reliability=None,
            )
        context = self._context_for(geometry, weak_context).to(u_phi_valid.device)
        fields = self.build_reliability_fields(
            u_phi_valid=u_phi_valid,
            u_psi_valid=u_psi_valid,
            projected_physical=projected_physical,
            context=context,
            config=self.config,
        )
        prediction = fields.w_phi * u_phi_valid + fields.w_psi * u_psi_valid
        if not torch.all(torch.isfinite(prediction)):
            raise RuntimeError(
                "Local weak-residual reliability produced a non-finite prediction."
            )
        return ComplexCrossAxisReconstructionResult(
            mode="local_weak_residual_reliability",
            u_pred_valid=prediction,
            u_equal_mean_valid=equal_mean,
            reliability=fields,
        )

    @classmethod
    def build_reliability_fields(
        cls,
        *,
        u_phi_valid: torch.Tensor,
        u_psi_valid: torch.Tensor,
        projected_physical: torch.Tensor,
        context: LocalWeakResidualReliabilityContext,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> LocalWeakResidualReliabilityFields:
        cls._validate_candidate_inputs(
            u_phi_valid,
            u_psi_valid,
            projected_physical,
            context.weak.num_points,
        )
        phi_residuals = assemble_directional_weak_residuals(
            u_valid=u_phi_valid,
            projected_physical=projected_physical,
            context=context.weak,
        )
        psi_residuals = assemble_directional_weak_residuals(
            u_valid=u_psi_valid,
            projected_physical=projected_physical,
            context=context.weak,
        )
        nodal_mass = context.weak.x.nodal_mass + context.weak.y.nodal_mass
        phi_indicator_raw = phi_residuals.full.square() / (
            nodal_mass.unsqueeze(0) + float(config.eps)
        )
        psi_indicator_raw = psi_residuals.full.square() / (
            nodal_mass.unsqueeze(0) + float(config.eps)
        )
        return cls.fields_from_raw_indicators(
            phi_residuals_x=phi_residuals.x,
            phi_residuals_y=phi_residuals.y,
            psi_residuals_x=psi_residuals.x,
            psi_residuals_y=psi_residuals.y,
            nodal_mass=nodal_mass,
            phi_indicator_raw=phi_indicator_raw,
            psi_indicator_raw=psi_indicator_raw,
            context=context,
            config=config,
        )

    @classmethod
    def reweight_reliability_fields(
        cls,
        fields: LocalWeakResidualReliabilityFields,
        *,
        context: LocalWeakResidualReliabilityContext,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> LocalWeakResidualReliabilityFields:
        updated = cls._weights_from_raw_indicators(
            fields.phi_indicator_raw,
            fields.psi_indicator_raw,
            num_points=context.weak.num_points,
            edges=context.edges,
            degree=context.degree,
            connected=context.connected,
            config=config,
        )
        return replace(
            fields,
            phi_indicator=updated[0],
            psi_indicator=updated[1],
            sample_floor=updated[2],
            theta=updated[3],
            w_phi=updated[4],
            w_psi=updated[5],
            support_mask=updated[6],
        )

    @classmethod
    def fields_from_raw_indicators(
        cls,
        *,
        phi_residuals_x: torch.Tensor,
        phi_residuals_y: torch.Tensor,
        psi_residuals_x: torch.Tensor,
        psi_residuals_y: torch.Tensor,
        nodal_mass: torch.Tensor,
        phi_indicator_raw: torch.Tensor,
        psi_indicator_raw: torch.Tensor,
        context: LocalWeakResidualReliabilityContext,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> LocalWeakResidualReliabilityFields:
        (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            w_psi,
            support_mask,
        ) = cls._weights_from_raw_indicators(
            phi_indicator_raw,
            psi_indicator_raw,
            num_points=context.weak.num_points,
            edges=context.edges,
            degree=context.degree,
            connected=context.connected,
            config=config,
        )
        return LocalWeakResidualReliabilityFields(
            phi_x_residual=phi_residuals_x,
            phi_y_residual=phi_residuals_y,
            phi_full_residual=phi_residuals_x + phi_residuals_y,
            psi_x_residual=psi_residuals_x,
            psi_y_residual=psi_residuals_y,
            psi_full_residual=psi_residuals_x + psi_residuals_y,
            nodal_mass=nodal_mass,
            phi_indicator_raw=phi_indicator_raw,
            psi_indicator_raw=psi_indicator_raw,
            phi_indicator=phi_indicator,
            psi_indicator=psi_indicator,
            sample_floor=sample_floor,
            theta=theta,
            w_phi=w_phi,
            w_psi=w_psi,
            support_mask=support_mask,
        )

    @classmethod
    def _weights_from_raw_indicators(
        cls,
        phi_indicator_raw: torch.Tensor,
        psi_indicator_raw: torch.Tensor,
        *,
        num_points: int,
        edges: torch.Tensor,
        degree: torch.Tensor,
        connected: torch.Tensor,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if phi_indicator_raw.shape != psi_indicator_raw.shape:
            raise ValueError("Candidate indicators must have matching shape.")
        if phi_indicator_raw.dim() != 2:
            raise ValueError("Candidate indicators must have shape (B, P).")
        if phi_indicator_raw.shape[1] != num_points:
            raise ValueError("Candidate indicators do not match weak-context points.")
        if torch.any(phi_indicator_raw < 0.0) or torch.any(psi_indicator_raw < 0.0):
            raise ValueError("Candidate indicators must be non-negative.")

        phi_indicator = cls._smooth_indicator_graph(
            phi_indicator_raw,
            edges=edges,
            degree=degree,
            connected=connected,
            config=config,
        )
        psi_indicator = cls._smooth_indicator_graph(
            psi_indicator_raw,
            edges=edges,
            degree=degree,
            connected=connected,
            config=config,
        )
        mean_indicator = 0.5 * (
            phi_indicator.mean(dim=1, keepdim=True)
            + psi_indicator.mean(dim=1, keepdim=True)
        )
        sample_floor = float(config.relative_floor) * mean_indicator + float(config.eps)
        denominator = phi_indicator + psi_indicator + 2.0 * sample_floor
        theta = float(config.gamma) * (psi_indicator - phi_indicator) / denominator
        w_phi = 0.5 * (1.0 + theta)
        w_psi = 1.0 - w_phi
        support_mask = theta.abs() > 10.0 * float(config.eps)

        fields = (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            w_psi,
        )
        if not all(torch.all(torch.isfinite(field)) for field in fields):
            raise RuntimeError(
                "Local weak-residual reliability produced non-finite fields."
            )
        if torch.any(w_phi < 0.0) or torch.any(w_phi > 1.0):
            raise RuntimeError("Reliability weights must be in [0, 1].")
        if torch.any(w_psi < 0.0) or torch.any(w_psi > 1.0):
            raise RuntimeError("Reliability weights must be in [0, 1].")
        if not torch.allclose(
            w_phi + w_psi,
            torch.ones_like(w_phi),
            atol=1.0e-12,
            rtol=1.0e-12,
        ):
            raise RuntimeError("Reliability weights must sum to one.")
        return (
            phi_indicator,
            psi_indicator,
            sample_floor,
            theta,
            w_phi,
            w_psi,
            support_mask,
        )

    @classmethod
    def weights_from_geometry(
        cls,
        phi_indicator_raw: torch.Tensor,
        psi_indicator_raw: torch.Tensor,
        *,
        geometry: ComplexGeometryMetadata,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply the production graph smoother and reliability partition."""
        edges = torch.cat((geometry.x_edges, geometry.y_edges), dim=0)
        reference = phi_indicator_raw
        degree = reference.new_zeros((geometry.num_points,))
        if edges.numel():
            edges = edges.to(reference.device)
            ones = degree.new_ones((edges.shape[0],))
            degree.index_add_(0, edges[:, 0], ones)
            degree.index_add_(0, edges[:, 1], ones)
        return cls._weights_from_raw_indicators(
            phi_indicator_raw,
            psi_indicator_raw,
            num_points=geometry.num_points,
            edges=edges,
            degree=degree,
            connected=degree > 0,
            config=config,
        )

    @classmethod
    def _smooth_indicator(
        cls,
        values: torch.Tensor,
        context: LocalWeakResidualReliabilityContext,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> torch.Tensor:
        return cls._smooth_indicator_graph(
            values,
            edges=context.edges,
            degree=context.degree,
            connected=context.connected,
            config=config,
        )

    @staticmethod
    def _smooth_indicator_graph(
        values: torch.Tensor,
        *,
        edges: torch.Tensor,
        degree: torch.Tensor,
        connected: torch.Tensor,
        config: ComplexCrossAxisReconstructionConfig,
    ) -> torch.Tensor:
        current = values.clone()
        if config.smoothing_steps == 0 or edges.numel() == 0:
            return current
        degree = degree.to(device=values.device, dtype=values.dtype)
        connected = connected.to(device=values.device)
        edges = edges.to(values.device)
        for _ in range(config.smoothing_steps):
            neighbor_sum = torch.zeros_like(current)
            neighbor_sum.index_add_(1, edges[:, 0], current[:, edges[:, 1]])
            neighbor_sum.index_add_(1, edges[:, 1], current[:, edges[:, 0]])
            neighbor_mean = neighbor_sum / degree.clamp_min(1.0).unsqueeze(0)
            relaxed = (1.0 - float(config.smoothing_relaxation)) * current + float(
                config.smoothing_relaxation
            ) * neighbor_mean
            current = torch.where(connected.unsqueeze(0), relaxed, current)
        return current

    @staticmethod
    def _validate_candidate_inputs(
        u_phi_valid: torch.Tensor,
        u_psi_valid: torch.Tensor,
        projected_physical: torch.Tensor,
        num_points: int,
    ) -> None:
        if u_phi_valid.dim() != 2 or u_phi_valid.shape[-1] != num_points:
            raise ValueError("u_phi_valid must have shape (B, P).")
        if u_psi_valid.shape != u_phi_valid.shape:
            raise ValueError("u_psi_valid must match u_phi_valid shape.")
        if projected_physical.shape != (
            u_phi_valid.shape[0],
            2,
            num_points,
        ):
            raise ValueError("projected_physical must have shape (B, 2, P).")
        if not all(
            torch.all(torch.isfinite(field))
            for field in (u_phi_valid, u_psi_valid, projected_physical)
        ):
            raise ValueError("Cross-axis reconstruction inputs must be finite.")
