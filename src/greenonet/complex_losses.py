from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import ComplexRelativeSplitConsistencyConfig


@dataclass(frozen=True)
class ComplexBoundaryEnergyContext:
    """Physical endpoint-to-interior edges for every connected segment."""

    point_indices: torch.Tensor
    physical_distance: torch.Tensor
    transverse_measure: torch.Tensor
    axis_id: torch.Tensor
    endpoint_coords: torch.Tensor
    segment_id: torch.Tensor
    side_id: torch.Tensor
    x_anchor_count: int
    y_anchor_count: int

    @property
    def total_anchors(self) -> int:
        return int(self.point_indices.numel())

    def to(self, device: torch.device | str) -> ComplexBoundaryEnergyContext:
        return replace(
            self,
            point_indices=self.point_indices.to(device),
            physical_distance=self.physical_distance.to(device),
            transverse_measure=self.transverse_measure.to(device),
            axis_id=self.axis_id.to(device),
            endpoint_coords=self.endpoint_coords.to(device),
            segment_id=self.segment_id.to(device),
            side_id=self.side_id.to(device),
        )


@dataclass(frozen=True)
class ComplexBoundaryEnergyLossResult:
    """Dirichlet endpoint contribution of the canonical split energy."""

    total: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    total_per_sample: torch.Tensor
    x_per_sample: torch.Tensor
    y_per_sample: torch.Tensor


@dataclass(frozen=True)
class ComplexEnergyLossResult:
    """Full-domain canonical bulk-plus-boundary energy."""

    total: torch.Tensor
    bulk: torch.Tensor
    boundary: torch.Tensor
    boundary_x: torch.Tensor
    boundary_y: torch.Tensor
    total_per_sample: torch.Tensor
    bulk_per_sample: torch.Tensor
    boundary_per_sample: torch.Tensor
    boundary_x_per_sample: torch.Tensor
    boundary_y_per_sample: torch.Tensor


@dataclass(frozen=True)
class ComplexRelativeSplitLossResult:
    """Source-normalized split energy and solution-value consistency."""

    loss: torch.Tensor
    energy_relative: torch.Tensor
    mass_relative: torch.Tensor
    loss_per_sample: torch.Tensor
    energy_relative_per_sample: torch.Tensor
    mass_relative_per_sample: torch.Tensor
    mass_unscaled_per_sample: torch.Tensor
    rhs_l2_squared_per_sample: torch.Tensor
    domain_length_scale: torch.Tensor


def build_boundary_energy_context(
    geometry: ComplexGeometryMetadata,
) -> ComplexBoundaryEnergyContext:
    """Connect every segment endpoint to its nearest represented interior node."""

    point_indices: list[int] = []
    physical_distance: list[float] = []
    transverse_measure: list[float] = []
    axis_id: list[int] = []
    endpoint_coords: list[list[float]] = []
    segment_ids: list[int] = []
    side_ids: list[int] = []

    x_anchor_count = _append_axis_boundary_context(
        geometry=geometry,
        axis="x",
        point_indices=point_indices,
        physical_distance=physical_distance,
        transverse_measure=transverse_measure,
        axis_id=axis_id,
        endpoint_coords=endpoint_coords,
        segment_ids=segment_ids,
        side_ids=side_ids,
    )
    y_anchor_count = _append_axis_boundary_context(
        geometry=geometry,
        axis="y",
        point_indices=point_indices,
        physical_distance=physical_distance,
        transverse_measure=transverse_measure,
        axis_id=axis_id,
        endpoint_coords=endpoint_coords,
        segment_ids=segment_ids,
        side_ids=side_ids,
    )
    dtype = geometry.coords_valid.dtype
    device = geometry.coords_valid.device
    return ComplexBoundaryEnergyContext(
        point_indices=torch.tensor(point_indices, dtype=torch.long, device=device),
        physical_distance=torch.tensor(
            physical_distance,
            dtype=dtype,
            device=device,
        ),
        transverse_measure=torch.tensor(
            transverse_measure,
            dtype=dtype,
            device=device,
        ),
        axis_id=torch.tensor(axis_id, dtype=torch.long, device=device),
        endpoint_coords=(
            torch.tensor(endpoint_coords, dtype=dtype, device=device)
            if endpoint_coords
            else torch.empty((0, 2), dtype=dtype, device=device)
        ),
        segment_id=torch.tensor(segment_ids, dtype=torch.long, device=device),
        side_id=torch.tensor(side_ids, dtype=torch.long, device=device),
        x_anchor_count=x_anchor_count,
        y_anchor_count=y_anchor_count,
    )


def physical_boundary_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    context: ComplexBoundaryEnergyContext,
) -> ComplexBoundaryEnergyLossResult:
    """Evaluate endpoint P1 edge energy with homogeneous boundary value zero."""

    residual, a_valid = _validate_energy_inputs(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
    )
    context = context.to(residual.device)
    if context.total_anchors == 0:
        zero_per_sample = residual.new_zeros((residual.shape[0],))
        zero = residual.new_zeros(())
        return ComplexBoundaryEnergyLossResult(
            total=zero,
            x=zero,
            y=zero,
            total_per_sample=zero_per_sample,
            x_per_sample=zero_per_sample,
            y_per_sample=zero_per_sample,
        )
    if torch.any(context.physical_distance <= 0.0):
        raise ValueError("Boundary endpoint distances must be positive.")
    indices = context.point_indices
    weights = (
        a_valid[:, indices]
        * context.transverse_measure.unsqueeze(0)
        / context.physical_distance.unsqueeze(0)
    )
    values = weights * residual[:, indices].square()
    x_mask = context.axis_id == 0
    y_mask = context.axis_id == 1
    x_per_sample = values[:, x_mask].sum(dim=-1)
    y_per_sample = values[:, y_mask].sum(dim=-1)
    total_per_sample = x_per_sample + y_per_sample
    return ComplexBoundaryEnergyLossResult(
        total=total_per_sample.mean(),
        x=x_per_sample.mean(),
        y=y_per_sample.mean(),
        total_per_sample=total_per_sample,
        x_per_sample=x_per_sample,
        y_per_sample=y_per_sample,
    )


def physical_edge_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
) -> torch.Tensor:
    residual, a_valid = _validate_energy_inputs(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
    )
    x_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.x_edges.to(residual.device),
        spacing=geometry.hx.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    y_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.y_edges.to(residual.device),
        spacing=geometry.hy.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    return _sum_edge_energy(x_values, y_values, residual)


def canonical_complex_energy_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    geometry: ComplexGeometryMetadata,
    boundary_context: ComplexBoundaryEnergyContext | None = None,
) -> ComplexEnergyLossResult:
    """Integrate the canonical split energy over all physical valid edges."""

    residual, a_valid = _validate_energy_inputs(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
    )
    x_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.x_edges.to(residual.device),
        spacing=geometry.hx.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    y_values = _edge_energy_values(
        residual=residual,
        a_valid=a_valid,
        edges=geometry.y_edges.to(residual.device),
        spacing=geometry.hy.to(residual.device),
        area=geometry.hx.to(residual.device) * geometry.hy.to(residual.device),
    )
    bulk_per_sample = _sum_edge_energy_per_sample(
        x_values,
        y_values,
        residual,
    )
    bulk = bulk_per_sample.mean()
    if boundary_context is None:
        boundary_context = build_boundary_energy_context(geometry)
    boundary = physical_boundary_energy_loss(
        u_phi_valid=u_phi_valid,
        u_psi_valid=u_psi_valid,
        a_valid=a_valid,
        context=boundary_context,
    )
    total_per_sample = bulk_per_sample + boundary.total_per_sample
    return ComplexEnergyLossResult(
        total=total_per_sample.mean(),
        bulk=bulk,
        boundary=boundary.total,
        boundary_x=boundary.x,
        boundary_y=boundary.y,
        total_per_sample=total_per_sample,
        bulk_per_sample=bulk_per_sample,
        boundary_per_sample=boundary.total_per_sample,
        boundary_x_per_sample=boundary.x_per_sample,
        boundary_y_per_sample=boundary.y_per_sample,
    )


def relative_split_consistency_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    rhs_valid: torch.Tensor,
    energy: ComplexEnergyLossResult,
    geometry: ComplexGeometryMetadata,
    config: ComplexRelativeSplitConsistencyConfig,
) -> ComplexRelativeSplitLossResult:
    """Normalize split energy and value mismatch by each sample source scale."""

    if u_phi_valid.shape != u_psi_valid.shape:
        raise ValueError("u_phi_valid and u_psi_valid must have matching shapes.")
    if u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if rhs_valid.shape != u_phi_valid.shape:
        raise ValueError("rhs_valid must match represented solution shape.")
    if energy.total_per_sample.shape != (u_phi_valid.shape[0],):
        raise ValueError("energy per-sample values do not match batch size.")

    area = geometry.hx.to(u_phi_valid.device) * geometry.hy.to(u_phi_valid.device)
    split_residual = u_phi_valid - u_psi_valid
    mass_unscaled_per_sample = split_residual.square().sum(dim=-1) * area
    rhs_l2_squared_per_sample = rhs_valid.square().sum(dim=-1) * area
    domain_length_scale = _domain_length_scale(
        geometry,
        reference=u_phi_valid,
    )
    denominator = rhs_l2_squared_per_sample + float(config.eps)
    energy_relative_per_sample = energy.total_per_sample / denominator
    mass_relative_per_sample = (
        mass_unscaled_per_sample / domain_length_scale.square()
    ) / denominator
    loss_per_sample = float(config.weight) * (
        energy_relative_per_sample
        + float(config.mass_weight) * mass_relative_per_sample
    )
    return ComplexRelativeSplitLossResult(
        loss=loss_per_sample.mean(),
        energy_relative=energy_relative_per_sample.mean(),
        mass_relative=mass_relative_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        energy_relative_per_sample=energy_relative_per_sample,
        mass_relative_per_sample=mass_relative_per_sample,
        mass_unscaled_per_sample=mass_unscaled_per_sample,
        rhs_l2_squared_per_sample=rhs_l2_squared_per_sample,
        domain_length_scale=domain_length_scale,
    )


def _validate_energy_inputs(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if u_phi_valid.shape != u_psi_valid.shape:
        raise ValueError("u_phi_valid and u_psi_valid must have matching shapes.")
    if u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if a_valid.dim() == 1:
        a_valid = a_valid.unsqueeze(0).expand_as(u_phi_valid)
    if a_valid.shape != u_phi_valid.shape:
        raise ValueError("a_valid must have shape (P,) or (B, P).")
    return u_phi_valid - u_psi_valid, a_valid


def _edge_energy_values(
    *,
    residual: torch.Tensor,
    a_valid: torch.Tensor,
    edges: torch.Tensor,
    spacing: torch.Tensor,
    area: torch.Tensor,
) -> torch.Tensor:
    if edges.numel() == 0:
        return residual.new_zeros((residual.shape[0], 0))
    left = edges[:, 0]
    right = edges[:, 1]
    derivative = (residual[:, right] - residual[:, left]) / spacing
    a_face = 0.5 * (a_valid[:, left] + a_valid[:, right])
    return a_face * derivative.pow(2) * area


def _sum_edge_energy(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if x_values.shape[-1] + y_values.shape[-1] == 0:
        return reference.new_zeros(())
    return (x_values.sum(dim=-1) + y_values.sum(dim=-1)).mean()


def _sum_edge_energy_per_sample(
    x_values: torch.Tensor,
    y_values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if x_values.shape[-1] + y_values.shape[-1] == 0:
        return reference.new_zeros((reference.shape[0],))
    return x_values.sum(dim=-1) + y_values.sum(dim=-1)


def _domain_length_scale(
    geometry: ComplexGeometryMetadata,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    x_span = geometry.y_transverse_max.to(
        reference.device
    ) - geometry.y_transverse_min.to(reference.device)
    y_span = geometry.x_transverse_max.to(
        reference.device
    ) - geometry.x_transverse_min.to(reference.device)
    scale = torch.maximum(x_span, y_span).to(dtype=reference.dtype)
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) <= 0.0:
        raise ValueError("Complex geometry domain length scale must be positive.")
    return scale


def _append_axis_boundary_context(
    *,
    geometry: ComplexGeometryMetadata,
    axis: str,
    point_indices: list[int],
    physical_distance: list[float],
    transverse_measure: list[float],
    axis_id: list[int],
    endpoint_coords: list[list[float]],
    segment_ids: list[int],
    side_ids: list[int],
) -> int:
    if axis == "x":
        ptr = geometry.x_recon_ptr
        local_t = geometry.x_recon_t
        valid_index = geometry.x_recon_valid_index
        length = geometry.x_segment_length
        lower = geometry.x_segment_left
        upper = geometry.x_segment_right
        fixed = geometry.x_segment_y
        measure = float(geometry.hy.item())
        numeric_axis_id = 0
    elif axis == "y":
        ptr = geometry.y_recon_ptr
        local_t = geometry.y_recon_t
        valid_index = geometry.y_recon_valid_index
        length = geometry.y_segment_length
        lower = geometry.y_segment_bottom
        upper = geometry.y_segment_top
        fixed = geometry.y_segment_x
        measure = float(geometry.hx.item())
        numeric_axis_id = 1
    else:
        raise ValueError(f"Unsupported boundary context axis: {axis}.")

    initial_count = len(point_indices)
    for segment_index in range(int(length.numel())):
        start = int(ptr[segment_index].item())
        stop = int(ptr[segment_index + 1].item())
        segment_indices = valid_index[start:stop]
        segment_t = local_t[start:stop]
        if segment_indices.numel() < 2:
            raise ValueError(f"{axis}-segment reconstruction requires two endpoints.")
        if (
            int(segment_indices[0].item()) != -1
            or int(segment_indices[-1].item()) != -1
        ):
            raise ValueError(
                f"{axis}-segment reconstruction endpoints must use valid_index=-1."
            )
        interior_positions = torch.nonzero(
            segment_indices >= 0,
            as_tuple=False,
        ).flatten()
        if interior_positions.numel() == 0:
            continue
        first_position = int(interior_positions[0].item())
        last_position = int(interior_positions[-1].item())
        first_t = float(segment_t[first_position].item())
        last_t = float(segment_t[last_position].item())
        segment_length = float(length[segment_index].item())
        distances = (segment_length * first_t, segment_length * (1.0 - last_t))
        if not all(math.isfinite(value) and value > 0.0 for value in distances):
            raise ValueError(
                f"{axis}-segment {segment_index} has non-positive boundary distance."
            )
        valid_points = (
            int(segment_indices[first_position].item()),
            int(segment_indices[last_position].item()),
        )
        endpoints = (
            float(lower[segment_index].item()),
            float(upper[segment_index].item()),
        )
        fixed_coordinate = float(fixed[segment_index].item())
        for side, (point_index, distance, endpoint) in enumerate(
            zip(valid_points, distances, endpoints, strict=True)
        ):
            point_indices.append(point_index)
            physical_distance.append(distance)
            transverse_measure.append(measure)
            axis_id.append(numeric_axis_id)
            endpoint_coords.append(
                [endpoint, fixed_coordinate]
                if axis == "x"
                else [fixed_coordinate, endpoint]
            )
            segment_ids.append(segment_index)
            side_ids.append(side)
    return len(point_indices) - initial_count


def relative_l2_valid(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes.")
    numerator = torch.linalg.vector_norm(pred - target, dim=-1)
    denominator = torch.linalg.vector_norm(target, dim=-1).clamp_min(eps)
    return cast(torch.Tensor, (numerator / denominator).mean())
