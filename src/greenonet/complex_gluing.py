from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import ComplexAdmissibilityGluingConfig


@dataclass(frozen=True)
class AxisTraceContext:
    """Precomputed axial-only trace stencils for one slice orientation."""

    lower_indices: torch.Tensor
    lower_weights: torch.Tensor
    upper_indices: torch.Tensor
    upper_weights: torch.Tensor
    face_indices: torch.Tensor
    integration_weight: torch.Tensor
    trace_coords: torch.Tensor
    transition_mask: torch.Tensor
    carrier_self_index: torch.Tensor
    carrier_indices: torch.Tensor
    carrier_weights: torch.Tensor
    carrier_coords: torch.Tensor
    boundary_indices: torch.Tensor
    boundary_weights: torch.Tensor
    boundary_face_indices: torch.Tensor
    boundary_integration_weight: torch.Tensor
    boundary_coords: torch.Tensor
    interface_count: int
    transition_interface_count: int

    def to(self, device: torch.device | str) -> AxisTraceContext:
        return replace(
            self,
            lower_indices=self.lower_indices.to(device),
            lower_weights=self.lower_weights.to(device),
            upper_indices=self.upper_indices.to(device),
            upper_weights=self.upper_weights.to(device),
            face_indices=self.face_indices.to(device),
            integration_weight=self.integration_weight.to(device),
            trace_coords=self.trace_coords.to(device),
            transition_mask=self.transition_mask.to(device),
            carrier_self_index=self.carrier_self_index.to(device),
            carrier_indices=self.carrier_indices.to(device),
            carrier_weights=self.carrier_weights.to(device),
            carrier_coords=self.carrier_coords.to(device),
            boundary_indices=self.boundary_indices.to(device),
            boundary_weights=self.boundary_weights.to(device),
            boundary_face_indices=self.boundary_face_indices.to(device),
            boundary_integration_weight=self.boundary_integration_weight.to(device),
            boundary_coords=self.boundary_coords.to(device),
        )


@dataclass(frozen=True)
class ComplexGluingContext:
    x: AxisTraceContext
    y: AxisTraceContext

    def to(self, device: torch.device | str) -> ComplexGluingContext:
        return ComplexGluingContext(x=self.x.to(device), y=self.y.to(device))


@dataclass(frozen=True)
class ComplexGluingLossResult:
    loss: torch.Tensor
    loss_per_sample: torch.Tensor
    self_loss: torch.Tensor
    self_loss_per_sample: torch.Tensor
    self_regular: torch.Tensor
    self_regular_per_sample: torch.Tensor
    self_transition: torch.Tensor
    self_transition_per_sample: torch.Tensor
    carrier_transition: torch.Tensor
    carrier_transition_per_sample: torch.Tensor
    x_self_rms: torch.Tensor
    y_self_rms: torch.Tensor
    x_carrier_rms: torch.Tensor
    y_carrier_rms: torch.Tensor
    x_self_residual: torch.Tensor
    y_self_residual: torch.Tensor
    x_carrier_residual: torch.Tensor
    y_carrier_residual: torch.Tensor
    transition_trace_fraction: torch.Tensor


@dataclass(frozen=True)
class _LineData:
    coordinate: float
    points: dict[int, int]
    segment_ids: tuple[int, ...]


def build_complex_gluing_context(
    geometry: ComplexGeometryMetadata,
    config: ComplexAdmissibilityGluingConfig,
) -> ComplexGluingContext:
    """Build reusable trace stencils from existing axial-line metadata."""

    return ComplexGluingContext(
        x=_build_axis_context(geometry, config, axis="x"),
        y=_build_axis_context(geometry, config, axis="y"),
    )


def complex_admissibility_gluing_loss(
    *,
    u_phi_valid: torch.Tensor,
    u_psi_valid: torch.Tensor,
    a_valid: torch.Tensor,
    context: ComplexGluingContext,
    config: ComplexAdmissibilityGluingConfig,
) -> ComplexGluingLossResult:
    """Penalize broken transverse traces without reference solution targets."""

    if u_phi_valid.shape != u_psi_valid.shape or u_phi_valid.dim() != 2:
        raise ValueError("u_phi_valid and u_psi_valid must have shape (B, P).")
    if a_valid.dim() == 1:
        a_valid = a_valid.unsqueeze(0).expand_as(u_phi_valid)
    if a_valid.shape != u_phi_valid.shape:
        raise ValueError("a_valid must have shape (P,) or (B, P).")

    context = context.to(u_phi_valid.device)
    x_stats = _axis_loss_terms(u_phi_valid, u_psi_valid, a_valid, context.x)
    y_stats = _axis_loss_terms(u_psi_valid, u_phi_valid, a_valid, context.y)

    regular_num = x_stats["regular_num"] + y_stats["regular_num"]
    regular_den = x_stats["regular_den"] + y_stats["regular_den"]
    transition_num = x_stats["transition_num"] + y_stats["transition_num"]
    transition_den = x_stats["transition_den"] + y_stats["transition_den"]
    all_num = regular_num + transition_num
    all_den = regular_den + transition_den
    eps = float(config.eps)
    regular = _safe_weighted_mean(regular_num, regular_den, eps=eps)
    transition = _safe_weighted_mean(transition_num, transition_den, eps=eps)
    fallback = _safe_weighted_mean(all_num, all_den, eps=eps)
    if float(regular_den.item()) > 0.0 and float(transition_den.item()) > 0.0:
        alpha = float(config.transition_fraction)
        self_per_sample = (1.0 - alpha) * regular + alpha * transition
    else:
        self_per_sample = fallback

    carrier_num = x_stats["carrier_num"] + y_stats["carrier_num"]
    carrier_den = x_stats["carrier_den"] + y_stats["carrier_den"]
    carrier_per_sample = _safe_weighted_mean(carrier_num, carrier_den, eps=eps)
    weighted_self = float(config.self_trace_weight) * self_per_sample
    weighted_carrier = float(config.transition_carrier_weight) * carrier_per_sample
    loss_per_sample = weighted_self + weighted_carrier

    trace_count = int(context.x.transition_mask.numel()) + int(
        context.y.transition_mask.numel()
    )
    transition_count = int(context.x.transition_mask.sum().item()) + int(
        context.y.transition_mask.sum().item()
    )
    fraction = u_phi_valid.new_tensor(
        0.0 if trace_count == 0 else transition_count / trace_count
    )
    return ComplexGluingLossResult(
        loss=loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        self_loss=weighted_self.mean(),
        self_loss_per_sample=weighted_self,
        self_regular=regular.mean(),
        self_regular_per_sample=regular,
        self_transition=transition.mean(),
        self_transition_per_sample=transition,
        carrier_transition=weighted_carrier.mean(),
        carrier_transition_per_sample=weighted_carrier,
        x_self_rms=_rms(x_stats["self_residual"]),
        y_self_rms=_rms(y_stats["self_residual"]),
        x_carrier_rms=_rms(x_stats["carrier_residual"]),
        y_carrier_rms=_rms(y_stats["carrier_residual"]),
        x_self_residual=x_stats["self_residual"],
        y_self_residual=y_stats["self_residual"],
        x_carrier_residual=x_stats["carrier_residual"],
        y_carrier_residual=y_stats["carrier_residual"],
        transition_trace_fraction=fraction,
    )


def _build_axis_context(
    geometry: ComplexGeometryMetadata,
    config: ComplexAdmissibilityGluingConfig,
    *,
    axis: Literal["x", "y"],
) -> AxisTraceContext:
    lines = _line_data(geometry, axis)
    intervals = _segment_intervals(geometry, axis)
    opposite_segment_id = (
        (geometry.y_segment_id if axis == "x" else geometry.x_segment_id).detach().cpu()
    )
    coords = geometry.coords_valid.detach().cpu()

    lower_indices: list[list[int]] = []
    lower_weights: list[list[float]] = []
    upper_indices: list[list[int]] = []
    upper_weights: list[list[float]] = []
    face_indices: list[list[int]] = []
    integration_weights: list[float] = []
    trace_coords: list[list[float]] = []
    transition_mask: list[bool] = []
    carrier_self_index: list[int] = []
    carrier_indices: list[list[int]] = []
    carrier_weights: list[list[float]] = []
    carrier_coords: list[list[float]] = []
    boundary_indices: list[list[int]] = []
    boundary_weights: list[list[float]] = []
    boundary_face_indices: list[list[int]] = []
    boundary_integration_weights: list[float] = []
    boundary_coords: list[list[float]] = []
    interface_count = max(len(lines) - 1, 0)
    transition_interface_count = 0

    primary_spacing = float((geometry.hx if axis == "x" else geometry.hy).item())
    for line_index in range(len(lines) - 1):
        lower = lines[line_index]
        upper = lines[line_index + 1]
        is_transition = _is_transition_interface(
            lower,
            upper,
            intervals,
            threshold=float(config.log_length_jump_threshold),
        )
        if is_transition:
            transition_interface_count += 1
        interface_coordinate = 0.5 * (lower.coordinate + upper.coordinate)
        transverse_gap = upper.coordinate - lower.coordinate
        if transverse_gap <= 0.0:
            raise ValueError("Axial line coordinates must be strictly increasing.")

        if line_index > 0 and line_index + 2 < len(lines):
            previous = lines[line_index - 1]
            following = lines[line_index + 2]
            shared = sorted(
                set(previous.points)
                & set(lower.points)
                & set(upper.points)
                & set(following.points)
            )
            for primary_index in shared:
                lower_pair = [
                    previous.points[primary_index],
                    lower.points[primary_index],
                ]
                upper_pair = [
                    upper.points[primary_index],
                    following.points[primary_index],
                ]
                lower_indices.append(lower_pair)
                lower_weights.append(
                    _linear_weights(
                        previous.coordinate,
                        lower.coordinate,
                        interface_coordinate,
                    )
                )
                upper_indices.append(upper_pair)
                upper_weights.append(
                    _linear_weights(
                        upper.coordinate,
                        following.coordinate,
                        interface_coordinate,
                    )
                )
                central_pair = [
                    lower.points[primary_index],
                    upper.points[primary_index],
                ]
                face_indices.append(central_pair)
                integration_weights.append(primary_spacing / transverse_gap)
                if axis == "x":
                    trace_coords.append(
                        [float(coords[central_pair[0], 0].item()), interface_coordinate]
                    )
                else:
                    trace_coords.append(
                        [interface_coordinate, float(coords[central_pair[0], 1].item())]
                    )
                transition_mask.append(is_transition)
                if is_transition and (
                    int(opposite_segment_id[central_pair[0]].item())
                    == int(opposite_segment_id[central_pair[1]].item())
                ):
                    carrier_self_index.append(len(lower_indices) - 1)
                    carrier_indices.append(central_pair)
                    carrier_weights.append(
                        _linear_weights(
                            lower.coordinate,
                            upper.coordinate,
                            interface_coordinate,
                        )
                    )
                    carrier_coords.append(trace_coords[-1])

        if is_transition:
            _append_boundary_anchors(
                geometry=geometry,
                axis=axis,
                lines=lines,
                line_index=line_index,
                coords=coords,
                primary_spacing=primary_spacing,
                transverse_gap=transverse_gap,
                boundary_indices=boundary_indices,
                boundary_weights=boundary_weights,
                boundary_face_indices=boundary_face_indices,
                boundary_integration_weights=boundary_integration_weights,
                boundary_coords=boundary_coords,
            )

    dtype = geometry.coords_valid.dtype
    device = geometry.coords_valid.device
    return AxisTraceContext(
        lower_indices=_index_tensor(lower_indices, 2, device),
        lower_weights=_float_tensor(lower_weights, 2, dtype, device),
        upper_indices=_index_tensor(upper_indices, 2, device),
        upper_weights=_float_tensor(upper_weights, 2, dtype, device),
        face_indices=_index_tensor(face_indices, 2, device),
        integration_weight=torch.as_tensor(
            integration_weights, dtype=dtype, device=device
        ),
        trace_coords=_float_tensor(trace_coords, 2, dtype, device),
        transition_mask=torch.as_tensor(
            transition_mask, dtype=torch.bool, device=device
        ),
        carrier_self_index=torch.as_tensor(
            carrier_self_index, dtype=torch.long, device=device
        ),
        carrier_indices=_index_tensor(carrier_indices, 2, device),
        carrier_weights=_float_tensor(carrier_weights, 2, dtype, device),
        carrier_coords=_float_tensor(carrier_coords, 2, dtype, device),
        boundary_indices=_index_tensor(boundary_indices, 2, device),
        boundary_weights=_float_tensor(boundary_weights, 2, dtype, device),
        boundary_face_indices=_index_tensor(boundary_face_indices, 2, device),
        boundary_integration_weight=torch.as_tensor(
            boundary_integration_weights, dtype=dtype, device=device
        ),
        boundary_coords=_float_tensor(boundary_coords, 2, dtype, device),
        interface_count=interface_count,
        transition_interface_count=transition_interface_count,
    )


def _line_data(
    geometry: ComplexGeometryMetadata,
    axis: Literal["x", "y"],
) -> list[_LineData]:
    coords = geometry.coords_valid.detach().cpu()
    if axis == "x":
        transverse_grid = geometry.valid_grid_y_index.detach().cpu()
        primary_grid = geometry.valid_grid_x_index.detach().cpu()
        segment_id = geometry.x_segment_id.detach().cpu()
        coordinate_column = 1
    else:
        transverse_grid = geometry.valid_grid_x_index.detach().cpu()
        primary_grid = geometry.valid_grid_y_index.detach().cpu()
        segment_id = geometry.y_segment_id.detach().cpu()
        coordinate_column = 0

    groups: dict[int, list[int]] = {}
    for point_index, grid_index in enumerate(transverse_grid.tolist()):
        groups.setdefault(int(grid_index), []).append(point_index)
    result = []
    for grid_index in sorted(groups):
        point_indices = groups[grid_index]
        coordinate = float(coords[point_indices[0], coordinate_column].item())
        points = {
            int(primary_grid[point_index].item()): point_index
            for point_index in point_indices
        }
        segment_ids = tuple(
            sorted(
                {int(segment_id[point_index].item()) for point_index in point_indices}
            )
        )
        result.append(
            _LineData(
                coordinate=coordinate,
                points=points,
                segment_ids=segment_ids,
            )
        )
    return result


def _segment_intervals(
    geometry: ComplexGeometryMetadata,
    axis: Literal["x", "y"],
) -> dict[int, tuple[float, float, float]]:
    if axis == "x":
        left = geometry.x_segment_left
        right = geometry.x_segment_right
        length = geometry.x_segment_length
    else:
        left = geometry.y_segment_bottom
        right = geometry.y_segment_top
        length = geometry.y_segment_length
    return {
        index: (
            float(left[index].item()),
            float(right[index].item()),
            float(length[index].item()),
        )
        for index in range(int(length.numel()))
    }


def _is_transition_interface(
    lower: _LineData,
    upper: _LineData,
    intervals: dict[int, tuple[float, float, float]],
    *,
    threshold: float,
) -> bool:
    lower_degree = {segment_id: 0 for segment_id in lower.segment_ids}
    upper_degree = {segment_id: 0 for segment_id in upper.segment_ids}
    matched_pairs: list[tuple[int, int]] = []
    tolerance = 1.0e-12
    for lower_id in lower.segment_ids:
        lower_left, lower_right, _lower_length = intervals[lower_id]
        for upper_id in upper.segment_ids:
            upper_left, upper_right, _upper_length = intervals[upper_id]
            overlap = min(lower_right, upper_right) - max(lower_left, upper_left)
            if overlap > tolerance:
                lower_degree[lower_id] += 1
                upper_degree[upper_id] += 1
                matched_pairs.append((lower_id, upper_id))
    topology_change = (
        len(lower.segment_ids) != len(upper.segment_ids)
        or any(degree != 1 for degree in lower_degree.values())
        or any(degree != 1 for degree in upper_degree.values())
    )
    if topology_change:
        return True
    for lower_id, upper_id in matched_pairs:
        lower_length = intervals[lower_id][2]
        upper_length = intervals[upper_id][2]
        jump = abs(math.log(lower_length * lower_length) - math.log(upper_length**2))
        if jump > threshold:
            return True
    return False


def _append_boundary_anchors(
    *,
    geometry: ComplexGeometryMetadata,
    axis: Literal["x", "y"],
    lines: list[_LineData],
    line_index: int,
    coords: torch.Tensor,
    primary_spacing: float,
    transverse_gap: float,
    boundary_indices: list[list[int]],
    boundary_weights: list[list[float]],
    boundary_face_indices: list[list[int]],
    boundary_integration_weights: list[float],
    boundary_coords: list[list[float]],
) -> None:
    lower = lines[line_index]
    upper = lines[line_index + 1]
    only_lower = sorted(set(lower.points) - set(upper.points))
    only_upper = sorted(set(upper.points) - set(lower.points))
    for primary_index, side in (
        *((value, "lower") for value in only_lower),
        *((value, "upper") for value in only_upper),
    ):
        if side == "lower":
            if line_index == 0 or primary_index not in lines[line_index - 1].points:
                continue
            near_line = lower
            far_line = lines[line_index - 1]
        else:
            if (
                line_index + 2 >= len(lines)
                or primary_index not in lines[line_index + 2].points
            ):
                continue
            near_line = upper
            far_line = lines[line_index + 2]
        near_index = near_line.points[primary_index]
        far_index = far_line.points[primary_index]
        boundary_coordinate = _opposite_axis_boundary_between(
            geometry,
            axis=axis,
            point_index=near_index,
            lower=lower.coordinate,
            upper=upper.coordinate,
        )
        if boundary_coordinate is None:
            continue
        boundary_indices.append([far_index, near_index])
        boundary_weights.append(
            _linear_weights(
                far_line.coordinate,
                near_line.coordinate,
                boundary_coordinate,
            )
        )
        boundary_face_indices.append([near_index, near_index])
        boundary_integration_weights.append(primary_spacing / transverse_gap)
        primary_coordinate = (
            float(coords[near_index, 0].item())
            if axis == "x"
            else float(coords[near_index, 1].item())
        )
        if axis == "x":
            boundary_coords.append([primary_coordinate, boundary_coordinate])
        else:
            boundary_coords.append([boundary_coordinate, primary_coordinate])


def _opposite_axis_boundary_between(
    geometry: ComplexGeometryMetadata,
    *,
    axis: Literal["x", "y"],
    point_index: int,
    lower: float,
    upper: float,
) -> float | None:
    if axis == "x":
        segment_id = int(geometry.y_segment_id[point_index].item())
        candidates = (
            float(geometry.y_segment_bottom[segment_id].item()),
            float(geometry.y_segment_top[segment_id].item()),
        )
    else:
        segment_id = int(geometry.x_segment_id[point_index].item())
        candidates = (
            float(geometry.x_segment_left[segment_id].item()),
            float(geometry.x_segment_right[segment_id].item()),
        )
    tolerance = 1.0e-10 * max(1.0, abs(lower), abs(upper))
    inside = [
        value for value in candidates if lower - tolerance <= value <= upper + tolerance
    ]
    if not inside:
        return None
    midpoint = 0.5 * (lower + upper)
    return min(inside, key=lambda value: abs(value - midpoint))


def _axis_loss_terms(
    primary_solution: torch.Tensor,
    carrier_solution: torch.Tensor,
    a_valid: torch.Tensor,
    context: AxisTraceContext,
) -> dict[str, torch.Tensor]:
    batch_size = primary_solution.shape[0]
    lower_trace = _apply_stencil(
        primary_solution, context.lower_indices, context.lower_weights
    )
    upper_trace = _apply_stencil(
        primary_solution, context.upper_indices, context.upper_weights
    )
    self_residual = lower_trace - upper_trace
    face_a = _face_coefficient(a_valid, context.face_indices)
    weighted = self_residual.square() * face_a * context.integration_weight.unsqueeze(0)
    regular = ~context.transition_mask
    transition = context.transition_mask
    regular_num = weighted[:, regular].sum(dim=-1)
    transition_num = weighted[:, transition].sum(dim=-1)
    regular_den = context.integration_weight[regular].sum()
    transition_den = context.integration_weight[transition].sum()

    carrier_residual_parts: list[torch.Tensor] = []
    carrier_weight_parts: list[torch.Tensor] = []
    carrier_a_parts: list[torch.Tensor] = []
    if context.carrier_self_index.numel() > 0:
        carrier = _apply_stencil(
            carrier_solution,
            context.carrier_indices,
            context.carrier_weights,
        )
        selected_lower = lower_trace[:, context.carrier_self_index]
        selected_upper = upper_trace[:, context.carrier_self_index]
        carrier_residual_parts.extend(
            (selected_lower - carrier, selected_upper - carrier)
        )
        selected_weight = context.integration_weight[context.carrier_self_index]
        selected_a = face_a[:, context.carrier_self_index]
        carrier_weight_parts.extend((selected_weight, selected_weight))
        carrier_a_parts.extend((selected_a, selected_a))
    if context.boundary_indices.numel() > 0:
        boundary_trace = _apply_stencil(
            primary_solution,
            context.boundary_indices,
            context.boundary_weights,
        )
        carrier_residual_parts.append(boundary_trace)
        carrier_weight_parts.append(context.boundary_integration_weight)
        carrier_a_parts.append(
            _face_coefficient(a_valid, context.boundary_face_indices)
        )
    if carrier_residual_parts:
        carrier_residual = torch.cat(carrier_residual_parts, dim=-1)
        carrier_weight = torch.cat(carrier_weight_parts, dim=0)
        carrier_a = torch.cat(carrier_a_parts, dim=-1)
        carrier_num = (
            carrier_residual.square() * carrier_a * carrier_weight.unsqueeze(0)
        ).sum(dim=-1)
        carrier_den = carrier_weight.sum()
    else:
        carrier_residual = primary_solution.new_zeros((batch_size, 0))
        carrier_num = primary_solution.new_zeros((batch_size,))
        carrier_den = primary_solution.new_zeros(())
    return {
        "regular_num": regular_num,
        "regular_den": regular_den,
        "transition_num": transition_num,
        "transition_den": transition_den,
        "carrier_num": carrier_num,
        "carrier_den": carrier_den,
        "self_residual": self_residual,
        "carrier_residual": carrier_residual,
    }


def _apply_stencil(
    values: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return values.new_zeros((values.shape[0], 0))
    return (values[:, indices] * weights.unsqueeze(0)).sum(dim=-1)


def _face_coefficient(a_valid: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return a_valid.new_zeros((a_valid.shape[0], 0))
    return a_valid[:, indices].mean(dim=-1)


def _safe_weighted_mean(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    if float(denominator.item()) <= 0.0:
        return numerator.new_zeros(numerator.shape)
    return numerator / denominator.clamp_min(eps)


def _rms(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.square().mean().sqrt()


def _linear_weights(left: float, right: float, query: float) -> list[float]:
    if right == left:
        raise ValueError("Trace interpolation coordinates must be distinct.")
    fraction = (query - left) / (right - left)
    return [1.0 - fraction, fraction]


def _index_tensor(
    values: list[list[int]],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    if not values:
        return torch.empty((0, width), dtype=torch.long, device=device)
    return torch.tensor(values, dtype=torch.long, device=device)


def _float_tensor(
    values: list[list[float]],
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not values:
        return torch.empty((0, width), dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, device=device)
