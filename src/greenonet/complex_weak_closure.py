from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import ComplexGeometryMetadata


@dataclass(frozen=True)
class ComplexWeakAxisContext:
    """Precomputed P1 element data for one directional weak operator."""

    element_valid_index: torch.Tensor
    element_length: torch.Tensor
    local_operator: torch.Tensor
    local_mass: torch.Tensor
    nodal_mass: torch.Tensor

    def to(self, device: torch.device | str) -> ComplexWeakAxisContext:
        return type(self)(
            element_valid_index=self.element_valid_index.to(device),
            element_length=self.element_length.to(device),
            local_operator=self.local_operator.to(device),
            local_mass=self.local_mass.to(device),
            nodal_mass=self.nodal_mass.to(device),
        )


@dataclass(frozen=True)
class ComplexDirectionalWeakContext:
    """Batch-shared x/y directional weak operator discretization."""

    x: ComplexWeakAxisContext
    y: ComplexWeakAxisContext
    num_points: int
    point_area: torch.Tensor

    def to(self, device: torch.device | str) -> ComplexDirectionalWeakContext:
        return type(self)(
            x=self.x.to(device),
            y=self.y.to(device),
            num_points=self.num_points,
            point_area=self.point_area.to(device),
        )


@dataclass(frozen=True)
class ComplexWeakClosureResult:
    """Source-normalized directional weak residual loss and audit tensors."""

    loss: torch.Tensor
    x_loss: torch.Tensor
    y_loss: torch.Tensor
    loss_per_sample: torch.Tensor
    x_loss_per_sample: torch.Tensor
    y_loss_per_sample: torch.Tensor
    x_residual: torch.Tensor
    y_residual: torch.Tensor
    rhs_l2_squared_per_sample: torch.Tensor


@dataclass(frozen=True)
class ComplexDirectionalWeakResiduals:
    """Assembled x/y weak residuals for one candidate solution field."""

    x: torch.Tensor
    y: torch.Tensor

    @property
    def full(self) -> torch.Tensor:
        """Return the full-operator weak residual on the shared valid nodes."""

        return self.x + self.y


def build_directional_weak_context(
    geometry: ComplexGeometryMetadata,
    coeffs: CoefficientFunctions,
) -> ComplexDirectionalWeakContext:
    """Build sample-independent P1 operators on every connected axial segment."""

    return ComplexDirectionalWeakContext(
        x=_build_axis_context(geometry, coeffs, axis="x"),
        y=_build_axis_context(geometry, coeffs, axis="y"),
        num_points=geometry.num_points,
        point_area=geometry.hx * geometry.hy,
    )


def directional_weak_operator_closure_loss(
    *,
    u_pred_valid: torch.Tensor,
    projected_physical: torch.Tensor,
    rhs_valid: torch.Tensor,
    context: ComplexDirectionalWeakContext,
    eps: float,
) -> ComplexWeakClosureResult:
    """Evaluate Bx(u_pred)-phi and By(u_pred)-psi in nodal test spaces."""

    if u_pred_valid.dim() != 2:
        raise ValueError("u_pred_valid must have shape (B, P).")
    if u_pred_valid.shape[-1] != context.num_points:
        raise ValueError("u_pred_valid point count does not match weak context.")
    if projected_physical.shape != (
        u_pred_valid.shape[0],
        2,
        context.num_points,
    ):
        raise ValueError("projected_physical must have shape (B, 2, P).")
    if rhs_valid.shape != u_pred_valid.shape:
        raise ValueError("rhs_valid must match u_pred_valid shape.")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool):
        raise TypeError("eps must be numeric.")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    context = context.to(u_pred_valid.device)
    residuals = assemble_directional_weak_residuals(
        u_valid=u_pred_valid,
        projected_physical=projected_physical,
        context=context,
    )
    x_residual = residuals.x
    y_residual = residuals.y
    x_dual_per_sample = (x_residual.square() / (context.x.nodal_mass + float(eps))).sum(
        dim=-1
    )
    y_dual_per_sample = (y_residual.square() / (context.y.nodal_mass + float(eps))).sum(
        dim=-1
    )
    rhs_l2_squared_per_sample = rhs_valid.square().sum(dim=-1) * context.point_area.to(
        dtype=rhs_valid.dtype
    )
    denominator = rhs_l2_squared_per_sample + float(eps)
    x_loss_per_sample = x_dual_per_sample / denominator
    y_loss_per_sample = y_dual_per_sample / denominator
    loss_per_sample = 0.5 * (x_loss_per_sample + y_loss_per_sample)
    return ComplexWeakClosureResult(
        loss=loss_per_sample.mean(),
        x_loss=x_loss_per_sample.mean(),
        y_loss=y_loss_per_sample.mean(),
        loss_per_sample=loss_per_sample,
        x_loss_per_sample=x_loss_per_sample,
        y_loss_per_sample=y_loss_per_sample,
        x_residual=x_residual,
        y_residual=y_residual,
        rhs_l2_squared_per_sample=rhs_l2_squared_per_sample,
    )


def assemble_directional_weak_residuals(
    *,
    u_valid: torch.Tensor,
    projected_physical: torch.Tensor,
    context: ComplexDirectionalWeakContext,
) -> ComplexDirectionalWeakResiduals:
    """Apply both axial weak operators without solving a global system."""

    if u_valid.dim() != 2:
        raise ValueError("u_valid must have shape (B, P).")
    if u_valid.shape[-1] != context.num_points:
        raise ValueError("u_valid point count does not match weak context.")
    if projected_physical.shape != (
        u_valid.shape[0],
        2,
        context.num_points,
    ):
        raise ValueError("projected_physical must have shape (B, 2, P).")
    if not torch.all(torch.isfinite(u_valid)):
        raise ValueError("u_valid must contain only finite values.")
    if not torch.all(torch.isfinite(projected_physical)):
        raise ValueError("projected_physical must contain only finite values.")

    context = context.to(u_valid.device)
    x_residual = _assemble_axis_residual(
        u_valid=u_valid,
        source_valid=projected_physical[:, 0],
        context=context.x,
    )
    y_residual = _assemble_axis_residual(
        u_valid=u_valid,
        source_valid=projected_physical[:, 1],
        context=context.y,
    )
    return ComplexDirectionalWeakResiduals(x=x_residual, y=y_residual)


def _build_axis_context(
    geometry: ComplexGeometryMetadata,
    coeffs: CoefficientFunctions,
    *,
    axis: Literal["x", "y"],
) -> ComplexWeakAxisContext:
    if axis == "x":
        ptr = geometry.x_recon_ptr
        t_nodes = geometry.x_recon_t
        valid_index = geometry.x_recon_valid_index
        segment_left = geometry.x_segment_left
        segment_length = geometry.x_segment_length
        segment_fixed = geometry.x_segment_y
        transverse_measure = geometry.hy
        segment_count = geometry.num_x_segments
    else:
        ptr = geometry.y_recon_ptr
        t_nodes = geometry.y_recon_t
        valid_index = geometry.y_recon_valid_index
        segment_left = geometry.y_segment_bottom
        segment_length = geometry.y_segment_length
        segment_fixed = geometry.y_segment_x
        transverse_measure = geometry.hx
        segment_count = geometry.num_y_segments

    element_indices: list[torch.Tensor] = []
    element_lengths: list[torch.Tensor] = []
    midpoint_x: list[torch.Tensor] = []
    midpoint_y: list[torch.Tensor] = []
    for segment_index in range(segment_count):
        start = int(ptr[segment_index].item())
        end = int(ptr[segment_index + 1].item())
        if end - start < 2:
            raise ValueError("Weak operator segment requires at least two nodes.")
        segment_t = t_nodes[start:end]
        delta_t = segment_t[1:] - segment_t[:-1]
        if torch.any(delta_t <= 0.0):
            raise ValueError("Weak operator segment nodes must be increasing.")
        physical_length = segment_length[segment_index] * delta_t
        if torch.any(physical_length <= 0.0):
            raise ValueError("Weak operator element lengths must be positive.")
        midpoint_primary = segment_left[segment_index] + segment_length[
            segment_index
        ] * (0.5 * (segment_t[1:] + segment_t[:-1]))
        fixed = segment_fixed[segment_index].expand_as(midpoint_primary)
        if axis == "x":
            midpoint_x.append(midpoint_primary)
            midpoint_y.append(fixed)
        else:
            midpoint_x.append(fixed)
            midpoint_y.append(midpoint_primary)
        element_indices.append(
            torch.stack(
                (valid_index[start : end - 1], valid_index[start + 1 : end]),
                dim=-1,
            )
        )
        element_lengths.append(physical_length)

    index = torch.cat(element_indices, dim=0)
    length = torch.cat(element_lengths, dim=0)
    x = torch.cat(midpoint_x, dim=0)
    y = torch.cat(midpoint_y, dim=0)
    a = _evaluate_coefficient(coeffs.a_fun, x, y, "a")
    b_function = coeffs.bx_fun if axis == "x" else coeffs.by_fun
    b = _evaluate_coefficient(b_function, x, y, f"b{axis}")
    c = _evaluate_coefficient(coeffs.c_fun, x, y, "c")
    transverse = transverse_measure.to(dtype=length.dtype, device=length.device)

    diffusion = (a * transverse / length)[:, None, None] * length.new_tensor(
        [[1.0, -1.0], [-1.0, 1.0]]
    )
    convection = (0.5 * b * transverse)[:, None, None] * length.new_tensor(
        [[-1.0, 1.0], [-1.0, 1.0]]
    )
    base_mass = length.new_tensor([[2.0, 1.0], [1.0, 2.0]])
    local_mass = (transverse * length / 6.0)[:, None, None] * base_mass
    reaction = (0.5 * c)[:, None, None] * local_mass
    local_operator = diffusion + convection + reaction
    nodal_mass = _assemble_nodal_mass(
        element_valid_index=index,
        local_mass=local_mass,
        num_points=geometry.num_points,
    )
    if torch.any(nodal_mass <= 0.0):
        raise ValueError(
            f"Every valid point must have positive {axis}-direction nodal mass."
        )
    return ComplexWeakAxisContext(
        element_valid_index=index,
        element_length=length,
        local_operator=local_operator,
        local_mass=local_mass,
        nodal_mass=nodal_mass,
    )


def _evaluate_coefficient(
    function: object,
    x: torch.Tensor,
    y: torch.Tensor,
    field_name: str,
) -> torch.Tensor:
    if not callable(function):
        raise TypeError(f"{field_name} coefficient must be callable.")
    value = function(x, y)
    tensor = torch.as_tensor(value, dtype=x.dtype, device=x.device)
    if tensor.ndim == 0:
        tensor = tensor.expand_as(x)
    if tensor.shape != x.shape:
        raise ValueError(
            f"{field_name} coefficient returned shape {tuple(tensor.shape)}; "
            f"expected {tuple(x.shape)}."
        )
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{field_name} coefficient contains non-finite values.")
    return tensor


def _assemble_nodal_mass(
    *,
    element_valid_index: torch.Tensor,
    local_mass: torch.Tensor,
    num_points: int,
) -> torch.Tensor:
    nodal_mass = local_mass.new_zeros((num_points,))
    row_mass = local_mass.sum(dim=-1)
    for local_row in range(2):
        index = element_valid_index[:, local_row]
        mask = index >= 0
        nodal_mass = nodal_mass.scatter_add(
            0,
            index[mask],
            row_mass[mask, local_row],
        )
    return nodal_mass


def _assemble_axis_residual(
    *,
    u_valid: torch.Tensor,
    source_valid: torch.Tensor,
    context: ComplexWeakAxisContext,
) -> torch.Tensor:
    element_u = _gather_element_values(u_valid, context.element_valid_index)
    element_source = _gather_element_values(
        source_valid,
        context.element_valid_index,
    )
    local_residual = torch.einsum(
        "eij,bej->bei",
        context.local_operator,
        element_u,
    ) - torch.einsum(
        "eij,bej->bei",
        context.local_mass,
        element_source,
    )
    residual = u_valid.new_zeros(u_valid.shape)
    for local_row in range(2):
        index = context.element_valid_index[:, local_row]
        mask = index >= 0
        batch_index = index[mask].unsqueeze(0).expand(u_valid.shape[0], -1)
        residual = residual.scatter_add(
            1,
            batch_index,
            local_residual[:, mask, local_row],
        )
    return residual


def _gather_element_values(
    values: torch.Tensor,
    element_valid_index: torch.Tensor,
) -> torch.Tensor:
    padded = torch.cat((values, values.new_zeros((values.shape[0], 1))), dim=-1)
    endpoint_index = values.shape[-1]
    safe_index = torch.where(
        element_valid_index >= 0,
        element_valid_index,
        element_valid_index.new_full(element_valid_index.shape, endpoint_index),
    )
    return padded[:, safe_index]
