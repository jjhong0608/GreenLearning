from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch

from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import ComplexGeometryMetadata


@dataclass(frozen=True)
class IntervalBranchCoefficients:
    """Unit-interval coefficient samples for one segment family."""

    a_unit: torch.Tensor
    ap_unit: torch.Tensor
    b_unit: torch.Tensor
    c_unit: torch.Tensor

    def as_coupling_branch(self) -> torch.Tensor:
        return torch.stack((self.a_unit, self.b_unit, self.c_unit), dim=1)


def unit_branch_grid(
    branch_input_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if branch_input_dim < 2:
        raise ValueError("branch_input_dim must be at least 2.")
    return torch.linspace(0.0, 1.0, branch_input_dim, dtype=dtype, device=device)


def physical_interval_coordinates(
    left: torch.Tensor,
    right: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return left.unsqueeze(-1) + (right - left).unsqueeze(-1) * t.unsqueeze(0)


def transform_interval_coefficients(
    *,
    a_phys: torch.Tensor,
    ap_phys: torch.Tensor,
    b_phys: torch.Tensor,
    c_phys: torch.Tensor,
    length: torch.Tensor,
) -> IntervalBranchCoefficients:
    length = length.unsqueeze(-1)
    return IntervalBranchCoefficients(
        a_unit=a_phys,
        ap_unit=length * ap_phys,
        b_unit=length * b_phys,
        c_unit=length.pow(2) * c_phys,
    )


def transform_interval_source(
    f_phys: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    return length.unsqueeze(-1).pow(2) * f_phys


def build_segment_branch_samples(
    geometry: ComplexGeometryMetadata,
    coeffs: CoefficientFunctions,
    *,
    axis: Literal["x", "y"],
    branch_input_dim: int,
    dtype: torch.dtype,
    device: torch.device | str | None = None,
) -> IntervalBranchCoefficients:
    t = unit_branch_grid(branch_input_dim, dtype=dtype, device=device)
    if axis == "x":
        left = geometry.x_segment_left.to(device=device, dtype=dtype)
        right = geometry.x_segment_right.to(device=device, dtype=dtype)
        fixed = geometry.x_segment_y.to(device=device, dtype=dtype)
        length = geometry.x_segment_length.to(device=device, dtype=dtype)
        x = physical_interval_coordinates(left, right, t)
        y = fixed.unsqueeze(-1).expand_as(x)
        ap_fun = coeffs.apx_fun
        b_fun = coeffs.bx_fun
    else:
        left = geometry.y_segment_bottom.to(device=device, dtype=dtype)
        right = geometry.y_segment_top.to(device=device, dtype=dtype)
        fixed = geometry.y_segment_x.to(device=device, dtype=dtype)
        length = geometry.y_segment_length.to(device=device, dtype=dtype)
        y = physical_interval_coordinates(left, right, t)
        x = fixed.unsqueeze(-1).expand_as(y)
        ap_fun = coeffs.apy_fun
        b_fun = coeffs.by_fun

    a_phys = coeffs.a_fun(x, y).to(dtype=dtype, device=device)
    ap_phys = ap_fun(x, y).to(dtype=dtype, device=device)
    b_phys = b_fun(x, y).to(dtype=dtype, device=device)
    c_phys = coeffs.c_fun(x, y).to(dtype=dtype, device=device)
    return transform_interval_coefficients(
        a_phys=a_phys,
        ap_phys=ap_phys,
        b_phys=b_phys,
        c_phys=c_phys,
        length=length,
    )


def evaluate_green_pairs(
    green_model: torch.nn.Module,
    *,
    a_unit: torch.Tensor,
    ap_unit: torch.Tensor,
    b_unit: torch.Tensor,
    c_unit: torch.Tensor,
    t_eval: torch.Tensor,
    eta_eval: torch.Tensor,
) -> torch.Tensor:
    """Evaluate GreenNet on unit interval coordinates without length rescaling."""

    trunk_grid = torch.stack(
        torch.meshgrid(t_eval, eta_eval, indexing="ij"),
        dim=-1,
    )
    pair_forward = getattr(green_model, "forward_pairs", None)
    if callable(pair_forward):
        output = pair_forward(
            trunk_grid,
            a_unit,
            ap_unit,
            b_unit,
            c_unit,
        )
        if output.shape[0] == 1:
            return cast(torch.Tensor, output[0])
        return cast(torch.Tensor, output)

    if a_unit.dim() != 1:
        raise ValueError("Fallback GreenNet evaluation accepts one line at a time.")
    if t_eval.numel() != a_unit.numel() or eta_eval.numel() != a_unit.numel():
        raise ValueError(
            "Fallback GreenNet evaluation requires eval grids to match branch samples."
        )
    return cast(
        torch.Tensor,
        green_model(
            trunk_grid=trunk_grid,
            a_vals=a_unit.view(1, 1, -1),
            ap_vals=ap_unit.view(1, 1, -1),
            b_vals=b_unit.view(1, 1, -1),
            c_vals=c_unit.view(1, 1, -1),
        )[0, 0],
    )
