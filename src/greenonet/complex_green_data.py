from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from greenonet.axial import AxialLines
from greenonet.backward_sampler import BackwardSampler
from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.green_interval import (
    build_segment_branch_samples,
    physical_interval_coordinates,
    unit_branch_grid,
)
from greenonet.numerics import IntegrationRule
from greenonet.sampler import ForwardSampler


SamplerMode = Literal["forward", "backward"]
AxisName = Literal["x", "y"]
LineFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ComplexGreenData:
    """Flat interval data for complex-geometry GreenNet training."""

    unit_grid: Tensor
    physical_coords: Tensor
    axis_id: Tensor
    segment_id: Tensor
    left: Tensor
    right: Tensor
    fixed: Tensor
    length: Tensor
    solution: Tensor
    source: Tensor
    a_vals: Tensor
    ap_vals: Tensor
    b_vals: Tensor
    c_vals: Tensor

    @property
    def num_samples(self) -> int:
        return int(self.solution.shape[0])

    @property
    def num_intervals(self) -> int:
        return int(self.solution.shape[1])

    @property
    def branch_input_dim(self) -> int:
        return int(self.unit_grid.numel())


@dataclass(frozen=True)
class ComplexGreenItem:
    """One generated source/solution sample with shared interval metadata."""

    unit_grid: Tensor
    physical_coords: Tensor
    axis_id: Tensor
    segment_id: Tensor
    left: Tensor
    right: Tensor
    fixed: Tensor
    length: Tensor
    solution: Tensor
    source: Tensor
    a_vals: Tensor
    ap_vals: Tensor
    b_vals: Tensor
    c_vals: Tensor


@dataclass(frozen=True)
class ComplexGreenBatch:
    """Batched complex GreenNet tensors."""

    unit_grid: Tensor
    physical_coords: Tensor
    axis_id: Tensor
    segment_id: Tensor
    left: Tensor
    right: Tensor
    fixed: Tensor
    length: Tensor
    solution: Tensor
    source: Tensor
    a_vals: Tensor
    ap_vals: Tensor
    b_vals: Tensor
    c_vals: Tensor

    def to(self, device: torch.device | str) -> ComplexGreenBatch:
        return ComplexGreenBatch(
            unit_grid=self.unit_grid.to(device),
            physical_coords=self.physical_coords.to(device),
            axis_id=self.axis_id.to(device),
            segment_id=self.segment_id.to(device),
            left=self.left.to(device),
            right=self.right.to(device),
            fixed=self.fixed.to(device),
            length=self.length.to(device),
            solution=self.solution.to(device),
            source=self.source.to(device),
            a_vals=self.a_vals.to(device),
            ap_vals=self.ap_vals.to(device),
            b_vals=self.b_vals.to(device),
            c_vals=self.c_vals.to(device),
        )


class ComplexGreenDataset(Dataset[ComplexGreenItem]):
    """Dataset wrapper for flat connected-interval GreenNet samples."""

    def __init__(self, data: ComplexGreenData) -> None:
        super().__init__()
        if data.solution.shape != data.source.shape:
            raise ValueError("solution and source must have matching shapes.")
        if data.solution.dim() != 3:
            raise ValueError("solution/source must have shape (samples, intervals, M).")
        expected_interval_shape = (data.num_intervals, data.branch_input_dim)
        for field_name, values in (
            ("a_vals", data.a_vals),
            ("ap_vals", data.ap_vals),
            ("b_vals", data.b_vals),
            ("c_vals", data.c_vals),
        ):
            if values.shape != expected_interval_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_interval_shape}."
                )
        self.data = data

    def __len__(self) -> int:
        return self.data.num_samples

    def __getitem__(self, index: int) -> ComplexGreenItem:
        return ComplexGreenItem(
            unit_grid=self.data.unit_grid,
            physical_coords=self.data.physical_coords,
            axis_id=self.data.axis_id,
            segment_id=self.data.segment_id,
            left=self.data.left,
            right=self.data.right,
            fixed=self.data.fixed,
            length=self.data.length,
            solution=self.data.solution[index],
            source=self.data.source[index],
            a_vals=self.data.a_vals,
            ap_vals=self.data.ap_vals,
            b_vals=self.data.b_vals,
            c_vals=self.data.c_vals,
        )


def complex_green_collate_fn(batch: Sequence[ComplexGreenItem]) -> ComplexGreenBatch:
    if not batch:
        raise ValueError("Cannot collate an empty complex GreenNet batch.")
    first = batch[0]
    return ComplexGreenBatch(
        unit_grid=first.unit_grid,
        physical_coords=first.physical_coords,
        axis_id=first.axis_id,
        segment_id=first.segment_id,
        left=first.left,
        right=first.right,
        fixed=first.fixed,
        length=first.length,
        solution=torch.stack([item.solution for item in batch], dim=0),
        source=torch.stack([item.source for item in batch], dim=0),
        a_vals=first.a_vals,
        ap_vals=first.ap_vals,
        b_vals=first.b_vals,
        c_vals=first.c_vals,
    )


class ComplexGreenDataBuilder:
    """Generate unit-interval GreenNet samples for all complex geometry segments."""

    def __init__(
        self,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
        *,
        branch_input_dim: int,
        samples_per_interval: int,
        sampler_mode: SamplerMode,
        scale_length: float | tuple[float, float],
        deterministic: bool,
        integration_rule: IntegrationRule,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        if samples_per_interval <= 0:
            raise ValueError("samples_per_interval must be positive.")
        self.geometry = geometry
        self.coeffs = coeffs
        self.branch_input_dim = branch_input_dim
        self.samples_per_interval = samples_per_interval
        self.sampler_mode = sampler_mode
        self.scale_length = scale_length
        self.deterministic = deterministic
        self.integration_rule = integration_rule
        self.dtype = dtype
        self.device = torch.device(device)
        self.unit_grid = unit_branch_grid(
            branch_input_dim,
            dtype=dtype,
            device=self.device,
        )

    def build(self) -> ComplexGreenData:
        interval_meta = self._build_interval_metadata()
        a_vals, ap_vals, b_vals, c_vals = self._build_branch_coefficients()
        solution, source = self._sample_interval_data(
            left=interval_meta.left,
            fixed=interval_meta.fixed,
            length=interval_meta.length,
            axis_id=interval_meta.axis_id,
        )
        return ComplexGreenData(
            unit_grid=self.unit_grid.detach().cpu(),
            physical_coords=interval_meta.physical_coords.detach().cpu(),
            axis_id=interval_meta.axis_id.detach().cpu(),
            segment_id=interval_meta.segment_id.detach().cpu(),
            left=interval_meta.left.detach().cpu(),
            right=interval_meta.right.detach().cpu(),
            fixed=interval_meta.fixed.detach().cpu(),
            length=interval_meta.length.detach().cpu(),
            solution=solution.detach().cpu(),
            source=source.detach().cpu(),
            a_vals=a_vals.detach().cpu(),
            ap_vals=ap_vals.detach().cpu(),
            b_vals=b_vals.detach().cpu(),
            c_vals=c_vals.detach().cpu(),
        )

    def _build_branch_coefficients(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_coeffs = build_segment_branch_samples(
            self.geometry,
            self.coeffs,
            axis="x",
            branch_input_dim=self.branch_input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        y_coeffs = build_segment_branch_samples(
            self.geometry,
            self.coeffs,
            axis="y",
            branch_input_dim=self.branch_input_dim,
            dtype=self.dtype,
            device=self.device,
        )
        return (
            torch.cat((x_coeffs.a_unit, y_coeffs.a_unit), dim=0),
            torch.cat((x_coeffs.ap_unit, y_coeffs.ap_unit), dim=0),
            torch.cat((x_coeffs.b_unit, y_coeffs.b_unit), dim=0),
            torch.cat((x_coeffs.c_unit, y_coeffs.c_unit), dim=0),
        )

    @dataclass(frozen=True)
    class _IntervalMetadata:
        physical_coords: Tensor
        axis_id: Tensor
        segment_id: Tensor
        left: Tensor
        right: Tensor
        fixed: Tensor
        length: Tensor

    def _build_interval_metadata(self) -> _IntervalMetadata:
        x_left = self.geometry.x_segment_left.to(self.device, dtype=self.dtype)
        x_right = self.geometry.x_segment_right.to(self.device, dtype=self.dtype)
        x_fixed = self.geometry.x_segment_y.to(self.device, dtype=self.dtype)
        x_length = self.geometry.x_segment_length.to(self.device, dtype=self.dtype)
        x_s = physical_interval_coordinates(x_left, x_right, self.unit_grid)
        x_coords = torch.stack((x_s, x_fixed.unsqueeze(-1).expand_as(x_s)), dim=-1)

        y_left = self.geometry.y_segment_bottom.to(self.device, dtype=self.dtype)
        y_right = self.geometry.y_segment_top.to(self.device, dtype=self.dtype)
        y_fixed = self.geometry.y_segment_x.to(self.device, dtype=self.dtype)
        y_length = self.geometry.y_segment_length.to(self.device, dtype=self.dtype)
        y_s = physical_interval_coordinates(y_left, y_right, self.unit_grid)
        y_coords = torch.stack((y_fixed.unsqueeze(-1).expand_as(y_s), y_s), dim=-1)

        num_x = self.geometry.num_x_segments
        num_y = self.geometry.num_y_segments
        return self._IntervalMetadata(
            physical_coords=torch.cat((x_coords, y_coords), dim=0),
            axis_id=torch.cat(
                (
                    torch.zeros(num_x, dtype=torch.long, device=self.device),
                    torch.ones(num_y, dtype=torch.long, device=self.device),
                )
            ),
            segment_id=torch.cat(
                (
                    torch.arange(num_x, dtype=torch.long, device=self.device),
                    torch.arange(num_y, dtype=torch.long, device=self.device),
                )
            ),
            left=torch.cat((x_left, y_left), dim=0),
            right=torch.cat((x_right, y_right), dim=0),
            fixed=torch.cat((x_fixed, y_fixed), dim=0),
            length=torch.cat((x_length, y_length), dim=0),
        )

    def _sample_interval_data(
        self,
        *,
        left: Tensor,
        fixed: Tensor,
        length: Tensor,
        axis_id: Tensor,
    ) -> tuple[Tensor, Tensor]:
        sampler = self._make_sampler()
        num_intervals = int(axis_id.numel())
        solution = torch.empty(
            (self.samples_per_interval, num_intervals, self.branch_input_dim),
            dtype=self.dtype,
            device=self.device,
        )
        source = torch.empty_like(solution)
        for interval_idx in range(num_intervals):
            functions = self._unit_coefficient_functions(
                axis="x" if int(axis_id[interval_idx].item()) == 0 else "y",
                left=left[interval_idx],
                fixed=fixed[interval_idx],
                length=length[interval_idx],
            )
            for sample_idx in range(self.samples_per_interval):
                u, f, _a, _ap, _b, _c = sampler.generate_sample(
                    self.unit_grid,
                    functions[0],
                    functions[1],
                    functions[2],
                    functions[3],
                )
                solution[sample_idx, interval_idx] = u
                source[sample_idx, interval_idx] = f
        return solution, source

    def _make_sampler(self) -> ForwardSampler | BackwardSampler:
        if self.sampler_mode == "forward":
            sampler_cls: type[ForwardSampler] | type[BackwardSampler] = ForwardSampler
        elif self.sampler_mode == "backward":
            sampler_cls = BackwardSampler
        else:
            raise ValueError(f"Unsupported sampler_mode: {self.sampler_mode}")
        return sampler_cls(
            axial_lines=AxialLines(),
            data_size_per_each_line=self.samples_per_interval,
            scale_length=self.scale_length,
            device=self.device,
            dtype=self.dtype,
            deterministic=self.deterministic,
            integration_rule=self.integration_rule,
        )

    def _unit_coefficient_functions(
        self,
        *,
        axis: AxisName,
        left: Tensor,
        fixed: Tensor,
        length: Tensor,
    ) -> tuple[LineFunction, LineFunction, LineFunction, LineFunction]:
        def physical_coords(t: Tensor) -> tuple[Tensor, Tensor]:
            s = left + length * t
            if axis == "x":
                return s, fixed.expand_as(s)
            return fixed.expand_as(s), s

        def a_line(t: Tensor) -> Tensor:
            x, y = physical_coords(t)
            return self.coeffs.a_fun(x, y).to(device=t.device, dtype=t.dtype)

        def ap_line(t: Tensor) -> Tensor:
            x, y = physical_coords(t)
            ap_fun = self.coeffs.apx_fun if axis == "x" else self.coeffs.apy_fun
            return length * ap_fun(x, y).to(device=t.device, dtype=t.dtype)

        def b_line(t: Tensor) -> Tensor:
            x, y = physical_coords(t)
            b_fun = self.coeffs.bx_fun if axis == "x" else self.coeffs.by_fun
            return length * b_fun(x, y).to(device=t.device, dtype=t.dtype)

        def c_line(t: Tensor) -> Tensor:
            x, y = physical_coords(t)
            return length.pow(2) * self.coeffs.c_fun(x, y).to(
                device=t.device,
                dtype=t.dtype,
            )

        return a_line, ap_line, b_line, c_line


def generate_complex_green_data(
    geometry: ComplexGeometryMetadata,
    coeffs: CoefficientFunctions,
    *,
    branch_input_dim: int,
    samples_per_interval: int,
    sampler_mode: SamplerMode,
    scale_length: float | tuple[float, float],
    deterministic: bool,
    integration_rule: IntegrationRule,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> ComplexGreenData:
    return ComplexGreenDataBuilder(
        geometry,
        coeffs,
        branch_input_dim=branch_input_dim,
        samples_per_interval=samples_per_interval,
        sampler_mode=sampler_mode,
        scale_length=scale_length,
        deterministic=deterministic,
        integration_rule=integration_rule,
        dtype=dtype,
        device=device,
    ).build()
