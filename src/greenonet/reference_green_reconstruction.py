"""Diagnostic diffusion reference; never used for source optimization."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.greens import ExactGreenFunction


@dataclass
class DiffusionReferenceBuilder:
    geometry: ComplexGeometryMetadata
    coefficients: CoefficientFunctions
    device: torch.device

    @staticmethod
    def refine(t: Tensor, factor: int) -> Tensor:
        if factor < 1 or t.ndim != 1 or not torch.all(t[1:] > t[:-1]):
            raise ValueError("Require increasing nodes and positive refinement factor")
        fractions = torch.arange(factor, dtype=t.dtype, device=t.device) / factor
        inner = t[:-1, None] + (t[1:] - t[:-1])[:, None] * fractions
        return torch.cat((inner.flatten(), t[-1:]))

    @staticmethod
    def kernel(t: Tensor, a: Tensor, factor: int) -> Tensor:
        if a.shape != t.shape or not torch.isfinite(a).all() or torch.any(a <= 0):
            raise ValueError("Diffusion must be finite, positive and match nodes")
        if factor < 1 or (t.numel() - 1) % factor:
            raise ValueError("Invalid refinement factor")
        if not torch.isfinite(t).all() or not torch.all(t[1:] > t[:-1]):
            raise ValueError("Nodes must be finite and strictly increasing")
        if factor == 1:
            return ExactGreenFunction(t, a)()
        # Same two cumulative trapezoids as ExactGreenFunction, avoiding a dense
        # fine-grid kernel: only the original target/source nodes are retained.
        e = 1 / a
        p = torch.cat((t.new_zeros(1), torch.cumulative_trapezoid(e, t)))
        q = torch.cat(
            (-torch.cumulative_trapezoid(e.flip(0), t.flip(0)).flip(0), t.new_zeros(1))
        )
        p, q = p[::factor], q[::factor]
        nodes = t[::factor]
        return torch.where(
            nodes[:, None] < nodes[None, :],
            p[:, None] * q[None, :] / q[0],
            p[None, :] * q[:, None] / q[0],
        )

    @staticmethod
    def response_matrix(
        kernel: Tensor, weights: Tensor, interior: Tensor, length: float | Tensor
    ) -> Tensor:
        return kernel[interior][:, interior] * (weights[interior] * length**2)[None]

    @torch.no_grad()
    def build(self, factor: int) -> FrozenBidirectionalResponseOperator:
        return FrozenBidirectionalResponseOperator(
            x=self._axis("x", factor), y=self._axis("y", factor)
        )

    def _axis(
        self, axis: Literal["x", "y"], factor: int
    ) -> FrozenAxialResponseOperator:
        geo = self.geometry
        ptr = getattr(geo, f"{axis}_recon_ptr")
        nodes = getattr(geo, f"{axis}_recon_t").to(self.device, torch.float64)
        weights = getattr(geo, f"{axis}_recon_weight").to(self.device, torch.float64)
        valid = getattr(geo, f"{axis}_recon_valid_index").to(self.device)
        lengths = getattr(geo, f"{axis}_segment_length").to(self.device, torch.float64)
        starts = (geo.x_segment_left if axis == "x" else geo.y_segment_bottom).to(
            self.device, torch.float64
        )
        fixed = (geo.x_segment_y if axis == "x" else geo.y_segment_x).to(
            self.device, torch.float64
        )
        blocks = []
        for j in range(len(ptr) - 1):
            start, end = int(ptr[j]), int(ptr[j + 1])
            local = valid[start:end]
            interior = torch.nonzero(local >= 0).flatten()
            if not interior.numel():
                continue
            t = self.refine(nodes[start:end], factor)
            varying = starts[j] + lengths[j] * t
            constant = fixed[j].expand_as(t)
            x, y = (varying, constant) if axis == "x" else (constant, varying)
            coeff = self.coefficients
            for value in (coeff.bx_fun(x, y), coeff.by_fun(x, y), coeff.c_fun(x, y)):
                if torch.any(value.abs() > 1e-12):
                    raise ValueError("This diagnostic requires reaction-free diffusion")
            kernel = self.kernel(t, coeff.a_fun(x, y), factor)
            blocks.append(
                AxialResponseBlock(
                    local[interior],
                    self.response_matrix(
                        kernel, weights[start:end], interior, lengths[j]
                    ),
                )
            )
        return FrozenAxialResponseOperator(axis, geo.num_points, tuple(blocks))
