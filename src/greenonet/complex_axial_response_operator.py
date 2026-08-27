from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_reconstruction import evaluate_segment_green_kernel


@dataclass(frozen=True)
class AxialResponseBlock:
    """One connected-segment physical-source response matrix."""

    valid_indices: torch.Tensor
    matrix: torch.Tensor

    def __post_init__(self) -> None:
        if self.valid_indices.dim() != 1:
            raise ValueError("valid_indices must be one-dimensional.")
        if self.valid_indices.dtype != torch.long:
            raise ValueError("valid_indices must use torch.long dtype.")
        expected = int(self.valid_indices.numel())
        if self.matrix.shape != (expected, expected):
            raise ValueError(
                "Axial response block matrix must be square and match its indices."
            )
        if not torch.all(torch.isfinite(self.matrix)):
            raise ValueError("Axial response block contains non-finite values.")


@dataclass(frozen=True)
class TangentColumnGramTerms:
    """Column-wise self and cross terms for the tangent response operator."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


@dataclass(frozen=True)
class FrozenAxialResponseOperator:
    """Blockwise frozen Green response and transpose without a global matrix."""

    axis: Literal["x", "y"]
    point_count: int
    blocks: tuple[AxialResponseBlock, ...]

    def __post_init__(self) -> None:
        if self.axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'.")
        if self.point_count < 1:
            raise ValueError("point_count must be positive.")
        assignment_count = torch.zeros(self.point_count, dtype=torch.int64)
        for block in self.blocks:
            indices = block.valid_indices.detach().cpu()
            if indices.numel() and (
                int(indices.min().item()) < 0
                or int(indices.max().item()) >= self.point_count
            ):
                raise ValueError("Axial response block contains invalid point indices.")
            assignment_count.index_add_(
                0,
                indices,
                torch.ones_like(indices, dtype=torch.int64),
            )
        if torch.any(assignment_count != 1):
            raise ValueError(
                f"Every valid point must belong to exactly one {self.axis}-axis block."
            )

    @property
    def dtype(self) -> torch.dtype:
        return self.blocks[0].matrix.dtype

    @property
    def device(self) -> torch.device:
        return self.blocks[0].matrix.device

    def forward(self, source_physical: torch.Tensor) -> torch.Tensor:
        """Apply H_axis to batched physical directional sources."""

        self._validate_input(source_physical, "source_physical")
        output = torch.zeros_like(source_physical)
        for block in self.blocks:
            indices = block.valid_indices
            output[:, indices] = source_physical[:, indices] @ block.matrix.T
        return output

    def adjoint(self, response_dual: torch.Tensor) -> torch.Tensor:
        """Apply H_axis^T to batched valid-point response duals."""

        self._validate_input(response_dual, "response_dual")
        output = torch.zeros_like(response_dual)
        for block in self.blocks:
            indices = block.valid_indices
            output[:, indices] = response_dual[:, indices] @ block.matrix
        return output

    def column_gain_squared(
        self,
        *,
        point_mass: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return diag(H_axis^T M H_axis) from cached segment blocks."""

        mass = torch.as_tensor(
            point_mass,
            dtype=self.dtype,
            device=self.device,
        )
        if mass.numel() != 1 or not torch.isfinite(mass):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((mass <= 0.0).item()):
            raise ValueError("point_mass must be positive.")
        gain = torch.zeros(self.point_count, dtype=self.dtype, device=self.device)
        for block in self.blocks:
            local_gain = mass * block.matrix.square().sum(dim=0)
            gain.index_copy_(0, block.valid_indices, local_gain)
        if not torch.all(torch.isfinite(gain)) or torch.any(gain < 0.0):
            raise ValueError("Column response gains must be finite and non-negative.")
        return gain.detach()

    def diagonal_response(self) -> torch.Tensor:
        """Scatter each segment block diagonal to global source order."""

        diagonal = torch.zeros(
            self.point_count,
            dtype=self.dtype,
            device=self.device,
        )
        for block in self.blocks:
            diagonal.index_copy_(0, block.valid_indices, torch.diagonal(block.matrix))
        return diagonal.detach()

    def _validate_input(self, values: torch.Tensor, name: str) -> None:
        if values.dim() != 2 or values.shape[1] != self.point_count:
            raise ValueError(f"{name} must have shape (B, {self.point_count}).")
        if values.dtype != self.dtype or values.device != self.device:
            raise ValueError(
                f"{name} must share dtype and device with the response operator."
            )
        if not torch.all(torch.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")

    def statistics(self) -> dict[str, bool | int | str]:
        matrix_entries = sum(block.matrix.numel() for block in self.blocks)
        return {
            "axis": self.axis,
            "point_count": self.point_count,
            "segment_block_count": len(self.blocks),
            "local_matrix_entry_count": matrix_entries,
            "global_matrix_materialized": False,
        }


@dataclass(frozen=True)
class FrozenBidirectionalResponseOperator:
    """Frozen x/y physical-source response pair used by projection and diagnostics."""

    x: FrozenAxialResponseOperator
    y: FrozenAxialResponseOperator

    def __post_init__(self) -> None:
        if self.x.point_count != self.y.point_count:
            raise ValueError("x and y response operators must share a point count.")
        if self.x.dtype != self.y.dtype or self.x.device != self.y.device:
            raise ValueError("x and y response operators must share dtype and device.")

    @property
    def point_count(self) -> int:
        return self.x.point_count

    def forward_pair(self, physical_sources: torch.Tensor) -> torch.Tensor:
        if physical_sources.dim() != 3 or physical_sources.shape[1:] != (
            2,
            self.point_count,
        ):
            raise ValueError(
                f"physical_sources must have shape (B, 2, {self.point_count})."
            )
        return torch.stack(
            (
                self.x.forward(physical_sources[:, 0]),
                self.y.forward(physical_sources[:, 1]),
            ),
            dim=1,
        )

    def tangent_gradient(
        self,
        mismatch: torch.Tensor,
        *,
        point_mass: torch.Tensor | float,
    ) -> torch.Tensor:
        """Return (H_x+H_y)^T M mismatch for a tangent source update."""

        mass = torch.as_tensor(
            point_mass,
            dtype=mismatch.dtype,
            device=mismatch.device,
        )
        if mass.numel() != 1 or not torch.isfinite(mass):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((mass <= 0.0).item()):
            raise ValueError("point_mass must be positive.")
        weighted = mass * mismatch
        return self.x.adjoint(weighted) + self.y.adjoint(weighted)

    def column_gain_squared(
        self,
        *,
        point_mass: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.x.column_gain_squared(point_mass=point_mass),
            self.y.column_gain_squared(point_mass=point_mass),
        )

    def tangent_column_gram_terms(
        self,
        *,
        point_mass: torch.Tensor | float,
    ) -> TangentColumnGramTerms:
        """Return a, b, c without assembling a global response matrix."""

        mass = torch.as_tensor(
            point_mass,
            dtype=self.x.dtype,
            device=self.x.device,
        )
        if mass.numel() != 1 or not torch.isfinite(mass):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((mass <= 0.0).item()):
            raise ValueError("point_mass must be positive.")
        a, b = self.column_gain_squared(point_mass=mass)

        x_block_id, x_local = self._block_membership(self.x)
        y_block_id, y_local = self._block_membership(self.y)
        groups: dict[tuple[int, int], list[int]] = {}
        for point_index in range(self.point_count):
            key = (
                int(x_block_id[point_index].item()),
                int(y_block_id[point_index].item()),
            )
            groups.setdefault(key, []).append(point_index)

        c = torch.zeros(self.point_count, dtype=self.x.dtype, device=self.x.device)
        for (x_id, y_id), global_points in groups.items():
            points = torch.tensor(
                global_points,
                dtype=torch.long,
                device=self.x.device,
            )
            x_positions = x_local.index_select(0, points)
            y_positions = y_local.index_select(0, points)
            x_block = self.x.blocks[x_id]
            y_block = self.y.blocks[y_id]
            x_overlap = x_block.matrix.index_select(0, x_positions).index_select(
                1, x_positions
            )
            y_overlap = y_block.matrix.index_select(0, y_positions).index_select(
                1, y_positions
            )
            local_cross = mass * (x_overlap * y_overlap).sum(dim=0)
            c.index_copy_(0, points, local_cross)
        if not torch.all(torch.isfinite(c)):
            raise ValueError("Cross-axis tangent column Gram terms must be finite.")
        return TangentColumnGramTerms(a=a, b=b, c=c.detach())

    @staticmethod
    def _block_membership(
        operator: FrozenAxialResponseOperator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_id = torch.empty(
            operator.point_count,
            dtype=torch.long,
            device=operator.device,
        )
        local_position = torch.empty_like(block_id)
        for index, block in enumerate(operator.blocks):
            count = int(block.valid_indices.numel())
            block_id.index_fill_(0, block.valid_indices, index)
            local_position.index_copy_(
                0,
                block.valid_indices,
                torch.arange(count, dtype=torch.long, device=operator.device),
            )
        return block_id, local_position


class FrozenAxialResponseOperatorBuilder:
    """Build segment-local response blocks from the production Green kernel."""

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> FrozenBidirectionalResponseOperator:
        return FrozenBidirectionalResponseOperator(
            x=cls._build_axis(
                green_model=green_model,
                geometry=geometry,
                green_branch=x_green_branch,
                axis="x",
            ),
            y=cls._build_axis(
                green_model=green_model,
                geometry=geometry,
                green_branch=y_green_branch,
                axis="y",
            ),
        )

    @staticmethod
    def _build_axis(
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        green_branch: torch.Tensor,
        axis: Literal["x", "y"],
    ) -> FrozenAxialResponseOperator:
        if green_branch.dim() != 4 or green_branch.shape[2] != 4:
            raise ValueError("green_branch must have shape (B, S, 4, M).")
        ptr = geometry.x_recon_ptr if axis == "x" else geometry.y_recon_ptr
        node_t = geometry.x_recon_t if axis == "x" else geometry.y_recon_t
        node_weight = (
            geometry.x_recon_weight if axis == "x" else geometry.y_recon_weight
        )
        node_valid_index = (
            geometry.x_recon_valid_index
            if axis == "x"
            else geometry.y_recon_valid_index
        )
        segment_length = (
            geometry.x_segment_length if axis == "x" else geometry.y_segment_length
        )
        device = green_branch.device
        dtype = green_branch.dtype
        ptr = ptr.to(device=device)
        node_t = node_t.to(device=device, dtype=dtype)
        node_weight = node_weight.to(device=device, dtype=dtype)
        node_valid_index = node_valid_index.to(device=device)
        segment_length = segment_length.to(device=device, dtype=dtype)
        segment_count = int(ptr.numel()) - 1
        if green_branch.shape[1] != segment_count:
            raise ValueError("Green branch and reconstruction segment counts differ.")
        blocks: list[AxialResponseBlock] = []
        for segment_index in range(segment_count):
            start = int(ptr[segment_index].item())
            end = int(ptr[segment_index + 1].item())
            local_indices = node_valid_index[start:end]
            interior = torch.nonzero(local_indices >= 0, as_tuple=False).flatten()
            if interior.numel() == 0:
                continue
            local_t = node_t[start:end]
            kernel = evaluate_segment_green_kernel(
                green_model=green_model,
                green_branch=green_branch,
                segment_index=segment_index,
                node_t=local_t,
                dtype=dtype,
                device=device,
            )
            interior_kernel = kernel.index_select(0, interior).index_select(
                1,
                interior,
            )
            physical_source_scale = (
                node_weight[start:end].index_select(0, interior)
                * segment_length[segment_index].square()
            )
            matrix = interior_kernel * physical_source_scale.unsqueeze(0)
            blocks.append(
                AxialResponseBlock(
                    valid_indices=local_indices.index_select(0, interior).to(
                        dtype=torch.long
                    ),
                    matrix=matrix,
                )
            )
        return FrozenAxialResponseOperator(
            axis=axis,
            point_count=geometry.num_points,
            blocks=tuple(blocks),
        )
