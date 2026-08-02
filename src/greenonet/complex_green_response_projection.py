from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.complex_reconstruction import evaluate_segment_green_kernel
from greenonet.config import ColumnDiagonalGreenResponseProjectionConfig


@dataclass(frozen=True)
class ColumnDiagonalGreenResponseContext:
    """Sample-independent source-column response costs on valid points."""

    gamma_x_squared: torch.Tensor
    gamma_y_squared: torch.Tensor
    regularized_gamma_x_squared: torch.Tensor
    regularized_gamma_y_squared: torch.Tensor
    correction_weight_phi: torch.Tensor
    correction_weight_psi: torch.Tensor
    point_mass: torch.Tensor
    gain_squared_eps: float
    gain_exponent: float

    @classmethod
    def from_gain_squared(
        cls,
        *,
        gamma_x_squared: torch.Tensor,
        gamma_y_squared: torch.Tensor,
        point_mass: torch.Tensor | float,
        gain_squared_eps: float,
        gain_exponent: float = 1.0,
    ) -> ColumnDiagonalGreenResponseContext:
        config = ColumnDiagonalGreenResponseProjectionConfig(
            gain_squared_eps=gain_squared_eps,
            gain_exponent=gain_exponent,
        )
        if gamma_x_squared.dim() != 1 or gamma_y_squared.dim() != 1:
            raise ValueError("Column response gains must be one-dimensional.")
        if gamma_x_squared.shape != gamma_y_squared.shape:
            raise ValueError("x and y column response gains must have matching shapes.")
        if gamma_x_squared.device != gamma_y_squared.device:
            raise ValueError("x and y column response gains must share a device.")
        if gamma_x_squared.dtype != gamma_y_squared.dtype:
            raise ValueError("x and y column response gains must share a dtype.")
        if not torch.all(torch.isfinite(gamma_x_squared)) or not torch.all(
            torch.isfinite(gamma_y_squared)
        ):
            raise ValueError("Column response gains must be finite.")
        if torch.any(gamma_x_squared < 0.0) or torch.any(gamma_y_squared < 0.0):
            raise ValueError(
                "Column response gain squared values must be non-negative."
            )

        point_mass_tensor = torch.as_tensor(
            point_mass,
            dtype=gamma_x_squared.dtype,
            device=gamma_x_squared.device,
        )
        if point_mass_tensor.numel() != 1 or not torch.isfinite(point_mass_tensor):
            raise ValueError("point_mass must be a finite scalar.")
        if bool((point_mass_tensor <= 0.0).item()):
            raise ValueError("point_mass must be positive.")

        eps = float(config.gain_squared_eps)
        exponent = float(config.gain_exponent)
        regularized_x = gamma_x_squared + eps
        regularized_y = gamma_y_squared + eps
        if exponent == 0.0:
            weight_phi = torch.full_like(regularized_x, 0.5)
        elif exponent == 1.0:
            weight_phi = regularized_y / (regularized_x + regularized_y)
        else:
            log_gain_ratio = torch.log(regularized_y) - torch.log(regularized_x)
            weight_phi = torch.sigmoid(exponent * log_gain_ratio)
        weight_psi = 1.0 - weight_phi
        return cls(
            gamma_x_squared=gamma_x_squared.detach(),
            gamma_y_squared=gamma_y_squared.detach(),
            regularized_gamma_x_squared=regularized_x.detach(),
            regularized_gamma_y_squared=regularized_y.detach(),
            correction_weight_phi=weight_phi.detach(),
            correction_weight_psi=weight_psi.detach(),
            point_mass=point_mass_tensor.detach(),
            gain_squared_eps=eps,
            gain_exponent=exponent,
        )

    @property
    def num_points(self) -> int:
        return int(self.gamma_x_squared.numel())

    def validate_for(self, reference: torch.Tensor) -> None:
        if reference.shape[-1] != self.num_points:
            raise ValueError(
                "Column response gain point count does not match projection input."
            )
        if reference.device != self.gamma_x_squared.device:
            raise ValueError(
                "Column response gains and projection input must share a device."
            )
        if reference.dtype != self.gamma_x_squared.dtype:
            raise ValueError(
                "Column response gains and projection input must share a dtype."
            )

    def statistics(self) -> dict[str, float | int]:
        eps = self.gain_squared_eps
        return {
            "gain_exponent": self.gain_exponent,
            "gamma_x_squared_min": float(self.gamma_x_squared.min().item()),
            "gamma_x_squared_max": float(self.gamma_x_squared.max().item()),
            "gamma_y_squared_min": float(self.gamma_y_squared.min().item()),
            "gamma_y_squared_max": float(self.gamma_y_squared.max().item()),
            "weight_phi_min": float(self.correction_weight_phi.min().item()),
            "weight_phi_max": float(self.correction_weight_phi.max().item()),
            "weight_psi_min": float(self.correction_weight_psi.min().item()),
            "weight_psi_max": float(self.correction_weight_psi.max().item()),
            "x_floored_point_count": int(
                torch.count_nonzero(self.gamma_x_squared <= eps).item()
            ),
            "y_floored_point_count": int(
                torch.count_nonzero(self.gamma_y_squared <= eps).item()
            ),
        }


def column_diagonal_gain_squared(
    *,
    kernel: torch.Tensor,
    source_weights: torch.Tensor,
    segment_length: torch.Tensor | float,
    point_mass: torch.Tensor | float,
) -> torch.Tensor:
    """Return diag(H^T M H) by summing over output rows of H."""

    if kernel.dim() != 2:
        raise ValueError("kernel must be a two-dimensional output-by-source matrix.")
    if source_weights.dim() != 1 or source_weights.numel() != kernel.shape[1]:
        raise ValueError("source_weights must match the kernel source-column count.")
    length = torch.as_tensor(
        segment_length,
        dtype=kernel.dtype,
        device=kernel.device,
    )
    mass = torch.as_tensor(point_mass, dtype=kernel.dtype, device=kernel.device)
    if length.numel() != 1 or not torch.isfinite(length) or bool((length <= 0).item()):
        raise ValueError("segment_length must be a finite positive scalar.")
    if mass.numel() != 1 or not torch.isfinite(mass) or bool((mass <= 0).item()):
        raise ValueError("point_mass must be a finite positive scalar.")
    response_columns = kernel * (source_weights * length.square()).unsqueeze(0)
    return mass * response_columns.square().sum(dim=0)


class ColumnDiagonalGreenResponseContextBuilder:
    """Build column gains segment-by-segment without a global response matrix."""

    def __init__(
        self,
        config: ColumnDiagonalGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.config = ColumnDiagonalGreenResponseProjectionConfig.from_raw(config)

    @torch.no_grad()
    def build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> ColumnDiagonalGreenResponseContext:
        point_mass = (geometry.hx * geometry.hy).to(
            device=x_green_branch.device,
            dtype=x_green_branch.dtype,
        )
        gamma_x = self._axis_gain_squared(
            green_model=green_model,
            ptr=geometry.x_recon_ptr,
            node_t=geometry.x_recon_t,
            node_weight=geometry.x_recon_weight,
            node_valid_index=geometry.x_recon_valid_index,
            segment_length=geometry.x_segment_length,
            green_branch=x_green_branch,
            point_mass=point_mass,
            point_count=geometry.num_points,
            axis="x",
        )
        gamma_y = self._axis_gain_squared(
            green_model=green_model,
            ptr=geometry.y_recon_ptr,
            node_t=geometry.y_recon_t,
            node_weight=geometry.y_recon_weight,
            node_valid_index=geometry.y_recon_valid_index,
            segment_length=geometry.y_segment_length,
            green_branch=y_green_branch,
            point_mass=point_mass,
            point_count=geometry.num_points,
            axis="y",
        )
        return ColumnDiagonalGreenResponseContext.from_gain_squared(
            gamma_x_squared=gamma_x,
            gamma_y_squared=gamma_y,
            point_mass=point_mass,
            gain_squared_eps=self.config.gain_squared_eps,
            gain_exponent=self.config.gain_exponent,
        )

    @staticmethod
    def _axis_gain_squared(
        *,
        green_model: torch.nn.Module,
        ptr: torch.Tensor,
        node_t: torch.Tensor,
        node_weight: torch.Tensor,
        node_valid_index: torch.Tensor,
        segment_length: torch.Tensor,
        green_branch: torch.Tensor,
        point_mass: torch.Tensor,
        point_count: int,
        axis: str,
    ) -> torch.Tensor:
        if green_branch.dim() != 4 or green_branch.shape[2] != 4:
            raise ValueError(f"{axis}_green_branch must have shape (B, S, 4, M).")
        segment_count = int(ptr.numel()) - 1
        if segment_count != int(segment_length.numel()):
            raise ValueError(f"{axis} reconstruction and segment counts do not match.")
        if green_branch.shape[1] != segment_count:
            raise ValueError(
                f"{axis} Green branch segment count does not match geometry."
            )

        device = green_branch.device
        dtype = green_branch.dtype
        ptr = ptr.to(device=device)
        node_t = node_t.to(device=device, dtype=dtype)
        node_weight = node_weight.to(device=device, dtype=dtype)
        node_valid_index = node_valid_index.to(device=device)
        segment_length = segment_length.to(device=device, dtype=dtype)
        gain = torch.zeros(point_count, dtype=dtype, device=device)
        assignment_count = torch.zeros(point_count, dtype=torch.int64, device=device)

        for segment_index in range(segment_count):
            start = int(ptr[segment_index].item())
            end = int(ptr[segment_index + 1].item())
            local_valid_index = node_valid_index[start:end]
            interior_nodes = torch.nonzero(
                local_valid_index >= 0,
                as_tuple=False,
            ).flatten()
            if interior_nodes.numel() == 0:
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
            valid_kernel = kernel.index_select(0, interior_nodes).index_select(
                1,
                interior_nodes,
            )
            valid_weights = node_weight[start:end].index_select(0, interior_nodes)
            local_gain = column_diagonal_gain_squared(
                kernel=valid_kernel,
                source_weights=valid_weights,
                segment_length=segment_length[segment_index],
                point_mass=point_mass,
            )
            global_index = local_valid_index.index_select(0, interior_nodes)
            if torch.any(global_index >= point_count):
                raise ValueError(f"{axis} reconstruction valid index is out of range.")
            gain.index_copy_(0, global_index, local_gain)
            assignment_count.index_add_(
                0,
                global_index,
                torch.ones_like(global_index, dtype=torch.int64),
            )

        if not torch.all(assignment_count == 1):
            raise ValueError(
                f"Each valid point must belong to exactly one {axis} reconstruction "
                "segment."
            )
        if not torch.all(torch.isfinite(gain)) or torch.any(gain < 0.0):
            raise ValueError(
                f"{axis} column response gains must be finite and non-negative."
            )
        return gain.detach()


class ColumnDiagonalGreenResponseContextCache:
    """Lazily build and reuse one frozen response-gain context per runtime."""

    def __init__(
        self,
        config: ColumnDiagonalGreenResponseProjectionConfig | dict[str, object],
    ) -> None:
        self.builder = ColumnDiagonalGreenResponseContextBuilder(config)
        self.context: ColumnDiagonalGreenResponseContext | None = None
        self.build_count = 0
        self.build_seconds = 0.0

    def get_or_build(
        self,
        *,
        green_model: torch.nn.Module,
        geometry: ComplexGeometryMetadata,
        x_green_branch: torch.Tensor,
        y_green_branch: torch.Tensor,
    ) -> ColumnDiagonalGreenResponseContext:
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
