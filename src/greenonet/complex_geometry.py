from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from typing_extensions import Self

import numpy as np
import torch


REQUIRED_GEOMETRY_KEYS: tuple[str, ...] = (
    "coords_valid",
    "valid_grid_y_index",
    "valid_grid_x_index",
    "x_segment_id",
    "y_segment_id",
    "x_local_t",
    "y_local_t",
    "x_segment_left",
    "x_segment_right",
    "x_segment_y",
    "x_segment_length",
    "y_segment_bottom",
    "y_segment_top",
    "y_segment_x",
    "y_segment_length",
    "x_recon_ptr",
    "x_recon_t",
    "x_recon_weight",
    "x_recon_valid_index",
    "y_recon_ptr",
    "y_recon_t",
    "y_recon_weight",
    "y_recon_valid_index",
    "x_edges",
    "y_edges",
    "hx",
    "hy",
)


@dataclass(frozen=True)
class ComplexGeometryMetadata:
    """Validated batch-shared geometry tensors for complex-domain CouplingNet."""

    coords_valid: torch.Tensor
    valid_grid_y_index: torch.Tensor
    valid_grid_x_index: torch.Tensor
    x_segment_id: torch.Tensor
    y_segment_id: torch.Tensor
    x_local_t: torch.Tensor
    y_local_t: torch.Tensor
    x_segment_left: torch.Tensor
    x_segment_right: torch.Tensor
    x_segment_y: torch.Tensor
    x_segment_length: torch.Tensor
    y_segment_bottom: torch.Tensor
    y_segment_top: torch.Tensor
    y_segment_x: torch.Tensor
    y_segment_length: torch.Tensor
    x_recon_ptr: torch.Tensor
    x_recon_t: torch.Tensor
    x_recon_weight: torch.Tensor
    x_recon_valid_index: torch.Tensor
    y_recon_ptr: torch.Tensor
    y_recon_t: torch.Tensor
    y_recon_weight: torch.Tensor
    y_recon_valid_index: torch.Tensor
    x_edges: torch.Tensor
    y_edges: torch.Tensor
    hx: torch.Tensor
    hy: torch.Tensor

    @property
    def num_points(self) -> int:
        return int(self.coords_valid.shape[0])

    @property
    def num_x_segments(self) -> int:
        return int(self.x_segment_length.shape[0])

    @property
    def num_y_segments(self) -> int:
        return int(self.y_segment_length.shape[0])

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
                if isinstance(value, torch.Tensor)
            }
        )

    def x_lengths_for_valid_points(self) -> torch.Tensor:
        return self.x_segment_length[self.x_segment_id]

    def y_lengths_for_valid_points(self) -> torch.Tensor:
        return self.y_segment_length[self.y_segment_id]


def load_complex_geometry(
    path: Path | str,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> ComplexGeometryMetadata:
    """Load and validate precomputed complex-geometry metadata from an NPZ file."""

    geometry_path = Path(path)
    if not geometry_path.is_file():
        raise FileNotFoundError(f"Complex geometry file does not exist: {path}")

    with np.load(geometry_path) as raw:
        missing = sorted(set(REQUIRED_GEOMETRY_KEYS) - set(raw.files))
        if missing:
            raise KeyError(
                f"Complex geometry NPZ is missing required keys: {', '.join(missing)}."
            )
        metadata = ComplexGeometryMetadata(
            coords_valid=_float_tensor(raw["coords_valid"], dtype, device),
            valid_grid_y_index=_long_tensor(raw["valid_grid_y_index"], device),
            valid_grid_x_index=_long_tensor(raw["valid_grid_x_index"], device),
            x_segment_id=_long_tensor(raw["x_segment_id"], device),
            y_segment_id=_long_tensor(raw["y_segment_id"], device),
            x_local_t=_float_tensor(raw["x_local_t"], dtype, device),
            y_local_t=_float_tensor(raw["y_local_t"], dtype, device),
            x_segment_left=_float_tensor(raw["x_segment_left"], dtype, device),
            x_segment_right=_float_tensor(raw["x_segment_right"], dtype, device),
            x_segment_y=_float_tensor(raw["x_segment_y"], dtype, device),
            x_segment_length=_float_tensor(raw["x_segment_length"], dtype, device),
            y_segment_bottom=_float_tensor(raw["y_segment_bottom"], dtype, device),
            y_segment_top=_float_tensor(raw["y_segment_top"], dtype, device),
            y_segment_x=_float_tensor(raw["y_segment_x"], dtype, device),
            y_segment_length=_float_tensor(raw["y_segment_length"], dtype, device),
            x_recon_ptr=_long_tensor(raw["x_recon_ptr"], device),
            x_recon_t=_float_tensor(raw["x_recon_t"], dtype, device),
            x_recon_weight=_float_tensor(raw["x_recon_weight"], dtype, device),
            x_recon_valid_index=_long_tensor(raw["x_recon_valid_index"], device),
            y_recon_ptr=_long_tensor(raw["y_recon_ptr"], device),
            y_recon_t=_float_tensor(raw["y_recon_t"], dtype, device),
            y_recon_weight=_float_tensor(raw["y_recon_weight"], dtype, device),
            y_recon_valid_index=_long_tensor(raw["y_recon_valid_index"], device),
            x_edges=_long_tensor(raw["x_edges"], device),
            y_edges=_long_tensor(raw["y_edges"], device),
            hx=_scalar_tensor(raw["hx"], dtype, device),
            hy=_scalar_tensor(raw["hy"], dtype, device),
        )

    _validate_complex_geometry(metadata)
    return metadata


def _float_tensor(
    value: np.ndarray,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _long_tensor(
    value: np.ndarray,
    device: torch.device | str | None,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.long, device=device)


def _scalar_tensor(
    value: np.ndarray,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.numel() != 1:
        raise ValueError("hx and hy must be scalar values.")
    return tensor.reshape(())


def _validate_complex_geometry(metadata: ComplexGeometryMetadata) -> None:
    point_count = _validate_point_arrays(metadata)
    _validate_segment_arrays(
        prefix="x",
        left=metadata.x_segment_left,
        right=metadata.x_segment_right,
        fixed=metadata.x_segment_y,
        length=metadata.x_segment_length,
    )
    _validate_segment_arrays(
        prefix="y",
        left=metadata.y_segment_bottom,
        right=metadata.y_segment_top,
        fixed=metadata.y_segment_x,
        length=metadata.y_segment_length,
    )
    _validate_segment_ids(
        metadata.x_segment_id,
        segment_count=metadata.num_x_segments,
        point_count=point_count,
        field_name="x_segment_id",
    )
    _validate_segment_ids(
        metadata.y_segment_id,
        segment_count=metadata.num_y_segments,
        point_count=point_count,
        field_name="y_segment_id",
    )
    _validate_strict_local_coordinates(metadata.x_local_t, "x_local_t", point_count)
    _validate_strict_local_coordinates(metadata.y_local_t, "y_local_t", point_count)
    _validate_reconstruction_arrays(
        prefix="x",
        ptr=metadata.x_recon_ptr,
        t=metadata.x_recon_t,
        weight=metadata.x_recon_weight,
        valid_index=metadata.x_recon_valid_index,
        segment_count=metadata.num_x_segments,
        point_count=point_count,
    )
    _validate_reconstruction_arrays(
        prefix="y",
        ptr=metadata.y_recon_ptr,
        t=metadata.y_recon_t,
        weight=metadata.y_recon_weight,
        valid_index=metadata.y_recon_valid_index,
        segment_count=metadata.num_y_segments,
        point_count=point_count,
    )
    _validate_edges(
        metadata.x_edges,
        point_count=point_count,
        segment_id=metadata.x_segment_id,
        field_name="x_edges",
    )
    _validate_edges(
        metadata.y_edges,
        point_count=point_count,
        segment_id=metadata.y_segment_id,
        field_name="y_edges",
    )
    if metadata.hx.item() <= 0.0 or metadata.hy.item() <= 0.0:
        raise ValueError("hx and hy must be positive.")


def _validate_point_arrays(metadata: ComplexGeometryMetadata) -> int:
    if metadata.coords_valid.dim() != 2 or metadata.coords_valid.shape[1] != 2:
        raise ValueError("coords_valid must have shape (P, 2).")
    point_count = int(metadata.coords_valid.shape[0])
    if point_count == 0:
        raise ValueError("coords_valid must contain at least one valid point.")
    for field_name, values in (
        ("valid_grid_y_index", metadata.valid_grid_y_index),
        ("valid_grid_x_index", metadata.valid_grid_x_index),
        ("x_segment_id", metadata.x_segment_id),
        ("y_segment_id", metadata.y_segment_id),
        ("x_local_t", metadata.x_local_t),
        ("y_local_t", metadata.y_local_t),
    ):
        if values.shape != (point_count,):
            raise ValueError(f"{field_name} must have shape ({point_count},).")
    if torch.any(metadata.valid_grid_y_index < 0) or torch.any(
        metadata.valid_grid_x_index < 0
    ):
        raise ValueError("valid grid indices must be non-negative.")
    return point_count


def _validate_segment_arrays(
    *,
    prefix: str,
    left: torch.Tensor,
    right: torch.Tensor,
    fixed: torch.Tensor,
    length: torch.Tensor,
) -> None:
    segment_count = int(length.shape[0])
    if length.dim() != 1 or segment_count == 0:
        raise ValueError(f"{prefix}_segment_length must have shape (S,) with S > 0.")
    for field_name, values in (
        (f"{prefix}_segment_left", left),
        (f"{prefix}_segment_right", right),
        (f"{prefix}_segment_fixed", fixed),
    ):
        if values.shape != (segment_count,):
            raise ValueError(f"{field_name} must have shape ({segment_count},).")
    if torch.any(length <= 0):
        raise ValueError(f"{prefix}_segment_length values must be positive.")
    if torch.any((right - left).abs() <= 0):
        raise ValueError(f"{prefix} segment endpoints must be distinct.")


def _validate_segment_ids(
    values: torch.Tensor,
    *,
    segment_count: int,
    point_count: int,
    field_name: str,
) -> None:
    if values.shape != (point_count,):
        raise ValueError(f"{field_name} must have shape ({point_count},).")
    if torch.any(values < 0) or torch.any(values >= segment_count):
        raise ValueError(f"{field_name} contains out-of-range segment ids.")


def _validate_strict_local_coordinates(
    values: torch.Tensor,
    field_name: str,
    point_count: int,
) -> None:
    if values.shape != (point_count,):
        raise ValueError(f"{field_name} must have shape ({point_count},).")
    if torch.any(values <= 0.0) or torch.any(values >= 1.0):
        raise ValueError(
            f"{field_name} must be strictly inside (0, 1); endpoints are represented "
            "only by reconstruction nodes with valid_index == -1."
        )


def _validate_reconstruction_arrays(
    *,
    prefix: str,
    ptr: torch.Tensor,
    t: torch.Tensor,
    weight: torch.Tensor,
    valid_index: torch.Tensor,
    segment_count: int,
    point_count: int,
) -> None:
    if ptr.shape != (segment_count + 1,):
        raise ValueError(f"{prefix}_recon_ptr must have shape ({segment_count + 1},).")
    if t.shape != weight.shape or t.shape != valid_index.shape:
        raise ValueError(
            f"{prefix}_recon_t, {prefix}_recon_weight, and "
            f"{prefix}_recon_valid_index must have the same shape."
        )
    if int(ptr[0].item()) != 0 or int(ptr[-1].item()) != int(t.numel()):
        raise ValueError(f"{prefix}_recon_ptr endpoints do not match node count.")
    if torch.any(ptr[1:] < ptr[:-1]):
        raise ValueError(f"{prefix}_recon_ptr must be monotone nondecreasing.")
    if torch.any((t < 0.0) | (t > 1.0)):
        raise ValueError(f"{prefix}_recon_t values must be in [0, 1].")
    if torch.any(weight < 0.0):
        raise ValueError(f"{prefix}_recon_weight values must be non-negative.")
    if torch.any((valid_index < -1) | (valid_index >= point_count)):
        raise ValueError(
            f"{prefix}_recon_valid_index values must be -1 or valid point indices."
        )
    for segment_index, start, end in _ptr_windows(ptr):
        if end - start < 2:
            raise ValueError(
                f"{prefix} reconstruction segment {segment_index} needs endpoints."
            )
        segment_valid = valid_index[start:end]
        segment_t = t[start:end]
        if int(segment_valid[0].item()) != -1 or int(segment_valid[-1].item()) != -1:
            raise ValueError(
                f"{prefix} reconstruction segment {segment_index} endpoints must "
                "have valid_index == -1."
            )
        if segment_t[0].item() != 0.0 or segment_t[-1].item() != 1.0:
            raise ValueError(
                f"{prefix} reconstruction segment {segment_index} must start at "
                "t=0 and end at t=1."
            )


def _ptr_windows(ptr: torch.Tensor) -> Iterable[tuple[int, int, int]]:
    ptr_cpu = ptr.detach().cpu()
    for index in range(int(ptr_cpu.numel()) - 1):
        yield index, int(ptr_cpu[index].item()), int(ptr_cpu[index + 1].item())


def _validate_edges(
    edges: torch.Tensor,
    *,
    point_count: int,
    segment_id: torch.Tensor,
    field_name: str,
) -> None:
    if edges.numel() == 0:
        if edges.shape != (0, 2):
            raise ValueError(f"{field_name} must have shape (E, 2).")
        return
    if edges.dim() != 2 or edges.shape[1] != 2:
        raise ValueError(f"{field_name} must have shape (E, 2).")
    if torch.any(edges < 0) or torch.any(edges >= point_count):
        raise ValueError(f"{field_name} contains out-of-range valid point indices.")
    left = edges[:, 0]
    right = edges[:, 1]
    if torch.any(left == right):
        raise ValueError(f"{field_name} must not contain self-edges.")
    if torch.any(segment_id[left] != segment_id[right]):
        raise ValueError(f"{field_name} must connect points on the same segment.")
