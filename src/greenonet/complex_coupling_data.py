from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import CouplingCoefficientTermsConfig
from greenonet.green_interval import (
    build_segment_branch_samples,
    physical_interval_coordinates,
    unit_branch_grid,
)
from greenonet.numerics import IntegrationRule, integrate


@dataclass(frozen=True)
class ComplexCouplingItem:
    geometry: ComplexGeometryMetadata
    rhs_valid: torch.Tensor
    sol_valid: torch.Tensor
    flux_valid: torch.Tensor
    has_flux: torch.Tensor
    a_valid: torch.Tensor
    x_source_branch: torch.Tensor
    y_source_branch: torch.Tensor
    x_source_amplitude: torch.Tensor
    y_source_amplitude: torch.Tensor
    x_coefficient_branch: torch.Tensor
    y_coefficient_branch: torch.Tensor
    x_green_branch: torch.Tensor
    y_green_branch: torch.Tensor
    sample_index: torch.Tensor
    file_stem: str


@dataclass(frozen=True)
class ComplexCouplingBatch:
    geometry: ComplexGeometryMetadata
    rhs_valid: torch.Tensor
    sol_valid: torch.Tensor
    flux_valid: torch.Tensor
    has_flux: torch.Tensor
    a_valid: torch.Tensor
    x_source_branch: torch.Tensor
    y_source_branch: torch.Tensor
    x_source_amplitude: torch.Tensor
    y_source_amplitude: torch.Tensor
    x_coefficient_branch: torch.Tensor
    y_coefficient_branch: torch.Tensor
    x_green_branch: torch.Tensor
    y_green_branch: torch.Tensor
    sample_indices: torch.Tensor
    file_stems: tuple[str, ...]

    def to(self, device: torch.device | str) -> ComplexCouplingBatch:
        return ComplexCouplingBatch(
            geometry=self.geometry.to(device),
            rhs_valid=self.rhs_valid.to(device),
            sol_valid=self.sol_valid.to(device),
            flux_valid=self.flux_valid.to(device),
            has_flux=self.has_flux.to(device),
            a_valid=self.a_valid.to(device),
            x_source_branch=self.x_source_branch.to(device),
            y_source_branch=self.y_source_branch.to(device),
            x_source_amplitude=self.x_source_amplitude.to(device),
            y_source_amplitude=self.y_source_amplitude.to(device),
            x_coefficient_branch=self.x_coefficient_branch.to(device),
            y_coefficient_branch=self.y_coefficient_branch.to(device),
            x_green_branch=self.x_green_branch.to(device),
            y_green_branch=self.y_green_branch.to(device),
            sample_indices=self.sample_indices.to(device),
            file_stems=self.file_stems,
        )


class ComplexCouplingDataset(Dataset[ComplexCouplingItem]):
    """Full-grid sample dataset gathered into complex-geometry valid-point order."""

    def __init__(
        self,
        data_dir: Path | str,
        geometry: ComplexGeometryMetadata,
        coeffs: CoefficientFunctions,
        *,
        branch_input_dim: int,
        dtype: torch.dtype = torch.float64,
        coefficient_terms: CouplingCoefficientTermsConfig | None = None,
        integration_rule: IntegrationRule = "trapezoid",
        source_amplitude_eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {self.data_dir}")
        self.geometry = geometry
        self.coeffs = coeffs
        self.dtype = dtype
        self.branch_input_dim = int(branch_input_dim)
        if self.branch_input_dim < 2:
            raise ValueError("branch_input_dim must be at least 2.")
        self.coefficient_terms = coefficient_terms or CouplingCoefficientTermsConfig()
        self.integration_rule = integration_rule
        self.source_amplitude_eps = float(source_amplitude_eps)
        self.branch_grid = unit_branch_grid(self.branch_input_dim, dtype=dtype)
        self.x_green_coefficients = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="x",
            branch_input_dim=self.branch_input_dim,
            dtype=dtype,
        )
        self.y_green_coefficients = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="y",
            branch_input_dim=self.branch_input_dim,
            dtype=dtype,
        )
        self.x_coefficient_branch = self._build_coefficient_branch(axis="x")
        self.y_coefficient_branch = self._build_coefficient_branch(axis="y")
        self.x_green_branch = torch.stack(
            (
                self.x_green_coefficients.a_unit,
                self.x_green_coefficients.ap_unit,
                self.x_green_coefficients.b_unit,
                self.x_green_coefficients.c_unit,
            ),
            dim=1,
        )
        self.y_green_branch = torch.stack(
            (
                self.y_green_coefficients.a_unit,
                self.y_green_coefficients.ap_unit,
                self.y_green_coefficients.b_unit,
                self.y_green_coefficients.c_unit,
            ),
            dim=1,
        )
        coords = geometry.coords_valid.to(dtype=dtype)
        self.a_valid = coeffs.a_fun(coords[:, 0], coords[:, 1]).to(dtype=dtype)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> ComplexCouplingItem:
        path = self.files[index]
        with np.load(path) as raw:
            missing = sorted({"rhs", "sol"} - set(raw.files))
            if missing:
                raise KeyError(f"{path} is missing required keys: {', '.join(missing)}")
            rhs_valid = self._gather_full_grid(raw["rhs"], "rhs", path)
            sol_valid = self._gather_full_grid(raw["sol"], "sol", path)
            flux_valid, has_flux = self._gather_optional_flux(raw, path)

        x_source_branch, x_source_amplitude = self._build_source_branch(
            rhs_valid,
            axis="x",
        )
        y_source_branch, y_source_amplitude = self._build_source_branch(
            rhs_valid,
            axis="y",
        )
        return ComplexCouplingItem(
            geometry=self.geometry,
            rhs_valid=rhs_valid,
            sol_valid=sol_valid,
            flux_valid=flux_valid,
            has_flux=torch.tensor(has_flux, dtype=torch.bool),
            a_valid=self.a_valid,
            x_source_branch=x_source_branch,
            y_source_branch=y_source_branch,
            x_source_amplitude=x_source_amplitude,
            y_source_amplitude=y_source_amplitude,
            x_coefficient_branch=self.x_coefficient_branch,
            y_coefficient_branch=self.y_coefficient_branch,
            x_green_branch=self.x_green_branch,
            y_green_branch=self.y_green_branch,
            sample_index=torch.tensor(index, dtype=torch.long),
            file_stem=path.stem,
        )

    def _build_coefficient_branch(
        self,
        *,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        coefficients = (
            self.x_green_coefficients if axis == "x" else self.y_green_coefficients
        )
        active = []
        if self.coefficient_terms.diffusion:
            active.append(coefficients.a_unit)
        if self.coefficient_terms.convection:
            active.append(coefficients.b_unit)
            active.append(self._build_transverse_convection_branch(axis=axis))
        if self.coefficient_terms.reaction:
            active.append(coefficients.c_unit)
        if not active:
            return coefficients.a_unit.new_empty(
                (
                    coefficients.a_unit.shape[0],
                    0,
                    coefficients.a_unit.shape[1],
                )
            )
        return torch.stack(active, dim=1)

    def _build_transverse_convection_branch(
        self,
        *,
        axis: Literal["x", "y"],
    ) -> torch.Tensor:
        t = self.branch_grid
        if axis == "x":
            left = self.geometry.x_segment_left.to(dtype=self.dtype)
            right = self.geometry.x_segment_right.to(dtype=self.dtype)
            fixed = self.geometry.x_segment_y.to(dtype=self.dtype)
            length = self.geometry.x_segment_length.to(dtype=self.dtype)
            x = physical_interval_coordinates(left, right, t)
            y = fixed.unsqueeze(-1).expand_as(x)
            b_transverse = self.coeffs.by_fun(x, y).to(dtype=self.dtype)
        else:
            left = self.geometry.y_segment_bottom.to(dtype=self.dtype)
            right = self.geometry.y_segment_top.to(dtype=self.dtype)
            fixed = self.geometry.y_segment_x.to(dtype=self.dtype)
            length = self.geometry.y_segment_length.to(dtype=self.dtype)
            y = physical_interval_coordinates(left, right, t)
            x = fixed.unsqueeze(-1).expand_as(y)
            b_transverse = self.coeffs.bx_fun(x, y).to(dtype=self.dtype)
        return length.unsqueeze(-1) * b_transverse

    def _build_source_branch(
        self,
        rhs_valid: torch.Tensor,
        *,
        axis: Literal["x", "y"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if axis == "x":
            ptr = self.geometry.x_recon_ptr
            t_nodes = self.geometry.x_recon_t
            valid_index = self.geometry.x_recon_valid_index
            segment_count = self.geometry.num_x_segments
        else:
            ptr = self.geometry.y_recon_ptr
            t_nodes = self.geometry.y_recon_t
            valid_index = self.geometry.y_recon_valid_index
            segment_count = self.geometry.num_y_segments

        branches = []
        for segment_index in range(segment_count):
            start = int(ptr[segment_index].item())
            end = int(ptr[segment_index + 1].item())
            segment_t = t_nodes[start:end].to(dtype=self.dtype)
            segment_valid = valid_index[start:end]
            values = torch.zeros_like(segment_t)
            mask = segment_valid >= 0
            if torch.any(mask):
                values[mask] = rhs_valid[segment_valid[mask]]
            branches.append(self._interpolate_unit_branch(segment_t, values))
        source_phys = torch.stack(branches, dim=0)
        source_profile_norm = integrate(
            source_phys.pow(2),
            x=self.branch_grid,
            dim=-1,
            rule=self.integration_rule,
        ).sqrt()
        source_amplitude = torch.maximum(
            source_profile_norm,
            source_profile_norm.new_full(
                source_profile_norm.shape,
                self.source_amplitude_eps,
            ),
        )
        return source_phys / source_amplitude.unsqueeze(-1), source_amplitude

    def _interpolate_unit_branch(
        self,
        t_nodes: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if t_nodes.numel() < 2:
            raise ValueError("Source branch interpolation requires at least two nodes.")
        idx = torch.searchsorted(t_nodes.contiguous(), self.branch_grid.contiguous())
        idx = idx.clamp(1, t_nodes.numel() - 1)
        left = idx - 1
        right = idx
        t_left = t_nodes[left]
        t_right = t_nodes[right]
        denom = (t_right - t_left).clamp_min(torch.finfo(self.dtype).eps)
        weight = (self.branch_grid - t_left) / denom
        return values[left] * (1.0 - weight) + values[right] * weight

    def _gather_optional_flux(
        self,
        raw: np.lib.npyio.NpzFile,
        path: Path,
    ) -> tuple[torch.Tensor, bool]:
        if {"phi", "psi"}.issubset(raw.files):
            phi = self._gather_full_grid(raw["phi"], "phi", path)
            psi = self._gather_full_grid(raw["psi"], "psi", path)
            return torch.stack((phi, psi), dim=0), True
        if {"uxx", "uyy"}.issubset(raw.files):
            phi = self._gather_full_grid(raw["uxx"], "uxx", path)
            psi = self._gather_full_grid(raw["uyy"], "uyy", path)
            return torch.stack((phi, psi), dim=0), True
        empty = torch.zeros((2, self.geometry.num_points), dtype=self.dtype)
        return empty, False

    def _gather_full_grid(
        self,
        array: np.ndarray,
        field_name: str,
        path: Path,
    ) -> torch.Tensor:
        if array.ndim != 2:
            raise ValueError(f"{path}:{field_name} must be a 2D full-grid array.")
        y_index = self.geometry.valid_grid_y_index.detach().cpu().numpy()
        x_index = self.geometry.valid_grid_x_index.detach().cpu().numpy()
        if (
            int(y_index.max(initial=0)) >= array.shape[0]
            or int(x_index.max(initial=0)) >= array.shape[1]
        ):
            raise ValueError(
                f"{path}:{field_name} shape {array.shape} does not cover geometry "
                "valid_grid_y_index/x_index."
            )
        gathered = array[y_index, x_index]
        return torch.as_tensor(gathered, dtype=self.dtype)


def complex_coupling_collate_fn(
    items: Sequence[ComplexCouplingItem],
) -> ComplexCouplingBatch:
    if not items:
        raise ValueError("Cannot collate an empty complex coupling batch.")
    geometry = items[0].geometry
    return ComplexCouplingBatch(
        geometry=geometry,
        rhs_valid=torch.stack([item.rhs_valid for item in items], dim=0),
        sol_valid=torch.stack([item.sol_valid for item in items], dim=0),
        flux_valid=torch.stack([item.flux_valid for item in items], dim=0),
        has_flux=torch.stack([item.has_flux for item in items], dim=0),
        a_valid=torch.stack([item.a_valid for item in items], dim=0),
        x_source_branch=torch.stack([item.x_source_branch for item in items], dim=0),
        y_source_branch=torch.stack([item.y_source_branch for item in items], dim=0),
        x_source_amplitude=torch.stack(
            [item.x_source_amplitude for item in items], dim=0
        ),
        y_source_amplitude=torch.stack(
            [item.y_source_amplitude for item in items], dim=0
        ),
        x_coefficient_branch=torch.stack(
            [item.x_coefficient_branch for item in items],
            dim=0,
        ),
        y_coefficient_branch=torch.stack(
            [item.y_coefficient_branch for item in items],
            dim=0,
        ),
        x_green_branch=torch.stack([item.x_green_branch for item in items], dim=0),
        y_green_branch=torch.stack([item.y_green_branch for item in items], dim=0),
        sample_indices=torch.stack([item.sample_index for item in items], dim=0),
        file_stems=tuple(item.file_stem for item in items),
    )
