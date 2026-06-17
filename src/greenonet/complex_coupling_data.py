from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from greenonet.coefficients import CoefficientFunctions
from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.green_interval import build_segment_branch_samples


@dataclass(frozen=True)
class ComplexCouplingItem:
    geometry: ComplexGeometryMetadata
    rhs_valid: torch.Tensor
    sol_valid: torch.Tensor
    flux_valid: torch.Tensor
    has_flux: torch.Tensor
    a_valid: torch.Tensor
    x_branch: torch.Tensor
    y_branch: torch.Tensor
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
    x_branch: torch.Tensor
    y_branch: torch.Tensor
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
            x_branch=self.x_branch.to(device),
            y_branch=self.y_branch.to(device),
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
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {self.data_dir}")
        self.geometry = geometry
        self.coeffs = coeffs
        self.dtype = dtype
        self.x_green_coefficients = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="x",
            branch_input_dim=branch_input_dim,
            dtype=dtype,
        )
        self.y_green_coefficients = build_segment_branch_samples(
            geometry,
            coeffs,
            axis="y",
            branch_input_dim=branch_input_dim,
            dtype=dtype,
        )
        self.x_branch = self.x_green_coefficients.as_coupling_branch()
        self.y_branch = self.y_green_coefficients.as_coupling_branch()
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

        return ComplexCouplingItem(
            geometry=self.geometry,
            rhs_valid=rhs_valid,
            sol_valid=sol_valid,
            flux_valid=flux_valid,
            has_flux=torch.tensor(has_flux, dtype=torch.bool),
            a_valid=self.a_valid,
            x_branch=self.x_branch,
            y_branch=self.y_branch,
            x_green_branch=self.x_green_branch,
            y_green_branch=self.y_green_branch,
            sample_index=torch.tensor(index, dtype=torch.long),
            file_stem=path.stem,
        )

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
        x_branch=torch.stack([item.x_branch for item in items], dim=0),
        y_branch=torch.stack([item.y_branch for item in items], dim=0),
        x_green_branch=torch.stack([item.x_green_branch for item in items], dim=0),
        y_green_branch=torch.stack([item.y_green_branch for item in items], dim=0),
        sample_indices=torch.stack([item.sample_index for item in items], dim=0),
        file_stems=tuple(item.file_stem for item in items),
    )
