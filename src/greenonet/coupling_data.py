from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from greenonet.axial import AxialLines, make_square_axial_lines
from greenonet.numerics import IntegrationRule, integrate


def _pad_flux(arr: np.ndarray) -> np.ndarray:
    """Pad a (255,255) flux array to (257,257) via edge replication."""
    return np.pad(arr, pad_width=1, mode="edge")


class CouplingDataset(
    Dataset[
        Tuple[
            torch.Tensor,  # coords
            torch.Tensor,  # rhs_raw
            torch.Tensor,  # rhs_tilde
            torch.Tensor,  # rhs_norm
            torch.Tensor,  # sol
            torch.Tensor,  # flux
            torch.Tensor,  # kappa
            torch.Tensor,  # b
            torch.Tensor,  # c
            torch.Tensor,  # ap
            tuple[torch.Tensor, ...],  # boundary payload
        ]
    ]
):
    """
    Dataset for coupling net, sampling npz files with rhs/sol/uxx/uyy on a uniform grid.

    Each item returns:
        coords: (2, n_lines, m_points, 2)
        rhs_raw: (2, n_lines, m_points)
        rhs_tilde: (2, n_lines, m_points)  # normalized per line
        rhs_norm: (2, n_lines)             # line-wise L2 norms
        sol: (2, n_lines, m_points)
        flux: (2, n_lines, m_points)  # x/y axial flux divergence targets
        kappa, b, c: (2, n_lines, m_points)
        ap: (2, n_lines, m_points)
        boundary payload:
            boundary_coords: (2, 2, m_points, 2)
            boundary_rhs_raw: (2, 2, m_points)
            boundary_rhs_tilde: (2, 2, m_points)
            boundary_rhs_norm: (2, 2)
            boundary_kappa/b/c: (2, 2, m_points)
    """

    def __init__(
        self,
        data_dir: Path,
        step_size: float,
        n_points_per_line: int | None = None,
        dtype: torch.dtype = torch.float64,
        integration_rule: IntegrationRule = "simpson",
        a_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        b_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        c_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        ap_fun_x: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        ap_fun_y: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.integration_rule = integration_rule
        self.files: List[Path] = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No npz files found in {data_dir}")
        self.step_size = step_size
        self.n_points_per_line = n_points_per_line
        self.a_fun = a_fun
        self.b_fun = b_fun
        self.c_fun = c_fun
        self.ap_fun_x = ap_fun_x
        self.ap_fun_y = ap_fun_y

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _sample_lines(array: np.ndarray, lines: AxialLines, axis: int) -> torch.Tensor:
        """
        Sample a grid array along axial lines.

        axis=0 -> x-lines (varying x, fixed y), axis=1 -> y-lines (varying y, fixed x)
        """
        grid_res = array.shape[0] - 1  # 256 for 257 grid
        samples = []
        if axis == 0:
            for x_line in lines.xaxial_lines:
                y_idx = int(round(x_line.y_coordinate * grid_res))
                x_coords = x_line.x_coordinates.cpu().numpy()
                x_idx = np.round(x_coords * grid_res).astype(int)
                samples.append(array[y_idx, x_idx])
        else:
            for y_line in lines.yaxial_lines:
                x_idx = int(round(y_line.x_coordinate * grid_res))
                y_coords = y_line.y_coordinates.cpu().numpy()
                y_idx = np.round(y_coords * grid_res).astype(int)
                samples.append(array[y_idx, x_idx])
        stacked = np.stack(samples, axis=0)  # (n_lines, m_points)
        return torch.from_numpy(stacked)

    @staticmethod
    def _boundary_coords(lines: AxialLines, dtype: torch.dtype) -> torch.Tensor:
        x_coords = lines.xaxial_lines[0].x_coordinates.to(dtype=dtype)
        y_coords = lines.yaxial_lines[0].y_coordinates.to(dtype=dtype)
        x_boundary = torch.stack(
            (
                torch.stack((x_coords, torch.zeros_like(x_coords)), dim=-1),
                torch.stack((x_coords, torch.ones_like(x_coords)), dim=-1),
            ),
            dim=0,
        )
        y_boundary = torch.stack(
            (
                torch.stack((torch.zeros_like(y_coords), y_coords), dim=-1),
                torch.stack((torch.ones_like(y_coords), y_coords), dim=-1),
            ),
            dim=0,
        )
        return torch.stack((x_boundary, y_boundary), dim=0)

    @staticmethod
    def _sample_boundary_lines(
        array: np.ndarray, lines: AxialLines, axis: int
    ) -> torch.Tensor:
        grid_res = array.shape[0] - 1
        if axis == 0:
            x_coords = lines.xaxial_lines[0].x_coordinates.cpu().numpy()
            x_idx = np.round(x_coords * grid_res).astype(int)
            samples = [array[0, x_idx], array[grid_res, x_idx]]
        else:
            y_coords = lines.yaxial_lines[0].y_coordinates.cpu().numpy()
            y_idx = np.round(y_coords * grid_res).astype(int)
            samples = [array[y_idx, 0], array[y_idx, grid_res]]
        return torch.from_numpy(np.stack(samples, axis=0))

    @staticmethod
    def _sample_lines_fun(
        fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lines: AxialLines,
        axis: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        samples = []
        if axis == 0:
            for x_line in lines.xaxial_lines:
                x = x_line.x_coordinates
                y = torch.full_like(x, x_line.y_coordinate)
                samples.append(fun(x, y))
        else:
            for y_line in lines.yaxial_lines:
                y = y_line.y_coordinates
                x = torch.full_like(y, y_line.x_coordinate)
                samples.append(fun(x, y))
        return torch.stack(samples, dim=0).to(dtype=dtype)

    @staticmethod
    def _sample_boundary_lines_fun(
        fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        lines: AxialLines,
        axis: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if axis == 0:
            x = lines.xaxial_lines[0].x_coordinates
            samples = [
                fun(x, torch.zeros_like(x)),
                fun(x, torch.ones_like(x)),
            ]
        else:
            y = lines.yaxial_lines[0].y_coordinates
            samples = [
                fun(torch.zeros_like(y), y),
                fun(torch.ones_like(y), y),
            ]
        return torch.stack(samples, dim=0).to(dtype=dtype)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
    ]:
        data = np.load(self.files[index])
        sol = data["sol"]
        rhs = data["rhs"]
        uxx = _pad_flux(data["uxx"])
        uyy = _pad_flux(data["uyy"])

        lines = make_square_axial_lines(
            step_size=self.step_size, n_points_per_line=self.n_points_per_line
        )
        coords = torch.stack(
            (
                torch.stack([line.coordinates for line in lines.xaxial_lines]),
                torch.stack([line.coordinates for line in lines.yaxial_lines]),
            ),
            dim=0,
        )  # (2, n_lines, m, 2)

        rhs_x = self._sample_lines(rhs, lines, axis=0)
        rhs_y = self._sample_lines(rhs, lines, axis=1)
        rhs_boundary_x = self._sample_boundary_lines(rhs, lines, axis=0)
        rhs_boundary_y = self._sample_boundary_lines(rhs, lines, axis=1)
        sol_x = self._sample_lines(sol, lines, axis=0)
        sol_y = self._sample_lines(sol, lines, axis=1)
        flux_x = self._sample_lines(uxx, lines, axis=0)
        flux_y = self._sample_lines(uyy, lines, axis=1)

        rhs_t = torch.stack((rhs_x, rhs_y), dim=0).to(dtype=self.dtype)
        boundary_rhs_t = torch.stack((rhs_boundary_x, rhs_boundary_y), dim=0).to(
            dtype=self.dtype
        )
        sol_t = torch.stack((sol_x, sol_y), dim=0).to(dtype=self.dtype)
        flux_t = torch.stack((flux_x, flux_y), dim=0).to(dtype=self.dtype)

        # Line-wise L2 norm of rhs via Simpson rule
        eps = torch.tensor(1e-12, dtype=self.dtype)
        x_axis = lines.xaxial_lines[0].x_coordinates
        y_axis = lines.yaxial_lines[0].y_coordinates
        rhs_norm_x = (
            integrate(
                rhs_t[0].pow(2),
                x=x_axis,
                dim=-1,
                rule=self.integration_rule,
            )
            .sqrt()
            .clamp_min(eps)
        )
        rhs_norm_y = (
            integrate(
                rhs_t[1].pow(2),
                x=y_axis,
                dim=-1,
                rule=self.integration_rule,
            )
            .sqrt()
            .clamp_min(eps)
        )
        rhs_norm = torch.stack((rhs_norm_x, rhs_norm_y), dim=0)  # (2, n_lines)
        rhs_tilde = rhs_t / rhs_norm.unsqueeze(-1)
        boundary_norm_x = (
            integrate(
                boundary_rhs_t[0].pow(2),
                x=x_axis,
                dim=-1,
                rule=self.integration_rule,
            )
            .sqrt()
            .clamp_min(eps)
        )
        boundary_norm_y = (
            integrate(
                boundary_rhs_t[1].pow(2),
                x=y_axis,
                dim=-1,
                rule=self.integration_rule,
            )
            .sqrt()
            .clamp_min(eps)
        )
        boundary_rhs_norm = torch.stack((boundary_norm_x, boundary_norm_y), dim=0)
        boundary_rhs_tilde = boundary_rhs_t / boundary_rhs_norm.unsqueeze(-1)
        if self.a_fun is not None:
            kappa_x = self._sample_lines_fun(
                self.a_fun, lines, axis=0, dtype=self.dtype
            )
            kappa_y = self._sample_lines_fun(
                self.a_fun, lines, axis=1, dtype=self.dtype
            )
            kappa_t = torch.stack((kappa_x, kappa_y), dim=0)
            boundary_kappa_x = self._sample_boundary_lines_fun(
                self.a_fun, lines, axis=0, dtype=self.dtype
            )
            boundary_kappa_y = self._sample_boundary_lines_fun(
                self.a_fun, lines, axis=1, dtype=self.dtype
            )
            boundary_kappa_t = torch.stack((boundary_kappa_x, boundary_kappa_y), dim=0)
        else:
            kappa_t = torch.ones_like(rhs_t)
            boundary_kappa_t = torch.ones_like(boundary_rhs_t)
        if self.b_fun is not None:
            b_x = self._sample_lines_fun(self.b_fun, lines, axis=0, dtype=self.dtype)
            b_y = self._sample_lines_fun(self.b_fun, lines, axis=1, dtype=self.dtype)
            b_t = torch.stack((b_x, b_y), dim=0)
            boundary_b_x = self._sample_boundary_lines_fun(
                self.b_fun, lines, axis=0, dtype=self.dtype
            )
            boundary_b_y = self._sample_boundary_lines_fun(
                self.b_fun, lines, axis=1, dtype=self.dtype
            )
            boundary_b_t = torch.stack((boundary_b_x, boundary_b_y), dim=0)
        else:
            b_t = torch.zeros_like(rhs_t)
            boundary_b_t = torch.zeros_like(boundary_rhs_t)
        if self.c_fun is not None:
            c_x = self._sample_lines_fun(self.c_fun, lines, axis=0, dtype=self.dtype)
            c_y = self._sample_lines_fun(self.c_fun, lines, axis=1, dtype=self.dtype)
            c_t = torch.stack((c_x, c_y), dim=0)
            boundary_c_x = self._sample_boundary_lines_fun(
                self.c_fun, lines, axis=0, dtype=self.dtype
            )
            boundary_c_y = self._sample_boundary_lines_fun(
                self.c_fun, lines, axis=1, dtype=self.dtype
            )
            boundary_c_t = torch.stack((boundary_c_x, boundary_c_y), dim=0)
        else:
            c_t = torch.zeros_like(rhs_t)
            boundary_c_t = torch.zeros_like(boundary_rhs_t)
        if self.ap_fun_x is not None and self.ap_fun_y is not None:
            ap_x = self._sample_lines_fun(
                self.ap_fun_x, lines, axis=0, dtype=self.dtype
            )
            ap_y = self._sample_lines_fun(
                self.ap_fun_y, lines, axis=1, dtype=self.dtype
            )
            ap_t = torch.stack((ap_x, ap_y), dim=0)
        else:
            ap_t = torch.zeros_like(rhs_t)
        coords = coords.to(dtype=self.dtype)
        boundary_payload = (
            self._boundary_coords(lines, dtype=self.dtype),
            boundary_rhs_t,
            boundary_rhs_tilde,
            boundary_rhs_norm,
            boundary_kappa_t,
            boundary_b_t,
            boundary_c_t,
        )
        return (
            coords,
            rhs_t,
            rhs_tilde,
            rhs_norm,
            sol_t,
            flux_t,
            kappa_t,
            b_t,
            c_t,
            ap_t,
            boundary_payload,
        )


CouplingBoundaryItem = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

CouplingCoreItem = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]

CouplingItem = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CouplingBoundaryItem,
]


def split_coupling_batch(
    batch: tuple[object, ...],
) -> tuple[CouplingCoreItem, CouplingBoundaryItem | None]:
    if len(batch) == 10:
        return batch, None  # type: ignore[return-value]
    if len(batch) == 11:
        core = batch[:10]
        boundary = batch[10]
        if not isinstance(boundary, tuple) or len(boundary) != 7:
            raise ValueError("Coupling boundary payload must contain seven tensors.")
        return core, boundary  # type: ignore[return-value]
    raise ValueError(f"Expected Coupling batch with 10 or 11 fields, got {len(batch)}.")


def coupling_collate_fn(batch: Sequence[tuple[object, ...]]) -> tuple[object, ...]:
    first_core, first_boundary = split_coupling_batch(tuple(batch[0]))
    del first_core
    has_boundary = first_boundary is not None
    split_items = [split_coupling_batch(tuple(item)) for item in batch]
    if any((boundary is not None) != has_boundary for _core, boundary in split_items):
        raise ValueError(
            "Cannot collate mixed Coupling items with and without boundary payload."
        )
    core_items = [core for core, _boundary in split_items]
    coords, rhs_raw, rhs_tilde, rhs_norm, sol, flux, kappa, b_vals, c_vals, ap = zip(
        *core_items
    )
    coords_packed = coords[0]

    def stack(fields: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(fields), dim=0)

    collated = (
        coords_packed,
        stack(rhs_raw),
        stack(rhs_tilde),
        stack(rhs_norm),
        stack(sol),
        stack(flux),
        stack(kappa),
        stack(b_vals),
        stack(c_vals),
        stack(ap),
    )
    if not has_boundary:
        return collated

    boundary_items: list[CouplingBoundaryItem] = []
    for _core, boundary in split_items:
        if boundary is None:
            continue
        boundary_items.append(boundary)
    (
        boundary_coords,
        boundary_rhs,
        boundary_tilde,
        boundary_norm,
        boundary_a,
        boundary_b,
        boundary_c,
    ) = zip(*boundary_items)
    boundary_payload = (
        boundary_coords[0],
        stack(boundary_rhs),
        stack(boundary_tilde),
        stack(boundary_norm),
        stack(boundary_a),
        stack(boundary_b),
        stack(boundary_c),
    )
    return (*collated, boundary_payload)
