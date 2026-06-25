from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from greenonet.sampler import TrainingData


class AxialDataset(
    Dataset[tuple[torch.Tensor, ...]]
):
    """
    Dataset that packs axial-line data.

    coords: shared grid (2, n_lines, m_points, 2) -> (axis, line_idx, point_idx, xy)
    solutions/coeffs/sources: pre-batched tensors (B, 2, n_lines, m_points)
    """

    def __init__(self, training_data: TrainingData) -> None:
        super().__init__()
        # coordinates already shaped (2, n_lines, m_points, 2)
        self.coords = training_data.COORDS
        # fields shaped (B, 2, n_lines, m_points)
        self.solutions = training_data.U
        self.sources = training_data.F
        self.sources_fine = training_data.F_FINE
        self.source_fine_grid = training_data.F_FINE_GRID
        self.a_vals = training_data.A
        self.ap_vals = training_data.AP
        self.b_vals = training_data.B
        self.c_vals = training_data.C
        self.num_samples: int = int(self.solutions.shape[0])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, ...]:
        base = (
            self.coords,  # (2, n, m, 2)
            self.solutions[index],  # (2, n, m)
            self.sources[index],  # (2, n, m)
        )
        coeffs = (
            self.a_vals[0],  # shared coefficients (2, n, m)
            self.ap_vals[0],
            self.b_vals[0],
            self.c_vals[0],
        )
        if self.sources_fine is None or self.source_fine_grid is None:
            return (*base, *coeffs)
        return (
            *base,
            self.sources_fine[index],
            self.source_fine_grid,
            *coeffs,
        )


AxialItem = tuple[torch.Tensor, ...]


def axial_collate_fn(batch: Sequence[AxialItem]) -> AxialItem:
    """
    Custom collate for AxialDataset to avoid duplicating coords.

    Returns:
        coords: (2, n, m, 2)
        solutions, sources: (B, 2, n, m)
        a, ap, b, c: (2, n, m) shared across batch
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    item_size = len(batch[0])
    if item_size not in {7, 9}:
        raise ValueError(f"Expected 7 or 9 axial fields, got {item_size}.")
    # All coords are identical; take the first
    coords_packed = batch[0][0]

    def stack(fields: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(fields), dim=0)

    if item_size == 7:
        _coords, solutions, sources, a_vals, ap_vals, b_vals, c_vals = zip(*batch)
        return (
            coords_packed,
            stack(solutions),
            stack(sources),
            a_vals[0],
            ap_vals[0],
            b_vals[0],
            c_vals[0],
        )
    (
        _coords,
        solutions,
        sources,
        sources_fine,
        source_fine_grid,
        a_vals,
        ap_vals,
        b_vals,
        c_vals,
    ) = zip(*batch)
    return (
        coords_packed,
        stack(solutions),
        stack(sources),
        stack(sources_fine),
        source_fine_grid[0],
        a_vals[0],
        ap_vals[0],
        b_vals[0],
        c_vals[0],
    )
