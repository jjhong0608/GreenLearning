from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset

from greenonet.sampler import TrainingData


class AxialDataset(
    Dataset[  # type: ignore[misc]
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ]
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
        self.a_vals = training_data.A
        self.ap_vals = training_data.AP
        self.b_vals = training_data.B
        self.c_vals = training_data.C
        self.num_samples: int = int(self.solutions.shape[0])

    def __len__(self) -> int:
        return self.num_samples

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
    ]:
        return (
            self.coords,  # (2, n, m, 2)
            self.solutions[index],  # (2, n, m)
            self.sources[index],  # (2, n, m)
            self.a_vals[0],  # shared coefficients (2, n, m)
            self.ap_vals[0],
            self.b_vals[0],
            self.c_vals[0],
        )


AxialItem = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def axial_collate_fn(batch: Sequence[AxialItem]) -> AxialItem:
    """
    Custom collate for AxialDataset to avoid duplicating coords.

    Returns:
        coords: (2, n, m, 2)
        solutions, sources: (B, 2, n, m)
        a, ap, b, c: (2, n, m) shared across batch
    """
    coords, solutions, sources, a_vals, ap_vals, b_vals, c_vals = zip(*batch)
    # All coords are identical; take the first
    coords_packed = coords[0]

    def stack(fields: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(fields), dim=0)

    return (
        coords_packed,
        stack(solutions),
        stack(sources),
        a_vals[0],
        ap_vals[0],
        b_vals[0],
        c_vals[0],
    )
