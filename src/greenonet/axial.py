from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class XAxialLine:
    idx: int
    y_coordinate: float
    x_coordinates: Tensor = field(default_factory=Tensor)
    coordinates: Tensor = field(default_factory=Tensor)
    sampling_num: int = -1


@dataclass
class YAxialLine:
    idx: int
    x_coordinate: float
    y_coordinates: Tensor = field(default_factory=Tensor)
    coordinates: Tensor = field(default_factory=Tensor)
    sampling_num: int = -1


@dataclass
class AxialLines:
    xaxial_lines: list[XAxialLine] = field(default_factory=list)
    yaxial_lines: list[YAxialLine] = field(default_factory=list)

    def reindexing(self) -> "AxialLines":
        for i, x_line in enumerate(self.xaxial_lines):
            x_line.idx = i
        for i, y_line in enumerate(self.yaxial_lines):
            y_line.idx = i
        return self

    def sort(self) -> "AxialLines":
        self.sort_xaxial_by_y()
        self.sort_yaxial_by_x()
        return self

    def sort_xaxial_by_y(self) -> "AxialLines":
        self.xaxial_lines.sort(key=lambda line: line.y_coordinate)
        return self

    def sort_yaxial_by_x(self) -> "AxialLines":
        self.yaxial_lines.sort(key=lambda line: line.x_coordinate)
        return self


def make_square_axial_lines(
    step_size: float,
    n_points_per_line: int | None = None,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
) -> AxialLines:
    """Generate uniformly spaced axial lines over a square domain."""

    x_min, x_max = x_range
    y_min, y_max = y_range
    n_lines_x = int(round((x_max - x_min) / step_size)) + 1
    n_lines_y = int(round((y_max - y_min) / step_size)) + 1
    n_points = (
        n_points_per_line
        if n_points_per_line is not None
        else int(round((x_max - x_min) / step_size)) + 1
    )

    line_positions_x = torch.linspace(x_min, x_max, n_lines_x, dtype=torch.float64)[
        1:-1
    ]
    line_positions_y = torch.linspace(y_min, y_max, n_lines_y, dtype=torch.float64)[
        1:-1
    ]
    x_line_coords = torch.linspace(x_min, x_max, n_points, dtype=torch.float64)
    y_line_coords = torch.linspace(y_min, y_max, n_points, dtype=torch.float64)

    x_lines: list[XAxialLine] = []
    y_lines: list[YAxialLine] = []

    for idx, y in enumerate(line_positions_y):
        coords = torch.stack((x_line_coords, torch.full_like(x_line_coords, y)), dim=1)
        x_lines.append(
            XAxialLine(
                idx=idx,
                y_coordinate=float(y),
                x_coordinates=x_line_coords.clone(),
                coordinates=coords,
                sampling_num=n_points,
            )
        )

    for idx, x in enumerate(line_positions_x):
        coords = torch.stack((torch.full_like(y_line_coords, x), y_line_coords), dim=1)
        y_lines.append(
            YAxialLine(
                idx=idx,
                x_coordinate=float(x),
                y_coordinates=y_line_coords.clone(),
                coordinates=coords,
                sampling_num=n_points,
            )
        )

    return AxialLines(xaxial_lines=x_lines, yaxial_lines=y_lines).sort().reindexing()
