from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from torch import Tensor

from greenonet.axial import AxialLines, XAxialLine, YAxialLine
from greenonet.numerics import IntegrationRule, integrate


@dataclass
class TrainingData:
    X: Tensor
    Y: Tensor
    U: Tensor
    F: Tensor
    A: Tensor
    AP: Tensor
    B: Tensor
    C: Tensor
    COORDS: Tensor


class ForwardSampler:
    """Sample axial data inspired by the original GreenONet codebase.

    Pipeline: build RBF mixture on a 1D line, enforce Dirichlet via linear
    interpolant, compute PDE residual terms (-∂x(a∂x u)+b∂x u+c u) and boundary
    correction, then normalize (|u|≤1) to stabilize training.
    """

    EPS = 1e-8  # avoid divide-by-zero when signals are near zero

    def __init__(
        self,
        axial_lines: AxialLines,
        data_size_per_each_line: int,
        scale_length: float | tuple[float, float],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
        deterministic: bool = True,
        integration_rule: IntegrationRule = "simpson",
    ) -> None:
        self.axial_lines = axial_lines
        self.data_size_per_each_line = data_size_per_each_line
        self.deterministic = deterministic
        self.scale_length = scale_length
        self.device = device
        self.dtype = dtype
        self.integration_rule = integration_rule

    def _sample_scale_length(self) -> float:
        if isinstance(self.scale_length, tuple):
            return float(
                torch.empty(1).uniform_(self.scale_length[0], self.scale_length[1])
            )
        return float(self.scale_length)

    def _centers(self, x: Tensor) -> Tensor:
        if self.deterministic:
            return torch.linspace(
                x[0].item(),
                x[-1].item(),
                x.size(0),
                device=self.device,
                dtype=self.dtype,
            )
        return (
            torch.randn(x.size(0), device=self.device, dtype=self.dtype)
            .mul(x[-1].item() - x[0].item())
            .add(x[0].item())
        )

    @staticmethod
    def _rbf_kernel(x: Tensor, y: Tensor, scale_length: float) -> Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist2 = diff.pow(2)
        return dist2.div(scale_length**2).mul(-0.5).exp()

    def _rbf_derivatives(
        self, x: Tensor, y: Tensor, scale_length: float, k: Tensor
    ) -> tuple[Tensor, Tensor]:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        k_x = -diff.div(scale_length**2).mul(k)
        k_xx = diff.pow(2).div(scale_length**4).sub(1.0 / (scale_length**2)).mul(k)
        return k_x, k_xx

    @staticmethod
    def _linear_interpolant(x: Tensor, u: Tensor) -> tuple[Tensor, Tensor]:
        ul = u[0].item()
        ur = u[-1].item()
        xl = x[0].item()
        xr = x[-1].item()
        slope = (ur - ul) / (xr - xl)
        boundary_interpolant = slope * (x - xl) + ul
        return boundary_interpolant, torch.full_like(x, slope)

    def _line_integral(self, x: Tensor, y: Tensor) -> Tensor:
        if self.integration_rule == "simpson" and x.numel() < 3:
            return integrate(y, x=x, dim=-1, rule="trapezoid")
        return integrate(y, x=x, dim=-1, rule=self.integration_rule)

    def _simpson_integral(self, x: Tensor, y: Tensor) -> Tensor:
        return self._line_integral(x, y)

    def _normalize_sample(
        self, x: Tensor, u: Tensor, f: Tensor
    ) -> tuple[Tensor, Tensor]:
        # Use the configured line integral on u^2 to keep |u| <= 1 and avoid degenerate scales.
        scale = max(self._line_integral(x, u**2).sqrt().item(), self.EPS)
        return u.div(scale), f.div(scale)

    def generate_sample(
        self,
        x: Tensor,
        a_fun: Callable[[Tensor], Tensor],
        ap_fun: Callable[[Tensor], Tensor],
        b_fun: Callable[[Tensor], Tensor],
        c_fun: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a single axial sample: returns (u, f, a, ap, b, c)."""
        x = x.to(device=self.device, dtype=self.dtype)
        alpha = torch.randn(x.shape, device=self.device, dtype=self.dtype)
        centers = self._centers(x)
        scale_length = self._sample_scale_length()

        k = self._rbf_kernel(x, centers, scale_length)
        k_x, k_xx = self._rbf_derivatives(x, centers, scale_length, k)

        u = k @ alpha
        du_dx = k_x @ alpha
        du_dxx = k_xx @ alpha

        boundary_interpolant, dl_dx = self._linear_interpolant(x, u)

        a_val = a_fun(x)
        ap_val = ap_fun(x)
        b_val = b_fun(x)
        c_val = c_fun(x)

        dv_dx = ap_val * du_dx + a_val * du_dxx
        dm_dx = ap_val * dl_dx

        f = -dv_dx + b_val * du_dx + c_val * u
        g = -dm_dx + b_val * dl_dx + c_val * boundary_interpolant

        u = u.sub(boundary_interpolant).detach()
        f = f.sub(g).detach()
        u, f = self._normalize_sample(x, u, f)

        return u, f, a_val.detach(), ap_val.detach(), b_val.detach(), c_val.detach()

    def _generate_sample_on_x_line(
        self,
        axial_line: XAxialLine,
        a_fun: Callable[[Tensor], Tensor],
        ap_fun: Callable[[Tensor], Tensor],
        b_fun: Callable[[Tensor], Tensor],
        c_fun: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = axial_line.x_coordinates
        u, f, a, ap, b, c = self.generate_sample(x, a_fun, ap_fun, b_fun, c_fun)
        coords = axial_line.coordinates.to(device=self.device, dtype=self.dtype)
        return u, f, a, ap, b, c, coords

    def _generate_sample_on_y_line(
        self,
        axial_line: YAxialLine,
        a_fun: Callable[[Tensor], Tensor],
        ap_fun: Callable[[Tensor], Tensor],
        b_fun: Callable[[Tensor], Tensor],
        c_fun: Callable[[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        y = axial_line.y_coordinates
        u, f, a, ap, b, c = self.generate_sample(y, a_fun, ap_fun, b_fun, c_fun)
        coords = axial_line.coordinates.to(device=self.device, dtype=self.dtype)
        return u, f, a, ap, b, c, coords

    def generate_dataset(
        self,
        a_fun: Callable[[Tensor, Tensor], Tensor],
        ap_fun: Callable[[Tensor, Tensor], Tensor],
        b_fun: Callable[[Tensor, Tensor], Tensor],
        c_fun: Callable[[Tensor, Tensor], Tensor],
        a_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
        ap_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
        b_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
        c_fun_y: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> TrainingData:
        """Generate stacked TrainingData for all x- and y-axis lines."""
        x_entries = []
        for x_line in self.axial_lines.xaxial_lines:
            a_line = lambda x, y_const=x_line.y_coordinate: a_fun(  # noqa: E731
                x, torch.full_like(x, y_const)
            )
            ap_line = lambda x, y_const=x_line.y_coordinate: ap_fun(  # noqa: E731
                x, torch.full_like(x, y_const)
            )
            b_line = lambda x, y_const=x_line.y_coordinate: b_fun(  # noqa: E731
                x, torch.full_like(x, y_const)
            )
            c_line = lambda x, y_const=x_line.y_coordinate: c_fun(  # noqa: E731
                x, torch.full_like(x, y_const)
            )
            for _ in range(self.data_size_per_each_line):
                x_entries.append(
                    self._generate_sample_on_x_line(
                        x_line, a_line, ap_line, b_line, c_line
                    )
                )

        y_entries = []
        for y_line in self.axial_lines.yaxial_lines:
            a_src = a_fun_y if a_fun_y is not None else a_fun
            ap_src = ap_fun_y if ap_fun_y is not None else ap_fun
            b_src = b_fun_y if b_fun_y is not None else b_fun
            c_src = c_fun_y if c_fun_y is not None else c_fun
            a_line = lambda y, x_const=y_line.x_coordinate: a_src(  # noqa: E731
                torch.full_like(y, x_const), y
            )
            ap_line = lambda y, x_const=y_line.x_coordinate: ap_src(  # noqa: E731
                torch.full_like(y, x_const), y
            )
            b_line = lambda y, x_const=y_line.x_coordinate: b_src(  # noqa: E731
                torch.full_like(y, x_const), y
            )
            c_line = lambda y, x_const=y_line.x_coordinate: c_src(  # noqa: E731
                torch.full_like(y, x_const), y
            )
            for _ in range(self.data_size_per_each_line):
                y_entries.append(
                    self._generate_sample_on_y_line(
                        y_line, a_line, ap_line, b_line, c_line
                    )
                )

        # Reshape into (k, 2, n_lines, m_points) assuming n_x == n_y
        n_x = len(self.axial_lines.xaxial_lines)
        n_y = len(self.axial_lines.yaxial_lines)
        if n_x != n_y:
            raise ValueError("Expected the same number of x- and y-axial lines")

        k = self.data_size_per_each_line

        def reshape_axis(entries: tuple[Tensor, ...]) -> torch.Tensor:
            stacked = torch.stack(entries, dim=0)  # (n_lines * k, m)
            m_points = stacked.shape[1]
            return stacked.reshape(n_x, k, m_points).permute(1, 0, 2)  # (k, n, m)

        ux, fx, ax, apx, bx, cx, coords_x = zip(*x_entries)
        uy, fy, ay, apy, by, cy, coords_y = zip(*y_entries)

        Ux = reshape_axis(ux)
        Uy = reshape_axis(uy)
        Fx = reshape_axis(fx)
        Fy = reshape_axis(fy)
        Ax = reshape_axis(ax)
        Ay = reshape_axis(ay)
        APx = reshape_axis(apx)
        APy = reshape_axis(apy)
        Bx = reshape_axis(bx)
        By = reshape_axis(by)
        Cx = reshape_axis(cx)
        Cy = reshape_axis(cy)

        def reshape_coords(entries: tuple[Tensor, ...]) -> torch.Tensor:
            stacked = torch.stack(entries, dim=0)  # (n_lines * k, m, 2)
            m_points = stacked.shape[1]
            return stacked.reshape(n_x, k, m_points, 2)

        coords_x_t = reshape_coords(coords_x)  # (n, k, m, 2)
        coords_y_t = reshape_coords(coords_y)
        # Coordinates are identical across repeated samples; keep one per line
        coords_x_single = coords_x_t[:, 0]  # (n, m, 2)
        coords_y_single = coords_y_t[:, 0]

        X = self.axial_lines.xaxial_lines[0].x_coordinates
        Y = self.axial_lines.yaxial_lines[0].y_coordinates

        U = torch.stack((Ux, Uy), dim=1)  # (k, 2, n, m)
        F = torch.stack((Fx, Fy), dim=1)
        A = torch.stack((Ax, Ay), dim=1)
        AP = torch.stack((APx, APy), dim=1)
        B = torch.stack((Bx, By), dim=1)
        C = torch.stack((Cx, Cy), dim=1)
        COORDS = torch.stack((coords_x_single, coords_y_single), dim=0)  # (2, n, m, 2)
        return TrainingData(X, Y, U, F, A, AP, B, C, COORDS)
