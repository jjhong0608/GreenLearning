"""Diagnostic Q1 weak PDE operator with compact interior test supports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
import torch
from scipy.sparse import coo_matrix


@dataclass
class InteriorWeakPDE:
    stiffness: torch.Tensor
    mass: torch.Tensor
    test_mass: torch.Tensor
    test_indices: np.ndarray
    cell_count: int
    point_count: int

    @classmethod
    def build(
        cls,
        coords: torch.Tensor,
        a_fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        order: int = 3,
        device: str | torch.device = "cpu",
    ) -> InteriorWeakPDE:
        xy = coords.detach().cpu().numpy()
        x, ix = np.unique(xy[:, 0], return_inverse=True)
        y, iy = np.unique(xy[:, 1], return_inverse=True)
        grid = np.full((len(y), len(x)), -1, dtype=int)
        grid[iy, ix] = np.arange(len(xy))
        cells = np.stack(
            (grid[:-1, :-1], grid[:-1, 1:], grid[1:, 1:], grid[1:, :-1]), -1
        ).reshape(-1, 4)
        cells = cells[(cells >= 0).all(1)]
        count = np.bincount(cells.ravel(), minlength=len(xy))
        tests = np.flatnonzero(count == 4)
        if not len(tests):
            raise ValueError("No fully supported interior Q1 test functions")
        vertex = xy[cells]
        hx, hy = vertex[:, 1, 0] - vertex[:, 0, 0], vertex[:, 3, 1] - vertex[:, 0, 1]
        assert np.all(hx > 0) and np.all(hy > 0)
        ke = np.zeros((len(cells), 4, 4))
        me = np.zeros_like(ke)
        nodes, weights = np.polynomial.legendre.leggauss(order)
        for r, wr in zip((nodes + 1) / 2, weights / 2, strict=True):
            for s, ws in zip((nodes + 1) / 2, weights / 2, strict=True):
                n = np.array([(1 - r) * (1 - s), r * (1 - s), r * s, (1 - r) * s])
                dx = np.array([-(1 - s), 1 - s, s, -s])[None] / hx[:, None]
                dy = np.array([-(1 - r), -r, r, 1 - r])[None] / hy[:, None]
                q = torch.from_numpy(vertex[:, 0] + np.stack((r * hx, s * hy), 1))
                a = torch.as_tensor(a_fun(q[:, 0], q[:, 1])).expand(len(cells)).numpy()
                if not np.all(np.isfinite(a) & (a > 0)):
                    raise ValueError("Diffusion must be positive")
                weight = hx * hy * wr * ws
                ke += (a * weight)[:, None, None] * (
                    dx[:, :, None] * dx[:, None, :] + dy[:, :, None] * dy[:, None, :]
                )
                me += weight[:, None, None] * n[None, :, None] * n[None, None, :]
        row = np.broadcast_to(cells[:, :, None], ke.shape).ravel()
        col = np.broadcast_to(cells[:, None, :], ke.shape).ravel()
        matrices = []
        for values in (ke, me):
            matrix = (
                coo_matrix((values.ravel(), (row, col)), shape=(len(xy), len(xy)))
                .tocsr()[tests]
                .tocoo()
            )
            indices = torch.from_numpy(np.stack((matrix.row, matrix.col))).long()
            matrices.append(
                torch.sparse_coo_tensor(
                    indices,
                    torch.from_numpy(matrix.data),
                    matrix.shape,
                    check_invariants=True,
                )
                .coalesce()
                .to(device)
            )
        lump = torch.sparse.sum(matrices[1], dim=1).to_dense()
        return cls(matrices[0], matrices[1], lump, tests, len(cells), len(xy))

    def apply(self, values: torch.Tensor, *, load: bool = False) -> torch.Tensor:
        matrix = self.mass if load else self.stiffness
        shape = values.shape[:-1]
        result: torch.Tensor = torch.sparse.mm(
            matrix, values.reshape(-1, self.point_count).T
        ).T
        return result.reshape(*shape, len(self.test_indices)) / self.test_mass.sqrt()

    def residual(self, u: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return self.apply(u) - self.apply(f, load=True)


class EuclideanMetric:
    @staticmethod
    def inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left * right).sum(-1)


class ManufacturedResult(TypedDict):
    domain: str
    n: int
    relative_weak_defect: float
    points: int
    tests: int


def manufactured_checks() -> list[ManufacturedResult]:
    results: list[ManufacturedResult] = []
    for domain in ("square", "disk"):
        for n in (16, 32, 64):
            line = torch.linspace(0, 1, n + 1, dtype=torch.float64)
            yy, xx = torch.meshgrid(line, line, indexing="ij")
            coords = torch.stack((xx.ravel(), yy.ravel()), 1)
            if domain == "disk":
                coords -= 0.5
                coords = coords[coords.square().sum(1) < 0.25]
            x, y = coords.T
            if domain == "square":

                def a_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return torch.ones_like(x)

                u = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
                f = 2 * torch.pi**2 * u
            else:

                def a_fun(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return 1 + x * x + y * y

                radius_squared = x * x + y * y
                u = (0.25 - radius_squared) ** 2
                f = 2 - 12 * radius_squared - 24 * radius_squared**2
            op = InteriorWeakPDE.build(coords, a_fun)
            residual = op.residual(u[None], f[None])
            relative = float(residual.norm() / op.apply(f[None], load=True).norm())
            results.append(
                dict(
                    domain=domain,
                    n=n,
                    relative_weak_defect=relative,
                    points=len(coords),
                    tests=len(op.test_indices),
                )
            )
    for domain in ("square", "disk"):
        errors = [r["relative_weak_defect"] for r in results if r["domain"] == domain]
        if not (errors[1] < 0.4 * errors[0] and errors[2] < 0.4 * errors[1]):
            raise RuntimeError("Manufactured weak residual did not converge")
    return results
