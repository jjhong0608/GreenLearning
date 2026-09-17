import importlib
from pathlib import Path

import torch

from cli.weak_pde_fixed_space import (
    InteriorWeakPDE,
    EuclideanMetric,
    manufactured_checks,
)


def test_manufactured_weak_pde_converges():
    assert len(manufactured_checks()) == 6


def test_interior_q1_affine_patch_and_quadratic_source():
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    y, x = torch.meshgrid(t, t, indexing="ij")
    coords = torch.stack((x.ravel(), y.ravel()), 1)
    op = InteriorWeakPDE.build(coords, lambda x, y: torch.ones_like(x))
    u = coords[:, 0] ** 2 + coords[:, 1] ** 2
    r = op.residual(u[None], torch.full_like(u[None], -4))
    torch.testing.assert_close(r, torch.zeros_like(r), atol=1e-12, rtol=0)


def test_shared_space_refits_dense_objectives(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "cli"))
    refit = importlib.import_module("audit_fixed_tangent_subspace").refit_subspace
    torch.manual_seed(5)
    z = torch.randn(4, 2, 7, dtype=torch.float64)
    c = torch.randn(4, 2, 9, dtype=torch.float64)
    w = torch.randn(4, 2, 11, dtype=torch.float64)
    c0 = torch.randn(2, 9, dtype=torch.float64)
    w0 = torch.randn(2, 11, dtype=torch.float64)
    for target, design in [
        (c0, c),
        (w0, w),
        (torch.cat((c0, w0), -1), torch.cat((c, w), -1)),
    ]:
        fitted = refit(target, z, design, EuclideanMetric())
        for b in range(2):
            for k in range(1, 5):
                alpha = torch.linalg.lstsq(design[:k, b].T, -target[b]).solution
                torch.testing.assert_close(
                    fitted.deltas[k - 1, b], alpha @ z[:k, b], atol=1e-10, rtol=1e-10
                )


def test_exact_poisson_green_reconstruction_has_consistent_weak_load():
    t = torch.linspace(0, 1, 33, dtype=torch.float64)
    y, x = torch.meshgrid(t, t, indexing="ij")
    coords = torch.stack((x.ravel(), y.ravel()), 1)
    green = torch.minimum(t[:, None], t[None, :]) * (
        1 - torch.maximum(t[:, None], t[None, :])
    )
    weights = torch.ones_like(t) / 32
    weights[[0, -1]] *= 0.5
    phi, psi = 2 * y * (1 - y), 2 * x * (1 - x)
    ux = phi @ (green * weights).T
    uy = (green * weights) @ psi
    exact = x * (1 - x) * y * (1 - y)
    torch.testing.assert_close(ux, exact, atol=1e-14, rtol=0)
    torch.testing.assert_close(uy, exact, atol=1e-14, rtol=0)
    op = InteriorWeakPDE.build(coords, lambda x, y: torch.ones_like(x))
    defect = op.residual(((ux + uy) / 2).reshape(1, -1), (phi + psi).reshape(1, -1))
    torch.testing.assert_close(defect, torch.zeros_like(defect), atol=1e-12, rtol=0)
