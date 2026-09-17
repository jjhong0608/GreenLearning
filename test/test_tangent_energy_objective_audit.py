"""Audit-only energy metric and correction tests."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "cli"))
    return importlib.import_module("audit_tangent_energy_objective")


def test_edge_energy_and_adjoint(monkeypatch):
    from greenonet.complex_losses import _edge_energy_values

    mod = module(monkeypatch)
    geometry = SimpleNamespace(
        hx=torch.tensor(0.2),
        hy=torch.tensor(0.3),
        x_edges=torch.tensor([[0, 1], [1, 2]]),
        y_edges=torch.tensor([[0, 2]]),
    )
    a = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    m = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    metric = mod.ResponseMetric(geometry, a, "energy")
    expected = sum(
        _edge_energy_values(
            residual=m, a_valid=a, edges=e, spacing=h, area=geometry.hx * geometry.hy
        ).sum(-1)
        for e, h in [(geometry.x_edges, geometry.hx), (geometry.y_edges, geometry.hy)]
    )
    torch.testing.assert_close(metric.inner(m, m), expected)
    (gradient,) = torch.autograd.grad(expected.sum() / 2, m)
    torch.testing.assert_close(metric.apply(m), gradient)
    torch.testing.assert_close(metric.apply(torch.ones_like(m)), torch.zeros_like(m))


def test_dense_small_problem_and_l2_energy_degeneracy(monkeypatch):
    mod = module(monkeypatch)
    s = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))
    context = SimpleNamespace(
        point_mass=1.0,
        denominator=torch.ones(3, dtype=torch.float64),
        tangent_gradient=lambda m: m @ s,
        response_operator=SimpleNamespace(forward_pair=lambda pair: pair @ (s / 2)),
    )
    geometry = SimpleNamespace(
        hx=1.0,
        hy=1.0,
        x_edges=torch.tensor([[0, 1], [1, 2]]),
        y_edges=torch.empty((0, 2), dtype=torch.int64),
    )
    mismatch = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float64)
    for objective in ("l2", "energy"):
        metric = mod.ResponseMetric(geometry, torch.ones_like(mismatch), objective)
        result = mod.metric_subspace(context, mismatch, metric, 4)
        residual = mismatch + result.deltas[-1] @ s
        assert metric.inner(residual, residual).item() < 1e-20
        assert not result.direction_active[-1].any()


def test_streamed_energy_diagonal_and_relative_damping(monkeypatch):
    mod = module(monkeypatch)
    h = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [2.0, 1.0, 1.0]], dtype=torch.float64
    )
    geometry = SimpleNamespace(
        hx=1.0,
        hy=1.0,
        x_edges=torch.tensor([[0, 1], [1, 2]]),
        y_edges=torch.empty((0, 2), dtype=torch.int64),
    )
    metric = mod.ResponseMetric(
        geometry, torch.ones((1, 3), dtype=torch.float64), "energy"
    )
    context = SimpleNamespace(
        denominator=torch.ones(3, dtype=torch.float64),
        relative_lambda=0.01,
        denominator_relative_eps=1e-12,
        response_operator=SimpleNamespace(forward_pair=lambda pair: pair @ h.T),
    )
    actual, denominator = mod.energy_separable_diagonal(context, metric, chunk_size=2)
    q = metric.apply(torch.eye(3, dtype=torch.float64)).T
    expected = 2 * torch.diagonal(h.T @ q @ h)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(denominator, expected + (0.01 + 1e-12) * expected.mean())
