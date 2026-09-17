import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "cli"))
    return importlib.import_module("audit_mixed_energy")


def test_edge_features_equal_existing_metric(monkeypatch):
    mod = load(monkeypatch)
    old = importlib.import_module("audit_tangent_energy_objective")
    geometry = SimpleNamespace(
        hx=0.5,
        hy=0.25,
        x_edges=torch.tensor([[0, 1], [2, 3]]),
        y_edges=torch.tensor([[0, 2], [1, 3]]),
    )
    a = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]], dtype=torch.float64)
    metric = mod.EnergyFeatures(geometry, a)
    u = torch.randn(5, 2, 4, dtype=torch.float64)
    original = old.ResponseMetric(geometry, a, "energy")
    expected = torch.stack([original.inner(v, v) for v in u])
    torch.testing.assert_close(metric.apply(u).square().sum(-1), expected)
    assert torch.count_nonzero(metric.apply(torch.ones_like(u))) == 0


@pytest.mark.parametrize("weight", [0.0, 0.01, 0.1, 1.0, None])
def test_mixed_refit_matches_independent_dense_solve(monkeypatch, weight):
    mod = load(monkeypatch)
    torch.manual_seed(18)
    z = torch.randn(3, 2, 7, dtype=torch.float64)
    c, e = (
        torch.randn(3, 2, 9, dtype=torch.float64),
        torch.randn(3, 2, 11, dtype=torch.float64),
    )
    c0, e0 = (
        torch.randn(2, 9, dtype=torch.float64),
        torch.randn(2, 11, dtype=torch.float64),
    )
    target, design = mod.features(c0, e0, weight), mod.features(c, e, weight)
    result = mod.refit_subspace(target, z, design, mod.EuclideanMetric())
    for k in range(1, 4):
        for b in range(2):
            alpha = torch.linalg.lstsq(design[:k, b].T, -target[b]).solution
            torch.testing.assert_close(
                result.deltas[k - 1, b], alpha @ z[:k, b], rtol=1e-10, atol=1e-11
            )
    torch.testing.assert_close(
        target.square().sum(-1),
        mod.objective(c0.square().sum(-1), e0.square().sum(-1), weight),
    )
