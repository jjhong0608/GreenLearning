"""Fixed-space refitting agrees with a tiny independent dense least squares."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import torch


def test_fixed_subspace_minimization(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "cli"))
    mod = importlib.import_module("audit_fixed_tangent_subspace")
    torch.manual_seed(24)
    dtype = torch.float64
    geometry = SimpleNamespace(
        hx=1.0,
        hy=1.0,
        x_edges=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        y_edges=torch.empty((0, 2), dtype=torch.int64),
    )
    a = torch.ones((1, 4), dtype=dtype)
    sources = torch.randn((2, 1, 4), dtype=dtype)
    s = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype))
    responses = sources @ s.T
    mismatch = torch.randn((1, 4), dtype=dtype)
    for objective in ("l2", "energy"):
        metric = mod.ResponseMetric(geometry, a, objective)
        result = mod.refit_subspace(mismatch, sources, responses, metric)
        for k in (1, 2):
            v = responses[:k, 0].T
            rhs = mismatch[0]
            if objective == "energy":
                v, rhs = v[1:] - v[:-1], rhs[1:] - rhs[:-1]
            coefficients = torch.linalg.lstsq(v, -rhs).solution
            expected = sources[:k, 0].T @ coefficients
            torch.testing.assert_close(
                result.deltas[k - 1, 0], expected, rtol=1e-10, atol=1e-12
            )
