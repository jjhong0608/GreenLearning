"""Contracts for the isolated frozen initialization audit."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def load_audit():
    path = Path(__file__).parents[1] / "cli/audit_equal_split_initialization.py"
    spec = importlib.util.spec_from_file_location("equal_split_audit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_split_is_physical_and_balanced():
    module = load_audit()
    rhs = torch.tensor([[2.0, -3.0, 0.0]], dtype=torch.float64)
    pair = module.equal_split(rhs)
    torch.testing.assert_close(pair.sum(1), rhs, rtol=0, atol=0)
    torch.testing.assert_close(pair[:, 0], pair[:, 1], rtol=0, atol=0)


def test_equal_split_preparation_bypasses_network():
    module = load_audit()
    audit = object.__new__(module.InitializationAudit)
    # A missing model is intentional: this route must not call CouplingNet.
    audit.response_operator = SimpleNamespace(forward_pair=lambda pair: 2 * pair)
    audit.tangent_context = SimpleNamespace(tangent_gradient=lambda m: 3 * m)
    rhs = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
    prepared = audit.prepare(SimpleNamespace(rhs_valid=rhs), "equal_split")
    torch.testing.assert_close(prepared.symmetric_physical.sum(1), rhs)
    torch.testing.assert_close(prepared.mismatch, torch.zeros_like(rhs))
    torch.testing.assert_close(prepared.gradient, torch.zeros_like(rhs))


def test_zero_correction_and_prefix_selection():
    module = load_audit()
    audit = object.__new__(module.InitializationAudit)
    audit.response_operator = SimpleNamespace(forward_pair=lambda pair: pair)
    audit._cross_axis_reconstructor = SimpleNamespace(reconstruct=lambda **kw: kw)
    rhs = torch.tensor([[4.0, -2.0]], dtype=torch.float64)
    prepared = SimpleNamespace(
        gradient=torch.ones_like(rhs), symmetric_physical=module.equal_split(rhs)
    )
    batch = SimpleNamespace(rhs_valid=rhs, geometry=None, weak_context=None)
    delta, solution, _ = audit.candidate(batch, prepared, None, 0)
    torch.testing.assert_close(delta, torch.zeros_like(rhs))
    torch.testing.assert_close(solution, module.equal_split(rhs))
    result = SimpleNamespace(deltas=torch.stack((rhs / 8, rhs / 4)))
    delta, solution, _ = audit.candidate(batch, prepared, result, 2)
    torch.testing.assert_close(delta, rhs / 4)
    torch.testing.assert_close(solution.sum(1), rhs)
