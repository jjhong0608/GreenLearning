from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cli.audit_normalized_tangent import NormalizedAudit, normalized_step
from cli.audit_uniform_tangent import uniform_context
from greenonet.complex_axial_response_operator import (
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_tangent_projection import SymmetricTangentGreenResponseContext
from test.test_complex_tangent_subspace_audit import _context, _response_operator
from test.test_complex_projection_response_audit import _write_fixture
from greenonet.complex_tangent_subspace_audit import TangentSubspaceAuditRequest


def evaluate(context, mismatch, dimension=2, scale=1.0):
    return normalized_step(
        context=context,
        mismatch=mismatch,
        gradient=context.tangent_gradient(mismatch),
        max_dimension=dimension,
        relative_eps=1e-12,
        direction_scale=scale,
    )


@pytest.mark.parametrize("scale", [1e-100, 1e100, -3.0])
def test_source_scaling_preserves_correction_and_activity(scale):
    context = _context()
    m = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64)
    baseline, scaled = evaluate(context, m), evaluate(context, m, scale=scale)
    torch.testing.assert_close(scaled.deltas, baseline.deltas, atol=1e-12, rtol=1e-10)
    assert torch.equal(scaled.direction_active, baseline.direction_active)


def test_small_identity_directions_and_pair_relation():
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=_context().response_operator,
        point_mass=1e-8,
        config={},
    )
    m = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64)
    identity = replace(context, denominator=torch.ones_like(context.denominator))
    result = evaluate(identity, m)
    scaled = evaluate(uniform_context(context), m)
    assert result.direction_active.all()
    torch.testing.assert_close(result.deltas, scaled.deltas, atol=1e-12, rtol=1e-10)
    for j in range(2):
        pair = context.response_operator.forward_pair(
            torch.stack((result.directions[j], result.directions[j]), 1)
        )
        torch.testing.assert_close(
            pair, result.directional_responses[j], rtol=1e-10, atol=1e-9
        )
    assert result.costs[1] <= result.costs[0]


def test_duplicate_and_zero_response_directions():
    matrix = torch.eye(3, dtype=torch.float64)
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=FrozenBidirectionalResponseOperator(
            x=_response_operator(axis="x", matrix=matrix),
            y=_response_operator(axis="y", matrix=matrix),
        ),
        point_mass=1.0,
        config={},
    )
    m = torch.tensor([[1.0, 0, 0], [0, 0, 0]], dtype=torch.float64, requires_grad=True)
    result = evaluate(context, m, dimension=4)
    assert result.direction_active[:, 0].tolist() == [True, False, False, False]
    assert not result.direction_active[:, 1].any()
    derivative = torch.autograd.grad(result.deltas.square().sum(), m)[0]
    assert torch.isfinite(derivative).all()
    zero_operator = FrozenBidirectionalResponseOperator(
        x=_response_operator(axis="x", matrix=matrix),
        y=_response_operator(axis="y", matrix=-matrix),
    )
    null_result = evaluate(replace(context, response_operator=zero_operator), m)
    assert not null_result.direction_active.any()
    assert torch.count_nonzero(null_result.deltas) == 0


@pytest.mark.parametrize("identity", [False, True])
def test_backward_gradcheck(identity):
    context = _context()
    if identity:
        context = replace(context, denominator=torch.ones_like(context.denominator))
    m = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: evaluate(context, x).deltas[-1], (m,), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_nearly_duplicate_candidate_is_rejected():
    matrix = torch.eye(3, dtype=torch.float64)
    operator = FrozenBidirectionalResponseOperator(
        x=_response_operator(axis="x", matrix=matrix),
        y=_response_operator(axis="y", matrix=matrix),
    )
    first = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)
    repeated = torch.tensor([[1.0, 1e-10, 0]], dtype=torch.float64)
    # Deliberately inject an almost repeated second candidate to test MGS,
    # independently of the PDE residual recurrence.
    context = SimpleNamespace(
        point_mass=torch.tensor(1.0, dtype=torch.float64),
        denominator=torch.ones(3, dtype=torch.float64),
        response_operator=operator,
        tangent_gradient=lambda _: repeated,
    )
    result = normalized_step(
        context=context,
        mismatch=first,
        gradient=first,
        max_dimension=2,
        relative_eps=1e-12,
    )
    assert result.direction_active[:, 0].tolist() == [True, False]


def test_frozen_fixture_uses_prototype_without_changing_production(tmp_path):
    import greenonet.complex_tangent_subspace_audit as module

    original = module.matrix_free_krylov_subspace_audit
    config, coupling, green, geometry, test, coefficients = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "eta_strategy": "closed_loop_exact_line_search"
        },
    }
    config.write_text(json.dumps(payload))
    checkpoint = coupling.read_bytes()
    audit = NormalizedAudit(
        TangentSubspaceAuditRequest(
            config=config,
            coupling_checkpoint=coupling,
            green_checkpoint=green,
            geometry=geometry,
            test_path=test,
            coefficients=coefficients,
            outdir=tmp_path / "audit",
            device="cpu",
            max_subspace_dimension=4,
        )
    )
    result = audit.run()
    assert result["maximum_balance_error"] < 1e-12
    assert len(audit.prototype_checks) == 3
    assert all(row["all_backward_finite"] for row in audit.prototype_checks)
    assert module.matrix_free_krylov_subspace_audit is original
    assert coupling.read_bytes() == checkpoint
