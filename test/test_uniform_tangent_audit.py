from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from cli.audit_uniform_tangent import UniformTangentAudit, uniform_context
from greenonet.complex_tangent_subspace_audit import TangentSubspaceAuditRequest
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
    matrix_free_krylov_subspace_step,
)
from test.test_complex_projection_response_audit import _write_fixture
from test.test_complex_tangent_subspace_audit import _context


def test_uniform_changes_only_active_denominator() -> None:
    context = _context()
    uniform = uniform_context(context)
    assert uniform.response_operator is context.response_operator
    torch.testing.assert_close(
        uniform.denominator,
        context.separable_denominator.mean().expand_as(context.denominator),
    )
    assert uniform.denominator.unique().numel() == 1
    assert context.denominator.unique().numel() > 1
    with pytest.raises(ValueError, match="finite positive"):
        uniform_context(replace(context, separable_denominator=-context.denominator))


def test_identity_scaling_guard_and_small_epsilon_control() -> None:
    base = _context()
    base = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=base.response_operator,
        point_mass=1e-8,
        config={"eta_strategy": "closed_loop_exact_line_search"},
    )
    mismatch = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64)
    gradient = base.tangent_gradient(mismatch)
    identity = replace(base, denominator=torch.ones_like(base.denominator))

    def evaluate(context, eps):
        return matrix_free_krylov_subspace_step(
            context=context,
            mismatch=mismatch,
            gradient=gradient,
            max_dimension=2,
            relative_eps=eps,
            monotonicity_relative_tol=1e-10,
        )

    guarded = evaluate(identity, 1e-12)
    resolved = evaluate(identity, 1e-28)
    scaled = evaluate(uniform_context(base), 1e-28)
    assert not guarded.direction_active[1].any()
    assert resolved.direction_active.all()
    torch.testing.assert_close(resolved.deltas, scaled.deltas, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("include_identity", [False, True])
def test_csv_frozen_uniform_audit(tmp_path: Path, include_identity: bool) -> None:
    config, coupling, green, geometry, test, coefficients = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "subspace_dimension": 4,
            "eta_strategy": "closed_loop_exact_line_search",
        },
    }
    config.write_text(json.dumps(payload))
    before = coupling.read_bytes()
    request = TangentSubspaceAuditRequest(
        config=config,
        coupling_checkpoint=coupling,
        green_checkpoint=green,
        geometry=geometry,
        test_path=test,
        coefficients=coefficients,
        outdir=tmp_path / "audit",
        device="cpu",
        batch_size=1,
        max_subspace_dimension=4,
    )
    audit = UniformTangentAudit(request)
    audit.include_identity = include_identity
    summary = audit.run()
    assert coupling.read_bytes() == before
    assert summary["maximum_balance_error"] < 1e-12
    assert summary["operator_equivalence_max_abs"] < 1e-10
    assert summary["context_build_count"] == 1
    with (request.outdir / "per_sample.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    expected = {"uniform", "separable"} | ({"identity"} if include_identity else set())
    assert {row["preconditioner"] for row in rows} == expected
    for name in expected:
        group = {row["method_id"]: row for row in rows if row["preconditioner"] == name}
        if name == "identity":
            assert float(group["k2_unconstrained"]["active_denominator_min"]) == 1.0
            assert float(group["k2_unconstrained"]["active_denominator_max"]) == 1.0
        j1 = float(group["k1_uncapped"]["response_mismatch_cost"])
        j2 = float(group["k2_unconstrained"]["response_mismatch_cost"])
        assert j2 <= j1 * (1 + 1e-10)
    assert not list(request.outdir.rglob("*.html"))
    assert not list(request.outdir.rglob("*.safetensors"))
    with pytest.raises(FileExistsError):
        UniformTangentAudit(request).run()
