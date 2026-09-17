from dataclasses import asdict, replace
import json

import pytest
import torch

from cli.audit_normalized_tangent import normalized_step
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig as Config
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
    matrix_free_krylov_subspace_step,
)
from test.test_complex_tangent_subspace_audit import _context


def response_config(k=2, variant="separable", **kwargs):
    return Config(
        subspace_dimension=k,
        max_subspace_dimension=max(k, 8),
        direction_normalization="response",
        eta_strategy="closed_loop_exact_line_search",
        eta_cap_enabled=False,
        preconditioner_variant=variant,
        **kwargs,
    )


def test_defaults_and_roundtrip():
    assert Config().direction_normalization == "legacy"
    config = response_config(1, "identity")
    assert asdict(Config.from_raw(asdict(config))) == asdict(config)


@pytest.mark.parametrize("value", [True, "1e-12", 0, -1, 1, float("nan"), float("inf")])
def test_invalid_independence(value):
    with pytest.raises((TypeError, ValueError)):
        response_config(direction_independence_relative_eps=value)


def test_invalid_modes():
    with pytest.raises(ValueError, match="identity"):
        Config(preconditioner_variant="identity")
    with pytest.raises(ValueError, match="uncapped"):
        Config(direction_normalization="response")
    with pytest.raises(ValueError, match="uncapped"):
        Config(
            direction_normalization="response",
            eta_strategy="closed_loop_exact_line_search",
        )


@pytest.mark.parametrize("k", [1, 2, 4, 10, 64])
@pytest.mark.parametrize("variant", ["identity", "separable"])
def test_prototype_equivalence_and_backward(k, variant):
    original = _context()
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=original.response_operator,
        point_mass=original.point_mass,
        config=response_config(k, variant),
    )
    if variant == "identity":
        assert torch.equal(context.denominator, torch.ones_like(context.denominator))
    m = torch.tensor(
        [[1.2, -0.4, 0.8], [0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    g = context.tangent_gradient(m)
    expected = normalized_step(
        context=context, mismatch=m, gradient=g, max_dimension=k, relative_eps=1e-12
    )
    actual = matrix_free_krylov_subspace_step(
        context=context, mismatch=m, gradient=g, max_dimension=k, relative_eps=1e-12
    )
    torch.testing.assert_close(actual.deltas, expected.deltas, rtol=1e-12, atol=1e-12)
    assert torch.equal(actual.direction_active, expected.direction_active)
    step = context.tangent_step(mismatch=m, gradient=g)
    torch.testing.assert_close(step.delta, actual.final_delta)
    assert step.subspace_result is not None
    assert step.eta_star is None
    assert torch.isfinite(torch.autograd.grad(step.delta.square().sum(), m)[0]).all()


def test_legacy_dispatch_is_unchanged():
    context = _context()
    m = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64)
    g = context.tangent_gradient(m)
    args = dict(mismatch=m, gradient=g, max_dimension=2, relative_eps=1e-12)
    a = matrix_free_krylov_subspace_step(context=context, **args)
    b = matrix_free_krylov_subspace_step(
        context=replace(context, direction_normalization="legacy"), **args
    )
    assert torch.equal(a.deltas, b.deltas)


@pytest.mark.parametrize(
    "threshold,cap,expected", [(0.99, False, 2), (0.01, False, 1), (0.01, True, None)]
)
def test_auto_k_validates_resolved_normalized_config(
    tmp_path, threshold, cap, expected
):
    from greenonet.complex_geometry import load_complex_geometry
    from greenonet.complex_tangent_geometry_selection import (
        GeometryTangentDimensionResolver,
    )
    from greenonet.config import BalanceProjectionConfig, CouplingModelConfig
    from test.complex_fixtures import write_geometry_npz

    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    config = replace(
        response_config(2),
        eta_cap_enabled=cap,
        geometry_k_selection={
            "enabled": True,
            "global_reach_threshold": threshold,
            "pointwise_tail_reach_threshold": threshold,
        },
    )
    model = CouplingModelConfig(
        balance_projection=BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response=config,
        )
    )

    def resolve():
        return GeometryTangentDimensionResolver.resolve(
            model_config=model,
            geometry=load_complex_geometry(geometry_path),
            geometry_path=geometry_path,
        )

    if expected is None:
        with pytest.raises(ValueError, match="uncapped"):
            resolve()
    else:
        result = (
            resolve().model_config.balance_projection.symmetric_tangent_green_response
        )
        assert result.subspace_dimension == expected
        assert result.direction_normalization == "response"


def test_nearly_dependent_production_direction_is_inactive():
    context = replace(_context(), direction_normalization="response")
    m = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    candidates = iter((m, m + torch.tensor([[0.0, 1e-10, 0.0]], dtype=m.dtype)))
    result = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=m,
        gradient=context.tangent_gradient(m),
        max_dimension=2,
        relative_eps=1e-12,
        inverse_preconditioner=lambda _: next(candidates),
    )
    assert result.direction_active[:, 0].tolist() == [True, False]
    assert result.coefficients[1, 0] == 0


@pytest.mark.parametrize("scale", [1e-100, 1e100, -3.0])
def test_production_scaling_and_gradcheck(scale):
    context = replace(_context(), direction_normalization="response")
    m = torch.tensor([[1.2, -0.4, 0.8]], dtype=torch.float64, requires_grad=True)

    def evaluate(value, factor):
        return matrix_free_krylov_subspace_step(
            context=context,
            mismatch=value,
            gradient=context.tangent_gradient(value),
            max_dimension=2,
            relative_eps=1e-12,
            inverse_preconditioner=lambda g: factor * g / context.denominator,
        )

    original, scaled = evaluate(m, 1.0), evaluate(m, scale)
    torch.testing.assert_close(scaled.deltas, original.deltas, rtol=1e-10, atol=1e-12)
    assert torch.equal(scaled.direction_active, original.direction_active)
    assert torch.autograd.gradcheck(
        lambda value: evaluate(value, scale).final_delta, (m,)
    )


def test_production_null_response_and_separate_thresholds():
    from greenonet.complex_axial_response_operator import (
        FrozenBidirectionalResponseOperator,
    )
    from test.test_complex_tangent_subspace_audit import _response_operator

    matrix = torch.eye(3, dtype=torch.float64)
    context = replace(
        _context(),
        direction_normalization="response",
        response_operator=FrozenBidirectionalResponseOperator(
            x=_response_operator(axis="x", matrix=matrix),
            y=_response_operator(axis="y", matrix=-matrix),
        ),
    )
    m = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    result = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=m,
        gradient=context.tangent_gradient(m),
        max_dimension=4,
        relative_eps=1e-12,
    )
    assert not result.direction_active.any()
    assert torch.count_nonzero(result.deltas) == 0
    assert torch.isfinite(torch.autograd.grad(result.final_delta.sum(), m)[0]).all()
    ordinary = replace(_context(), direction_normalization="response")
    g = ordinary.tangent_gradient(m)
    a = matrix_free_krylov_subspace_step(
        context=ordinary, mismatch=m, gradient=g, max_dimension=1, relative_eps=1e-12
    )
    b = matrix_free_krylov_subspace_step(
        context=ordinary, mismatch=m, gradient=g, max_dimension=1, relative_eps=0.1
    )
    assert torch.equal(a.direction_active, b.direction_active)
    torch.testing.assert_close(b.coefficients, a.coefficients * (1 + 1e-12) / 1.1)


@pytest.mark.parametrize(
    "k,variant", [(1, "identity"), (2, "identity"), (1, "separable"), (4, "separable")]
)
def test_normalized_artifact_export(tmp_path, monkeypatch, k, variant):
    import numpy as np
    from greenonet.complex_coupling_artifacts import export_complex_coupling_artifacts
    from greenonet.coupling_artifacts import CouplingArtifactRequest
    from test.test_complex_projection_response_audit import (
        _write_fixture,
        _patch_static_export,
    )

    _patch_static_export(monkeypatch)
    config, checkpoint, green, geometry, data, coefficients = _write_fixture(tmp_path)
    payload = json.loads(config.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": asdict(response_config(k, variant)),
    }
    for name in ("post_line_search_stationarity", "response_trust"):
        payload["coupling_training"][name] = {"enabled": True, "weight": 1e-4}
    config.write_text(json.dumps(payload))
    out = tmp_path / "export"
    result = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config,
            coupling_checkpoint=checkpoint,
            green_checkpoint=green,
            outdir=out,
            device="cpu",
            selected_samples=(0,),
        )
    )
    encoded = json.dumps(result)
    assert '"direction_normalization": "response"' in encoded
    assert "unit_response" in encoded
    assert "eta_star_statistics" not in encoded
    with np.load(out / "data/symmetric_tangent_green_response_fields.npz") as arrays:
        assert arrays["direction_normalization"].item() == "response"
        if variant == "identity":
            assert np.all(arrays["denominator"] == 1)
