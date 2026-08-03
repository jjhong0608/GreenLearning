from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_symmetric_tangent_audit import (
    SymmetricTangentAuditRequest,
    run_symmetric_tangent_response_audit,
)
from test.test_complex_projection_response_audit import (
    _patch_static_export,
    _write_fixture,
)


def _operator(
    axis: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> FrozenAxialResponseOperator:
    return FrozenAxialResponseOperator(
        axis=axis,  # type: ignore[arg-type]
        point_count=3,
        blocks=(
            AxialResponseBlock(
                valid_indices=torch.tensor([0, 2], dtype=torch.long),
                matrix=first,
            ),
            AxialResponseBlock(
                valid_indices=torch.tensor([1], dtype=torch.long),
                matrix=second,
            ),
        ),
    )


def test_frozen_axial_response_forward_and_adjoint_are_dual() -> None:
    dtype = torch.float64
    operator = _operator(
        "x",
        torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=dtype),
        torch.tensor([[4.0]], dtype=dtype),
    )
    source = torch.tensor([[1.5, -2.0, 0.25]], dtype=dtype)
    dual = torch.tensor([[0.5, 3.0, -1.5]], dtype=dtype)

    response = operator.forward(source)
    adjoint = operator.adjoint(dual)

    assert torch.sum(response * dual) == pytest.approx(
        torch.sum(source * adjoint).item()
    )


def test_tangent_gradient_matches_finite_difference() -> None:
    dtype = torch.float64
    x_operator = _operator(
        "x",
        torch.tensor([[1.0, 0.2], [-0.4, 1.5]], dtype=dtype),
        torch.tensor([[0.8]], dtype=dtype),
    )
    y_operator = _operator(
        "y",
        torch.tensor([[0.7, -0.1], [0.3, 1.2]], dtype=dtype),
        torch.tensor([[1.1]], dtype=dtype),
    )
    operator = FrozenBidirectionalResponseOperator(x_operator, y_operator)
    p_tilde = torch.tensor([[0.2, -0.5, 1.0]], dtype=dtype)
    q_tilde = torch.tensor([[0.8, 0.25, -0.2]], dtype=dtype)
    direction = torch.tensor([[0.4, -0.7, 0.3]], dtype=dtype)
    point_mass = torch.tensor(0.125, dtype=dtype)
    mismatch = x_operator.forward(p_tilde) - y_operator.forward(q_tilde)
    gradient = operator.tangent_gradient(mismatch, point_mass=point_mass)

    def objective(delta: torch.Tensor) -> torch.Tensor:
        updated = mismatch + x_operator.forward(delta) + y_operator.forward(delta)
        return 0.5 * point_mass * updated.square().sum()

    step = 1.0e-6
    finite_difference = (objective(step * direction) - objective(-step * direction)) / (
        2.0 * step
    )
    analytic = torch.sum(gradient * direction)

    assert finite_difference.item() == pytest.approx(
        analytic.item(),
        rel=1.0e-8,
        abs=1.0e-10,
    )


def test_request_rejects_invalid_eta_lambda_and_duplicates(tmp_path: Path) -> None:
    common = {
        "config": tmp_path / "config.json",
        "coupling_checkpoint": tmp_path / "coupling.safetensors",
        "green_checkpoint": tmp_path / "green.safetensors",
        "outdir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="etas values must be greater than"):
        SymmetricTangentAuditRequest(**common, etas=(0.0,))
    with pytest.raises(ValueError, match="relative_lambdas values must be at least"):
        SymmetricTangentAuditRequest(**common, relative_lambdas=(-1.0,))
    with pytest.raises(ValueError, match="must not contain duplicate"):
        SymmetricTangentAuditRequest(**common, etas=(0.1, 0.1))


def test_checkpoint_backed_tangent_audit_writes_balanced_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
        coefficient_path,
    ) = _write_fixture(tmp_path)
    outdir = tmp_path / "tangent_audit"
    summary = run_symmetric_tangent_response_audit(
        SymmetricTangentAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            etas=(0.05, 0.1),
            relative_lambdas=(0.0, 0.1),
            selected_samples=(0,),
            batch_size=1,
        )
    )

    assert summary["diagnostic"] == (
        "symmetric_tangent_response_gradient_posthoc_audit"
    )
    assert summary["matrix_policy"]["global_matrix_materialized"] is False
    assert summary["matrix_policy"]["global_matrix_solve"] is False
    assert summary["matrix_policy"]["response_context_build_count"] == 1
    assert summary["matrix_policy"]["response_operator_build_count"] == 1
    assert summary["matrix_policy"]["operator_production_equivalence_max_abs"] < 1.0e-10
    assert summary["reference_policy"]["sol_and_flux_used_for_update"] is False
    assert len(summary["aggregate_metrics"]) == 6
    assert (outdir / "metrics" / "per_sample_tangent_sweep.csv").is_file()
    assert (outdir / "figures" / "aggregate" / "eta_lambda_sweep.json").is_file()
    assert (outdir / "diagnosis_report.md").is_file()

    raw_path = outdir / "data" / "selected_symmetric_tangent_audit.npz"
    assert raw_path.is_file()
    with np.load(raw_path, allow_pickle=False) as raw:
        assert {
            "tangent_gradient",
            "tangent_preconditioner_base",
            "tangent_delta",
            "candidate_physical",
            "candidate_solution",
            "candidate_prediction",
        }.issubset(raw.files)
        expected_rhs = np.broadcast_to(
            raw["rhs"][None, :, None, :],
            raw["candidate_physical"].shape,
        )
        np.testing.assert_allclose(
            raw["candidate_physical"].sum(axis=2, keepdims=True),
            expected_rhs[:, :, :1],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_checkpoint_backed_tangent_audit_accepts_symmetric_trained_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
        coefficient_path,
    ) = _write_fixture(tmp_path)
    payload = json.loads(config_path.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_path.write_text(json.dumps(payload))

    outdir = tmp_path / "symmetric_trained_tangent_audit"
    summary = run_symmetric_tangent_response_audit(
        SymmetricTangentAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            etas=(0.05, 0.1),
            relative_lambdas=(0.0, 0.1),
            selected_samples=(0,),
            batch_size=1,
        )
    )

    assert summary["configured_projection_mode"] == "physical_symmetric"
    assert summary["automated_findings"]["configured_projection_method"] == (
        "symmetric"
    )
    assert "configured_column" not in summary["aggregate_metrics"]
    assert len(summary["aggregate_metrics"]) == 5


def test_cli_module_imports_and_exposes_runner() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "cli"
        / "audit_symmetric_tangent_response.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_symmetric_tangent_response_cli",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(module.AuditSymmetricTangentResponseCLI)
