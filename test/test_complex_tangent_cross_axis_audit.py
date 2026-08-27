from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from greenonet.complex_axial_response_operator import (
    AxialResponseBlock,
    FrozenAxialResponseOperator,
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_tangent_cross_axis_audit import (
    MatrixFreeCrossAxisCouplingAnalyzer,
    TangentCrossAxisAuditRequest,
    run_tangent_cross_axis_audit,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from test.test_complex_projection_response_audit import (
    _patch_static_export,
    _write_fixture,
)


def _operator(axis: str, matrix: torch.Tensor) -> FrozenAxialResponseOperator:
    return FrozenAxialResponseOperator(
        axis=axis,  # type: ignore[arg-type]
        point_count=matrix.shape[0],
        blocks=(
            AxialResponseBlock(
                valid_indices=torch.arange(matrix.shape[0], dtype=torch.long),
                matrix=matrix,
            ),
        ),
    )


def _explicit_context() -> tuple[
    SymmetricTangentGreenResponseContext,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    dtype = torch.float64
    x_matrix = torch.tensor(
        [[1.0, 0.2, -0.1], [0.4, 0.9, 0.3], [0.1, -0.2, 0.8]],
        dtype=dtype,
    )
    y_matrix = torch.tensor(
        [[0.7, -0.1, 0.3], [0.2, 1.1, -0.2], [-0.3, 0.4, 0.9]],
        dtype=dtype,
    )
    mass = 0.25
    context = SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=FrozenBidirectionalResponseOperator(
            x=_operator("x", x_matrix),
            y=_operator("y", y_matrix),
        ),
        point_mass=mass,
        config={
            "eta": 0.01,
            "eta_strategy": "closed_loop_exact_line_search",
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
        },
    )
    return context, x_matrix, y_matrix, mass


def test_matrix_free_cross_axis_actions_match_dense_reference() -> None:
    context, x_matrix, y_matrix, mass = _explicit_context()
    analyzer = MatrixFreeCrossAxisCouplingAnalyzer(
        context,
        metric_eps=1.0e-30,
        confidence_z=1.96,
    )
    values = torch.tensor(
        [[1.2, -0.4, 0.8], [-0.3, 1.5, 0.2]],
        dtype=torch.float64,
    )
    actions = analyzer.actions(values)
    cross = mass * x_matrix.T @ y_matrix
    cross_transpose = cross.T
    self_hessian = mass * (x_matrix.T @ x_matrix + y_matrix.T @ y_matrix)
    tangent_hessian = self_hessian + cross + cross_transpose

    torch.testing.assert_close(actions.cross, values @ cross.T)
    torch.testing.assert_close(actions.cross_transpose, values @ cross_transpose.T)
    torch.testing.assert_close(
        actions.symmetric_cross,
        values @ (cross + cross_transpose).T,
    )
    torch.testing.assert_close(actions.self_hessian, values @ self_hessian.T)
    torch.testing.assert_close(actions.tangent_hessian, values @ tangent_hessian.T)
    torch.testing.assert_close(
        context.cross_axis_inner_product,
        torch.diagonal(cross),
    )


def test_rademacher_frobenius_and_actual_direction_metrics_match_dense_math() -> None:
    context, x_matrix, y_matrix, mass = _explicit_context()
    analyzer = MatrixFreeCrossAxisCouplingAnalyzer(
        context,
        metric_eps=1.0e-30,
        confidence_z=1.96,
    )
    estimates, rows = analyzer.estimate_frobenius(
        probe_count=4096,
        probe_batch_size=128,
        probe_seed=17,
    )
    cross = mass * x_matrix.T @ y_matrix
    cross_frobenius_squared = float(cross.square().sum().item())
    diagonal_energy = float(torch.diagonal(cross).square().sum().item())
    expected_off_fraction = 1.0 - diagonal_energy / cross_frobenius_squared

    assert len(rows) == 4096
    assert estimates["cross"]["frobenius_norm_squared_estimate"] == pytest.approx(
        cross_frobenius_squared,
        rel=0.03,
    )
    assert estimates["cross"]["off_diagonal_fraction"] == pytest.approx(
        expected_off_fraction,
        abs=0.03,
    )

    gradients = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.2, 0.4, 0.8]],
        dtype=torch.float64,
    )
    denominators = context.separable_denominator
    sample_rows = analyzer.sample_direction_rows(
        gradients=gradients,
        denominators=denominators,
        sample_indices=torch.tensor([4, 9], dtype=torch.long),
    )
    direction = gradients / denominators.unsqueeze(0)
    self_hessian = mass * (x_matrix.T @ x_matrix + y_matrix.T @ y_matrix)
    symmetric_cross = cross + cross.T
    tangent_hessian = self_hessian + symmetric_cross
    expected_cross = direction @ symmetric_cross.T
    expected_tangent = direction @ tangent_hessian.T
    expected_ratio = torch.linalg.vector_norm(
        expected_cross, dim=1
    ) / torch.linalg.vector_norm(expected_tangent, dim=1)

    assert [row["sample_id"] for row in sample_rows] == [4, 9]
    assert sample_rows[0]["cross_to_tangent_action_ratio"] == pytest.approx(
        float(expected_ratio[0].item())
    )
    assert sample_rows[1]["cross_to_tangent_action_ratio"] == pytest.approx(
        float(expected_ratio[1].item())
    )


def test_cross_axis_audit_request_validation(tmp_path: Path) -> None:
    common = {
        "config": tmp_path / "config.json",
        "coupling_checkpoint": tmp_path / "coupling.safetensors",
        "green_checkpoint": tmp_path / "green.safetensors",
        "outdir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="probe_count"):
        TangentCrossAxisAuditRequest(**common, probe_count=0)
    with pytest.raises(ValueError, match="probe_seed"):
        TangentCrossAxisAuditRequest(**common, probe_seed=-1)
    with pytest.raises(ValueError, match="preconditioner_variant"):
        TangentCrossAxisAuditRequest(
            **common,
            preconditioner_variant="invalid",  # type: ignore[arg-type]
        )


def test_checkpoint_backed_cross_axis_audit_writes_reference_free_outputs(
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
    outdir = tmp_path / "cross_axis_audit"

    summary = run_tangent_cross_axis_audit(
        TangentCrossAxisAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            batch_size=1,
            probe_count=16,
            probe_batch_size=4,
            posthoc_tangent_override=True,
        )
    )

    assert summary["diagnostic"] == ("matrix_free_cross_axis_gram_and_direction_audit")
    assert summary["sample_count"] == 1
    assert summary["matrix_policy"]["global_matrix_materialized"] is False
    assert summary["matrix_policy"]["global_gram_materialized"] is False
    assert summary["matrix_policy"]["global_linear_solve"] is False
    assert summary["reference_policy"] == {
        "sol_used": False,
        "phi_used": False,
        "psi_used": False,
        "rhs_used_to_form_symmetric_proposal": True,
    }
    assert summary["projection_provenance"]["training_mode"] == "physical_symmetric"
    assert summary["projection_provenance"]["explicit_posthoc_override"] is True
    assert summary["operator_global"]["probe_contract"]["count"] == 16
    assert (outdir / "metrics" / "probe_cross_axis_coupling.csv").is_file()
    assert (outdir / "metrics" / "per_sample_cross_axis_coupling.csv").is_file()
    assert (outdir / "data" / "cross_axis_operator_fields.npz").is_file()
    assert (outdir / "figures" / "cross_axis_coupling_audit.html").is_file()
    assert (outdir / "diagnosis_report.md").is_file()


def test_cross_axis_cli_module_imports_and_exposes_runner() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "cli"
        / "audit_tangent_cross_axis_coupling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_tangent_cross_axis_coupling_cli",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(module.AuditTangentCrossAxisCouplingCLI)
