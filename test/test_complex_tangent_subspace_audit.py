from __future__ import annotations

import csv
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
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContext,
)
from greenonet.complex_tangent_preconditioner_audit import (
    TangentPreconditionerAuditRequest,
    run_tangent_preconditioner_audit,
)
from greenonet.complex_tangent_subspace_audit import (
    TangentSubspaceAuditRequest,
    matrix_free_krylov_k2_step,
    matrix_free_krylov_subspace_audit,
    run_tangent_subspace_audit,
)
from test.test_complex_projection_response_audit import (
    _patch_static_export,
    _write_fixture,
)


def _response_operator(
    *,
    axis: str,
    matrix: torch.Tensor,
) -> FrozenAxialResponseOperator:
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


def _context() -> SymmetricTangentGreenResponseContext:
    dtype = torch.float64
    x = _response_operator(
        axis="x",
        matrix=torch.tensor(
            [[1.0, 0.4, -0.1], [0.2, 1.3, 0.3], [0.0, -0.2, 0.8]],
            dtype=dtype,
        ),
    )
    y = _response_operator(
        axis="y",
        matrix=torch.tensor(
            [[0.7, -0.3, 0.2], [0.1, 0.9, -0.4], [0.25, 0.15, 1.1]],
            dtype=dtype,
        ),
    )
    return SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=FrozenBidirectionalResponseOperator(x=x, y=y),
        point_mass=torch.tensor(0.25, dtype=dtype),
        config={
            "eta": 0.015,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-12,
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
        },
    )


def test_matrix_free_k2_never_increases_uncapped_k1_response_cost() -> None:
    context = _context()
    mismatch = torch.tensor(
        [[1.2, -0.4, 0.8], [-0.3, 1.5, 0.2]],
        dtype=torch.float64,
    )
    gradient = context.tangent_gradient(mismatch)

    result = matrix_free_krylov_k2_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        relative_eps=1.0e-12,
        monotonicity_relative_tol=1.0e-10,
    )

    assert torch.all(result.cost_k2 <= result.cost_k1 * (1.0 + 1.0e-10))
    assert torch.any(result.cost_k2 < result.cost_k1)
    torch.testing.assert_close(
        result.mismatch_k2,
        mismatch
        + context.response_operator.forward_pair(
            torch.stack((result.delta_k2, result.delta_k2), dim=1)
        ).sum(dim=1),
    )
    assert torch.all(torch.isfinite(result.delta_k2))


def test_matrix_free_k4_preserves_k2_and_monotonically_reduces_response() -> None:
    context = _context()
    mismatch = torch.tensor(
        [[1.2, -0.4, 0.8], [-0.3, 1.5, 0.2]],
        dtype=torch.float64,
    )
    gradient = context.tangent_gradient(mismatch)
    k2 = matrix_free_krylov_k2_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        relative_eps=1.0e-12,
        monotonicity_relative_tol=1.0e-10,
    )
    k4 = matrix_free_krylov_subspace_audit(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=4,
        relative_eps=1.0e-12,
        monotonicity_relative_tol=1.0e-10,
    )

    torch.testing.assert_close(k4.deltas[0], k2.delta_k1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k4.deltas[1], k2.delta_k2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k4.costs[0], k2.cost_k1, rtol=0.0, atol=0.0)
    torch.testing.assert_close(k4.costs[1], k2.cost_k2, rtol=0.0, atol=0.0)
    assert k4.directions.shape == (4, 2, 3)
    assert k4.response_gram.shape == (2, 4, 4)
    assert torch.all(k4.costs[1:] <= k4.costs[:-1] * (1.0 + 1.0e-10))
    assert torch.all(k4.response_orthogonality_max[2:] < 1.0e-8)
    reconstructed = mismatch.unsqueeze(0) + torch.stack(
        [
            context.response_operator.forward_pair(
                torch.stack((delta, delta), dim=1)
            ).sum(dim=1)
            for delta in k4.deltas
        ],
        dim=0,
    )
    torch.testing.assert_close(k4.mismatches, reconstructed)


def test_tangent_subspace_request_validation(tmp_path: Path) -> None:
    common = {
        "config": tmp_path / "config.json",
        "coupling_checkpoint": tmp_path / "coupling.safetensors",
        "green_checkpoint": tmp_path / "green.safetensors",
        "outdir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="batch_size"):
        TangentSubspaceAuditRequest(**common, batch_size=0)
    with pytest.raises(ValueError, match="subspace_relative_eps"):
        TangentSubspaceAuditRequest(**common, subspace_relative_eps=0.0)
    with pytest.raises(ValueError, match="duplicates"):
        TangentSubspaceAuditRequest(**common, selected_samples=(0, 0))
    with pytest.raises(ValueError, match="max_subspace_dimension"):
        TangentSubspaceAuditRequest(**common, max_subspace_dimension=1)
    assert (
        TangentSubspaceAuditRequest(
            **common, max_subspace_dimension=5
        ).max_subspace_dimension
        == 5
    )


def test_checkpoint_backed_k1_k4_audit_writes_metrics_and_balanced_fields(
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
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "eta": 0.015,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-12,
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
        },
    }
    config_path.write_text(json.dumps(payload))
    outdir = tmp_path / "k1_k4_audit"

    summary = run_tangent_subspace_audit(
        TangentSubspaceAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            selected_samples=(0,),
            batch_size=1,
            max_subspace_dimension=4,
        )
    )

    assert summary["diagnostic"] == ("matrix_free_k1_k4_tangent_subspace_posthoc_audit")
    assert summary["sample_count"] == 1
    assert summary["matrix_policy"]["global_matrix_materialized"] is False
    assert summary["matrix_policy"]["global_matrix_solve"] is False
    assert summary["matrix_policy"]["subspace_dimension"] == 4
    assert summary["matrix_policy"]["response_context_build_count"] == 1
    assert summary["reference_policy"]["sol_and_flux_used_for_correction"] is False
    assert len(summary["aggregate_metrics"]) == 6
    assert (
        summary["paired_comparisons"]["k2_vs_k1_uncapped"]["response_mismatch_cost"][
            "worsened_sample_count"
        ]
        == 0
    )
    assert (
        summary["paired_comparisons"]["k2_vs_k1_uncapped"]["response_mismatch_cost"][
            "max_worsening"
        ]
        == 0.0
    )
    assert (
        summary["paired_comparisons"]["k3_vs_k2_unconstrained"][
            "response_mismatch_cost"
        ]["worsened_sample_count"]
        == 0
    )
    assert (
        summary["paired_comparisons"]["k4_vs_k3_unconstrained"][
            "response_mismatch_cost"
        ]["worsened_sample_count"]
        == 0
    )

    with (outdir / "metrics" / "per_sample_k1_k4.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all("rel_u_phi" in row and "rel_u_psi" in row for row in rows)
    production = next(row for row in rows if row["method_id"] == "k1_production")
    uncapped = next(row for row in rows if row["method_id"] == "k1_uncapped")
    assert float(production["eta_applied"]) <= 0.015
    assert float(uncapped["eta_applied"]) == pytest.approx(float(uncapped["eta_star"]))

    raw_path = outdir / "data" / "selected_k1_k4_tangent_subspace.npz"
    with np.load(raw_path, allow_pickle=False) as raw:
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
        assert raw["direction_0"].shape == raw["direction_1"].shape
        assert raw["directions"].shape[0] == 4
        assert raw["coefficients"].shape[0] == 4
        assert raw["subspace_costs"].shape[0] == 4
        assert raw["candidate_physical"].shape[0] == 6
    assert (outdir / "diagnosis_report.md").is_file()


def test_checkpoint_backed_four_preconditioner_audit_reuses_raw_output(
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
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "subspace_dimension": 4,
            "eta": 0.015,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-12,
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
            "cross_axis_relative_eps": 1.0e-12,
        },
    }
    config_path.write_text(json.dumps(payload))
    outdir = tmp_path / "preconditioner_4x4_audit"

    summary = run_tangent_preconditioner_audit(
        TangentPreconditionerAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            selected_samples=(0,),
            batch_size=1,
        )
    )

    assert summary["raw_output"]["computed_once_per_batch"] is True
    assert summary["raw_output"]["shared_by_all_cells"] is True
    assert summary["tangent_context"]["response_operator_instance_count"] == 1
    assert summary["tangent_context"]["preconditioner_context_count"] == 4
    assert summary["phase_a_spatial_preflight"]["all_finite"] is True
    assert summary["phase_a_spatial_preflight"]["all_denominators_positive"] is True
    assert summary["operator_actions"]["global_matrix_materialized"] is False
    assert summary["operator_actions"]["global_linear_solve"] is False
    assert len(summary["phase_b_aggregate"]) == 21

    with (outdir / "metrics" / "per_sample_preconditioner_k1_k4.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert {row["method_id"] for row in rows} >= {
        "physical_symmetric",
        "sep_k1_capped",
        "sep_k4",
        "exact_k4",
        "abs_k4",
        "q_k4",
    }
    for prefix in ("sep", "exact", "abs", "q"):
        costs = [
            float(
                next(row for row in rows if row["method_id"] == f"{prefix}_k{k}")[
                    "response_mismatch_cost"
                ]
            )
            for k in range(1, 5)
        ]
        assert all(
            next_cost <= cost * (1.0 + 1.0e-10)
            for cost, next_cost in zip(costs, costs[1:])
        )

    with np.load(
        outdir / "data" / "selected_preconditioner_k1_k4.npz",
        allow_pickle=False,
    ) as raw:
        assert raw["candidate_physical"].shape[0] == 21
        expected = np.broadcast_to(
            raw["rhs"][None, :, None, :],
            raw["candidate_physical"].shape,
        )
        np.testing.assert_allclose(
            raw["candidate_physical"].sum(axis=2, keepdims=True),
            expected[:, :, :1],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    report_path = outdir / "diagnosis_report.md"
    assert report_path.is_file()
    report = report_path.read_text()
    assert (
        f"`{summary['findings']['lowest_loss_energy_optimized_mean_method']}` has "
        "the lowest mean optimized energy"
    ) in report
    assert "`separable K=4` has the lowest optimized energy" not in report


def test_four_preconditioner_audit_supports_explicit_physical_training_override(
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
    context_path = tmp_path / "shared_tangent_context.safetensors"

    with pytest.raises(ValueError, match="posthoc_tangent_override"):
        run_tangent_preconditioner_audit(
            TangentPreconditionerAuditRequest(
                config=config_path,
                coupling_checkpoint=coupling_checkpoint,
                green_checkpoint=green_checkpoint,
                outdir=tmp_path / "rejected",
                geometry=geometry_path,
                test_path=test_path,
                coefficients=coefficient_path,
                selected_samples=(0,),
                batch_size=1,
            )
        )

    summary = run_tangent_preconditioner_audit(
        TangentPreconditionerAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=tmp_path / "physical_override",
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            tangent_context=context_path,
            selected_samples=(0,),
            batch_size=1,
            posthoc_tangent_override=True,
        )
    )

    assert summary["posthoc_training_projection_override"] is True
    assert summary["projection_provenance"]["training_mode"] == "physical_symmetric"
    assert summary["projection_provenance"]["audit_mode"] == (
        "symmetric_tangent_green_response"
    )
    assert (
        summary["projection_provenance"]["audit_tangent_config"]["eta_strategy"]
        == "closed_loop_exact_line_search"
    )
    assert summary["tangent_context"]["source"] == "built"
    assert summary["tangent_context"]["save_count"] == 1
    assert context_path.is_file()
    with (
        tmp_path
        / "physical_override"
        / "metrics"
        / "per_sample_preconditioner_k1_k4.csv"
    ).open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["training_projection_mode"] for row in rows} == {"physical_symmetric"}
    assert {row["posthoc_training_projection_override"] for row in rows} == {"1"}


def test_cli_module_imports_and_exposes_runner() -> None:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "audit_tangent_subspace.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_tangent_subspace_cli",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert callable(module.AuditTangentSubspaceCLI)

    preconditioner_path = (
        Path(__file__).resolve().parents[1] / "cli" / "audit_tangent_preconditioners.py"
    )
    preconditioner_spec = importlib.util.spec_from_file_location(
        "audit_tangent_preconditioners_cli",
        preconditioner_path,
    )
    assert preconditioner_spec is not None and preconditioner_spec.loader is not None
    preconditioner_module = importlib.util.module_from_spec(preconditioner_spec)
    sys.modules[preconditioner_spec.name] = preconditioner_module
    preconditioner_spec.loader.exec_module(preconditioner_module)
    assert callable(preconditioner_module.AuditTangentPreconditionersCLI)
