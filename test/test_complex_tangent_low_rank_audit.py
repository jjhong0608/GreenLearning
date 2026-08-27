from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from greenonet.complex_tangent_low_rank_audit import (
    LowRankSpectralAuditRequest,
    LowRankSpectralContext,
    MatrixFreeScaledTangentOperator,
    RandomizedSpectralContextBuilder,
    run_tangent_low_rank_audit,
)
from greenonet.complex_tangent_projection import (
    matrix_free_krylov_subspace_step,
)
from test.test_complex_projection_response_audit import (
    _patch_static_export,
    _write_fixture,
)
from test.test_complex_tangent_subspace_audit import _context


def _dense_scaled_operator() -> tuple[torch.Tensor, MatrixFreeScaledTangentOperator]:
    context = _context()
    operator = MatrixFreeScaledTangentOperator(context)
    identity = torch.eye(3, dtype=torch.float64)
    dense = operator.apply(identity).transpose(0, 1)
    return dense, operator


def test_low_rank_request_validation(tmp_path: Path) -> None:
    common = {
        "config": tmp_path / "config.json",
        "coupling_checkpoint": tmp_path / "coupling.safetensors",
        "green_checkpoint": tmp_path / "green.safetensors",
        "outdir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="strictly increasing"):
        LowRankSpectralAuditRequest(**common, ranks=(4, 2))
    with pytest.raises(ValueError, match="duplicates"):
        LowRankSpectralAuditRequest(**common, ranks=(2, 2))
    assert (
        LowRankSpectralAuditRequest(
            **common, max_subspace_dimension=5
        ).max_subspace_dimension
        == 5
    )
    with pytest.raises(ValueError, match="max_subspace_dimension"):
        LowRankSpectralAuditRequest(**common, max_subspace_dimension=0)
    with pytest.raises(ValueError, match="probe_count"):
        LowRankSpectralAuditRequest(**common, probe_count=0)
    with pytest.raises(ValueError, match="benchmark_repeats"):
        LowRankSpectralAuditRequest(**common, benchmark_repeats=0)
    with pytest.raises(ValueError, match="complement_scale"):
        LowRankSpectralAuditRequest(**common, complement_scale="invalid")  # type: ignore[arg-type]


def test_scaled_operator_matches_explicit_dense_matrix() -> None:
    dense, operator = _dense_scaled_operator()
    values = torch.tensor(
        [[0.2, -0.7, 1.1], [1.3, 0.4, -0.2]],
        dtype=torch.float64,
    )

    actual = operator.apply(values)
    expected = values @ dense.transpose(0, 1)

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(dense, dense.transpose(0, 1))
    assert operator.global_matrix_materialized is False
    assert operator.global_matrix_solve is False


def test_full_rank_randomized_context_recovers_toy_spectrum() -> None:
    dense, operator = _dense_scaled_operator()
    builder = RandomizedSpectralContextBuilder(
        operator=operator,
        ranks=(1, 2, 3),
        oversampling=0,
        power_iterations=1,
        probe_count=16,
        seed=17,
        eigenvalue_relative_floor=1.0e-12,
    )

    context = builder.build()
    expected = torch.linalg.eigvalsh(dense).flip(0)

    torch.testing.assert_close(
        context.eigenvalues, expected, rtol=1.0e-10, atol=1.0e-12
    )
    torch.testing.assert_close(
        context.basis.transpose(0, 1) @ context.basis,
        torch.eye(3, dtype=torch.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert context.ritz_residual_norm.max().item() < 1.0e-10
    assert context.setup_operator_application_count >= 3


def test_low_rank_inverse_action_matches_explicit_spectral_formula() -> None:
    context = _context()
    dense_scaled, operator = _dense_scaled_operator()
    eigenvalues, basis = torch.linalg.eigh(dense_scaled)
    order = torch.argsort(eigenvalues, descending=True)
    spectral = LowRankSpectralContext.from_eigenpairs(
        operator=operator,
        ranks=(1, 2),
        eigenvalues=eigenvalues[order],
        basis=basis[:, order],
        eigenvalue_relative_floor=1.0e-12,
    )
    gradient = torch.tensor(
        [[0.8, -0.2, 1.4], [-0.5, 0.6, 0.1]],
        dtype=torch.float64,
    )
    rank = 2

    actual = spectral.apply_inverse(gradient, rank=rank)
    d_inv_sqrt = context.denominator.rsqrt()
    u = spectral.basis[:, :rank]
    lam = spectral.eigenvalues[:rank]
    whitened = gradient * d_inv_sqrt
    expected_white = (
        whitened + (whitened @ u) @ torch.diag(lam.reciprocal() - 1.0) @ u.T
    )
    expected = expected_white * d_inv_sqrt

    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    shifted = spectral.apply_inverse(
        gradient,
        rank=rank,
        complement_scale="next_ritz",
    )
    tail = spectral.eigenvalues[rank]
    expected_shifted_white = (
        whitened + (whitened @ u) @ torch.diag(tail / lam - 1.0) @ u.T
    )
    torch.testing.assert_close(
        shifted,
        expected_shifted_white * d_inv_sqrt,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_diagonal_callback_preserves_existing_k4_result() -> None:
    context = _context()
    mismatch = torch.tensor(
        [[1.2, -0.4, 0.8], [-0.3, 1.5, 0.2]],
        dtype=torch.float64,
    )
    gradient = context.tangent_gradient(mismatch)
    baseline = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=4,
        relative_eps=1.0e-12,
        monotonicity_relative_tol=1.0e-10,
    )
    callback = matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=gradient,
        max_dimension=4,
        relative_eps=1.0e-12,
        monotonicity_relative_tol=1.0e-10,
        inverse_preconditioner=lambda values: (
            values * context.denominator.reciprocal().unsqueeze(0)
        ),
    )

    torch.testing.assert_close(callback.deltas, baseline.deltas, rtol=0.0, atol=0.0)
    torch.testing.assert_close(callback.costs, baseline.costs, rtol=0.0, atol=0.0)


def test_checkpoint_backed_low_rank_k1_k4_audit_writes_complete_schema(
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
            "subspace_dimension": 1,
            "eta": 0.015,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 1.0e-12,
            "relative_lambda": 0.01,
            "denominator_relative_eps": 1.0e-12,
        },
    }
    config_path.write_text(json.dumps(payload))
    outdir = tmp_path / "low_rank_k1_k4"

    summary = run_tangent_low_rank_audit(
        LowRankSpectralAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            ranks=(1, 2),
            max_subspace_dimension=4,
            oversampling=1,
            power_iterations=0,
            probe_count=4,
            batch_size=1,
            selected_samples=(0,),
            benchmark_warmup=0,
            benchmark_repeats=1,
        )
    )

    assert summary["diagnostic"] == "matrix_free_low_rank_spectral_k1_k4_posthoc_audit"
    assert summary["sample_count"] == 1
    assert summary["matrix_policy"]["global_matrix_materialized"] is False
    assert summary["matrix_policy"]["global_matrix_solve"] is False
    assert summary["matrix_policy"]["response_context_build_count"] == 1
    assert summary["reference_policy"]["sol_and_flux_used_for_correction"] is False
    assert len(summary["aggregate_metrics"]) == 1 + 3 * 4
    assert summary["spectral_context"]["stored_max_rank"] == 2

    with (outdir / "metrics" / "per_sample_low_rank_k1_k4.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 13
    assert {row["preconditioner"] for row in rows} == {
        "none",
        "diagonal",
        "low_rank_1",
        "low_rank_2",
    }
    assert all("rel_u_phi" in row and "rel_u_psi" in row for row in rows)
    assert (outdir / "metrics" / "spectral_decay.csv").is_file()
    assert (outdir / "metrics" / "per_sample_gradient_coverage.csv").is_file()
    assert (outdir / "metrics" / "runtime_benchmark.csv").is_file()
    assert (outdir / "data" / "spectral_context.npz").is_file()
    assert (outdir / "diagnosis_report.md").is_file()
    runtime_rows = summary["runtime_benchmark"]
    diagonal_runtime = [
        row for row in runtime_rows if row["preconditioner_prefix"] == "diag"
    ]
    low_rank_runtime = [
        row for row in runtime_rows if row["preconditioner_prefix"] != "diag"
    ]
    assert all(row["spectral_setup_applicable"] is False for row in diagonal_runtime)
    assert all(row["break_even_optimizer_steps"] == 0.0 for row in diagonal_runtime)
    assert all(row["spectral_setup_applicable"] is True for row in low_rank_runtime)


def test_low_rank_cli_imports() -> None:
    cli_path = Path(__file__).resolve().parents[1] / "cli" / "audit_tangent_low_rank.py"
    spec = importlib.util.spec_from_file_location(
        "audit_tangent_low_rank_cli", cli_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "AuditTangentLowRankCLI")
