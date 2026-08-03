from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_green_response_projection import (
    ColumnDiagonalGreenResponseContext,
)
from greenonet.complex_projection_response_audit import (
    ComplexProjectionResponseAudit,
    ProjectionResponseAuditEvaluation,
    ProjectionResponseAuditRequest,
    ProjectionTransitionEdges,
    run_complex_projection_response_audit,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingModelConfig,
    ModelConfig,
    TransverseTrunkConfig,
)
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.model import GreenONetModel
from test.complex_fixtures import write_coefficients, write_complex_config


def _patch_static_export(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def _load_annular_geometry_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_annular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "projection_response_make_annular_geometry",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_annular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_annular_geometry(path: Path) -> Path:
    module = _load_annular_geometry_module()
    module.AnnularGeometryBuilder(
        module.AnnularGeometryConfig(
            inner_radius=0.4,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        )
    ).write()
    return path


def _write_sample(data_dir: Path, geometry_path: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with np.load(geometry_path, allow_pickle=False) as geometry:
        shape = (len(geometry["grid_y"]), len(geometry["grid_x"]))
    grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    np.savez(
        data_dir / "sample_0000.npz",
        rhs=1.0 + 0.01 * grid,
        sol=2.0 + 0.02 * grid,
        phi=0.6 + 0.006 * grid,
        psi=0.4 + 0.004 * grid,
    )


def _write_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    geometry_path = _write_annular_geometry(tmp_path / "geometry.npz")
    coefficient_path = write_coefficients(tmp_path / "coefficients.py")
    test_path = tmp_path / "test"
    _write_sample(test_path, geometry_path)

    coupling_config = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(
            mode="column_diagonal_green_response"
        ),
        axis_1d_trunk=Axis1DTrunkConfig(
            enabled=True,
            num_frequencies=2,
            max_frequency=2.0,
            transverse_trunk=TransverseTrunkConfig(
                enabled=True,
                fusion="product",
                length_context=True,
            ),
        ),
    )
    green_config = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_checkpoint = tmp_path / "coupling.safetensors"
    green_checkpoint = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_config).state_dict(),
        coupling_checkpoint,
    )
    save_model_with_config(
        GreenONetModel(green_config),
        green_config,
        green_checkpoint,
    )
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=test_path,
        coefficient_path=coefficient_path,
    )
    payload = json.loads(config_path.read_text())
    payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "column_diagonal_green_response",
        "column_diagonal_green_response": {
            "gain_squared_eps": 1.0e-12,
            "gain_exponent": 0.25,
        },
    }
    config_path.write_text(json.dumps(payload))
    return (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
        coefficient_path,
    )


def test_request_validation_requires_symmetric_and_full_column_endpoints(
    tmp_path: Path,
) -> None:
    common = {
        "config": tmp_path / "config.json",
        "coupling_checkpoint": tmp_path / "coupling.safetensors",
        "green_checkpoint": tmp_path / "green.safetensors",
        "outdir": tmp_path / "out",
    }
    with pytest.raises(ValueError, match="include 0.0 and 1.0"):
        ProjectionResponseAuditRequest(**common, alphas=(0.0, 0.25))
    with pytest.raises(ValueError, match="duplicate"):
        ProjectionResponseAuditRequest(**common, alphas=(0.0, 0.25, 0.25, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        ProjectionResponseAuditRequest(
            **common,
            transition_log_threshold=-1.0,
        )


def test_alpha_one_minimizes_column_diagonal_surrogate() -> None:
    alphas = (0.0, 0.25, 0.5, 1.0)
    gamma_x = torch.tensor([1.0, 4.0], dtype=torch.float64)
    gamma_y = torch.tensor([9.0, 1.0], dtype=torch.float64)
    contexts = tuple(
        ColumnDiagonalGreenResponseContext.from_gain_squared(
            gamma_x_squared=gamma_x,
            gamma_y_squared=gamma_y,
            point_mass=1.0,
            gain_squared_eps=1.0e-12,
            gain_exponent=alpha,
        )
        for alpha in alphas
    )
    residual = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
    correction = torch.stack(
        [
            torch.stack(
                (
                    context.correction_weight_phi.unsqueeze(0) * residual,
                    context.correction_weight_psi.unsqueeze(0) * residual,
                ),
                dim=1,
            )
            for context in contexts
        ],
        dim=0,
    )
    evaluation = ProjectionResponseAuditEvaluation(
        sample_ids=torch.tensor([0]),
        file_stems=("sample_0000",),
        has_solution=torch.tensor([False]),
        has_flux=torch.tensor([False]),
        rhs=torch.zeros((1, 2), dtype=torch.float64),
        sol=torch.zeros((1, 2), dtype=torch.float64),
        flux_target=torch.zeros((1, 2, 2), dtype=torch.float64),
        raw_response=torch.zeros((1, 2, 2), dtype=torch.float64),
        raw_physical=torch.zeros((1, 2, 2), dtype=torch.float64),
        raw_balance_residual=residual,
        weights_phi=torch.stack(
            [context.correction_weight_phi for context in contexts]
        ),
        projected_physical=correction,
        correction_physical=correction,
        correction_response=correction,
        correction_solution=correction,
        raw_solution=torch.zeros((1, 2, 2), dtype=torch.float64),
    )
    empty_edges = torch.empty((0, 2), dtype=torch.long)
    rows = ComplexProjectionResponseAudit.build_metric_rows(
        ComplexProjectionResponseAudit.__new__(ComplexProjectionResponseAudit),
        evaluation=evaluation,
        contexts=contexts,
        edges=ProjectionTransitionEdges(
            phi_transition=empty_edges,
            psi_transition=empty_edges,
            phi_regular=empty_edges,
            psi_regular=empty_edges,
        ),
        point_mass=torch.tensor(1.0, dtype=torch.float64),
        alphas=alphas,
        eps=1.0e-12,
    )
    costs = {
        float(row["gain_exponent"]): float(row["diagonal_surrogate_cost"])
        for row in rows
    }
    assert costs[1.0] == pytest.approx(min(costs.values()))


def test_annulus_transition_edges_use_cross_axis_length_jumps(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(_write_annular_geometry(tmp_path / "geometry.npz"))
    edges = ComplexProjectionResponseAudit.build_transition_edges(
        geometry,
        threshold=np.log(2.0),
    )
    assert edges.phi_transition.shape[0] > 0
    assert edges.psi_transition.shape[0] > 0
    assert edges.phi_transition.shape[1] == 2
    assert edges.psi_transition.shape[1] == 2


def test_candidate_audit_separates_raw_common_mode_from_balanced_difference() -> None:
    alphas = (0.0, 0.25, 1.0)
    rhs = torch.tensor([[2.0, 2.0]], dtype=torch.float64)
    raw = torch.tensor([[[2.0, 0.0], [0.0, 0.0]]], dtype=torch.float64)
    target = torch.tensor([[[2.0, 1.0], [0.0, 1.0]]], dtype=torch.float64)
    projected = torch.stack(
        (
            target,
            torch.tensor([[[1.5, 1.25], [0.5, 0.75]]], dtype=torch.float64),
            torch.tensor([[[1.0, 1.5], [1.0, 0.5]]], dtype=torch.float64),
        ),
        dim=0,
    )
    evaluation = ProjectionResponseAuditEvaluation(
        sample_ids=torch.tensor([0]),
        file_stems=("sample_0000",),
        has_solution=torch.tensor([False]),
        has_flux=torch.tensor([True]),
        rhs=rhs,
        sol=torch.zeros((1, 2), dtype=torch.float64),
        flux_target=target,
        raw_response=raw,
        raw_physical=raw,
        raw_balance_residual=rhs - raw.sum(dim=1),
        weights_phi=torch.full((3, 2), 0.5, dtype=torch.float64),
        projected_physical=projected,
        correction_physical=projected - raw.unsqueeze(0),
        correction_response=projected - raw.unsqueeze(0),
        correction_solution=torch.zeros((3, 1, 2, 2), dtype=torch.float64),
        raw_solution=torch.zeros((1, 2, 2), dtype=torch.float64),
    )
    transition = torch.tensor([[0, 1]], dtype=torch.long)
    empty = torch.empty((0, 2), dtype=torch.long)
    rows = ComplexProjectionResponseAudit.build_candidate_rows(
        ComplexProjectionResponseAudit.__new__(ComplexProjectionResponseAudit),
        evaluation=evaluation,
        edges=ProjectionTransitionEdges(
            phi_transition=transition,
            psi_transition=empty,
            phi_regular=empty,
            psi_regular=empty,
        ),
        configured_alpha=0.25,
        alphas=alphas,
        eps=1.0e-12,
    )

    assert len(rows) == 1
    row = rows[0]
    assert float(row["raw_pair_rel_target"]) > 0.0
    assert float(row["symmetric_pair_rel_target"]) == pytest.approx(0.0)
    assert float(row["configured_pair_rel_target"]) > 0.0
    assert float(row["raw_difference_rel_target"]) == pytest.approx(0.0)
    assert float(row["raw_balance_residual_rel_rhs"]) > 0.0


def test_checkpoint_backed_audit_writes_balanced_outputs(
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
    outdir = tmp_path / "audit"
    summary = run_complex_projection_response_audit(
        ProjectionResponseAuditRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            alphas=(0.0, 0.25, 1.0),
            selected_samples=(0,),
            batch_size=1,
        )
    )

    assert summary["diagnostic"] == "column_diagonal_green_response_posthoc_audit"
    assert summary["matrix_policy"]["green_response_context_build_count"] == 1
    assert summary["matrix_policy"]["row_norm_used"] is False
    assert summary["matrix_policy"]["global_matrix_solve"] is False
    assert summary["reference_policy"]["sol_and_flux_used_for_projection"] is False
    assert summary["projection_weight_geometry"]["alpha_0"][
        "transition_weight_jump_rms"
    ] == pytest.approx(0.0)
    assert (
        summary["projection_weight_geometry"]["alpha_1"]["transition_weight_jump_rms"]
        > 0.0
    )
    assert (
        summary["automated_findings"]["alpha_1_diagonal_surrogate_optimal_sample_count"]
        == 1
    )
    assert (outdir / "metrics" / "per_sample_projection_response_audit.csv").is_file()
    assert (outdir / "metrics" / "per_sample_directional_candidate_audit.csv").is_file()
    assert (outdir / "diagnosis_report.md").is_file()
    assert (outdir / "figures" / "geometry" / "correction_weight_phi.json").is_file()
    assert (
        outdir / "figures" / "selected" / "sample_0000_directional_candidate_audit.json"
    ).is_file()
    assert summary["directional_candidate_audit"]["flux_target_sample_count"] == 1
    assert (
        "raw_pair_rel_target"
        in summary["directional_candidate_audit"]["aggregate_metrics"]
    )
    raw_path = outdir / "data" / "selected_projection_response_audit.npz"
    assert raw_path.is_file()
    with np.load(raw_path, allow_pickle=False) as raw:
        assert {
            "alpha_values",
            "raw_balance_residual",
            "symmetric_balanced_physical",
            "configured_projected_physical",
            "configured_correction_physical",
            "correction_physical",
            "correction_response",
            "correction_solution",
            "phi_transition_edges",
            "psi_transition_edges",
        }.issubset(raw.files)
        expected_rhs = np.broadcast_to(
            raw["rhs"][None, ...],
            raw["projected_physical"].sum(axis=2).shape,
        )
        np.testing.assert_allclose(
            raw["projected_physical"].sum(axis=2),
            expected_rhs,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
