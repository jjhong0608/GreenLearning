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

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_smooth_blend_diagnostics import FixedSmoothBlendConfig
from greenonet.complex_weak_residual_blend_diagnostics import (
    WeakResidualBlendComparison,
    WeakResidualBlendComparisonRequest,
    WeakResidualBlendEvaluation,
    WeakResidualReliabilityBlendConfig,
    run_weak_residual_blend_comparison,
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
from test.complex_fixtures import (
    write_coefficients,
    write_complex_config,
    write_geometry_npz,
)


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
        "weak_residual_make_annular_geometry",
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


def _write_annular_sample(data_dir: Path, geometry_path: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with np.load(geometry_path, allow_pickle=False) as geometry:
        shape = (len(geometry["grid_y"]), len(geometry["grid_x"]))
    grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    np.savez(
        data_dir / "sample_0000.npz",
        rhs=1.0 + grid,
        sol=2.0 + grid,
        phi=3.0 + grid,
        psi=4.0 + grid,
    )


def _write_diagnostic_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    geometry_path = _write_annular_geometry(tmp_path / "geometry.npz")
    coefficient_path = write_coefficients(tmp_path / "coefficients.py")
    test_path = tmp_path / "test"
    _write_annular_sample(test_path, geometry_path)

    coupling_config = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
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
        "mode": "physical_symmetric",
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


def test_weak_residual_reliability_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="gamma"):
        WeakResidualReliabilityBlendConfig(gamma=1.1)
    with pytest.raises(ValueError, match="smoothing_steps"):
        WeakResidualReliabilityBlendConfig(smoothing_steps=-1)
    with pytest.raises(ValueError, match="smoothing_relaxation"):
        WeakResidualReliabilityBlendConfig(smoothing_relaxation=0.0)
    with pytest.raises(ValueError, match="relative_floor"):
        WeakResidualReliabilityBlendConfig(relative_floor=-1.0)
    with pytest.raises(ValueError, match="eps"):
        WeakResidualReliabilityBlendConfig(eps=0.0)


def test_weak_residual_weights_prefer_the_lower_indicator_candidate(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    config = WeakResidualReliabilityBlendConfig(
        gamma=1.0,
        smoothing_steps=0,
        relative_floor=0.0,
    )
    phi_indicator = torch.tensor([[1.0, 4.0, 2.0]], dtype=torch.float64)
    psi_indicator = torch.tensor([[4.0, 1.0, 2.0]], dtype=torch.float64)

    _, _, _, theta, w_phi, _ = WeakResidualBlendComparison._weights_from_raw_indicators(
        geometry,
        phi_indicator,
        psi_indicator,
        config,
    )
    w_psi = 1.0 - w_phi

    assert w_phi[0, 0] > 0.5
    assert w_phi[0, 1] < 0.5
    assert w_phi[0, 2] == pytest.approx(0.5)
    assert theta[0, 2] == pytest.approx(0.0)
    torch.testing.assert_close(w_phi + w_psi, torch.ones_like(w_phi))


def test_weak_residual_weights_do_not_depend_on_reference_solution(
    tmp_path: Path,
) -> None:
    geometry = load_complex_geometry(write_geometry_npz(tmp_path / "geometry.npz"))
    coeffs = load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py"))
    u_phi = torch.tensor([[0.0, 1.0, -0.5]], dtype=torch.float64)
    u_psi = torch.tensor([[0.25, -0.25, 0.75]], dtype=torch.float64)
    projected = torch.tensor(
        [[[1.0, 0.5, -0.5], [0.25, 0.75, 1.0]]],
        dtype=torch.float64,
    )

    def evaluation(sol: torch.Tensor) -> WeakResidualBlendEvaluation:
        baseline = 0.5 * (u_phi + u_psi)
        return WeakResidualBlendEvaluation(
            sample_ids=torch.tensor([0]),
            file_stems=("sample_0000",),
            sol=sol,
            u_phi=u_phi,
            u_psi=u_psi,
            baseline=baseline,
            blend=baseline,
            rhs=projected.sum(dim=1),
            projected_physical=projected,
        )

    first = WeakResidualBlendComparison.build_weak_residual_reliability_fields(
        geometry,
        coeffs,
        evaluation(torch.zeros_like(u_phi)),
        WeakResidualReliabilityBlendConfig(smoothing_steps=0),
    )
    second = WeakResidualBlendComparison.build_weak_residual_reliability_fields(
        geometry,
        coeffs,
        evaluation(torch.full_like(u_phi, 999.0)),
        WeakResidualReliabilityBlendConfig(smoothing_steps=0),
    )

    torch.testing.assert_close(first.phi_full_residual, second.phi_full_residual)
    torch.testing.assert_close(first.psi_full_residual, second.psi_full_residual)
    torch.testing.assert_close(first.w_phi, second.w_phi)
    torch.testing.assert_close(first.w_psi, second.w_psi)


def test_checkpoint_backed_weak_residual_comparison_writes_outputs(
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
    ) = _write_diagnostic_fixture(tmp_path)
    outdir = tmp_path / "comparison"

    summary = run_weak_residual_blend_comparison(
        WeakResidualBlendComparisonRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            coefficients=coefficient_path,
            selected_samples=(0,),
            batch_size=1,
            blend=FixedSmoothBlendConfig(
                weight_construction="compact_c2_ramp",
                ramp_gamma=0.5,
                ramp_width=0.5,
                transition_dilation_steps=0,
            ),
            weak_residual=WeakResidualReliabilityBlendConfig(
                smoothing_steps=0,
            ),
            weak_sweep=True,
            weak_sweep_gammas=(0.5,),
            weak_sweep_relative_floors=(0.1,),
            weak_sweep_smoothing_steps=(0,),
        )
    )

    assert summary["diagnostic"] == ("local_weak_residual_reliability_blend_comparison")
    assert summary["status"] == "posthoc_exploratory"
    assert summary["production_code_changed"] is False
    estimator = summary["estimators"]["local_weak_residual_reliability"]
    assert estimator["uses_sol"] is False
    assert estimator["uses_flux_targets"] is False
    assert estimator["requires_global_matrix_solve"] is False
    assert set(summary["aggregate_metrics"]) == {
        "equal_mean",
        "geometry_c2",
        "mismatch_seam_c2",
        "weak_residual_reliability",
    }
    assert summary["weak_parameter_sweep"]["row_count"] == 1
    assert (
        outdir / "metrics" / "per_sample_weak_residual_blend_comparison.csv"
    ).is_file()
    assert (outdir / "metrics" / "weak_residual_parameter_sweep.csv").is_file()
    raw_path = outdir / "data" / "selected_weak_residual_blend_arrays.npz"
    assert raw_path.is_file()
    with np.load(raw_path) as raw:
        assert {
            "u_equal_mean",
            "u_geometry_c2",
            "u_mismatch_seam_c2",
            "u_weak_residual_reliability",
            "weak_phi_full_residual",
            "weak_psi_full_residual",
            "weak_phi_indicator",
            "weak_psi_indicator",
            "weak_w_phi",
            "weak_w_psi",
        }.issubset(raw.files)
        np.testing.assert_allclose(raw["weak_w_phi"] + raw["weak_w_psi"], 1.0)
    assert (outdir / "figures" / "aggregate" / "four_way_rel_sol.json").is_file()
    assert (
        outdir / "figures" / "selected" / "sample_0000_weak_residual_comparison.json"
    ).is_file()
