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
from greenonet.complex_length_response_diagnostics import (
    ComplexLengthResponseDiagnostic,
    ComplexLengthResponseDiagnosticRequest,
    ExactGreenReconstructionMixin,
    run_complex_length_response_diagnostics,
)
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    CouplingModelConfig,
    ModelConfig,
    TransverseTrunkConfig,
)
from greenonet.greens import ExactGreenFunction
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.model import GreenONetModel
from test.complex_fixtures import write_complex_config


def _load_annular_geometry_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_annular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "diagnostic_test_make_annular_geometry",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_annular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_annulus(tmp_path: Path) -> Path:
    module = _load_annular_geometry_module()
    path = tmp_path / "annulus.npz"
    module.AnnularGeometryBuilder(
        module.AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        )
    ).write()
    return path


def _write_poisson_coefficients(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "import torch",
                "def a_fun(x, y): return torch.ones_like(x)",
                "def apx_fun(x, y): return torch.zeros_like(x)",
                "def apy_fun(x, y): return torch.zeros_like(x)",
                "def bx_fun(x, y): return torch.zeros_like(x)",
                "def by_fun(x, y): return torch.zeros_like(x)",
                "def c_fun(x, y): return torch.zeros_like(x)",
            ]
        )
    )
    return path


def _write_annulus_sample(
    data_dir: Path,
    geometry_path: Path,
    *,
    include_flux: bool = True,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    with np.load(geometry_path) as raw:
        grid_x = raw["grid_x"]
        grid_y = raw["grid_y"]
        valid_y = raw["valid_grid_y_index"]
        valid_x = raw["valid_grid_x_index"]
    shape = (grid_y.size, grid_x.size)
    rhs = np.zeros(shape, dtype=np.float64)
    sol = np.zeros(shape, dtype=np.float64)
    rhs[valid_y, valid_x] = 1.0
    payload = {"rhs": rhs, "sol": sol}
    if include_flux:
        phi = np.zeros(shape, dtype=np.float64)
        psi = np.zeros(shape, dtype=np.float64)
        phi[valid_y, valid_x] = 0.5
        psi[valid_y, valid_x] = 0.5
        payload.update(phi=phi, psi=psi)
    path = data_dir / "sample_000000.npz"
    np.savez(path, **payload)
    return path


def _write_diagnostic_fixture(
    tmp_path: Path,
    *,
    include_flux: bool = True,
) -> tuple[Path, Path, Path, Path, Path]:
    geometry_path = _write_annulus(tmp_path)
    coefficient_path = _write_poisson_coefficients(tmp_path / "coefficients.py")
    test_path = tmp_path / "test"
    _write_annulus_sample(test_path, geometry_path, include_flux=include_flux)

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
    )


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def test_exact_poisson_segment_reconstruction_and_length_squared_response():
    node_t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    node_weight = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64)
    source = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    kernel = ExactGreenFunction(node_t, torch.ones_like(node_t))()

    length_one, physical_one = (
        ExactGreenReconstructionMixin.reconstruct_segment_with_kernel(
            source_physical=source,
            node_weight_unit=node_weight,
            kernel_unit=kernel,
            length=1.0,
        )
    )
    length_two, physical_two = (
        ExactGreenReconstructionMixin.reconstruct_segment_with_kernel(
            source_physical=source,
            node_weight_unit=node_weight,
            kernel_unit=kernel,
            length=2.0,
        )
    )

    torch.testing.assert_close(length_one, physical_one)
    torch.testing.assert_close(length_two, physical_two)
    assert length_one[0, 1].item() == pytest.approx(0.125)
    assert length_two[0, 1].item() == pytest.approx(0.5)
    torch.testing.assert_close(length_two, 4.0 * length_one)


def test_transition_inference_finds_lines_on_both_sides(tmp_path):
    geometry_path = _write_annulus(tmp_path)
    request = ComplexLengthResponseDiagnosticRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "out",
        selected_samples=(0,),
    )
    diagnostic = ComplexLengthResponseDiagnostic(request)
    diagnostic.geometry_path = geometry_path
    diagnostic.geometry = load_complex_geometry(geometry_path)

    transition = diagnostic._infer_transition_geometry()

    assert transition.inner_radius == pytest.approx(0.5)
    assert transition.horizontal_split_coordinate == pytest.approx(0.25)
    assert transition.horizontal_one_segment_coordinate == pytest.approx(0.75)
    assert transition.vertical_split_coordinate == pytest.approx(0.25)
    assert transition.vertical_one_segment_coordinate == pytest.approx(0.75)
    assert transition.horizontal_length_jump_ratio > 1.0
    assert transition.vertical_length_squared_jump_ratio > 1.0


def test_diagnostic_writes_expected_outputs_and_keeps_targets_evaluation_only(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
    ) = _write_diagnostic_fixture(tmp_path)
    outdir = tmp_path / "diagnostics"

    summary = run_complex_length_response_diagnostics(
        ComplexLengthResponseDiagnosticRequest(
            config=config_path,
            coupling_checkpoint=coupling_checkpoint,
            green_checkpoint=green_checkpoint,
            outdir=outdir,
            geometry=geometry_path,
            test_path=test_path,
            selected_samples=(0,),
            include_rel_sol_quantiles=True,
        )
    )

    assert summary["num_samples"] == 1
    assert summary["selected_samples"] == [0]
    assert summary["projection_mode"] == "physical_symmetric"
    assert summary["output_contract_version"] == 6
    assert summary["uses_reference_targets_for_training"] is False
    assert summary["reference_fields_role"] == "evaluation_only"
    assert summary["unit_physical_equivalence"]["passed"] is True
    assert summary["unit_physical_equivalence"]["max_absolute_difference"] <= 1e-10
    assert summary["exact_green_reference_kinds"] == ["diffusion"]
    assert (outdir / "summary.json").is_file()
    assert (outdir / "metrics" / "per_sample_length_response.csv").is_file()
    assert (outdir / "metrics" / "per_segment_length_response.csv").is_file()
    assert (outdir / "metrics" / "transition_zone_metrics.csv").is_file()
    raw_path = outdir / "data" / "selected_diagnostic_arrays.npz"
    assert raw_path.is_file()
    with np.load(raw_path) as raw:
        keys = set(raw.files)
    for suffix in (
        "_physical_source_error_phi",
        "_response_source_error_phi",
        "_exact_u_pred",
        "_learned_u_pred",
        "_learned_minus_exact_mean",
        "_target_exact_closure_mean",
        "_raw_response_constraint_residual",
        "_response_constraint_residual",
    ):
        assert any(key.endswith(suffix) for key in keys)
    assert (outdir / "figures" / "segment_response_gain.json").is_file()


def test_diagnostic_rejects_samples_without_flux_targets(tmp_path, monkeypatch):
    _patch_static_export(monkeypatch)
    (
        config_path,
        coupling_checkpoint,
        green_checkpoint,
        geometry_path,
        test_path,
    ) = _write_diagnostic_fixture(tmp_path, include_flux=False)

    with pytest.raises(ValueError, match="require phi/psi flux targets"):
        run_complex_length_response_diagnostics(
            ComplexLengthResponseDiagnosticRequest(
                config=config_path,
                coupling_checkpoint=coupling_checkpoint,
                green_checkpoint=green_checkpoint,
                outdir=tmp_path / "diagnostics",
                geometry=geometry_path,
                test_path=test_path,
                selected_samples=(0,),
                include_rel_sol_quantiles=False,
            )
        )
