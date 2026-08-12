from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_artifacts import (
    ComplexCouplingArtifactExporter,
    ComplexDomainBoundaryOverlay,
    ComplexSelectedSample,
    export_complex_coupling_artifacts,
)
from greenonet.complex_coupling_model import ComplexCouplingNet
from greenonet.complex_geometry import load_complex_geometry
from greenonet.config import (
    Axis1DTrunkConfig,
    BalanceProjectionConfig,
    ComplexCrossAxisReconstructionConfig,
    ComplexPreProjectionFusionConfig,
    CouplingCoefficientTermsConfig,
    CouplingModelConfig,
    ModelConfig,
    TransverseTrunkConfig,
)
from greenonet.coupling_artifacts import CouplingArtifactRequest
from greenonet.io import save_model_with_config, save_state_dict_safetensors
from greenonet.model import GreenONetModel
from test.complex_fixtures import (
    write_coefficients,
    write_complex_config,
    write_geometry_npz,
    write_sample_npz,
    write_visualization_mesh_npz,
)


def _patch_static_export(monkeypatch) -> None:
    def fake_write_image(self: go.Figure, path: str) -> None:
        del self
        Path(path).write_text("static image placeholder")

    monkeypatch.setattr(go.Figure, "write_image", fake_write_image)


def _marker_for_field(outdir: Path, field: str) -> dict:
    figure = json.loads(
        (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).read_text()
    )
    return figure["data"][0]["marker"]


def _coefficient_figure(outdir: Path, field: str) -> dict:
    return json.loads(
        (outdir / "figures" / "coefficients" / f"{field}.json").read_text()
    )


def _boundary_overlay(*, enabled: bool = True) -> ComplexDomainBoundaryOverlay:
    return ComplexDomainBoundaryOverlay.from_endpoint_coords(
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        enabled=enabled,
        theme="plotly_white",
    )


def _assert_boundary_trace(figure: dict, *, expected_count: int) -> None:
    trace = figure["data"][-1]
    assert trace["type"] == "scatter"
    assert trace["mode"] == "markers"
    assert trace["name"] == "Domain boundary"
    assert trace["showlegend"] is True
    assert trace["marker"]["symbol"] == "circle-open"
    assert trace["marker"]["size"] == 5.0
    assert isinstance(trace["marker"]["color"], str)
    assert "colorscale" not in trace["marker"]
    assert "showscale" not in trace["marker"]
    assert "customdata" not in trace
    assert _plotly_array_size(trace["x"]) == expected_count
    assert _plotly_array_size(trace["y"]) == expected_count
    assert "Domain boundary" in trace["hovertemplate"]


def _plotly_array_size(values: list | dict) -> int:
    if isinstance(values, list):
        return len(values)
    buffer = base64.b64decode(values["bdata"])
    return int(np.frombuffer(buffer, dtype=np.dtype(values["dtype"])).size)


def _plotly_array(values: list | dict) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values)
    buffer = base64.b64decode(values["bdata"])
    return np.frombuffer(buffer, dtype=np.dtype(values["dtype"]))


def test_complex_domain_boundary_overlay_contract() -> None:
    endpoint_coords = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    overlay = ComplexDomainBoundaryOverlay.from_endpoint_coords(
        endpoint_coords,
        enabled=True,
        theme="plotly_white",
    )
    figure = go.Figure(
        go.Scattergl(
            x=[0.5],
            y=[0.5],
            mode="markers",
            marker={"color": [2.0], "colorscale": "Viridis", "showscale": True},
        )
    )

    overlay.add_to_figure(figure)

    payload = figure.to_plotly_json()
    assert len(payload["data"]) == 2
    assert payload["data"][0]["showlegend"] is False
    assert payload["data"][0]["marker"]["color"] == [2.0]
    assert payload["data"][0]["marker"]["showscale"] is True
    _assert_boundary_trace(payload, expected_count=2)
    assert overlay.summary() == {
        "enabled": True,
        "representation": "open_markers",
        "coordinate_source": ("canonical_boundary_energy_context.endpoint_coords"),
        "point_count": 2,
        "scalar_values_included": False,
        "included_in_metrics": False,
    }
    assert overlay.coords.flags.writeable is False

    disabled = ComplexDomainBoundaryOverlay.from_endpoint_coords(
        endpoint_coords,
        enabled=False,
        theme="plotly_white",
    )
    disabled_figure = go.Figure(go.Scattergl(x=[0.5], y=[0.5], mode="markers"))
    before = disabled_figure.to_plotly_json()
    disabled.add_to_figure(disabled_figure)
    assert disabled_figure.to_plotly_json() == before

    dark = ComplexDomainBoundaryOverlay.from_endpoint_coords(
        endpoint_coords,
        enabled=True,
        theme="plotly_dark",
    )
    assert dark.marker_color == "#ECEFF1"
    with pytest.raises(ValueError, match="cannot be empty"):
        ComplexDomainBoundaryOverlay.from_endpoint_coords(
            np.empty((0, 2)),
            enabled=True,
            theme="plotly_white",
        )
    with pytest.raises(ValueError, match="must be finite"):
        ComplexDomainBoundaryOverlay.from_endpoint_coords(
            np.asarray([[np.nan, 0.0]]),
            enabled=True,
            theme="plotly_white",
        )

    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        ComplexDomainBoundaryOverlay.from_endpoint_coords(
            np.zeros((2, 3)),
            enabled=True,
            theme="plotly_white",
        )


def test_directional_robust_color_ranges_are_shared_and_zero_centered(
    tmp_path: Path,
) -> None:
    exporter = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=tmp_path / "config.json",
            coupling_checkpoint=tmp_path / "coupling.safetensors",
            green_checkpoint=tmp_path / "green.safetensors",
            outdir=tmp_path / "artifacts",
            directional_color_quantile=0.75,
        )
    )
    arrays = {
        "sol": np.asarray([0.0, 1.0, 2.0]),
        "u_pred": np.asarray([-1.0, 0.5, 3.0]),
        "u_pred_error": np.asarray([-2.0, 0.0, 1.0]),
        "rhs": np.asarray([-5.0, 1.0, 7.0]),
        "phi": np.asarray([-100.0, -1.0, 0.0, 1.0]),
        "target_phi": np.asarray([2.0, 3.0, 4.0, 100.0]),
        "psi": np.asarray([-4.0, 0.0, 8.0, 12.0]),
        "target_psi": np.asarray([-8.0, -2.0, 2.0, 4.0]),
        "phi_error": np.asarray([-100.0, -2.0, 0.0, 4.0]),
        "psi_error": np.asarray([-6.0, -1.0, 3.0, 80.0]),
    }

    ranges = exporter._color_ranges_for_sample(arrays)
    phi_joined = np.concatenate((arrays["target_phi"], arrays["phi"]))
    expected_phi = tuple(np.quantile(phi_joined, (0.25, 0.75)))
    assert ranges["phi"] is ranges["target_phi"]
    assert ranges["phi"].plotly_kwargs() == {
        "cmin": pytest.approx(expected_phi[0]),
        "cmax": pytest.approx(expected_phi[1]),
    }
    assert ranges["psi"] is ranges["target_psi"]
    phi_error_limit = float(np.quantile(np.abs(arrays["phi_error"]), 0.75))
    assert ranges["phi_error"].plotly_kwargs() == {
        "cmin": pytest.approx(-phi_error_limit),
        "cmax": pytest.approx(phi_error_limit),
    }
    assert ranges["rhs"].plotly_kwargs() == {"cmin": -5.0, "cmax": 7.0}
    stats = ranges["phi"].field_summary(arrays["phi"])
    assert stats["saturated_point_count"] > 0
    assert stats["full_min"] == -100.0

    legacy = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=tmp_path / "config.json",
            coupling_checkpoint=tmp_path / "coupling.safetensors",
            green_checkpoint=tmp_path / "green.safetensors",
            outdir=tmp_path / "legacy_artifacts",
            directional_color_quantile=1.0,
        )
    )._color_ranges_for_sample(arrays)
    assert legacy["phi"].plotly_kwargs() == {"cmin": -100.0, "cmax": 100.0}
    assert legacy["phi_error"].plotly_kwargs() == {
        "cmin": -100.0,
        "cmax": 100.0,
    }


def test_scalar_mesh_omits_optional_target_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    exporter = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=tmp_path / "config.json",
            coupling_checkpoint=tmp_path / "coupling.safetensors",
            green_checkpoint=tmp_path / "green.safetensors",
            outdir=tmp_path / "artifacts",
        )
    )
    arrays = {
        "coords_valid": np.asarray(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]],
            dtype=np.float64,
        ),
        "sol": np.asarray([0.1, 0.2, 0.3]),
        "u_pred": np.asarray([0.11, 0.19, 0.31]),
        "u_pred_error": np.asarray([0.01, -0.01, 0.01]),
        "rhs": np.asarray([1.0, 2.0, 3.0]),
        "phi": np.asarray([0.6, 1.1, 1.7]),
        "psi": np.asarray([0.4, 0.9, 1.3]),
    }
    sample = ComplexSelectedSample(sample_id=0, file_stem="sample_0000", arrays=arrays)
    color_ranges = {0: exporter._color_ranges_for_sample(arrays)}

    paths, fields = exporter._write_scalar_mesh_figures(
        [sample],
        mesh,
        "plotly_white",
        _boundary_overlay(),
        color_ranges,
    )

    assert fields == ("sol", "u_pred", "u_pred_error", "rhs", "phi", "psi")
    assert len(paths) == 6
    assert not (tmp_path / "artifacts" / "figures" / "mesh" / "target_phi").exists()

    directional = exporter._scalar_mesh_figure(
        title="phi without boundary outline",
        field="phi",
        visualization_mesh=mesh,
        valid_values=arrays["phi"],
        theme="plotly_white",
        signed=False,
        color_range=color_ranges[0]["phi"],
        boundary_overlay=_boundary_overlay(enabled=False),
    ).to_plotly_json()
    assert len(directional["data"]) == 2
    assert all(trace.get("name") != "Domain boundary" for trace in directional["data"])


def test_robust_color_range_uses_finite_values_and_handles_constants() -> None:
    finite_range = ComplexCouplingArtifactExporter._shared_color_range(
        {"phi": np.asarray([np.nan, -2.0, 4.0, np.inf, -np.inf])},
        ("phi",),
        group="phi",
        policy="shared_lower_upper_quantile",
        quantile=1.0,
    )
    assert finite_range is not None
    assert finite_range.plotly_kwargs() == {"cmin": -2.0, "cmax": 4.0}

    constant_range = ComplexCouplingArtifactExporter._shared_color_range(
        {"phi": np.asarray([3.5, 3.5, 3.5])},
        ("phi",),
        group="phi",
        policy="shared_lower_upper_quantile",
        quantile=0.99,
    )
    assert constant_range is not None
    assert constant_range.plotly_kwargs() == {}
    assert constant_range.field_summary(np.asarray([3.5, 3.5])) == {
        "group": "phi",
        "policy": "shared_lower_upper_quantile",
        "quantile": 0.99,
        "full_min": 3.5,
        "full_max": 3.5,
        "display_cmin": 3.5,
        "display_cmax": 3.5,
        "finite_point_count": 2,
        "saturated_point_count": 0,
        "saturated_point_fraction": 0.0,
    }


def test_solution_mesh_cache_copy_respects_generated_data_flag(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    mesh_path, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    exporter = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=tmp_path / "config.json",
            coupling_checkpoint=tmp_path / "coupling.safetensors",
            green_checkpoint=tmp_path / "green.safetensors",
            outdir=tmp_path / "artifacts",
            visualization_mesh=mesh_path,
            save_generated_data=False,
        )
    )

    assert exporter._copy_visualization_mesh(mesh) is None
    assert not (tmp_path / "artifacts" / "data" / "visualization_mesh.npz").exists()


def _write_zero_coefficients(path: Path) -> Path:
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


def _write_spatial_coefficients(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "def a_fun(x, y): return 1.0 + x + 2.0 * y",
                "def apx_fun(x, y): return 1.0 + 0.0 * x",
                "def apy_fun(x, y): return 2.0 + 0.0 * y",
                "def bx_fun(x, y): return x - y",
                "def by_fun(x, y): return 2.0 * x + y",
                "def c_fun(x, y): return x + 3.0 * y - 1.0",
            ]
        )
    )
    return path


def test_coefficient_mesh_evaluates_every_physical_vertex_directly(
    tmp_path: Path,
) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coefficient_path = _write_spatial_coefficients(tmp_path / "coefficients.py")
    exporter = ComplexCouplingArtifactExporter(
        CouplingArtifactRequest(
            config=tmp_path / "config.json",
            coupling_checkpoint=tmp_path / "coupling.safetensors",
            green_checkpoint=tmp_path / "green.safetensors",
            outdir=tmp_path / "artifacts",
        )
    )
    geometry = load_complex_geometry(geometry_path)
    coefficients = load_coefficient_functions(coefficient_path)

    fields = exporter._evaluate_coefficient_mesh_fields(
        mesh,
        geometry,
        coefficients,
    )

    assert fields is not None
    x = mesh.vertices[:, 0]
    y = mesh.vertices[:, 1]
    np.testing.assert_array_equal(fields.coords, mesh.vertices)
    np.testing.assert_allclose(fields.a, 1.0 + x + 2.0 * y)
    np.testing.assert_allclose(fields.bx, x - y)
    np.testing.assert_allclose(fields.by, 2.0 * x + y)
    np.testing.assert_allclose(fields.b_magnitude, np.hypot(x - y, 2.0 * x + y))
    np.testing.assert_allclose(fields.c, x + 3.0 * y - 1.0)
    assert fields.a[0] == pytest.approx(1.0)
    assert fields.a[mesh.auxiliary_vertices[0]] == pytest.approx(2.8)
    assert (
        exporter._evaluate_coefficient_mesh_fields(None, geometry, coefficients) is None
    )


def test_coefficient_mesh_figures_share_color_contract_and_larger_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coefficient_path = _write_spatial_coefficients(tmp_path / "coefficients.py")
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
    )
    exporter = ComplexCouplingArtifactExporter(request)
    geometry = load_complex_geometry(geometry_path)
    coefficients = load_coefficient_functions(coefficient_path)
    valid_fields = exporter._evaluate_coefficient_fields(geometry, coefficients)
    mesh_fields = exporter._evaluate_coefficient_mesh_fields(
        mesh,
        geometry,
        coefficients,
    )
    assert mesh_fields is not None
    terms = CouplingCoefficientTermsConfig(
        diffusion=True,
        convection=True,
        reaction=True,
    )

    _, coefficient_fields = exporter._write_coefficient_figures(
        valid_fields,
        terms,
        "plotly_white",
        _boundary_overlay(),
        mesh_fields,
    )
    paths, mesh_figure_fields = exporter._write_coefficient_mesh_figures(
        mesh,
        mesh_fields,
        coefficient_fields,
        "plotly_white",
    )

    assert mesh_figure_fields == (
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
    )
    assert len(paths) == 5
    assert all("convection_vector" not in path for path in paths)
    scatter = _coefficient_figure(request.outdir, "convection_bx")
    mesh_figure = json.loads(
        (
            request.outdir
            / "figures"
            / "coefficients"
            / "mesh"
            / "convection_bx_mesh.json"
        ).read_text()
    )
    mesh_trace = mesh_figure["data"][0]
    hover_trace = mesh_figure["data"][1]
    assert mesh_trace["intensitymode"] == "vertex"
    assert _plotly_array_size(mesh_trace["intensity"]) == mesh.vertex_count
    assert _plotly_array_size(hover_trace["customdata"]) == mesh.vertex_count
    assert "b_x=%{customdata[0]:.6g}" in hover_trace["hovertemplate"]
    assert all(trace.get("name") != "Domain boundary" for trace in mesh_figure["data"])
    assert mesh_trace["colorscale"] == scatter["data"][0]["marker"]["colorscale"]
    assert mesh_trace["cmin"] == pytest.approx(scatter["data"][0]["marker"]["cmin"])
    assert mesh_trace["cmax"] == pytest.approx(scatter["data"][0]["marker"]["cmax"])
    layout = mesh_figure["layout"]
    assert layout["width"] == 900
    assert layout["height"] == 800
    assert layout["margin"] == {
        "l": 10,
        "r": 70,
        "t": 65,
        "b": 10,
    }
    assert layout["scene"]["aspectratio"]["x"] == pytest.approx(1.5)
    assert layout["scene"]["aspectratio"]["y"] == pytest.approx(1.5)
    assert layout["scene"]["aspectratio"]["z"] == pytest.approx(0.01)
    assert layout["scene"]["camera"]["projection"]["type"] == "orthographic"
    for field in mesh_figure_fields:
        field_scatter = _coefficient_figure(request.outdir, field)["data"][0]["marker"]
        field_mesh = json.loads(
            (
                request.outdir
                / "figures"
                / "coefficients"
                / "mesh"
                / f"{field}_mesh.json"
            ).read_text()
        )["data"][0]
        assert field_mesh["colorscale"] == field_scatter["colorscale"]
        assert field_mesh.get("cmin") == field_scatter.get("cmin")
        assert field_mesh.get("cmax") == field_scatter.get("cmax")


@pytest.mark.parametrize(
    ("vertices", "expected_x", "expected_y"),
    [
        (
            np.asarray([[-2.0, -0.25], [2.0, -0.25], [2.0, 0.25], [-2.0, 0.25]]),
            1.5,
            0.1875,
        ),
        (
            np.asarray([[-0.25, -2.0], [0.25, -2.0], [0.25, 2.0], [-0.25, 2.0]]),
            0.1875,
            1.5,
        ),
    ],
)
def test_mesh_layout_preserves_extreme_physical_aspect_ratios(
    vertices: np.ndarray,
    expected_x: float,
    expected_y: float,
) -> None:
    layout = ComplexCouplingArtifactExporter._mesh_layout(
        title="aspect ratio fixture",
        theme="plotly_white",
        vertices=vertices,
    ).to_plotly_json()

    assert layout["scene"]["aspectratio"] == pytest.approx(
        {"x": expected_x, "y": expected_y, "z": 0.01}
    )


def test_complex_artifact_export_writes_outputs_without_cross_fields(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    visualization_mesh_path, visualization_mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        pre_projection_fusion=ComplexPreProjectionFusionConfig(
            enabled=True,
            mode="residual",
            hidden_dim=8,
            depth=1,
            final_layer_init_scale=0.0,
        ),
        coefficient_terms=CouplingCoefficientTermsConfig(
            diffusion=True,
            convection=True,
            reaction=True,
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(), coupling_path
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coeff_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["coefficient_terms"] = {
        "diffusion": True,
        "convection": True,
        "reaction": True,
    }
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_payload["coupling_model"]["pre_projection_fusion"] = {
        "enabled": True,
        "mode": "residual",
        "hidden_dim": 8,
        "depth": 1,
        "eps": 1e-12,
        "final_layer_init_scale": 0.0,
    }
    config_payload["coupling_training"]["relative_split_consistency"] = {
        "enabled": True,
        "weight": 2.0,
        "mass_weight": 3.0,
        "eps": 1e-12,
    }
    config_payload["coupling_training"]["canonical_energy"] = {"boundary_weight": 0.0}
    config_payload["coupling_training"]["weak_operator_closure"] = {
        "enabled": True,
        "weight": 4.0,
        "eps": 1e-12,
    }
    config_payload["coupling_training"]["optimizer"] = {
        "name": "soap",
        "betas": [0.95, 0.95],
        "profile_step_time": True,
        "soap": {
            "precondition_frequency": 10,
            "max_precondition_dim": 64,
        },
    }
    config_payload["coupling_training"]["best_physics_checkpoint"] = {"enabled": True}
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            device="cpu",
            theme="plotly_white",
            coefficient_vector_max_points=2,
            visualization_mesh=visualization_mesh_path,
        )
    )

    assert summary["geometry_mode"] == "complex"
    assert summary["selected_samples"] == [0]
    assert summary["training_source"] == {"mode": "npz", "indexed_gp": None}
    assert summary["reference_diagnostics"] == {
        "training": True,
        "validation": True,
    }
    reproducibility = summary["training_reproducibility"]
    assert reproducibility["available"] is True
    assert reproducibility["stage"] == "coupling"
    assert reproducibility["base_seed"] == 7
    assert reproducibility["deterministic_algorithms"] is True
    assert reproducibility["source_seed_independent"] is True
    assert set(reproducibility["subseeds"]) == {
        "model",
        "runtime",
        "loader_train",
    }
    assert summary["artifact_dataset_contract"] == "full_reference_test_npz"
    boundary_summary = summary["domain_boundary_overlay"]
    assert boundary_summary == {
        "enabled": True,
        "representation": "open_markers",
        "coordinate_source": ("canonical_boundary_energy_context.endpoint_coords"),
        "point_count": 8,
        "scalar_values_included": False,
        "included_in_metrics": False,
    }
    assert "cross_consistency" not in json.dumps(summary)
    assert (outdir / "summary.json").exists()
    assert (outdir / "metrics" / "per_sample_metrics.csv").exists()
    assert (outdir / "data" / "selected_raw_arrays.npz").exists()
    expected_figure_fields = {
        "rhs",
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
        "u_pred_error",
        "u_phi_error",
        "u_psi_error",
        "u_split_mismatch",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
        "weak_residual_x",
        "weak_residual_y",
        "split_mass_relative_contribution",
        "fusion_base_difference",
        "fusion_network_output_physical",
        "fusion_residual_physical",
        "fusion_fused_difference",
        "fusion_delta_from_base",
    }
    assert set(summary["figure_fields"]) == expected_figure_fields
    assert summary["error_convention"] == "signed_difference"
    assert summary["solution_prediction"] == "u_pred=0.5*(u_phi+u_psi)"
    assert summary["raw_output_space"] == "reference_response"
    assert summary["output_contract_version"] == 6
    assert summary["optimizer"]["name"] == "soap"
    assert summary["optimizer"]["betas"] == (0.95, 0.95)
    assert summary["optimizer"]["soap"]["precondition_frequency"] == 10
    assert summary["optimizer"]["upstream_commit"] == (
        "a1e553530fde97d0e6b307d7c82ac6d38b072340"
    )
    assert summary["balance_projection"]["enabled"] is True
    assert summary["balance_projection"]["mode"] == "physical_symmetric"
    assert summary["balance_projection"]["space"] == "physical_source"
    assert summary["balance_projection"]["uses_reference_targets"] is False
    assert "p=P_raw/Lx^2" in summary["balance_projection"]["formula"]
    assert summary["pre_projection_fusion"]["enabled"] is True
    assert summary["pre_projection_fusion"]["space"] == ("physical_directional_source")
    assert summary["pre_projection_fusion"]["architecture"] == (
        "single_nonlinear_fusion_mlp"
    )
    assert summary["pre_projection_fusion"]["mode"] == "residual"
    assert summary["pre_projection_fusion"]["input"] == [
        "base_difference_over_safe_source_scale",
        "rhs_over_safe_source_scale",
    ]
    assert summary["pre_projection_fusion"]["hidden_dim"] == 8
    assert summary["pre_projection_fusion"]["depth"] == 1
    assert summary["pre_projection_fusion"]["identity_skip"] is True
    assert summary["pre_projection_fusion"]["final_layer_initialization"] == (
        "scaled_torch_linear_default"
    )
    assert summary["pre_projection_fusion"]["final_layer_init_scale"] == 0.0
    assert summary["pre_projection_fusion"]["explicit_geometry_features"] is False
    assert summary["pre_projection_fusion"]["learned_linear_branch"] is False
    assert summary["pre_projection_fusion"]["learned_gate"] is False
    assert summary["pre_projection_fusion"]["pre_projection_balance_constructed"]
    assert summary["pre_projection_fusion"]["uses_reference_targets"] is False
    assert summary["reconstruction_response_input"] == {
        "phi": "projected Phi is used directly",
        "psi": "projected Psi is used directly",
        "additional_length_scaling": False,
    }
    assert summary["reference_targets_used_for_training"] is False
    assert summary["canonical_boundary_energy"] == {
        "enabled": True,
        "optimization_enabled": False,
        "weight": 0.0,
        "diagnostic_always_reported": True,
        "definition": "endpoint_p1_edge",
        "formula": "a_i * r_i^2 * h_perp / d_endpoint",
        "coefficient_evaluation": "one_sided_nearest_valid_point",
        "endpoint_value": 0.0,
        "anchor_count": 8,
        "x_anchor_count": 4,
        "y_anchor_count": 4,
        "covers_all_connected_segment_endpoints": True,
        "uses_reference_targets": False,
    }
    assert summary["canonical_energy"] == {
        "enabled": True,
        "domain": "all_valid_same_segment_edges",
        "bulk_formula": (
            "sum_edges arithmetic_mean(a)*(delta(u_phi-u_psi)/h_axis)^2*hx*hy"
        ),
        "boundary_included": True,
        "boundary_weight": 0.0,
        "optimized_formula": "bulk + boundary_weight * boundary",
        "optimized_metric": "loss_energy_optimized",
        "unweighted_canonical_metric": "loss_energy_consistency",
        "transition_partition": False,
        "checkpoint_metric": "loss_energy_optimized",
        "uses_reference_targets": False,
    }
    assert "loss_energy_optimized_mean" in summary["aggregate_metrics"]
    assert summary["canonical_energy"]["boundary_weight"] == pytest.approx(0.0)
    assert "length_jump_balance" not in summary
    assert summary["relative_split_consistency"] == {
        "enabled": True,
        "weight": 2.0,
        "mass_weight": 3.0,
        "eps": 1e-12,
        "source_normalization": "physical_rhs_l2_squared",
        "domain_length_scale": "max_global_extent",
        "uses_reference_targets": False,
    }
    assert summary["weak_operator_closure"] == {
        "enabled": True,
        "weight": 4.0,
        "eps": 1e-12,
        "trial_solution": "u_equal_mean=0.5*(u_phi+u_psi)",
        "test_space": "directional_segment_p1_nodal",
        "coefficient_evaluation": "direct_at_physical_element_midpoints",
        "reaction_split": "c/2_per_direction",
        "uses_reference_targets": False,
    }
    assert summary["checkpoint_selection"] == {
        "best_energy": True,
        "best_physics": True,
        "reference_metric_used": False,
    }
    assert summary["non_error_color_range_policy"] == (
        "solution_full_range_and_directional_robust_quantile"
    )
    assert summary["non_error_color_range_groups"]["solution"] == [
        "sol",
        "u_pred",
        "u_phi",
        "u_psi",
    ]
    assert summary["optional_flux_targets_exported"] is True
    directional_range = summary["directional_color_range"]
    assert directional_range["configured_quantile"] is None
    assert directional_range["resolved_quantile"] == pytest.approx(0.99)
    assert directional_range["value_policy"] == "shared_lower_upper_quantile"
    assert directional_range["error_policy"] == "symmetric_absolute_quantile"
    sample_ranges = directional_range["samples"]["sample_0000_sample_0000"]
    assert set(sample_ranges) == {
        "rhs",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
    }
    assert sample_ranges["phi"]["display_cmin"] == pytest.approx(
        sample_ranges["target_phi"]["display_cmin"]
    )
    assert sample_ranges["phi_error"]["display_cmin"] == pytest.approx(
        -sample_ranges["phi_error"]["display_cmax"]
    )
    assert summary["coefficient_figure_fields"] == [
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
        "convection_vector",
    ]
    assert summary["coefficient_figure_count"] == 6
    coefficient_mesh_fields = [
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
    ]
    assert summary["coefficient_mesh_figure_fields"] == coefficient_mesh_fields
    assert summary["coefficient_mesh_figure_count"] == 5
    assert summary["coefficient_mesh_evaluation"] == (
        "direct_at_visualization_mesh_vertices"
    )
    assert summary["coefficient_mesh_boundary_value_source"] == (
        "direct_physical_coefficient_function"
    )
    assert summary["coefficient_mesh_intensity_mode"] == "vertex"
    assert summary["mesh_scene_scale"] == pytest.approx(1.5)
    mesh_fields = [
        "sol",
        "u_pred",
        "u_pred_error",
        "rhs",
        "phi",
        "psi",
        "target_phi",
        "target_psi",
        "phi_error",
        "psi_error",
    ]
    assert summary["figure_count"] == len(expected_figure_fields) + 6 + 10 + 5
    assert summary["mesh_figure_fields"] == mesh_fields
    assert summary["mesh_figure_count"] == 10
    mesh_summary = summary["visualization_mesh"]
    assert mesh_summary["schema_version"] == 1
    assert mesh_summary["vertex_count"] == 8
    assert mesh_summary["valid_vertex_count"] == 3
    assert mesh_summary["boundary_vertex_count"] == 4
    assert mesh_summary["auxiliary_vertex_count"] == 1
    assert mesh_summary["valid_point_transfer"] == "exact_vertex_mapping"
    assert mesh_summary["boundary_value_source"] == "field_specific"
    assert mesh_summary["solution_boundary_value_source"] == (
        "prescribed_homogeneous_dirichlet"
    )
    assert mesh_summary["interior_scalar_boundary_value_source"] == "not_evaluated"
    assert mesh_summary["boundary_values_model_evaluated"] is False
    assert mesh_summary["field_space"] == "physical_scalar"
    assert mesh_summary["scene_scale"] == pytest.approx(1.5)
    assert mesh_summary["color_range_policy"]["directional_quantile"] == (
        pytest.approx(0.99)
    )
    assert mesh_summary["field_boundary_policy"]["sol/u_pred/u_pred_error"] == (
        "prescribed_homogeneous_dirichlet_zero_without_outline"
    )
    assert mesh_summary["included_in_metrics"] is False
    assert mesh_summary["raw_archive"] == "data/visualization_mesh.npz"
    assert (outdir / "data" / "visualization_mesh.npz").read_bytes() == (
        visualization_mesh_path.read_bytes()
    )
    assert summary["coefficient_field_space"] == "physical"
    assert summary["coefficient_evaluation"] == "direct_at_coords_valid"
    assert summary["coefficient_raw_archive"] == "data/coefficient_fields.npz"
    for field in coefficient_mesh_fields:
        assert (
            outdir / "figures" / "coefficients" / "mesh" / f"{field}_mesh.json"
        ).exists()
    assert summary["coefficient_vector"]["max_points"] == 2
    assert 0 < summary["coefficient_vector"]["selected_points"] <= 2
    assert summary["coefficient_vector"]["background_points"] == 3
    assert summary["coefficient_field_statistics"]["a"] == {
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
        "physical_nonzero": True,
        "constant": True,
        "branch_enabled": True,
        "figure_exported": True,
    }
    assert summary["coefficient_field_statistics"]["bx"]["min"] == 4.0
    assert summary["coefficient_field_statistics"]["by"]["max"] == 5.0
    assert summary["coefficient_field_statistics"]["c"]["mean"] == 6.0
    assert summary["coefficient_branch_channel_order"] == [
        "a",
        "b_primary",
        "b_transverse",
        "c",
    ]
    assert summary["coefficient_branch_convection"] == "primary_transverse"
    assert (
        summary["coefficient_branch_transverse_convection_scaling"]
        == "primary_segment_length"
    )
    assert summary["transverse_trunk"] == {
        "enabled": True,
        "fusion": "product",
        "length_context": True,
        "features": [
            "t_perpendicular",
            "log(L_perpendicular/L_ref)",
            "log(L_parallel/L_perpendicular)",
            "kappa",
        ],
    }
    for field in expected_figure_fields:
        figure_path = (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        )
        assert figure_path.exists()
        scalar_trace = json.loads(figure_path.read_text())["data"][0]
        assert "value=%{customdata[0]:.6e}" in scalar_trace["hovertemplate"]
        assert _plotly_array_size(scalar_trace["customdata"]) == 3
        _assert_boundary_trace(
            json.loads(figure_path.read_text()),
            expected_count=boundary_summary["point_count"],
        )
    for field in summary["coefficient_figure_fields"]:
        figure_path = outdir / "figures" / "coefficients" / f"{field}.json"
        assert figure_path.exists()
        _assert_boundary_trace(
            json.loads(figure_path.read_text()),
            expected_count=boundary_summary["point_count"],
        )

    with (outdir / "metrics" / "per_sample_metrics.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert rows
    assert float(rows[0]["boundary_weight"]) == pytest.approx(0.0)
    assert "loss_energy_optimized" in rows[0]
    assert "loss_energy_consistency" in rows[0]
    assert "loss_energy_boundary" in rows[0]
    assert "loss_energy_boundary_x" in rows[0]
    assert "loss_energy_boundary_y" in rows[0]
    assert all("cross" not in key for key in rows[0])

    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    assert any(key.endswith("_raw_response_phi") for key in raw.files)
    assert any(key.endswith("_raw_response_psi") for key in raw.files)
    assert any(key.endswith("_raw_physical_phi") for key in raw.files)
    assert any(key.endswith("_raw_physical_psi") for key in raw.files)
    assert any(key.endswith("_projected_response_phi") for key in raw.files)
    assert any(key.endswith("_projected_response_psi") for key in raw.files)
    assert any(key.endswith("_x_length_squared") for key in raw.files)
    assert any(key.endswith("_y_length_squared") for key in raw.files)
    assert any(key.endswith("_raw_difference") for key in raw.files)
    assert any(key.endswith("_projected_difference") for key in raw.files)
    assert any(key.endswith("_raw_response_constraint_residual") for key in raw.files)
    assert any(key.endswith("_response_constraint_residual") for key in raw.files)
    assert any(key.endswith("_base_raw_response_phi") for key in raw.files)
    assert any(key.endswith("_base_raw_response_psi") for key in raw.files)
    assert any(key.endswith("_fusion_base_physical_p") for key in raw.files)
    assert any(key.endswith("_fusion_base_physical_q") for key in raw.files)
    assert any(key.endswith("_fusion_base_difference") for key in raw.files)
    assert any(key.endswith("_fusion_normalized_difference") for key in raw.files)
    assert any(key.endswith("_fusion_normalized_rhs") for key in raw.files)
    assert any(key.endswith("_fusion_network_output_normalized") for key in raw.files)
    assert any(key.endswith("_fusion_network_output_physical") for key in raw.files)
    assert any(key.endswith("_fusion_residual_normalized") for key in raw.files)
    assert any(key.endswith("_fusion_residual_physical") for key in raw.files)
    assert any(key.endswith("_fusion_fused_difference") for key in raw.files)
    assert any(key.endswith("_fusion_delta_from_base") for key in raw.files)
    assert any(key.endswith("_fusion_pre_projection_phi") for key in raw.files)
    assert any(key.endswith("_fusion_pre_projection_psi") for key in raw.files)
    assert any(key.endswith("_fusion_source_scale") for key in raw.files)
    assert any(key.endswith("_fusion_safe_source_scale") for key in raw.files)
    assert any(
        key.endswith("_fusion_pre_projection_balance_residual") for key in raw.files
    )
    assert not any(key.endswith("_linear_difference_component") for key in raw.files)
    assert not any(key.endswith("_nonlinear_difference_component") for key in raw.files)
    assert not any(key.endswith("_combined_difference_component") for key in raw.files)
    assert not any(key.endswith("_fusion_gate") for key in raw.files)
    assert not any(key.endswith("_x_length_jump_score") for key in raw.files)
    assert not any(key.endswith("_y_length_jump_score") for key in raw.files)
    assert not any(key.endswith("_x_transition_edge_mask") for key in raw.files)
    assert not any(key.endswith("_y_transition_edge_mask") for key in raw.files)
    assert any(key.endswith("_x_transverse_length_context") for key in raw.files)
    assert any(key.endswith("_y_transverse_length_context") for key in raw.files)
    assert any(key.endswith("_weak_residual_x") for key in raw.files)
    assert any(key.endswith("_weak_residual_y") for key in raw.files)
    assert any(key.endswith("_weak_nodal_mass_x") for key in raw.files)
    assert any(key.endswith("_weak_nodal_mass_y") for key in raw.files)
    assert any(key.endswith("_split_mass_relative_contribution") for key in raw.files)
    assert any(key.endswith("_boundary_endpoint_coords") for key in raw.files)
    assert any(key.endswith("_boundary_split_residual") for key in raw.files)
    assert any(key.endswith("_boundary_physical_distance") for key in raw.files)
    for suffix in (
        "_u_pred",
        "_u_pred_error",
        "_u_phi_error",
        "_u_psi_error",
        "_u_split_mismatch",
        "_target_phi",
        "_target_psi",
        "_phi_error",
        "_psi_error",
    ):
        assert any(key.endswith(suffix) for key in raw.files)
    assert all("cross" not in key for key in raw.files)
    mesh_color_ranges: dict[str, tuple[float, float]] = {}
    for field in mesh_fields:
        mesh_figure_path = (
            outdir
            / "figures"
            / "mesh"
            / field
            / f"sample_0000_sample_0000_{field}_mesh.json"
        )
        assert mesh_figure_path.exists()
        assert mesh_figure_path.with_suffix(".html").exists()
        assert mesh_figure_path.with_suffix(".png").exists()
        assert mesh_figure_path.with_suffix(".pdf").exists()
        figure = json.loads(mesh_figure_path.read_text())
        mesh_trace = figure["data"][0]
        hover_trace = figure["data"][1]
        assert mesh_trace["type"] == "mesh3d"
        assert hover_trace["type"] == "scatter3d"
        assert hover_trace["mode"] == "markers"
        assert "value=%{customdata[0]:.6e}" in hover_trace["hovertemplate"]
        np.testing.assert_array_equal(
            _plotly_array(mesh_trace["i"]),
            visualization_mesh.triangles[:, 0],
        )
        np.testing.assert_array_equal(
            _plotly_array(mesh_trace["j"]),
            visualization_mesh.triangles[:, 1],
        )
        np.testing.assert_array_equal(
            _plotly_array(mesh_trace["k"]),
            visualization_mesh.triangles[:, 2],
        )
        assert figure["layout"]["scene"]["camera"]["projection"]["type"] == (
            "orthographic"
        )
        intensity = _plotly_array(mesh_trace["intensity"])
        raw_key = f"sample_0000_sample_0000_{field}"
        assert raw_key in raw.files
        hover_values = _plotly_array(hover_trace["customdata"])
        np.testing.assert_array_equal(hover_values[:3], raw[raw_key])
        if field in {"sol", "u_pred", "u_pred_error"}:
            assert mesh_trace["intensitymode"] == "vertex"
            assert len(figure["data"]) == 2
            np.testing.assert_array_equal(
                intensity[visualization_mesh.valid_to_vertex],
                raw[raw_key],
            )
            np.testing.assert_array_equal(
                intensity[visualization_mesh.boundary_vertex_mask],
                0.0,
            )
            np.testing.assert_array_equal(hover_values[3:], 0.0)
        else:
            assert mesh_trace["intensitymode"] == "cell"
            np.testing.assert_allclose(
                intensity,
                visualization_mesh.transfer_interior_cell_values(raw[raw_key]),
            )
            boundary_trace = figure["data"][2]
            assert boundary_trace["type"] == "scatter3d"
            assert boundary_trace["mode"] == "lines"
            assert "scalar unavailable" in boundary_trace["hovertemplate"]
        mesh_color_ranges[field] = (mesh_trace["cmin"], mesh_trace["cmax"])
    assert mesh_color_ranges["sol"] == mesh_color_ranges["u_pred"]
    assert mesh_color_ranges["u_pred_error"][0] == -mesh_color_ranges["u_pred_error"][1]
    for field in ("phi", "psi", "target_phi", "target_psi", "phi_error", "psi_error"):
        marker = _marker_for_field(outdir, field)
        assert mesh_color_ranges[field] == (marker["cmin"], marker["cmax"])
    coefficient_raw = np.load(outdir / "data" / "coefficient_fields.npz")
    assert set(coefficient_raw.files) == {
        "coords_valid",
        "a",
        "bx",
        "by",
        "b_magnitude",
        "c",
        "quiver_indices",
    }
    np.testing.assert_allclose(coefficient_raw["a"], 1.0)
    np.testing.assert_allclose(coefficient_raw["bx"], 4.0)
    np.testing.assert_allclose(coefficient_raw["by"], 5.0)
    np.testing.assert_allclose(coefficient_raw["b_magnitude"], np.sqrt(41.0))
    np.testing.assert_allclose(coefficient_raw["c"], 6.0)
    assert coefficient_raw["quiver_indices"].size <= 2

    bx_marker = _coefficient_figure(outdir, "convection_bx")["data"][0]["marker"]
    assert bx_marker["cmin"] == -4.0
    assert bx_marker["cmax"] == 4.0
    vector_figure = _coefficient_figure(outdir, "convection_vector")
    assert vector_figure["data"][0]["type"] == "scattergl"
    assert vector_figure["data"][0]["showlegend"] is False
    arrow_trace = next(
        trace
        for trace in reversed(vector_figure["data"])
        if trace.get("mode") == "lines"
    )
    arrow_dx = arrow_trace["x"][1] - arrow_trace["x"][0]
    arrow_dy = arrow_trace["y"][1] - arrow_trace["y"][0]
    assert arrow_dx / arrow_dy == pytest.approx(4.0 / 5.0)

    error_figure = json.loads(
        (
            outdir
            / "figures"
            / "u_pred_error"
            / "sample_0000_sample_0000_u_pred_error.json"
        ).read_text()
    )
    marker = error_figure["data"][0]["marker"]
    assert marker["colorscale"]
    assert marker["cmin"] == -marker["cmax"]

    solution_ranges = {
        (
            _marker_for_field(outdir, field)["cmin"],
            _marker_for_field(outdir, field)["cmax"],
        )
        for field in ("sol", "u_pred", "u_phi", "u_psi")
    }
    assert len(solution_ranges) == 1
    phi_range = (
        _marker_for_field(outdir, "target_phi")["cmin"],
        _marker_for_field(outdir, "target_phi")["cmax"],
    )
    assert phi_range == (
        _marker_for_field(outdir, "phi")["cmin"],
        _marker_for_field(outdir, "phi")["cmax"],
    )
    psi_range = (
        _marker_for_field(outdir, "target_psi")["cmin"],
        _marker_for_field(outdir, "target_psi")["cmax"],
    )
    assert psi_range == (
        _marker_for_field(outdir, "psi")["cmin"],
        _marker_for_field(outdir, "psi")["cmax"],
    )

    boundary_off_outdir = tmp_path / "artifacts_boundary_off"
    boundary_off_summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=boundary_off_outdir,
            device="cpu",
            theme="plotly_white",
            coefficient_vector_max_points=2,
            show_domain_boundary=False,
        )
    )
    assert boundary_off_summary["domain_boundary_overlay"] == {
        **boundary_summary,
        "enabled": False,
    }
    assert boundary_off_summary["figure_count"] == (
        summary["figure_count"]
        - summary["mesh_figure_count"]
        - summary["coefficient_mesh_figure_count"]
    )
    assert "coefficient_mesh_figure_count" not in boundary_off_summary
    assert "coefficient_mesh_figure_fields" not in boundary_off_summary
    assert boundary_off_summary["figure_fields"] == summary["figure_fields"]
    assert boundary_off_summary["aggregate_metrics"] == summary["aggregate_metrics"]
    assert "visualization_mesh" not in boundary_off_summary
    assert "mesh_figure_fields" not in boundary_off_summary
    assert not (boundary_off_outdir / "figures" / "mesh").exists()
    boundary_off_figure = json.loads(
        (
            boundary_off_outdir
            / "figures"
            / "u_pred_error"
            / "sample_0000_sample_0000_u_pred_error.json"
        ).read_text()
    )
    assert len(boundary_off_figure["data"]) == 1
    assert boundary_off_figure["data"][0]["type"] == "scattergl"
    assert "showlegend" not in boundary_off_figure["data"][0]
    assert (
        boundary_off_outdir / "metrics" / "per_sample_metrics.csv"
    ).read_bytes() == (outdir / "metrics" / "per_sample_metrics.csv").read_bytes()

    boundary_off_raw = np.load(boundary_off_outdir / "data" / "selected_raw_arrays.npz")
    assert set(boundary_off_raw.files) == set(raw.files)
    for key in raw.files:
        np.testing.assert_equal(boundary_off_raw[key], raw[key])


def test_column_diagonal_green_response_artifact_provenance_and_fields(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(
            mode="column_diagonal_green_response",
            column_diagonal_green_response={
                "gain_squared_eps": 2.0e-12,
                "gain_exponent": 0.25,
            },
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(), coupling_path
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coeff_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "column_diagonal_green_response",
        "column_diagonal_green_response": {
            "gain_squared_eps": 2.0e-12,
            "gain_exponent": 0.25,
        },
    }
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            selected_samples=(0,),
            device="cpu",
            theme="plotly_white",
        )
    )

    projection = summary["balance_projection"]
    response = projection["column_diagonal_green_response"]
    assert projection["mode"] == "column_diagonal_green_response"
    assert projection["raw_physical_difference_preserved"] is False
    assert response["active"] is True
    assert response["gain_squared_eps"] == 2.0e-12
    assert response["gain_exponent"] == 0.25
    assert response["fixed_exponent"] is True
    assert response["learnable_exponent"] is False
    assert "alpha=0.25" in projection["formula"]
    assert "sigmoid" in response["weight_formula"]
    assert response["gain_definition"] == "diag(H_s^T M_Omega H_s)"
    assert response["summation_axis"] == "output_rows_for_each_source_column"
    assert response["row_norm_used"] is False
    assert response["full_gram_solve"] is False
    assert response["global_response_matrix_materialized"] is False
    assert response["context_build_count"] == 1
    assert response["raw_archive"] == ("data/column_diagonal_green_response_fields.npz")
    assert summary["projection_figure_fields"] == [
        "gamma_x_squared",
        "gamma_y_squared",
        "correction_weight_phi",
    ]
    assert summary["projection_figure_count"] == 3

    response_fields = np.load(
        outdir / "data" / "column_diagonal_green_response_fields.npz"
    )
    assert set(response_fields.files) == {
        "gamma_x_squared",
        "gamma_y_squared",
        "regularized_gamma_x_squared",
        "regularized_gamma_y_squared",
        "correction_weight_phi",
        "correction_weight_psi",
        "gain_exponent",
    }
    assert response_fields["gain_exponent"].shape == ()
    assert response_fields["gain_exponent"].item() == 0.25
    np.testing.assert_allclose(
        response_fields["correction_weight_phi"]
        + response_fields["correction_weight_psi"],
        1.0,
    )
    for field in summary["projection_figure_fields"]:
        figure_path = outdir / "figures" / "balance_projection" / f"{field}.json"
        assert figure_path.is_file()
        _assert_boundary_trace(
            json.loads(figure_path.read_text()),
            expected_count=summary["domain_boundary_overlay"]["point_count"],
        )
    weight_figure = json.loads(
        (
            outdir / "figures" / "balance_projection" / "correction_weight_phi.json"
        ).read_text()
    )
    assert "gain exponent=0.25" in weight_figure["layout"]["title"]["text"]

    selected = np.load(outdir / "data" / "selected_raw_arrays.npz")
    for suffix in (
        "_projection_balance_residual_before",
        "_projection_correction_phi",
        "_projection_correction_psi",
        "_projection_correction_weight_phi",
        "_projection_correction_weight_psi",
        "_projection_difference_update",
    ):
        assert any(key.endswith(suffix) for key in selected.files)


@pytest.mark.parametrize(
    ("eta_strategy", "subspace_dimension"),
    [
        ("fixed", 1),
        ("closed_loop_exact_line_search", 1),
        ("closed_loop_exact_line_search", 2),
        ("closed_loop_exact_line_search", 3),
        ("closed_loop_exact_line_search", 4),
    ],
)
def test_symmetric_tangent_green_response_artifact_provenance_and_fields(
    tmp_path,
    monkeypatch,
    eta_strategy,
    subspace_dimension,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "subspace_dimension": subspace_dimension,
                "eta": 0.01,
                "eta_strategy": eta_strategy,
                "line_search_relative_eps": 3.0e-12,
                "relative_lambda": 0.1,
                "denominator_relative_eps": 2.0e-12,
            },
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(), coupling_path
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coeff_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "subspace_dimension": subspace_dimension,
            "eta": 0.01,
            "eta_strategy": eta_strategy,
            "line_search_relative_eps": 3.0e-12,
            "relative_lambda": 0.1,
            "denominator_relative_eps": 2.0e-12,
        },
    }
    if eta_strategy == "closed_loop_exact_line_search":
        config_payload["coupling_training"]["post_line_search_stationarity"] = {
            "enabled": True,
            "weight": 1.0e-3,
            "eps": 2.0e-12,
        }
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            selected_samples=(0,),
            device="cpu",
            theme="plotly_white",
        )
    )

    projection = summary["balance_projection"]
    tangent = projection["symmetric_tangent_green_response"]
    assert projection["mode"] == "symmetric_tangent_green_response"
    assert projection["uses_reference_targets"] is False
    assert tangent["active"] is True
    assert tangent["subspace_dimension"] == subspace_dimension
    assert tangent["eta"] == pytest.approx(0.01)
    assert tangent["eta_strategy"] == eta_strategy
    assert tangent["line_search_relative_eps"] == pytest.approx(3.0e-12)
    assert tangent["relative_lambda"] == pytest.approx(0.1)
    assert tangent["denominator_relative_eps"] == pytest.approx(2.0e-12)
    adaptive = eta_strategy == "closed_loop_exact_line_search"
    subspace = subspace_dimension >= 2
    assert tangent["fixed_parameters"] is (not adaptive and not subspace)
    assert tangent["sample_adaptive"] is (adaptive or subspace)
    assert tangent["batch_independent"] is True
    assert tangent["differentiable_eta"] is (adaptive and not subspace)
    assert tangent["differentiable_subspace_coefficients"] is subspace
    assert tangent["learnable_parameters"] is False
    assert tangent["row_norm_used"] is False
    assert tangent["global_response_matrix_materialized"] is False
    assert tangent["full_gram_solve"] is False
    assert tangent["context_build_count"] == 1
    assert tangent["raw_archive"] == (
        "data/symmetric_tangent_green_response_fields.npz"
    )
    stationarity_summary = summary["post_line_search_stationarity"]
    assert stationarity_summary["enabled"] is adaptive
    assert stationarity_summary["weight"] == pytest.approx(1.0e-3 if adaptive else 1.0)
    assert stationarity_summary["eps"] == pytest.approx(
        2.0e-12 if adaptive else 1.0e-12
    )
    assert stationarity_summary["eta_source"] == (
        "not_applicable" if subspace else "uncapped_eta_star"
    )
    assert stationarity_summary["forward_eta_source"] == (
        "not_applicable" if subspace else "capped_eta_applied"
    )
    assert stationarity_summary["matrix_free"] is True
    assert stationarity_summary["extra_adjoint_actions_per_enabled_batch"] == (
        0 if subspace else 1
    )
    assert stationarity_summary["global_response_matrix_materialized"] is False
    assert stationarity_summary["full_gram_solve"] is False
    assert stationarity_summary["uses_reference_targets"] is False
    assert "m0=H_x*p_tilde-H_y*q_tilde" in projection["formula"]
    assert summary["projection_figure_fields"] == [
        "tangent_preconditioner_base",
        "tangent_denominator",
    ]

    context_fields = np.load(
        outdir / "data" / "symmetric_tangent_green_response_fields.npz"
    )
    expected_context_fields = {
        "gamma_x_squared",
        "gamma_y_squared",
        "preconditioner_base",
        "denominator",
        "point_mass",
        "eta",
        "eta_strategy",
        "line_search_relative_eps",
        "relative_lambda",
        "denominator_relative_eps",
    }
    if subspace:
        expected_context_fields.update({"subspace_dimension", "eta_applicability"})
    assert set(context_fields.files) == expected_context_fields
    assert context_fields["eta"].item() == pytest.approx(0.01)
    assert context_fields["eta_strategy"].item() == eta_strategy
    assert context_fields["line_search_relative_eps"].item() == pytest.approx(3.0e-12)
    assert np.all(context_fields["denominator"] > 0.0)
    for field in summary["projection_figure_fields"]:
        figure_path = outdir / "figures" / "balance_projection" / f"{field}.json"
        assert figure_path.is_file()
        _assert_boundary_trace(
            json.loads(figure_path.read_text()),
            expected_count=summary["domain_boundary_overlay"]["point_count"],
        )

    selected = np.load(outdir / "data" / "selected_raw_arrays.npz")
    for suffix in (
        "_symmetric_physical_phi",
        "_symmetric_physical_psi",
        "_symmetric_u_phi",
        "_symmetric_u_psi",
        "_tangent_mismatch_pre",
        "_tangent_gradient",
        "_tangent_preconditioner_base",
        "_tangent_denominator",
        "_tangent_delta",
        "_tangent_mismatch_post",
    ):
        assert any(key.endswith(suffix) for key in selected.files)
    metric_rows = list(
        csv.DictReader((outdir / "metrics" / "per_sample_metrics.csv").open())
    )
    assert "tangent_response_mismatch_pre" in metric_rows[0]
    assert "tangent_response_mismatch_post" in metric_rows[0]
    if subspace:
        assert tangent["eta_role"] == "k1_only_not_applied"
        assert tangent["eta_applicability"] == "k1_only_not_applied"
        assert tangent["eta_cap_schedule"]["applicable"] is False
        assert tangent["eta_cap_schedule"]["training_policy"] == "not_applied"
        expected_direction_contract = (
            "two_jacobi_preconditioned_response_orthogonal_directions"
            if subspace_dimension == 2
            else (
                f"{subspace_dimension}_jacobi_preconditioned_"
                "response_orthogonal_directions"
            )
        )
        assert tangent["direction_contract"] == expected_direction_contract
        assert tangent["linear_solve_used"] is False
        assert context_fields["subspace_dimension"].item() == subspace_dimension
        assert context_fields["eta_applicability"].item() == ("k1_only_not_applied")
        assert "z0=D^-1*g" in projection["formula"]
        assert "eta_star_statistics" not in tangent
        assert "coefficient_0_statistics" in tangent
        assert "coefficient_1_statistics" in tangent
        assert "response_cost_k1_statistics" in tangent
        assert "response_cost_k2_statistics" in tangent
        assert "second_direction_active_fraction" in tangent
        for direction_index in range(subspace_dimension):
            assert f"coefficient_{direction_index}_statistics" in tangent
            assert f"response_cost_k{direction_index + 1}_statistics" in tangent
            assert f"direction_{direction_index}_active_fraction" in tangent
        for suffix in (
            "_tangent_direction_0",
            "_tangent_direction_1",
            "_tangent_response_direction_0",
            "_tangent_response_direction_1",
            "_tangent_coefficient_0",
            "_tangent_coefficient_1",
            "_tangent_second_direction_active",
            "_tangent_mismatch_k1",
            "_tangent_response_cost_k1",
            "_tangent_response_cost_k2",
            "_tangent_residual_gradient_post",
            "_tangent_stationarity_residual",
        ):
            assert any(key.endswith(suffix) for key in selected.files)
        assert metric_rows[0]["tangent_subspace_dimension"] == str(subspace_dimension)
        for direction_index in range(subspace_dimension):
            assert f"tangent_coefficient_{direction_index}" in metric_rows[0]
            assert f"tangent_direction_{direction_index}_active" in metric_rows[0]
            assert f"tangent_response_cost_k{direction_index + 1}" in metric_rows[0]
            for suffix in (
                f"_tangent_direction_{direction_index}",
                f"_tangent_directional_response_{direction_index}",
                f"_tangent_response_direction_{direction_index}",
                f"_tangent_coefficient_{direction_index}",
                f"_tangent_direction_{direction_index}_active",
                f"_tangent_mismatch_k{direction_index + 1}",
                f"_tangent_response_cost_k{direction_index + 1}",
            ):
                assert any(key.endswith(suffix) for key in selected.files)
    elif adaptive:
        assert tangent["eta_role"] == "final_safety_cap"
        assert tangent["eta_cap_schedule"]["validation_policy"] == "final_cap"
        assert tangent["eta_cap_schedule"]["post_warmup_behavior"] == ("hold_final_eta")
        assert "eta_star_statistics" in tangent
        assert "eta_applied_statistics" in tangent
        assert "eta_cap_hit_fraction" in tangent
        for suffix in (
            "_tangent_response_direction",
            "_tangent_line_search_numerator",
            "_tangent_line_search_denominator",
            "_tangent_eta_star",
            "_tangent_eta_applied",
            "_tangent_eta_cap",
            "_tangent_eta_capped",
            "_tangent_hessian_direction",
            "_tangent_stationarity_residual",
            "_tangent_stationarity_ratio",
        ):
            assert any(key.endswith(suffix) for key in selected.files)
        assert "tangent_eta_star" in metric_rows[0]
        assert "tangent_eta_applied" in metric_rows[0]
        assert "tangent_eta_capped" in metric_rows[0]
        assert "loss_tangent_post_line_search_stationarity" in metric_rows[0]
        assert "tangent_post_line_search_stationarity_ratio" in metric_rows[0]
    else:
        assert tangent["eta_role"] == "fixed_step"
        assert "eta_star_statistics" not in tangent
        assert not any(
            key.endswith("_tangent_stationarity_ratio") for key in selected.files
        )


def test_joint_response_trust_stationarity_artifact_records_shared_provenance(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coeff_path = write_coefficients(tmp_path / "coeffs.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "eta": 0.01,
                "eta_strategy": "closed_loop_exact_line_search",
                "line_search_relative_eps": 3.0e-12,
                "relative_lambda": 0.1,
                "denominator_relative_eps": 2.0e-12,
            },
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(), coupling_path
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coeff_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "symmetric_tangent_green_response",
        "symmetric_tangent_green_response": {
            "eta": 0.01,
            "eta_strategy": "closed_loop_exact_line_search",
            "line_search_relative_eps": 3.0e-12,
            "relative_lambda": 0.1,
            "denominator_relative_eps": 2.0e-12,
        },
    }
    config_payload["coupling_training"]["response_trust"] = {
        "enabled": True,
        "weight": 1.0e-3,
        "trust_weight": 0.01,
        "eps": 2.0e-12,
    }
    config_payload["coupling_training"]["post_line_search_stationarity"] = {
        "enabled": True,
        "weight": 1.0e-4,
        "eps": 3.0e-12,
    }
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            selected_samples=(0,),
            device="cpu",
            theme="plotly_white",
        )
    )

    response_summary = summary["response_trust"]
    assert response_summary["enabled"] is True
    assert response_summary["weight"] == pytest.approx(1.0e-3)
    assert response_summary["trust_weight"] == pytest.approx(0.01)
    assert response_summary["eps"] == pytest.approx(2.0e-12)
    assert response_summary["eta_source"] == "capped_eta_applied"
    assert response_summary["stationarity_diagnostic_computed"] is True
    assert response_summary["joint_stationarity_optimized"] is True
    assert response_summary["source_response_shared_with_stationarity"] is True
    assert response_summary["extra_forward_actions_per_enabled_batch"] == 1
    assert response_summary["extra_adjoint_actions_per_enabled_batch"] == 1
    assert response_summary["global_response_matrix_materialized"] is False
    assert response_summary["full_gram_solve"] is False
    assert response_summary["uses_reference_targets"] is False
    stationarity_summary = summary["post_line_search_stationarity"]
    assert stationarity_summary["enabled"] is True
    assert stationarity_summary["optimized"] is True
    assert stationarity_summary["diagnostic_computed"] is True
    assert stationarity_summary["weight"] == pytest.approx(1.0e-4)
    assert stationarity_summary["eps"] == pytest.approx(3.0e-12)
    assert stationarity_summary["optimization_normalization"] == (
        "source_response_energy"
    )
    assert stationarity_summary["legacy_ratio_optimized"] is False
    assert stationarity_summary["joint_response_trust_enabled"] is True
    assert (
        stationarity_summary[
            "shared_source_response_forward_actions_per_computed_batch"
        ]
        == 1
    )

    metric_rows = list(
        csv.DictReader((outdir / "metrics" / "per_sample_metrics.csv").open())
    )
    assert "loss_tangent_response_trust" in metric_rows[0]
    assert "tangent_response_trust_ratio" in metric_rows[0]
    assert "tangent_response_post_mismatch_ratio" in metric_rows[0]
    assert "tangent_response_correction_ratio" in metric_rows[0]
    assert "tangent_source_response_energy" in metric_rows[0]
    assert "tangent_post_line_search_stationarity_source_normalized" in metric_rows[0]
    assert "tangent_post_line_search_stationarity_ratio" in metric_rows[0]
    assert "tangent_stationarity_initial_source_ratio" in metric_rows[0]
    assert "loss_tangent_post_line_search_stationarity" in metric_rows[0]

    selected = np.load(outdir / "data" / "selected_raw_arrays.npz")
    for suffix in (
        "_tangent_source_response_phi",
        "_tangent_source_response_psi",
        "_tangent_source_response_energy_density",
        "_tangent_response_correction",
        "_tangent_source_response_energy",
        "_tangent_response_post_mismatch_ratio",
        "_tangent_response_correction_ratio",
        "_tangent_response_trust_ratio",
        "_tangent_stationarity_ratio",
        "_tangent_stationarity_source_normalized",
        "_tangent_stationarity_initial_source_ratio",
        "_tangent_stationarity_initial_preconditioned_energy",
        "_tangent_stationarity_residual_preconditioned_energy",
    ):
        assert any(key.endswith(suffix) for key in selected.files)
    for field in (
        "tangent_source_response_energy_density",
        "tangent_response_correction",
        "tangent_mismatch_post",
    ):
        assert field in summary["figure_fields"]
        assert (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).is_file()


def test_local_weak_reliability_artifact_uses_weighted_official_prediction(
    tmp_path,
    monkeypatch,
):
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coefficient_path = write_coefficients(tmp_path / "coefficients.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        cross_axis_reconstruction=ComplexCrossAxisReconstructionConfig(
            enabled=True,
            gamma=0.5,
            smoothing_steps=2,
            smoothing_relaxation=0.5,
            relative_floor=0.1,
        ),
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(),
        coupling_path,
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coefficient_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_payload["coupling_model"]["cross_axis_reconstruction"] = {
        "enabled": True,
        "mode": "local_weak_residual_reliability",
        "gamma": 0.5,
        "smoothing_steps": 2,
        "smoothing_relaxation": 0.5,
        "relative_floor": 0.1,
        "eps": 1e-12,
    }
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            selected_samples=(0,),
            device="cpu",
            theme="plotly_white",
        )
    )

    reliability = summary["cross_axis_reconstruction"]
    assert reliability["enabled"] is True
    assert reliability["mode"] == "local_weak_residual_reliability"
    assert reliability["uses_reference_targets"] is False
    assert reliability["affects_training_objective"] is False
    assert reliability["uses_global_matrix_solve"] is False
    assert reliability["requires_global_matrix_solve"] is False
    assert reliability["geometry_only_and_mismatch_modes_available"] is False
    assert reliability["geometry_only_mode_available"] is False
    assert reliability["mismatch_detected_mode_available"] is False
    assert reliability["context_build_count"] == 1
    assert summary["solution_prediction"] == ("u_pred=w_phi*u_phi+(1-w_phi)*u_psi")
    expected_figures = {
        "u_equal_mean_error",
        "weak_reliability_eta_phi_squared",
        "weak_reliability_eta_psi_squared",
        "weak_reliability_theta",
        "weak_reliability_weight_phi",
    }
    assert expected_figures.issubset(set(summary["figure_fields"]))
    for field in expected_figures:
        assert (
            outdir / "figures" / field / f"sample_0000_sample_0000_{field}.json"
        ).is_file()

    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    prefix = "sample_0000_sample_0000_"
    u_phi = raw[f"{prefix}u_phi"]
    u_psi = raw[f"{prefix}u_psi"]
    weight_phi = raw[f"{prefix}weak_reliability_weight_phi"]
    weight_psi = raw[f"{prefix}weak_reliability_weight_psi"]
    np.testing.assert_allclose(weight_phi + weight_psi, 1.0)
    np.testing.assert_allclose(
        raw[f"{prefix}u_pred"],
        weight_phi * u_phi + weight_psi * u_psi,
    )
    np.testing.assert_allclose(
        raw[f"{prefix}u_equal_mean"],
        0.5 * (u_phi + u_psi),
    )
    for field in (
        "weak_reliability_residual_phi_x",
        "weak_reliability_residual_phi_y",
        "weak_reliability_residual_phi_full",
        "weak_reliability_residual_psi_x",
        "weak_reliability_residual_psi_y",
        "weak_reliability_residual_psi_full",
        "weak_reliability_eta_phi_squared_raw",
        "weak_reliability_eta_psi_squared_raw",
        "weak_reliability_eta_phi_squared",
        "weak_reliability_eta_psi_squared",
        "weak_reliability_sample_floor",
        "weak_reliability_theta",
    ):
        assert f"{prefix}{field}" in raw.files

    with (outdir / "metrics" / "per_sample_metrics.csv").open() as fp:
        row = next(csv.DictReader(fp))
    assert "rel_sol" in row
    assert "rel_sol_equal_mean" in row
    assert "weak_weight_phi_mean" in row
    assert "weak_support_fraction" in row


def test_complex_coefficient_artifacts_distinguish_physical_and_branch_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coefficient_path = _write_zero_coefficients(tmp_path / "coefficients.py")
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
        coefficient_vector_max_points=2,
    )
    exporter = ComplexCouplingArtifactExporter(request)
    geometry = load_complex_geometry(geometry_path)
    coefficients = load_coefficient_functions(coefficient_path)
    fields = exporter._evaluate_coefficient_fields(geometry, coefficients)
    mesh_fields = exporter._evaluate_coefficient_mesh_fields(
        mesh,
        geometry,
        coefficients,
    )
    terms = CouplingCoefficientTermsConfig(
        diffusion=False,
        convection=False,
        reaction=False,
    )

    figure_fields = exporter._coefficient_figure_fields(fields, terms)
    statistics = exporter._coefficient_field_statistics(fields, terms, figure_fields)

    assert figure_fields == ("diffusion_a",)
    assert statistics["a"]["physical_nonzero"] is True
    assert statistics["a"]["branch_enabled"] is False
    assert statistics["a"]["figure_exported"] is True
    assert statistics["b_magnitude"]["physical_nonzero"] is False
    assert statistics["b_magnitude"]["figure_exported"] is False
    assert statistics["c"]["physical_nonzero"] is False
    assert statistics["c"]["figure_exported"] is False
    mesh_paths, mesh_figure_fields = exporter._write_coefficient_mesh_figures(
        mesh,
        mesh_fields,
        figure_fields,
        "plotly_white",
    )
    assert mesh_figure_fields == ("diffusion_a",)
    assert len(mesh_paths) == 1


@pytest.mark.parametrize("mode", ["residual", "absolute"])
def test_complex_artifact_records_single_fusion_mlp_semantics(
    tmp_path: Path,
    monkeypatch,
    mode: str,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    coefficient_path = _write_zero_coefficients(tmp_path / "coefficients.py")
    data_dir = tmp_path / "test_data"
    write_sample_npz(data_dir)
    coupling_cfg = CouplingModelConfig(
        branch_input_dim=4,
        hidden_dim=4,
        depth=1,
        dtype=torch.float64,
        balance_projection=BalanceProjectionConfig(mode="physical_symmetric"),
        pre_projection_fusion=ComplexPreProjectionFusionConfig(
            enabled=True,
            mode=mode,
            hidden_dim=8,
            depth=1,
            final_layer_init_scale=0.0,
        ),
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
    green_cfg = ModelConfig(
        hidden_dim=4,
        depth=1,
        branch_input_dim=4,
        use_green=False,
        dtype=torch.float64,
    )
    coupling_path = tmp_path / "complex_coupling.safetensors"
    green_path = tmp_path / "green.safetensors"
    save_state_dict_safetensors(
        ComplexCouplingNet(coupling_cfg).state_dict(),
        coupling_path,
    )
    save_model_with_config(GreenONetModel(green_cfg), green_cfg, green_path)
    config_path = write_complex_config(
        tmp_path / "config.json",
        geometry_path=geometry_path,
        train_path=None,
        test_path=data_dir,
        coefficient_path=coefficient_path,
    )
    config_payload = json.loads(config_path.read_text())
    config_payload["coupling_model"]["balance_projection"] = {
        "enabled": True,
        "mode": "physical_symmetric",
    }
    config_payload["coupling_model"]["pre_projection_fusion"] = {
        "enabled": True,
        "mode": mode,
        "hidden_dim": 8,
        "depth": 1,
        "eps": 1e-12,
        "final_layer_init_scale": 0.0,
    }
    config_path.write_text(json.dumps(config_payload))
    outdir = tmp_path / "artifacts"

    summary = export_complex_coupling_artifacts(
        CouplingArtifactRequest(
            config=config_path,
            coupling_checkpoint=coupling_path,
            green_checkpoint=green_path,
            outdir=outdir,
            device="cpu",
            theme="plotly_white",
        )
    )

    fusion_summary = summary["pre_projection_fusion"]
    assert fusion_summary["architecture"] == "single_nonlinear_fusion_mlp"
    assert fusion_summary["mode"] == mode
    assert fusion_summary["identity_skip"] is (mode == "residual")
    assert fusion_summary["final_layer_initialization"] == "scaled_torch_linear_default"
    assert fusion_summary["final_layer_init_scale"] == 0.0
    if mode == "residual":
        assert fusion_summary["formula"].startswith("d_fused=d_base+")
    else:
        assert fusion_summary["formula"].startswith("d_fused=A_safe*")
    assert fusion_summary["explicit_geometry_features"] is False
    assert fusion_summary["learned_linear_branch"] is False
    assert fusion_summary["learned_gate"] is False
    raw = np.load(outdir / "data" / "selected_raw_arrays.npz")
    assert any(key.endswith("_fusion_network_output_normalized") for key in raw.files)
    assert any(key.endswith("_fusion_network_output_physical") for key in raw.files)
    assert any(key.endswith("_fusion_fused_difference") for key in raw.files)
    assert any(key.endswith("_fusion_delta_from_base") for key in raw.files)
    assert any(key.endswith("_fusion_residual_normalized") for key in raw.files) is (
        mode == "residual"
    )
    assert any(key.endswith("_fusion_residual_physical") for key in raw.files) is (
        mode == "residual"
    )
    assert not any(key.endswith("_linear_difference_component") for key in raw.files)
    assert not any(key.endswith("_nonlinear_difference_component") for key in raw.files)
    assert not any(key.endswith("_combined_difference_component") for key in raw.files)
    assert not any(key.endswith("_linear_difference_correction") for key in raw.files)
    assert not any(
        key.endswith("_nonlinear_difference_correction") for key in raw.files
    )
    assert not any(key.endswith("_blended_difference_correction") for key in raw.files)
    assert not any(key.endswith("_fusion_gate") for key in raw.files)


def test_complex_coefficient_artifacts_export_enabled_zero_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_static_export(monkeypatch)
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coefficient_path = _write_zero_coefficients(tmp_path / "coefficients.py")
    request = CouplingArtifactRequest(
        config=tmp_path / "config.json",
        coupling_checkpoint=tmp_path / "coupling.safetensors",
        green_checkpoint=tmp_path / "green.safetensors",
        outdir=tmp_path / "artifacts",
        save_generated_data=False,
        coefficient_vector_max_points=1,
    )
    exporter = ComplexCouplingArtifactExporter(request)
    geometry = load_complex_geometry(geometry_path)
    coefficients = load_coefficient_functions(coefficient_path)
    fields = exporter._evaluate_coefficient_fields(geometry, coefficients)
    mesh_fields = exporter._evaluate_coefficient_mesh_fields(
        mesh,
        geometry,
        coefficients,
    )
    terms = CouplingCoefficientTermsConfig(
        diffusion=False,
        convection=True,
        reaction=True,
    )

    paths, figure_fields = exporter._write_coefficient_figures(
        fields,
        terms,
        "plotly_white",
        _boundary_overlay(),
    )
    exporter._write_coefficient_npz(fields)
    mesh_paths, mesh_figure_fields = exporter._write_coefficient_mesh_figures(
        mesh,
        mesh_fields,
        figure_fields,
        "plotly_white",
    )

    assert len(paths) == 6
    assert figure_fields == (
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
        "convection_vector",
    )
    assert mesh_figure_fields == (
        "diffusion_a",
        "reaction_c",
        "convection_bx",
        "convection_by",
        "convection_magnitude",
    )
    assert len(mesh_paths) == 5
    vector_figure = _coefficient_figure(request.outdir, "convection_vector")
    annotations = vector_figure["layout"]["annotations"]
    assert any("Zero convection field" in item["text"] for item in annotations)
    assert not (request.outdir / "data" / "coefficient_fields.npz").exists()
