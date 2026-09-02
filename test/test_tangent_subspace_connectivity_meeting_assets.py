from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "docs"
    / "meeting"
    / "tangent_subspace_connectivity"
    / "build_assets.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "tangent_subspace_connectivity_meeting_build_assets",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPECTED_ASSETS = {
    "annulus_transition_sample47_error_matrix.html",
    "geometry_all_domains_k_reach.html",
    "geometry_square_k_reach.html",
    "geometry_disk_k_reach.html",
    "geometry_annulus_k_reach.html",
    "geometry_pentagram_k_reach.html",
    "pentagram_trained_k_quality.html",
    "pentagram_cost_quality_tradeoff.html",
    "unit_square_training_size.html",
    "unit_square_paired_seed_errors.html",
    "pentagram_problem_coefficients.html",
    "pentagram_sample79_rhs.html",
    "pentagram_sample79_directional.html",
    "pentagram_sample79_solution.html",
    "pentagram_test_distribution.html",
    "unit_square_problem_coefficient.html",
    "unit_square_sample11_rhs.html",
    "unit_square_sample11_directional.html",
    "unit_square_sample11_solution.html",
    "unit_square_test_distribution.html",
}


def test_asset_builder_uses_only_frozen_offline_evidence(tmp_path: Path) -> None:
    module = _load_module()
    outdir = tmp_path / "assets"
    config = module.MeetingAssetConfig(
        project_root=PROJECT_ROOT,
        outdir=outdir,
        overwrite=True,
    )

    manifest = module.TangentSubspaceMeetingAssetBuilder(config).run()

    assert manifest["builder_version"] == 3
    assert manifest["offline_plotly"] is True
    assert manifest["model_inference_used"] is False
    assert set(manifest["assets"]) == EXPECTED_ASSETS
    assert manifest["geometry_contract"]["selected_k"] == {
        "square": 2,
        "disk": 2,
        "annulus": 4,
        "pentagram": 4,
    }
    assert manifest["geometry_contract"]["representative_reach_percent"][
        "annulus"
    ] == pytest.approx([0.009269558769, 31.2106043752, 98.2665925102, 100.0])
    assert manifest["geometry_contract"]["representative_reach_percent"][
        "pentagram"
    ] == pytest.approx([0.02187226597, 83.6176727909, 98.8188976378, 99.8687664042])
    assert manifest["pentagram_contract"]["rel_sol_percent"] == pytest.approx(
        [2.678, 1.590, 1.234, 1.112]
    )
    assert manifest["pentagram_contract"]["rel_u_phi_percent"] == pytest.approx(
        [5.022, 3.289, 2.552, 2.394]
    )
    assert manifest["pentagram_contract"]["rel_u_psi_percent"] == pytest.approx(
        [4.832, 2.711, 2.120, 1.865]
    )
    assert manifest["pentagram_contract"]["rel_flux_percent"] == pytest.approx(
        [46.558, 39.658, 35.261, 31.930]
    )
    assert manifest["pentagram_contract"]["forward_backward_ms"] == pytest.approx(
        [141.373, 211.807, 282.183, 361.773]
    )
    assert manifest["unit_square_contract"]["rel_sol_percent"] == pytest.approx(
        [0.4254767115, 0.3931179074, 0.3695945505, 0.3504999896]
    )
    assert manifest["unit_square_contract"]["selected_num_train"] == 4800
    assert manifest["unit_square_contract"]["paired_seed_improvements"] == [4, 4, 4]
    pentagram = manifest["benchmark_contracts"]["pentagram"]
    unit_square = manifest["benchmark_contracts"]["unit_square"]
    assert pentagram["test_sample_count"] == 100
    assert pentagram["representative_sample"] == 79
    assert pentagram["subspace_dimension"] == 4
    assert pentagram["coefficient_terms"] == {
        "diffusion": True,
        "convection": True,
        "reaction": True,
    }
    assert pentagram["distributions"]["rel_sol"]["mean"] == pytest.approx(
        0.011122212653144637
    )
    assert pentagram["distributions"]["rel_sol"]["median"] == pytest.approx(
        0.00966121209825456
    )
    assert pentagram["distributions"]["rel_sol"]["p95"] == pytest.approx(
        0.01842735980520419
    )
    assert unit_square["test_sample_count"] == 100
    assert unit_square["representative_sample"] == 11
    assert unit_square["subspace_dimension"] == 4
    assert unit_square["coefficient_terms"] == {
        "diffusion": False,
        "convection": False,
        "reaction": False,
    }
    assert unit_square["distributions"]["rel_sol"]["mean"] == pytest.approx(
        0.0034930418293645615
    )
    assert unit_square["distributions"]["rel_flux"]["p95"] == pytest.approx(
        0.037523682631437794
    )

    for name in (
        "pentagram_test_distribution.html",
        "unit_square_test_distribution.html",
    ):
        assert manifest["assets"][name]["metric_keys"] == [
            "rel_sol",
            "rel_flux",
            "loss_energy_consistency",
            "tangent_response_mismatch_ratio",
        ]

    assert (outdir / "manifest.json").is_file()
    assert (outdir / "plotly.min.js").is_file()
    for metadata in manifest["assets"].values():
        assert metadata["source_files"]
        assert metadata["metric_keys"]
        assert len(metadata["generated_sha256"]) == 64
        assert metadata["generated_size_bytes"] > 0
    for metadata in manifest["source_provenance"].values():
        assert len(metadata["sha256"]) == 64
        assert metadata["size_bytes"] > 0

    for html_path in outdir.glob("*.html"):
        html = html_path.read_text()
        if 'data-asset-kind="static-mesh-grid"' in html:
            assert "data:image/png;base64," in html
        else:
            assert "plotly.min.js" in html
        assert "cdn.plot.ly" not in html
        assert "https://" not in html
        assert "http://" not in html


def test_asset_builder_is_deterministic_with_explicit_div_ids(tmp_path: Path) -> None:
    module = _load_module()
    outdir = tmp_path / "assets"
    config = module.MeetingAssetConfig(
        project_root=PROJECT_ROOT,
        outdir=outdir,
        overwrite=True,
    )

    first = module.TangentSubspaceMeetingAssetBuilder(config).run()
    second = module.TangentSubspaceMeetingAssetBuilder(config).run()

    assert {
        name: value["generated_sha256"] for name, value in first["assets"].items()
    } == {name: value["generated_sha256"] for name, value in second["assets"].items()}


def test_distribution_figures_fit_the_responsive_iframe_height(tmp_path: Path) -> None:
    module = _load_module()
    config = module.MeetingAssetConfig(
        project_root=PROJECT_ROOT,
        outdir=tmp_path / "assets",
        overwrite=True,
    )
    builder = module.TangentSubspaceMeetingAssetBuilder(config)

    for artifact_root, representative_sample, label in (
        (builder.paths.pentagram_artifact_root, 79, "Pentagram"),
        (builder.paths.unit_square_artifact_root, 11, "Unit square"),
    ):
        figure, _ = builder._distribution_figure(
            artifact_root=artifact_root,
            representative_sample=representative_sample,
            label=label,
        )

        assert figure.layout.height is None
        assert figure.layout.margin.b == 72
        assert figure.layout.xaxis3.automargin is True
        assert figure.layout.xaxis4.automargin is True
        assert figure.layout.xaxis3.title.standoff == 8
        assert figure.layout.xaxis4.title.standoff == 8


def test_asset_builder_source_has_no_model_runtime_dependency() -> None:
    source = MODULE_PATH.read_text()

    assert "import torch" not in source
    assert "safetensors" not in source
    assert ".safetensors" not in source
    assert "load_state_dict" not in source
    assert "CouplingNet" not in source
