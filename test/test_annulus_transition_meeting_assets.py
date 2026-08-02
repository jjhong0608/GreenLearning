from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT / "docs" / "meeting" / "annulus_transition_error" / "build_assets.py"
)
SPEC = importlib.util.spec_from_file_location(
    "annulus_transition_meeting_build_assets", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

AnnulusMeetingAssetBuilder = MODULE.AnnulusMeetingAssetBuilder
MeetingAssetConfig = MODULE.MeetingAssetConfig


EXPECTED_HTML_ASSETS = {
    "annulus_transition_sample47_error_matrix.html",
    "annulus_transition_sample47_error_matrix_marked.html",
    "geometry_c2_method_sample0.html",
    "mismatch_seam_c2_method_sample0.html",
    "weak_residual_reliability_method_sample0.html",
    "poisson_four_way_rel_sol.html",
    "poisson_weak_sample47_inset.html",
    "poisson_cdr_rule_comparison.html",
    "poisson_weak_result_fields_sample0.html",
    "poisson_weak_result_errors_sample0.html",
    "cdr_weak_result_fields_sample9.html",
    "cdr_weak_result_errors_sample9.html",
}


def test_result_colorbar_places_title_beside_bar() -> None:
    colorbar = AnnulusMeetingAssetBuilder._result_colorbar("f", 0.77)

    assert colorbar["title"] == {
        "text": "f",
        "side": "right",
        "font": {"size": 13, "color": MODULE.INK},
    }
    assert colorbar["xanchor"] == "left"
    assert colorbar["xpad"] == 10
    assert colorbar["thickness"] == 13
    assert colorbar["tickfont"]["size"] == 11


def test_meeting_asset_builder_uses_locked_artifacts_offline(tmp_path: Path) -> None:
    outdir = tmp_path / "assets"
    config = MeetingAssetConfig(
        project_root=PROJECT_ROOT,
        outdir=outdir,
        overwrite=True,
    )

    manifest = AnnulusMeetingAssetBuilder(config).run()

    assert set(manifest["assets"]) == EXPECTED_HTML_ASSETS
    assert manifest["offline_plotly"] is True
    assert manifest["builder_version"] == 3
    assert len(manifest["plotly_bundle_provenance"]["sha256"]) == 64
    assert manifest["plotly_bundle_provenance"]["size_bytes"] > 0
    assert manifest["final_prediction_contract"].startswith(
        "u_weak_residual_reliability"
    )
    assert manifest["method_detail_contract"] == {
        "sample_id": 0,
        "geometry_c2": "known topology; sample independent",
        "mismatch_seam_c2": "u_phi-u_psi only; no reference target",
        "weak_residual_reliability": (
            "local full PDE defect; no reference target or global solve"
        ),
    }
    assert manifest["legacy_sample_47"]["sample_id"] == 47
    assert manifest["legacy_sample_47"]["source_error_limit"] == pytest.approx(
        1.2686799967180011
    )
    assert manifest["poisson"]["comparison"]["num_samples"] == 50
    assert (
        manifest["poisson"]["comparison"]["estimators"]["weak_residual_reliability"][
            "win_count_vs_equal"
        ]
        == 50
    )
    assert (
        manifest["cdr"]["comparison"]["estimators"]["weak_residual_reliability"][
            "win_count_vs_equal"
        ]
        == 49
    )
    assert manifest["poisson"]["representative_sample"][
        "weak_rel_sol"
    ] == pytest.approx(0.04824259717361187)
    assert manifest["cdr"]["representative_sample"]["weak_rel_sol"] == pytest.approx(
        0.04573526837658398
    )
    assert (
        manifest["length_response_diagnostic"]["unit_physical_equivalence"]["passed"]
        is True
    )

    assert (outdir / "manifest.json").is_file()
    assert (outdir / "plotly.min.js").is_file()

    provenance = manifest["source_provenance"]
    referenced_sources = {
        source
        for metadata in manifest["assets"].values()
        for source in metadata["source_files"]
    }
    referenced_sources.update(manifest["length_response_diagnostic"]["source_files"])
    assert referenced_sources.issubset(provenance)
    for record in provenance.values():
        assert len(record["sha256"]) == 64
        assert record["size_bytes"] > 0
    for metadata in manifest["assets"].values():
        assert len(metadata["generated_sha256"]) == 64
        assert metadata["generated_size_bytes"] > 0
    assert {path.name for path in outdir.glob("*.html")} == EXPECTED_HTML_ASSETS
    for html_path in outdir.glob("*.html"):
        html = html_path.read_text()
        assert "plotly.min.js" in html
        assert "cdn.plot.ly" not in html
        assert "https://" not in html
        assert "http://" not in html
        assert 'style="height:100vh; width:100vw;"' in html


def test_weak_archive_requires_final_prediction_field(tmp_path: Path) -> None:
    weak_root = tmp_path / "weak"
    archive_path = weak_root / "data" / "selected_weak_residual_blend_arrays.npz"
    archive_path.parent.mkdir(parents=True)
    coords = np.asarray([[0.0, 0.0], [0.1, 0.1]], dtype=np.float64)
    sample = np.asarray([[1.0, 2.0]], dtype=np.float64)
    projected = np.asarray([[[0.5, 1.0], [0.5, 1.0]]], dtype=np.float64)
    np.savez(
        archive_path,
        selected_sample_ids=np.asarray([0], dtype=np.int64),
        coords_valid=coords,
        sol=sample,
        rhs=sample,
        projected_physical=projected,
        u_phi=sample,
        u_psi=sample,
        u_equal_mean=sample,
        u_geometry_c2=sample,
        u_mismatch_seam_c2=sample,
        weak_w_phi=sample,
        weak_w_psi=sample,
    )
    builder = AnnulusMeetingAssetBuilder(
        MeetingAssetConfig(project_root=PROJECT_ROOT, outdir=tmp_path / "assets")
    )

    with pytest.raises(ValueError, match="u_weak_residual_reliability"):
        builder.load_weak_sample(weak_root, 0)
