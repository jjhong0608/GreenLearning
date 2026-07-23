from __future__ import annotations

import json
from pathlib import Path

from greenonet.complex_energy_nullspace import (
    ComplexEnergyNullspaceAnalyzer,
    ComplexEnergyNullspaceRequest,
)
from test.test_make_annular_geometry import (
    AnnularGeometryBuilder,
    AnnularGeometryConfig,
)


def _write_annulus(tmp_path: Path) -> Path:
    geometry_path = tmp_path / "annulus.npz"
    AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.125,
            out=geometry_path,
        )
    ).write()
    return geometry_path


def test_annulus_bulk_energy_has_one_constant_mode(tmp_path: Path) -> None:
    analyzer = ComplexEnergyNullspaceAnalyzer(
        ComplexEnergyNullspaceRequest(
            geometry=_write_annulus(tmp_path),
            outdir=tmp_path / "analysis",
        )
    )

    summary = analyzer.run()
    stages = {stage["name"]: stage for stage in summary["stages"]}

    assert summary["geometry"]["bulk_components"] == 1
    assert stages["bulk"]["nullity"] == 1


def test_annulus_general_boundary_anchor_removes_constant_mode(
    tmp_path: Path,
) -> None:
    analyzer = ComplexEnergyNullspaceAnalyzer(
        ComplexEnergyNullspaceRequest(
            geometry=_write_annulus(tmp_path),
            outdir=tmp_path / "analysis",
        )
    )

    summary = analyzer.run()
    stages = {stage["name"]: stage for stage in summary["stages"]}
    conclusions = summary["conclusions"]

    assert (
        summary["constraint_families"]["general_segment_boundary_anchor"]["row_count"]
        > 0
    )
    assert stages["bulk_plus_general_boundary"]["nullity"] == 0
    assert conclusions["general_boundary_anchor_required"]
    assert conclusions["general_boundary_anchor_sufficient"]
    assert conclusions["carrier_objective_required"] is False


def test_nullspace_analysis_writes_reproducible_outputs(tmp_path: Path) -> None:
    outdir = tmp_path / "analysis"
    analyzer = ComplexEnergyNullspaceAnalyzer(
        ComplexEnergyNullspaceRequest(
            geometry=_write_annulus(tmp_path),
            outdir=outdir,
        )
    )

    summary = analyzer.run()
    stored = json.loads((outdir / "summary.json").read_text())

    assert stored == summary
    assert (outdir / "nullspace_stages.csv").is_file()
    report = (outdir / "analysis_report.md").read_text()
    assert "General segment boundary anchors" in report
    assert "Carrier objective required" in report
