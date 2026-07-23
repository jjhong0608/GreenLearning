from __future__ import annotations

from pathlib import Path

from greenonet.complex_energy_refinement import (
    ComplexEnergyRefinementAnalyzer,
    ComplexEnergyRefinementRequest,
)
from test.test_make_annular_geometry import (
    AnnularGeometryBuilder,
    AnnularGeometryConfig,
)


def _annulus(path: Path, step_size: float) -> Path:
    AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=step_size,
            out=path,
        )
    ).write()
    return path


def test_persistent_annulus_jump_has_inverse_h_canonical_energy(tmp_path):
    geometries = tuple(
        _annulus(tmp_path / f"annulus_{index}.npz", step_size)
        for index, step_size in enumerate((0.125, 0.0625, 0.03125))
    )
    analyzer = ComplexEnergyRefinementAnalyzer(
        ComplexEnergyRefinementRequest(
            geometries=geometries,
            outdir=tmp_path / "audit",
            exponent_min=-1.25,
            exponent_max=-0.75,
            scaled_energy_relative_spread_max=0.35,
        )
    )

    summary = analyzer.run()

    assert summary["acceptance"]["passed"]
    assert -1.25 <= summary["canonical_energy_log_slope"] <= -0.75
    assert summary["canonical_h_times_energy_relative_spread"] <= 0.35
    assert (tmp_path / "audit" / "summary.json").is_file()
    assert (tmp_path / "audit" / "refinement_metrics.csv").is_file()
    assert (tmp_path / "audit" / "analysis_report.md").is_file()
