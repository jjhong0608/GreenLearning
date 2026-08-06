from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


FENICSX_AVAILABLE = (
    importlib.util.find_spec("dolfinx") is not None
    and importlib.util.find_spec("gmsh") is not None
)

pytestmark = pytest.mark.skipif(
    not FENICSX_AVAILABLE,
    reason="FEniCSx/Gmsh integration tests require the green_fenicsx environment.",
)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fenicsx_generator_writes_one_unit_disk_sample(tmp_path):
    make_circular = _load_module(
        "make_circular_geometry_for_fenicsx_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_circular_geometry.py",
    )
    make_fenicsx = _load_module(
        "make_fenicsx_samples_for_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py",
    )
    validate_samples = _load_module(
        "validate_complex_samples_for_fenicsx_test",
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py",
    )
    geometry_path = tmp_path / "geometry" / "unit_circle_h025.npz"
    make_circular.CircularGeometryBuilder(
        make_circular.CircularGeometryConfig(
            step_size=0.25,
            out=geometry_path,
        )
    ).write()
    coeffs_path = tmp_path / "coeffs.py"
    coeffs_path.write_text(
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
    gmsh_script = (
        Path(__file__).resolve().parents[1] / "examples" / "unit_circle_gmsh.py"
    )

    summary = make_fenicsx.MakeFenicsxSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "samples"),
            "--gmsh-script",
            str(gmsh_script),
            "--num-train",
            "1",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--mesh-size",
            "0.035",
            "--solution-degree",
            "3",
            "--target-degree",
            "2",
            "--coefficients",
            str(coeffs_path),
        ]
    )

    sample_path = tmp_path / "samples" / "train" / "sample_000000.npz"
    assert sample_path.is_file()
    assert (tmp_path / "samples" / "make_fenicsx_samples.log").is_file()
    assert (tmp_path / "samples" / "generation_summary.json").is_file()
    assert summary["num_samples"] == 1
    assert summary["vertex_coverage_max_distance"] is not None
    with np.load(sample_path) as raw:
        assert set(raw.files) == {"rhs", "sol", "phi", "psi"}
        assert raw["rhs"].shape == (9, 9)
        valid_mask = raw["rhs"] != 0.0
        assert np.any(valid_mask)
        assert np.isfinite(raw["sol"]).all()
        assert np.isfinite(raw["phi"]).all()
        assert np.isfinite(raw["psi"]).all()

    validation = validate_samples.ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(tmp_path / "samples"),
            "--splits",
            "train",
            "--coefficients",
            str(coeffs_path),
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "1e-2",
        ]
    )
    assert validation["max_balance_residual"] <= 1.0e-2


def test_fenicsx_generator_writes_parallel_unit_disk_samples(tmp_path):
    make_circular = _load_module(
        "make_circular_geometry_for_fenicsx_parallel_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_circular_geometry.py",
    )
    make_fenicsx = _load_module(
        "make_fenicsx_samples_for_parallel_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py",
    )
    validate_samples = _load_module(
        "validate_complex_samples_for_fenicsx_parallel_test",
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py",
    )
    geometry_path = tmp_path / "geometry" / "unit_circle_h025.npz"
    make_circular.CircularGeometryBuilder(
        make_circular.CircularGeometryConfig(
            step_size=0.25,
            out=geometry_path,
        )
    ).write()
    coeffs_path = tmp_path / "coeffs.py"
    coeffs_path.write_text(
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
    gmsh_script = (
        Path(__file__).resolve().parents[1] / "examples" / "unit_circle_gmsh.py"
    )

    summary = make_fenicsx.MakeFenicsxSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "samples_parallel"),
            "--gmsh-script",
            str(gmsh_script),
            "--num-train",
            "2",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--mesh-size",
            "0.035",
            "--solution-degree",
            "3",
            "--target-degree",
            "2",
            "--coefficients",
            str(coeffs_path),
            "--num-workers",
            "2",
            "--sample-seed-policy",
            "indexed",
        ]
    )

    sample_root = tmp_path / "samples_parallel"
    assert (sample_root / "train" / "sample_000000.npz").is_file()
    assert (sample_root / "train" / "sample_000001.npz").is_file()
    assert summary["num_samples"] == 2
    assert summary["sample_counts"] == {"train": 2, "valid": 0, "test": 0}
    assert summary["parallel"]["num_workers"] == 2
    assert summary["parallel"]["sample_seed_policy"] == "indexed"
    assert summary["parallel"]["task_count"] == 2

    validation = validate_samples.ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(sample_root),
            "--splits",
            "train",
            "--coefficients",
            str(coeffs_path),
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "5e-2",
        ]
    )
    assert validation["max_balance_residual"] <= 5.0e-2


def test_fenicsx_generator_matches_non_unit_geometry_radius(tmp_path):
    make_circular = _load_module(
        "make_circular_geometry_for_fenicsx_radius_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_circular_geometry.py",
    )
    make_fenicsx = _load_module(
        "make_fenicsx_samples_for_radius_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py",
    )
    validate_samples = _load_module(
        "validate_complex_samples_for_fenicsx_radius_test",
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py",
    )
    geometry_path = tmp_path / "geometry" / "circle_r15_h075.npz"
    make_circular.CircularGeometryBuilder(
        make_circular.CircularGeometryConfig(
            step_size=0.75,
            radius=1.5,
            out=geometry_path,
        )
    ).write()
    coeffs_path = tmp_path / "coeffs.py"
    coeffs_path.write_text(
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
    gmsh_script = (
        Path(__file__).resolve().parents[1] / "examples" / "unit_circle_gmsh.py"
    )

    summary = make_fenicsx.MakeFenicsxSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "samples_radius"),
            "--gmsh-script",
            str(gmsh_script),
            "--num-train",
            "1",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--mesh-size",
            "0.06",
            "--solution-degree",
            "3",
            "--target-degree",
            "2",
            "--coefficients",
            str(coeffs_path),
        ]
    )

    sample_root = tmp_path / "samples_radius"
    assert (sample_root / "train" / "sample_000000.npz").is_file()
    assert summary["geometry_metadata"]["radius"] == pytest.approx(1.5)
    assert summary["vertex_coverage_max_distance"] is not None
    with np.load(sample_root / "train" / "sample_000000.npz") as raw:
        assert raw["rhs"].shape == (5, 5)
        assert np.isfinite(raw["sol"]).all()

    validation = validate_samples.ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(sample_root),
            "--splits",
            "train",
            "--coefficients",
            str(coeffs_path),
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "1e-1",
        ]
    )
    assert validation["max_balance_residual"] <= 1.0e-1


def test_fenicsx_generator_writes_one_annulus_sample(tmp_path):
    make_annular = _load_module(
        "make_annular_geometry_for_fenicsx_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_annular_geometry.py",
    )
    make_fenicsx = _load_module(
        "make_fenicsx_samples_for_annulus_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py",
    )
    validate_samples = _load_module(
        "validate_complex_samples_for_annulus_test",
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py",
    )
    geometry_path = tmp_path / "geometry" / "annulus_r05_r10_h025.npz"
    make_annular.AnnularGeometryBuilder(
        make_annular.AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.25,
            out=geometry_path,
        )
    ).write()
    coeffs_path = tmp_path / "coeffs.py"
    coeffs_path.write_text(
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
    gmsh_script = Path(__file__).resolve().parents[1] / "examples" / "annulus_gmsh.py"

    summary = make_fenicsx.MakeFenicsxSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "samples_annulus"),
            "--gmsh-script",
            str(gmsh_script),
            "--num-train",
            "1",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--mesh-size",
            "0.035",
            "--solution-degree",
            "3",
            "--target-degree",
            "2",
            "--coefficients",
            str(coeffs_path),
        ]
    )

    sample_root = tmp_path / "samples_annulus"
    sample_path = sample_root / "train" / "sample_000000.npz"
    assert sample_path.is_file()
    assert summary["geometry_metadata"]["domain_type"] == "annulus"
    assert summary["geometry_metadata"]["inner_radius"] == pytest.approx(0.5)
    assert summary["geometry_metadata"]["outer_radius"] == pytest.approx(1.0)
    assert summary["vertex_coverage_max_distance"] is not None
    with np.load(sample_path) as raw:
        assert set(raw.files) == {"rhs", "sol", "phi", "psi"}
        assert raw["rhs"].shape == (9, 9)
        assert np.isfinite(raw["sol"]).all()
        assert np.isfinite(raw["phi"]).all()
        assert np.isfinite(raw["psi"]).all()

    validation = validate_samples.ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(sample_root),
            "--splits",
            "train",
            "--coefficients",
            str(coeffs_path),
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "1e-1",
        ]
    )
    assert validation["max_balance_residual"] <= 1.0e-1


def test_fenicsx_generator_writes_one_pentagram_sample(tmp_path):
    make_pentagram = _load_module(
        "make_pentagram_geometry_for_fenicsx_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_pentagram_geometry.py",
    )
    make_fenicsx = _load_module(
        "make_fenicsx_samples_for_pentagram_test",
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py",
    )
    validate_samples = _load_module(
        "validate_complex_samples_for_pentagram_test",
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py",
    )
    geometry_path = tmp_path / "geometry" / "pentagram_r10_h025.npz"
    make_pentagram.PentagramGeometryBuilder(
        make_pentagram.PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=0.25,
            out=geometry_path,
        )
    ).write()
    coeffs_path = tmp_path / "coeffs.py"
    coeffs_path.write_text(
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
    gmsh_script = Path(__file__).resolve().parents[1] / "examples" / "pentagram_gmsh.py"

    summary = make_fenicsx.MakeFenicsxSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(tmp_path / "samples_pentagram"),
            "--gmsh-script",
            str(gmsh_script),
            "--num-train",
            "1",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--mesh-size",
            "0.035",
            "--solution-degree",
            "3",
            "--target-degree",
            "2",
            "--coefficients",
            str(coeffs_path),
        ]
    )

    sample_root = tmp_path / "samples_pentagram"
    sample_path = sample_root / "train" / "sample_000000.npz"
    assert sample_path.is_file()
    assert summary["geometry_metadata"]["domain_type"] == "regular_pentagram"
    assert summary["geometry_metadata"]["outer_radius"] == pytest.approx(1.0)
    assert summary["geometry_metadata"]["inner_radius"] == pytest.approx(
        1.0 / make_pentagram.PentagramGeometryBuilder.GOLDEN_RATIO**2
    )
    assert summary["geometry_metadata"]["orientation_angle"] == pytest.approx(
        np.pi / 2.0
    )
    assert np.asarray(summary["geometry_metadata"]["boundary_vertices"]).shape == (
        10,
        2,
    )
    assert summary["vertex_coverage_max_distance"] is not None
    with np.load(sample_path) as raw:
        assert set(raw.files) == {"rhs", "sol", "phi", "psi"}
        assert raw["rhs"].shape == (9, 9)
        assert np.isfinite(raw["sol"]).all()
        assert np.isfinite(raw["phi"]).all()
        assert np.isfinite(raw["psi"]).all()

    validation = validate_samples.ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(sample_root),
            "--splits",
            "train",
            "--coefficients",
            str(coeffs_path),
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "1e-1",
        ]
    )
    assert validation["max_balance_residual"] <= 1.0e-1
