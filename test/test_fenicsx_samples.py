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
    geometry_path = tmp_path / "geometry" / "unit_circle_h05.npz"
    make_circular.CircularGeometryBuilder(
        make_circular.CircularGeometryConfig(
            step_size=0.5,
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
    gmsh_script = tmp_path / "disk_domain.py"
    gmsh_script.write_text(
        "\n".join(
            [
                "def build_domain(gmsh, context):",
                "    tag = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1.0, 1.0)",
                "    gmsh.model.occ.synchronize()",
                "    return {'surface_tags': [tag]}",
            ]
        )
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
            "0.35",
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
        assert raw["rhs"].shape == (5, 5)
        valid_mask = raw["rhs"] != 0.0
        assert np.any(valid_mask)
        assert np.isfinite(raw["sol"]).all()
        assert np.isfinite(raw["phi"]).all()
        assert np.isfinite(raw["psi"]).all()
