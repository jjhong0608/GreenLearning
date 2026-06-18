from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import load_complex_geometry
from greenonet.fenicsx_samples import (
    GaussianProcessSourceSampler,
    GeometryGridLoader,
    SampleWriter,
)
from test.complex_fixtures import write_coefficients, write_geometry_npz


def _load_cli_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_fenicsx_samples.py"
    )
    spec = importlib.util.spec_from_file_location("make_fenicsx_samples", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_fenicsx_samples.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_fenicsx_samples = _load_cli_module()
MakeFenicsxSamplesCLI = make_fenicsx_samples.MakeFenicsxSamplesCLI


def _write_grid_geometry(path: Path) -> Path:
    return write_geometry_npz(
        path,
        grid_x=np.linspace(0.0, 1.0, 5, dtype=np.float64),
        grid_y=np.linspace(0.0, 1.0, 5, dtype=np.float64),
    )


def test_geometry_grid_loader_requires_grid_axes(tmp_path):
    path = write_geometry_npz(tmp_path / "geometry.npz")

    with pytest.raises(KeyError, match="grid_x"):
        GeometryGridLoader().load(path)


def test_geometry_grid_loader_loads_grid_and_valid_indices(tmp_path):
    path = _write_grid_geometry(tmp_path / "geometry.npz")

    geometry = GeometryGridLoader().load(path)

    assert geometry.full_grid_shape == (5, 5)
    assert geometry.num_valid_points == 3
    np.testing.assert_allclose(geometry.grid_x, np.linspace(0.0, 1.0, 5))


def test_gp_source_sampler_is_reproducible_on_rectangular_grid():
    grid_x = np.linspace(-1.0, 1.0, 5)
    grid_y = np.linspace(-0.5, 0.5, 3)

    first = GaussianProcessSourceSampler(grid_x, grid_y, seed=123).sample()
    second = GaussianProcessSourceSampler(grid_x, grid_y, seed=123).sample()
    different = GaussianProcessSourceSampler(grid_x, grid_y, seed=124).sample()

    assert first.shape == (3, 5)
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, different)


def test_sample_writer_schema_is_readable_by_complex_coupling_dataset(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    geometry_grid = GeometryGridLoader().load(geometry_path)
    full = np.zeros(geometry_grid.full_grid_shape, dtype=np.float64)
    valid_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rhs = geometry_grid.valid_values_to_full_grid(valid_values)
    sol = geometry_grid.valid_values_to_full_grid(valid_values + 10.0)
    phi = geometry_grid.valid_values_to_full_grid(valid_values + 20.0)
    psi = geometry_grid.valid_values_to_full_grid(valid_values + 30.0)
    path = SampleWriter(
        tmp_path / "samples", geometry_grid.full_grid_shape
    ).write_sample(
        "train",
        0,
        rhs=rhs,
        sol=sol,
        phi=phi,
        psi=psi,
    )

    with np.load(path) as raw:
        assert set(raw.files) == {"rhs", "sol", "phi", "psi"}
        np.testing.assert_allclose(raw["rhs"] + full, rhs)

    dataset = ComplexCouplingDataset(
        path.parent,
        load_complex_geometry(geometry_path),
        load_coefficient_functions(write_coefficients(tmp_path / "coeffs.py")),
        branch_input_dim=4,
        dtype=torch.float64,
    )
    item = dataset[0]

    torch.testing.assert_close(
        item.rhs_valid,
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        item.flux_valid[0],
        torch.tensor([21.0, 22.0, 23.0], dtype=torch.float64),
    )


def test_make_fenicsx_samples_cli_rejects_invalid_domain_source(tmp_path):
    cli = MakeFenicsxSamplesCLI()
    common = [
        "--geometry",
        str(tmp_path / "geometry.npz"),
        "--out",
        str(tmp_path / "out"),
        "--num-train",
        "1",
        "--num-valid",
        "0",
        "--num-test",
        "0",
    ]

    with pytest.raises(SystemExit):
        cli.parse_config(common)
    with pytest.raises(SystemExit):
        cli.parse_config(
            [
                *common,
                "--gmsh-script",
                str(tmp_path / "domain.py"),
                "--msh",
                str(tmp_path / "domain.msh"),
            ]
        )


def test_make_fenicsx_samples_cli_rejects_negative_sample_count(tmp_path):
    cli = MakeFenicsxSamplesCLI()

    with pytest.raises(ValueError, match="num-train"):
        cli.parse_config(
            [
                "--geometry",
                str(tmp_path / "geometry.npz"),
                "--out",
                str(tmp_path / "out"),
                "--gmsh-script",
                str(tmp_path / "domain.py"),
                "--num-train",
                "-1",
                "--num-valid",
                "0",
                "--num-test",
                "0",
            ]
        )
