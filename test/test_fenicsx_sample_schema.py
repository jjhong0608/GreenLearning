from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
import torch

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import ComplexCouplingDataset
from greenonet.complex_geometry import load_complex_geometry
from greenonet.fenicsx_samples.fenicsx_runtime import GmshIOAdapter
from greenonet.fenicsx_samples import (
    FenicsxSampleConfig,
    GaussianProcessSourceSampler,
    GeometryGridLoader,
    SampleWriter,
    build_sample_tasks,
    derive_indexed_seed,
    partition_tasks,
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
    with np.load(path) as raw:
        payload = {
            key: raw[key] for key in raw.files if key not in {"grid_x", "grid_y"}
        }
    np.savez(path, **payload)

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


def test_gp_source_sampler_sample_with_seed_is_order_independent():
    grid_x = np.linspace(-1.0, 1.0, 5)
    grid_y = np.linspace(-0.5, 0.5, 3)
    sampler = GaussianProcessSourceSampler(grid_x, grid_y, seed=123)

    first = sampler.sample_with_seed(derive_indexed_seed(7, "train", 0))
    sampler.sample()
    second = sampler.sample_with_seed(derive_indexed_seed(7, "train", 0))

    np.testing.assert_allclose(first, second)


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


def test_sample_writer_rejects_existing_file_unless_overwrite(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    geometry_grid = GeometryGridLoader().load(geometry_path)
    rhs = np.zeros(geometry_grid.full_grid_shape, dtype=np.float64)
    writer = SampleWriter(tmp_path / "samples", geometry_grid.full_grid_shape)
    writer.write_sample("train", 0, rhs=rhs, sol=rhs, phi=rhs, psi=rhs)

    with pytest.raises(FileExistsError, match="Sample already exists"):
        writer.write_sample("train", 0, rhs=rhs, sol=rhs, phi=rhs, psi=rhs)

    overwritten = np.ones(geometry_grid.full_grid_shape, dtype=np.float64)
    SampleWriter(
        tmp_path / "samples",
        geometry_grid.full_grid_shape,
        overwrite=True,
    ).write_sample(
        "train",
        0,
        rhs=overwritten,
        sol=overwritten,
        phi=overwritten,
        psi=overwritten,
    )
    with np.load(tmp_path / "samples" / "train" / "sample_000000.npz") as raw:
        np.testing.assert_allclose(raw["rhs"], overwritten)


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


def test_fenicsx_sample_config_rejects_invalid_parallel_options(tmp_path):
    common = {
        "geometry": tmp_path / "geometry.npz",
        "out": tmp_path / "out",
        "gmsh_script": tmp_path / "domain.py",
        "msh": None,
        "num_train": 1,
        "num_valid": 0,
        "num_test": 0,
    }

    with pytest.raises(ValueError, match="num-workers"):
        FenicsxSampleConfig(**common, num_workers=0)
    with pytest.raises(ValueError, match="sample-seed-policy"):
        FenicsxSampleConfig(**common, sample_seed_policy="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="num-workers > 1"):
        FenicsxSampleConfig(**common, num_workers=2)
    with pytest.raises(ValueError, match="cannot both"):
        FenicsxSampleConfig(**common, overwrite=True, skip_existing=True)


def test_make_fenicsx_samples_cli_parses_parallel_options(tmp_path):
    cli = MakeFenicsxSamplesCLI()

    config = cli.parse_config(
        [
            "--geometry",
            str(tmp_path / "geometry.npz"),
            "--out",
            str(tmp_path / "out"),
            "--gmsh-script",
            str(tmp_path / "domain.py"),
            "--num-train",
            "1",
            "--num-valid",
            "0",
            "--num-test",
            "0",
            "--num-workers",
            "2",
            "--sample-seed-policy",
            "indexed",
            "--skip-existing",
        ]
    )

    assert config.num_workers == 2
    assert config.sample_seed_policy == "indexed"
    assert config.skip_existing is True


def test_sample_task_planner_and_partition_are_stable(tmp_path):
    config = FenicsxSampleConfig(
        geometry=tmp_path / "geometry.npz",
        out=tmp_path / "out",
        gmsh_script=tmp_path / "domain.py",
        msh=None,
        num_train=2,
        num_valid=1,
        num_test=1,
        sample_seed_policy="indexed",
    )

    tasks = build_sample_tasks(config)
    partitions = partition_tasks(tasks, num_workers=3)
    flattened = [task for _, batch in partitions for task in batch]

    assert [(task.split, task.index) for task in tasks] == [
        ("train", 0),
        ("train", 1),
        ("valid", 0),
        ("test", 0),
    ]
    assert [task.global_ordinal for task in tasks] == [0, 1, 2, 3]
    assert tasks[0].seed == derive_indexed_seed(config.seed, "train", 0)
    assert sorted(task.global_ordinal for task in flattened) == [0, 1, 2, 3]


def test_unit_circle_gmsh_example_imports():
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "unit_circle_gmsh.py"
    )
    spec = importlib.util.spec_from_file_location("unit_circle_gmsh", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load unit_circle_gmsh.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(module.build_domain)


def test_unit_circle_gmsh_radius_reader_uses_geometry_metadata(tmp_path):
    module_path = (
        Path(__file__).resolve().parents[1] / "examples" / "unit_circle_gmsh.py"
    )
    spec = importlib.util.spec_from_file_location(
        "unit_circle_gmsh_radius", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load unit_circle_gmsh.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "radius_only.npz"
    np.savez(path, radius=np.array(2.0, dtype=np.float64))

    assert module.UnitCircleDomainBuilder.radius_from_context(
        SimpleNamespace(geometry_path=path)
    ) == pytest.approx(2.0)
    assert module.UnitCircleDomainBuilder.radius_from_context(
        SimpleNamespace(geometry_path=None)
    ) == pytest.approx(1.0)

    bad_path = tmp_path / "bad_radius.npz"
    np.savez(bad_path, radius=np.array(0.0, dtype=np.float64))
    with pytest.raises(ValueError, match="positive"):
        module.UnitCircleDomainBuilder.radius_from_context(
            SimpleNamespace(geometry_path=bad_path)
        )


def test_gmsh_io_adapter_normalizes_tuple_return():
    class TupleGmshModule:
        @staticmethod
        def model_to_mesh(*args, **kwargs):
            return "mesh", "cell_tags", "facet_tags"

        @staticmethod
        def read_from_msh(*args, **kwargs):
            return "msh_mesh", "msh_cell_tags", "msh_facet_tags"

    adapter = GmshIOAdapter(TupleGmshModule())

    assert adapter.model_to_mesh("model", "comm", 0, gdim=2) == (
        "mesh",
        "cell_tags",
        "facet_tags",
    )
    assert adapter.read_from_msh("domain.msh", "comm", 0, gdim=2) == (
        "msh_mesh",
        "msh_cell_tags",
        "msh_facet_tags",
    )


def test_gmsh_io_adapter_normalizes_mesh_data_return():
    class MeshData:
        mesh = "mesh"
        cell_tags = "cell_tags"
        facet_tags = "facet_tags"

    class MeshDataGmshModule:
        @staticmethod
        def model_to_mesh(*args, **kwargs):
            return MeshData()

        @staticmethod
        def read_from_msh(*args, **kwargs):
            return MeshData()

    adapter = GmshIOAdapter(MeshDataGmshModule())

    assert adapter.model_to_mesh("model", "comm", 0, gdim=2) == (
        "mesh",
        "cell_tags",
        "facet_tags",
    )
