from __future__ import annotations

import json
import sys

import numpy as np
import pytest

from cli.make_complex_sources import MakeComplexSourcesCLI
from greenonet.complex_sources import (
    ComplexSourceGenerationConfig,
    ComplexSourceGenerator,
    GeometryGridLoader,
    IndexedGpComplexSourceProvider,
    IndexedGpParameters,
)
from test.complex_fixtures import write_geometry_npz


def test_source_generator_writes_rhs_only_and_matches_runtime_provider(tmp_path):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    out = tmp_path / "sources"
    config = ComplexSourceGenerationConfig(
        geometry=geometry_path,
        out=out,
        num_train=3,
        num_valid=1,
        lengthscale=0.2,
        amplitude=1.0,
        mean=0.0,
        seed=7,
        validate=True,
    )

    summary = ComplexSourceGenerator(config).run()

    sample_path = out / "train" / "sample_000002.npz"
    with np.load(sample_path, allow_pickle=False) as raw:
        assert raw.files == ["rhs"]
        stored_rhs = np.asarray(raw["rhs"])
    geometry = GeometryGridLoader().load(geometry_path)
    runtime = IndexedGpComplexSourceProvider(
        geometry,
        split="train",
        sample_count=3,
        parameters=IndexedGpParameters(
            seed=7,
            lengthscale=0.2,
            amplitude=1.0,
            mean=0.0,
        ),
    )
    np.testing.assert_array_equal(stored_rhs, runtime[2].rhs)
    assert summary["sample_counts"] == {"train": 3, "valid": 1}
    assert summary["seed_policy"] == "indexed"
    assert summary["outside_domain"] == 0.0
    assert (out / "generation_summary.json").is_file()


def test_source_generator_rejects_existing_sample_without_overwrite(tmp_path):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    out = tmp_path / "sources"
    config = ComplexSourceGenerationConfig(
        geometry=geometry_path,
        out=out,
        num_train=1,
        num_valid=0,
    )
    ComplexSourceGenerator(config).run()

    with pytest.raises(FileExistsError, match="Sample already exists"):
        ComplexSourceGenerator(config).run()

    overwritten = ComplexSourceGenerationConfig(
        geometry=geometry_path,
        out=out,
        num_train=1,
        num_valid=0,
        overwrite=True,
    )
    ComplexSourceGenerator(overwritten).run()
    payload = json.loads((out / "generation_summary.json").read_text())
    assert payload["sample_counts"]["train"] == 1


def test_source_cli_runs_without_importing_fenicsx_runtime(tmp_path):
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    out = tmp_path / "cli_sources"

    MakeComplexSourcesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--out",
            str(out),
            "--num-train",
            "1",
            "--num-valid",
            "1",
            "--seed",
            "3",
        ]
    )

    assert (out / "train" / "sample_000000.npz").is_file()
    assert (out / "valid" / "sample_000000.npz").is_file()
    assert (out / "make_complex_sources.log").is_file()
    assert "dolfinx" not in sys.modules
    assert "gmsh" not in sys.modules
    assert "petsc4py" not in sys.modules
