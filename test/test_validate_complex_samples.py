from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from greenonet.fenicsx_samples import GeometryGridLoader, SampleWriter
from test.complex_fixtures import write_geometry_npz


def _load_cli_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "validate_complex_samples.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_complex_samples", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load validate_complex_samples.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


validate_complex_samples = _load_cli_module()
ValidateComplexSamplesCLI = validate_complex_samples.ValidateComplexSamplesCLI


def _write_grid_geometry(path: Path) -> Path:
    return write_geometry_npz(
        path,
        grid_x=np.linspace(0.0, 1.0, 5, dtype=np.float64),
        grid_y=np.linspace(0.0, 1.0, 5, dtype=np.float64),
    )


def _write_balanced_sample(sample_root: Path, geometry_path: Path, split: str) -> Path:
    geometry = GeometryGridLoader().load(geometry_path)
    rhs = geometry.valid_values_to_full_grid(
        np.array([2.0, 4.0, 6.0], dtype=np.float64)
    )
    phi = geometry.valid_values_to_full_grid(
        np.array([0.5, 1.5, 2.5], dtype=np.float64)
    )
    psi = rhs - phi
    sol = geometry.valid_values_to_full_grid(
        np.array([10.0, 11.0, 12.0], dtype=np.float64)
    )
    return SampleWriter(sample_root, geometry.full_grid_shape).write_sample(
        split,
        0,
        rhs=rhs,
        sol=sol,
        phi=phi,
        psi=psi,
    )


def test_validate_complex_samples_accepts_valid_split(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    sample_root = tmp_path / "samples"
    _write_balanced_sample(sample_root, geometry_path, "train")

    summary = ValidateComplexSamplesCLI().run(
        [
            "--geometry",
            str(geometry_path),
            "--sample-root",
            str(sample_root),
            "--splits",
            "train",
            "--branch-input-dim",
            "4",
            "--max-balance-residual",
            "1e-12",
        ]
    )

    assert summary["max_balance_residual"] == pytest.approx(0.0)
    assert (sample_root / "validate_complex_samples.log").is_file()
    summary_path = sample_root / "validation_summary.json"
    assert summary_path.is_file()
    assert json.loads(summary_path.read_text())["dataset_probe"]["has_flux"] is True


def test_validate_complex_samples_rejects_missing_key(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    sample_root = tmp_path / "samples"
    sample_dir = sample_root / "train"
    sample_dir.mkdir(parents=True)
    grid_shape = GeometryGridLoader().load(geometry_path).full_grid_shape
    empty = np.zeros(grid_shape, dtype=np.float64)
    np.savez(sample_dir / "sample_000000.npz", rhs=empty, sol=empty, phi=empty)

    with pytest.raises(KeyError, match="missing keys"):
        ValidateComplexSamplesCLI().run(
            [
                "--geometry",
                str(geometry_path),
                "--sample-root",
                str(sample_root),
                "--splits",
                "train",
            ]
        )


def test_validate_complex_samples_rejects_wrong_shape(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    sample_root = tmp_path / "samples"
    sample_dir = sample_root / "train"
    sample_dir.mkdir(parents=True)
    empty = np.zeros((4, 5), dtype=np.float64)
    np.savez(
        sample_dir / "sample_000000.npz",
        rhs=empty,
        sol=empty,
        phi=empty,
        psi=empty,
    )

    with pytest.raises(ValueError, match="expected"):
        ValidateComplexSamplesCLI().run(
            [
                "--geometry",
                str(geometry_path),
                "--sample-root",
                str(sample_root),
                "--splits",
                "train",
            ]
        )


def test_validate_complex_samples_rejects_outside_domain_values(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    sample_root = tmp_path / "samples"
    sample_path = _write_balanced_sample(sample_root, geometry_path, "train")
    with np.load(sample_path) as raw:
        payload = {key: raw[key].copy() for key in raw.files}
    payload["rhs"][0, 0] = 1.0
    np.savez(sample_path, **payload)

    with pytest.raises(ValueError, match="outside-domain"):
        ValidateComplexSamplesCLI().run(
            [
                "--geometry",
                str(geometry_path),
                "--sample-root",
                str(sample_root),
                "--splits",
                "train",
            ]
        )


def test_validate_complex_samples_writes_summary_before_balance_failure(tmp_path):
    geometry_path = _write_grid_geometry(tmp_path / "geometry.npz")
    sample_root = tmp_path / "samples"
    sample_path = _write_balanced_sample(sample_root, geometry_path, "train")
    with np.load(sample_path) as raw:
        payload = {key: raw[key].copy() for key in raw.files}
    payload["psi"][:] = 0.0
    np.savez(sample_path, **payload)

    with pytest.raises(ValueError, match="Balance residual threshold"):
        ValidateComplexSamplesCLI().run(
            [
                "--geometry",
                str(geometry_path),
                "--sample-root",
                str(sample_root),
                "--splits",
                "train",
                "--max-balance-residual",
                "1e-12",
            ]
        )

    assert (sample_root / "validation_summary.json").is_file()
