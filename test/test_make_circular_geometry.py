from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from greenonet.complex_geometry import load_complex_geometry


def _load_cli_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_circular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("make_circular_geometry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_circular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_circular_geometry = _load_cli_module()
CircularGeometryBuilder = make_circular_geometry.CircularGeometryBuilder
CircularGeometryConfig = make_circular_geometry.CircularGeometryConfig
MakeCircularGeometryCLI = make_circular_geometry.MakeCircularGeometryCLI


def test_circular_geometry_loads_with_unit_circle_contract(tmp_path):
    path = tmp_path / "unit_circle_h05.npz"
    CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.5, out=path),
    ).write()

    geometry = load_complex_geometry(path)
    with np.load(path) as raw:
        coords = raw["coords_valid"]
        assert raw["domain_type"].item() == "unit_circle"
        assert raw["step_size"].item() == pytest.approx(0.5)
        assert raw["grid_x"].tolist() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
        assert np.all(coords[:, 0] ** 2 + coords[:, 1] ** 2 < 1.0 - 1e-12)
        assert np.all(np.abs(raw["x_segment_y"]) < 1.0 - 1e-12)
        assert np.all(np.abs(raw["y_segment_x"]) < 1.0 - 1e-12)

    assert geometry.num_points == 9
    assert geometry.num_x_segments == 3
    assert geometry.num_y_segments == 3
    assert geometry.x_edges.shape == (6, 2)
    assert geometry.y_edges.shape == (6, 2)


def test_circular_geometry_reconstruction_endpoints_and_weights(tmp_path):
    path = tmp_path / "unit_circle_h05.npz"
    CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.5, out=path),
    ).write()

    with np.load(path) as raw:
        for prefix in ("x", "y"):
            ptr = raw[f"{prefix}_recon_ptr"]
            t_values = raw[f"{prefix}_recon_t"]
            weights = raw[f"{prefix}_recon_weight"]
            valid_index = raw[f"{prefix}_recon_valid_index"]
            for start, end in zip(ptr[:-1], ptr[1:]):
                assert valid_index[start] == -1
                assert valid_index[end - 1] == -1
                assert t_values[start] == pytest.approx(0.0)
                assert t_values[end - 1] == pytest.approx(1.0)
                assert weights[start:end].sum() == pytest.approx(1.0)
                assert np.all(np.diff(t_values[start:end]) > 0.0)


def test_circular_geometry_edges_stay_within_segments(tmp_path):
    path = tmp_path / "unit_circle_h05.npz"
    CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.5, out=path),
    ).write()

    with np.load(path) as raw:
        x_edges = raw["x_edges"]
        y_edges = raw["y_edges"]
        assert np.all(
            raw["x_segment_id"][x_edges[:, 0]] == raw["x_segment_id"][x_edges[:, 1]]
        )
        assert np.all(
            raw["y_segment_id"][y_edges[:, 0]] == raw["y_segment_id"][y_edges[:, 1]]
        )


def test_circular_geometry_rejects_non_dividing_step_size(tmp_path):
    path = tmp_path / "bad.npz"
    builder = CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.3, out=path),
    )

    with pytest.raises(ValueError, match="2 / step_size"):
        builder.build()


def test_circular_geometry_overwrite_policy(tmp_path):
    path = tmp_path / "unit_circle_h05.npz"
    CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.5, out=path),
    ).write()

    with pytest.raises(FileExistsError, match="--overwrite"):
        CircularGeometryBuilder(
            CircularGeometryConfig(step_size=0.5, out=path),
        ).write()

    CircularGeometryBuilder(
        CircularGeometryConfig(step_size=0.5, out=path, overwrite=True),
    ).write()
    assert path.is_file()


def test_make_circular_geometry_cli_writes_log_and_validates(tmp_path):
    path = tmp_path / "geometry" / "unit_circle_h05.npz"

    output_path = MakeCircularGeometryCLI().run(
        [
            "--step-size",
            "0.5",
            "--out",
            str(path),
        ]
    )

    assert output_path == path
    assert path.is_file()
    assert (path.parent / "make_circular_geometry.log").is_file()
    load_complex_geometry(path)
