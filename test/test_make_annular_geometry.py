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
        Path(__file__).resolve().parents[1] / "cli" / "make_annular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location("make_annular_geometry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_annular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_annular_geometry = _load_cli_module()
AnnularGeometryBuilder = make_annular_geometry.AnnularGeometryBuilder
AnnularGeometryConfig = make_annular_geometry.AnnularGeometryConfig
MakeAnnularGeometryCLI = make_annular_geometry.MakeAnnularGeometryCLI


def _write_annulus(tmp_path: Path) -> Path:
    path = tmp_path / "annulus_r05_r10_h025.npz"
    AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        ),
    ).write()
    return path


def test_annular_geometry_loads_with_schema_and_metadata(tmp_path):
    path = _write_annulus(tmp_path)

    geometry = load_complex_geometry(path)
    with np.load(path) as raw:
        coords = raw["coords_valid"]
        radius_sq = coords[:, 0] ** 2 + coords[:, 1] ** 2
        assert raw["domain_type"].item() == "annulus"
        assert raw["inner_radius"].item() == pytest.approx(0.5)
        assert raw["outer_radius"].item() == pytest.approx(1.0)
        assert raw["center"].tolist() == pytest.approx([0.0, 0.0])
        assert raw["step_size"].item() == pytest.approx(0.25)
        assert raw["grid_x"].tolist() == pytest.approx(
            np.linspace(-1.0, 1.0, 9).tolist()
        )
        assert raw["grid_y"].tolist() == pytest.approx(
            np.linspace(-1.0, 1.0, 9).tolist()
        )
        assert np.all(radius_sq > 0.5**2 + 1e-12)
        assert np.all(radius_sq < 1.0**2 - 1e-12)

    assert geometry.num_points > 0
    assert geometry.num_x_segments > 0
    assert geometry.num_y_segments > 0


def test_annular_geometry_excludes_inner_and_outer_boundary_points(tmp_path):
    path = _write_annulus(tmp_path)

    with np.load(path) as raw:
        coords = {tuple(point) for point in raw["coords_valid"].tolist()}

    excluded_boundary_points = {
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
    }
    assert coords.isdisjoint(excluded_boundary_points)


def test_annular_geometry_splits_hole_crossing_axial_lines(tmp_path):
    path = _write_annulus(tmp_path)

    with np.load(path) as raw:
        x_segment_y = raw["x_segment_y"]
        y_segment_x = raw["y_segment_x"]
        assert np.count_nonzero(np.isclose(x_segment_y, 0.0)) == 2
        assert np.count_nonzero(np.isclose(y_segment_x, 0.0)) == 2
        assert np.count_nonzero(np.isclose(x_segment_y, 0.75)) == 1
        assert np.count_nonzero(np.isclose(y_segment_x, 0.75)) == 1

        center_x_segments = np.flatnonzero(np.isclose(x_segment_y, 0.0))
        center_y_segments = np.flatnonzero(np.isclose(y_segment_x, 0.0))
        assert np.any(raw["x_segment_right"][center_x_segments] <= 0.0)
        assert np.any(raw["x_segment_left"][center_x_segments] >= 0.0)
        assert np.any(raw["y_segment_top"][center_y_segments] <= 0.0)
        assert np.any(raw["y_segment_bottom"][center_y_segments] >= 0.0)


def test_annular_geometry_reconstruction_endpoints_and_weights(tmp_path):
    path = _write_annulus(tmp_path)

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


def test_annular_geometry_edges_stay_within_disconnected_segments(tmp_path):
    path = _write_annulus(tmp_path)

    with np.load(path) as raw:
        x_edges = raw["x_edges"]
        y_edges = raw["y_edges"]
        assert np.all(
            raw["x_segment_id"][x_edges[:, 0]] == raw["x_segment_id"][x_edges[:, 1]]
        )
        assert np.all(
            raw["y_segment_id"][y_edges[:, 0]] == raw["y_segment_id"][y_edges[:, 1]]
        )


def test_annular_geometry_rejects_invalid_radii(tmp_path):
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError, match="--inner-radius"):
        AnnularGeometryConfig(
            inner_radius=0.0,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        )
    with pytest.raises(ValueError, match="greater than --inner-radius"):
        AnnularGeometryConfig(
            inner_radius=1.0,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
        )


def test_annular_geometry_rejects_non_dividing_step_size(tmp_path):
    path = tmp_path / "bad_step.npz"
    builder = AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.3,
            out=path,
        ),
    )

    with pytest.raises(ValueError, match=r"2 \* outer_radius / step_size"):
        builder.build()


def test_annular_geometry_overwrite_policy(tmp_path):
    path = _write_annulus(tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        AnnularGeometryBuilder(
            AnnularGeometryConfig(
                inner_radius=0.5,
                outer_radius=1.0,
                step_size=0.25,
                out=path,
            ),
        ).write()

    AnnularGeometryBuilder(
        AnnularGeometryConfig(
            inner_radius=0.5,
            outer_radius=1.0,
            step_size=0.25,
            out=path,
            overwrite=True,
        ),
    ).write()
    assert path.is_file()


def test_make_annular_geometry_cli_writes_log_and_validates(tmp_path):
    path = tmp_path / "geometry" / "annulus_r05_r10_h025.npz"

    output_path = MakeAnnularGeometryCLI().run(
        [
            "--inner-radius",
            "0.5",
            "--outer-radius",
            "1.0",
            "--step-size",
            "0.25",
            "--out",
            str(path),
        ]
    )

    assert output_path == path
    assert path.is_file()
    assert (path.parent / "make_annular_geometry.log").is_file()
    load_complex_geometry(path)
