from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_sources.geometry import GeometryGridLoader


def _load_cli_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_pentagram_geometry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "make_pentagram_geometry", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_pentagram_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


make_pentagram_geometry = _load_cli_module()
MakePentagramGeometryCLI = make_pentagram_geometry.MakePentagramGeometryCLI
PentagramGeometryBuilder = make_pentagram_geometry.PentagramGeometryBuilder
PentagramGeometryConfig = make_pentagram_geometry.PentagramGeometryConfig


def _write_pentagram(tmp_path: Path, *, step_size: float = 0.125) -> Path:
    path = tmp_path / "pentagram_r10_h0125.npz"
    PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=step_size,
            out=path,
        )
    ).write()
    return path


def test_regular_pentagram_vertices_follow_fixed_contract(tmp_path):
    builder = PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=2.0,
            step_size=0.25,
            out=tmp_path / "unused.npz",
        )
    )

    vertices = builder.boundary_vertices()
    expected_inner = 2.0 / builder.GOLDEN_RATIO**2

    assert vertices.shape == (10, 2)
    assert vertices[0].tolist() == pytest.approx([0.0, 2.0])
    assert np.linalg.norm(vertices[0::2], axis=1) == pytest.approx(np.full(5, 2.0))
    assert np.linalg.norm(vertices[1::2], axis=1) == pytest.approx(
        np.full(5, expected_inner)
    )
    assert builder._signed_area(vertices) > 0.0
    builder._validate_polygon(vertices)


def test_pentagram_geometry_loads_with_schema_and_metadata(tmp_path):
    path = _write_pentagram(tmp_path)

    geometry = load_complex_geometry(path)
    raw_geometry = GeometryGridLoader().load(path)
    with np.load(path, allow_pickle=False) as raw:
        assert raw["domain_type"].item() == "regular_pentagram"
        assert raw["outer_radius"].item() == pytest.approx(1.0)
        assert raw["inner_radius"].item() == pytest.approx(
            1.0 / PentagramGeometryBuilder.GOLDEN_RATIO**2
        )
        assert raw["center"].tolist() == pytest.approx([0.0, 0.0])
        assert raw["orientation_angle"].item() == pytest.approx(math.pi / 2.0)
        assert raw["fill_rule"].item() == "filled_simple_decagon"
        assert not bool(raw["has_hole"].item())
        assert raw["boundary_vertices"].shape == (10, 2)
        assert raw["boundary_vertices"][0].tolist() == pytest.approx([0.0, 1.0])
        assert raw["grid_x"].tolist() == pytest.approx(
            np.linspace(-1.0, 1.0, 17).tolist()
        )
        assert raw["grid_y"].tolist() == pytest.approx(
            np.linspace(-1.0, 1.0, 17).tolist()
        )

    assert geometry.num_points > 0
    assert geometry.num_x_segments > 0
    assert geometry.num_y_segments > 0
    assert raw_geometry.metadata["domain_type"] == "regular_pentagram"
    assert raw_geometry.metadata["orientation_angle"] == pytest.approx(math.pi / 2.0)
    assert raw_geometry.metadata["fill_rule"] == "filled_simple_decagon"
    assert raw_geometry.metadata["has_hole"] is False
    assert np.asarray(raw_geometry.metadata["boundary_vertices"]).shape == (10, 2)


def test_pentagram_geometry_contains_center_and_excludes_boundary(tmp_path):
    path = _write_pentagram(tmp_path)

    with np.load(path, allow_pickle=False) as raw:
        coords = {tuple(point) for point in raw["coords_valid"].tolist()}
        vertices = raw["boundary_vertices"]

    assert (0.0, 0.0) in coords
    assert (0.0, 1.0) not in coords
    distances = PentagramGeometryBuilder._distance_to_boundary(
        vertices,
        vertices,
    )
    assert distances == pytest.approx(np.zeros(10), abs=1.0e-14)


def test_pentagram_geometry_preserves_disconnected_axial_segments(tmp_path):
    path = _write_pentagram(tmp_path)

    with np.load(path, allow_pickle=False) as raw:
        assert np.count_nonzero(np.isclose(raw["x_segment_y"], -0.5)) == 2
        assert np.count_nonzero(np.isclose(raw["y_segment_x"], -0.5)) == 2
        assert np.count_nonzero(np.isclose(raw["x_segment_y"], 0.0)) == 1
        assert np.count_nonzero(np.isclose(raw["y_segment_x"], 0.0)) == 1

        builder = PentagramGeometryBuilder(
            PentagramGeometryConfig(
                outer_radius=1.0,
                step_size=0.125,
                out=tmp_path / "unused.npz",
            )
        )
        intervals = builder._axis_intervals(
            raw["boundary_vertices"],
            fixed_coordinate=-0.5,
            coordinate_axis=0,
        )
        segment_indices = np.flatnonzero(np.isclose(raw["x_segment_y"], -0.5))
        stored = list(
            zip(
                raw["x_segment_left"][segment_indices],
                raw["x_segment_right"][segment_indices],
            )
        )
        assert np.asarray(stored) == pytest.approx(np.asarray(intervals))


def test_pentagram_scanline_splits_at_exact_reentrant_vertex(tmp_path):
    builder = PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=0.125,
            out=tmp_path / "unused.npz",
        )
    )
    vertices = builder.boundary_vertices()

    intervals = builder._axis_intervals(
        vertices,
        fixed_coordinate=float(vertices[5, 1]),
        coordinate_axis=0,
    )

    assert len(intervals) == 2
    assert intervals[0][1] == pytest.approx(0.0, abs=1.0e-14)
    assert intervals[1][0] == pytest.approx(0.0, abs=1.0e-14)


def test_pentagram_reconstruction_and_edges_remain_segment_local(tmp_path):
    path = _write_pentagram(tmp_path)

    with np.load(path, allow_pickle=False) as raw:
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

            edges = raw[f"{prefix}_edges"]
            segment_id = raw[f"{prefix}_segment_id"]
            assert np.all(segment_id[edges[:, 0]] == segment_id[edges[:, 1]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outer_radius": 0.0}, "--outer-radius"),
        ({"outer_radius": math.inf}, "finite"),
        ({"step_size": 0.0}, "--step-size"),
        ({"step_size": math.nan}, "finite"),
        ({"boundary_tol": -1.0}, "--boundary-tol"),
        ({"boundary_tol": math.inf}, "finite"),
        ({"boundary_tol": 0.5}, "inner radius"),
    ],
)
def test_pentagram_geometry_rejects_invalid_config(tmp_path, kwargs, message):
    values = {
        "outer_radius": 1.0,
        "step_size": 0.125,
        "boundary_tol": 1.0e-12,
        "out": tmp_path / "bad.npz",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        PentagramGeometryConfig(**values)


def test_pentagram_geometry_rejects_non_dividing_step_size(tmp_path):
    builder = PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=0.3,
            out=tmp_path / "bad_step.npz",
        )
    )

    with pytest.raises(ValueError, match=r"2 \* outer_radius / step_size"):
        builder.build()


def test_pentagram_geometry_rejects_grid_without_interior_points(tmp_path):
    builder = PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=2.0,
            out=tmp_path / "empty_grid.npz",
        )
    )

    with pytest.raises(ValueError, match="too large"):
        builder.build()


def test_pentagram_polygon_and_scanline_validation_fail_fast(tmp_path):
    builder = PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=0.125,
            out=tmp_path / "unused.npz",
        )
    )
    vertices = builder.boundary_vertices()
    duplicate = vertices.copy()
    duplicate[1] = duplicate[0]
    with pytest.raises(ValueError, match="distinct"):
        builder._validate_polygon(duplicate)

    with pytest.raises(ValueError, match="even number"):
        builder._pair_intersections([0.0, 0.5, 1.0])


def test_pentagram_geometry_overwrite_policy(tmp_path):
    path = _write_pentagram(tmp_path)

    with pytest.raises(FileExistsError, match="--overwrite"):
        PentagramGeometryBuilder(
            PentagramGeometryConfig(
                outer_radius=1.0,
                step_size=0.125,
                out=path,
            )
        ).write()

    PentagramGeometryBuilder(
        PentagramGeometryConfig(
            outer_radius=1.0,
            step_size=0.125,
            out=path,
            overwrite=True,
        )
    ).write()
    assert path.is_file()


def test_make_pentagram_geometry_cli_writes_log_and_validates(tmp_path):
    path = tmp_path / "geometry" / "pentagram_r10_h025.npz"

    output_path = MakePentagramGeometryCLI().run(
        [
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
    assert (path.parent / "make_pentagram_geometry.log").is_file()
    load_complex_geometry(path)


def test_make_pentagram_geometry_cli_keeps_inner_radius_and_orientation_fixed():
    cli = MakePentagramGeometryCLI()
    option_strings = {
        option for action in cli.parser._actions for option in action.option_strings
    }

    assert "--outer-radius" in option_strings
    assert "--inner-radius" not in option_strings
    assert "--orientation" not in option_strings
    assert "--orientation-angle" not in option_strings
