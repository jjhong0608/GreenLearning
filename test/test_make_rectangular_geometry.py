from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from greenonet.complex_geometry import load_complex_geometry


def _load_cli_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "cli" / "make_rectangular_geometry.py"
    )
    spec = importlib.util.spec_from_file_location(
        "make_rectangular_geometry",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load make_rectangular_geometry.py.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_gmsh_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "examples" / "rectangle_gmsh.py"
    spec = importlib.util.spec_from_file_location("rectangle_gmsh", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load rectangle_gmsh.py.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_rectangular_geometry = _load_cli_module()
RectangularGeometryBuilder = make_rectangular_geometry.RectangularGeometryBuilder
RectangularGeometryConfig = make_rectangular_geometry.RectangularGeometryConfig
MakeRectangularGeometryCLI = make_rectangular_geometry.MakeRectangularGeometryCLI
rectangle_gmsh = _load_gmsh_module()


def test_rectangular_geometry_loads_unit_square_contract(tmp_path):
    path = tmp_path / "unit_square_h025.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.25, out=path),
    ).write()

    geometry = load_complex_geometry(path)
    with np.load(path) as raw:
        expected_coords = np.array(
            [
                [0.25, 0.25],
                [0.50, 0.25],
                [0.75, 0.25],
                [0.25, 0.50],
                [0.50, 0.50],
                [0.75, 0.50],
                [0.25, 0.75],
                [0.50, 0.75],
                [0.75, 0.75],
            ],
            dtype=np.float64,
        )
        assert raw["domain_type"].item() == "rectangle"
        assert raw["grid_x"].tolist() == pytest.approx([0.0, 0.25, 0.50, 0.75, 1.0])
        assert raw["grid_y"].tolist() == pytest.approx([0.0, 0.25, 0.50, 0.75, 1.0])
        np.testing.assert_allclose(raw["coords_valid"], expected_coords)
        np.testing.assert_array_equal(
            raw["valid_grid_y_index"], [1, 1, 1, 2, 2, 2, 3, 3, 3]
        )
        np.testing.assert_array_equal(
            raw["valid_grid_x_index"], [1, 2, 3, 1, 2, 3, 1, 2, 3]
        )
        np.testing.assert_allclose(raw["x_segment_length"], 1.0)
        np.testing.assert_allclose(raw["y_segment_length"], 1.0)
        np.testing.assert_allclose(raw["x_local_t"], expected_coords[:, 0])
        np.testing.assert_allclose(raw["y_local_t"], expected_coords[:, 1])

    assert geometry.num_points == 9
    assert geometry.num_x_segments == 3
    assert geometry.num_y_segments == 3
    assert geometry.x_edges.shape == (6, 2)
    assert geometry.y_edges.shape == (6, 2)


def test_rectangular_geometry_supports_shifted_non_square_bounds(tmp_path):
    path = tmp_path / "rectangle_xm1_1_y2_3_h05.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(
            step_size=0.5,
            out=path,
            x_min=-1.0,
            x_max=1.0,
            y_min=2.0,
            y_max=3.0,
        ),
    ).write()

    geometry = load_complex_geometry(path)
    with np.load(path) as raw:
        np.testing.assert_allclose(
            raw["coords_valid"],
            [[-0.5, 2.5], [0.0, 2.5], [0.5, 2.5]],
        )
        np.testing.assert_allclose(raw["bounds"], [[-1.0, 1.0], [2.0, 3.0]])
        np.testing.assert_allclose(raw["center"], [0.0, 2.5])
        np.testing.assert_allclose(
            raw["boundary_vertices"],
            [[-1.0, 2.0], [1.0, 2.0], [1.0, 3.0], [-1.0, 3.0]],
        )
        assert raw["width"].item() == pytest.approx(2.0)
        assert raw["height"].item() == pytest.approx(1.0)
        assert not bool(raw["has_hole"].item())
        np.testing.assert_allclose(raw["x_segment_length"], [2.0])
        np.testing.assert_allclose(raw["y_segment_length"], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(raw["x_local_t"], [0.25, 0.5, 0.75])
        np.testing.assert_allclose(raw["y_local_t"], [0.5, 0.5, 0.5])

    assert geometry.num_points == 3
    assert geometry.num_x_segments == 1
    assert geometry.num_y_segments == 3
    assert geometry.x_edges.shape == (2, 2)
    assert geometry.y_edges.shape == (0, 2)


def test_rectangular_geometry_reconstruction_endpoints_and_unit_weights(tmp_path):
    path = tmp_path / "unit_square_h025.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.25, out=path),
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


def test_rectangular_geometry_edges_stay_within_segments(tmp_path):
    path = tmp_path / "unit_square_h025.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.25, out=path),
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


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"step_size": 0.3}, "x-interval exactly"),
        (
            {
                "step_size": 0.5,
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.25,
            },
            "y-interval exactly",
        ),
        ({"step_size": 1.0}, "interior x-grid points"),
    ],
)
def test_rectangular_geometry_rejects_invalid_grid_partition(
    tmp_path,
    config_kwargs,
    message,
):
    builder = RectangularGeometryBuilder(
        RectangularGeometryConfig(out=tmp_path / "bad.npz", **config_kwargs),
    )

    with pytest.raises(ValueError, match=message):
        builder.build()


@pytest.mark.parametrize(
    ("config_kwargs", "message"),
    [
        ({"step_size": 0.0}, "step-size must be positive"),
        ({"step_size": float("nan")}, "step-size must be finite"),
        ({"step_size": 0.25, "x_min": 1.0, "x_max": 1.0}, "x-max"),
        ({"step_size": 0.25, "y_min": 2.0, "y_max": 1.0}, "y-max"),
        ({"step_size": 0.25, "boundary_tol": -1.0}, "non-negative"),
        ({"step_size": 0.25, "boundary_tol": 0.25}, "smaller"),
    ],
)
def test_rectangular_geometry_rejects_invalid_config(
    tmp_path,
    config_kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        RectangularGeometryConfig(out=tmp_path / "bad.npz", **config_kwargs)


def test_rectangular_geometry_overwrite_policy(tmp_path):
    path = tmp_path / "unit_square_h05.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.5, out=path),
    ).write()

    with pytest.raises(FileExistsError, match="--overwrite"):
        RectangularGeometryBuilder(
            RectangularGeometryConfig(step_size=0.5, out=path),
        ).write()

    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.5, out=path, overwrite=True),
    ).write()
    assert path.is_file()


def test_make_rectangular_geometry_cli_writes_log_and_uses_default_unit_square(
    tmp_path,
):
    path = tmp_path / "geometry" / "unit_square_h025.npz"

    output_path = MakeRectangularGeometryCLI().run(
        [
            "--step-size",
            "0.25",
            "--out",
            str(path),
        ]
    )

    assert output_path == path
    assert path.is_file()
    assert (path.parent / "make_rectangular_geometry.log").is_file()
    geometry = load_complex_geometry(path)
    assert geometry.num_points == 9


def test_make_rectangular_geometry_cli_accepts_custom_bounds(tmp_path):
    path = tmp_path / "geometry" / "rectangle.npz"

    MakeRectangularGeometryCLI().run(
        [
            "--step-size",
            "0.5",
            "--x-min",
            "-1.0",
            "--x-max",
            "1.0",
            "--y-min",
            "2.0",
            "--y-max",
            "3.0",
            "--out",
            str(path),
        ]
    )

    with np.load(path) as raw:
        assert raw["x_min"].item() == pytest.approx(-1.0)
        assert raw["x_max"].item() == pytest.approx(1.0)
        assert raw["y_min"].item() == pytest.approx(2.0)
        assert raw["y_max"].item() == pytest.approx(3.0)


def test_rectangle_gmsh_reader_uses_generated_boundary_metadata(tmp_path):
    path = tmp_path / "rectangle.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(
            step_size=0.5,
            out=path,
            x_min=-1.0,
            x_max=1.0,
            y_min=2.0,
            y_max=3.0,
        ),
    ).write()

    metadata = rectangle_gmsh.RectangleDomainBuilder.metadata_from_context(
        SimpleNamespace(geometry_path=path)
    )

    assert metadata.x_min == pytest.approx(-1.0)
    assert metadata.x_max == pytest.approx(1.0)
    assert metadata.y_min == pytest.approx(2.0)
    assert metadata.y_max == pytest.approx(3.0)
    np.testing.assert_allclose(
        metadata.boundary_vertices,
        [[-1.0, 2.0], [1.0, 2.0], [1.0, 3.0], [-1.0, 3.0]],
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("domain_type", np.array("annulus"), "domain_type"),
        ("has_hole", np.array(True), "has_hole"),
        ("width", np.array(2.0), "width"),
        ("bounds", np.array([[0.0, 2.0], [0.0, 1.0]]), "bounds"),
        ("center", np.array([0.25, 0.5]), "center"),
        (
            "boundary_vertices",
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]),
            "counter-clockwise",
        ),
        ("boundary_vertices", np.zeros((3, 2)), "shape"),
    ],
)
def test_rectangle_gmsh_reader_rejects_inconsistent_metadata(
    tmp_path,
    key,
    value,
    message,
):
    reference_path = tmp_path / "reference_rectangle.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(step_size=0.25, out=reference_path),
    ).write()
    with np.load(reference_path, allow_pickle=False) as raw:
        payload = {name: np.array(raw[name], copy=True) for name in raw.files}
    payload[key] = value
    path = tmp_path / f"bad_{key}.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        rectangle_gmsh.RectangleDomainBuilder.metadata_from_context(
            SimpleNamespace(geometry_path=path)
        )


def test_rectangle_gmsh_reader_rejects_missing_metadata(tmp_path):
    path = tmp_path / "missing_rectangle_metadata.npz"
    np.savez(path, domain_type=np.array("rectangle"))

    with pytest.raises(KeyError, match="missing required keys"):
        rectangle_gmsh.RectangleDomainBuilder.metadata_from_context(
            SimpleNamespace(geometry_path=path)
        )


def test_rectangle_gmsh_reader_requires_geometry_path():
    with pytest.raises(ValueError, match="geometry_path"):
        rectangle_gmsh.RectangleDomainBuilder.metadata_from_context(
            SimpleNamespace(geometry_path=None)
        )


def test_rectangle_gmsh_builds_one_surface_with_four_boundary_tags(tmp_path):
    path = tmp_path / "rectangle.npz"
    RectangularGeometryBuilder(
        RectangularGeometryConfig(
            step_size=0.5,
            out=path,
            x_min=-1.0,
            x_max=1.0,
            y_min=2.0,
            y_max=3.0,
        ),
    ).write()

    class FakeOcc:
        def __init__(self):
            self.points = []
            self.lines = []
            self.curve_loops = []
            self.plane_surfaces = []
            self.synchronized = False

        def addPoint(self, x, y, z, meshSize=0.0):
            self.points.append((x, y, z, meshSize))
            return len(self.points)

        def addLine(self, start, end):
            self.lines.append((start, end))
            return 100 + len(self.lines)

        def addCurveLoop(self, line_tags):
            self.curve_loops.append(tuple(line_tags))
            return 201

        def addPlaneSurface(self, loop_tags):
            self.plane_surfaces.append(tuple(loop_tags))
            return 301

        def synchronize(self):
            self.synchronized = True

    class FakeOption:
        def __init__(self):
            self.values = []

        def setNumber(self, name, value):
            self.values.append((name, value))

    occ = FakeOcc()
    gmsh = SimpleNamespace(
        model=SimpleNamespace(occ=occ),
        option=FakeOption(),
    )
    result = rectangle_gmsh.build_domain(
        gmsh,
        SimpleNamespace(geometry_path=path, mesh_size=0.05),
    )

    assert result == {
        "surface_tags": [301],
        "boundary_tags": [101, 102, 103, 104],
    }
    np.testing.assert_allclose(
        np.asarray([point[:2] for point in occ.points]),
        [[-1.0, 2.0], [1.0, 2.0], [1.0, 3.0], [-1.0, 3.0]],
    )
    assert occ.lines == [(1, 2), (2, 3), (3, 4), (4, 1)]
    assert occ.curve_loops == [(101, 102, 103, 104)]
    assert occ.plane_surfaces == [(201,)]
    assert occ.synchronized
    assert gmsh.option.values == [("Mesh.CharacteristicLengthMax", 0.05)]
