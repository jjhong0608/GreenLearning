from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cli.make_complex_visualization_mesh import MakeComplexVisualizationMeshCLI
from greenonet.complex_visualization_mesh import (
    ComplexVisualizationMeshConfig,
    ComplexVisualizationMeshGenerator,
    MeshAdjacencyInterpolationMixin,
    load_complex_visualization_mesh,
)
from test.complex_fixtures import (
    write_geometry_npz,
    write_visualization_mesh_npz,
)


def test_visualization_mesh_round_trip_and_solution_transfer(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    mesh_path, expected = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    coords_valid = np.asarray(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]],
        dtype=np.float64,
    )

    mesh = load_complex_visualization_mesh(
        mesh_path,
        geometry_path=geometry_path,
        coords_valid=coords_valid,
    )
    values = np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    transferred = mesh.transfer_solution(values)

    np.testing.assert_array_equal(
        transferred[:, mesh.valid_to_vertex],
        values,
    )
    np.testing.assert_array_equal(
        transferred[:, mesh.boundary_vertex_mask],
        0.0,
    )
    np.testing.assert_allclose(
        transferred[:, mesh.auxiliary_vertex_mask],
        np.asarray([[2.25], [3.0]]),
    )
    cell_values = mesh.transfer_interior_cell_values(values)
    np.testing.assert_allclose(
        cell_values,
        np.asarray(
            [
                [3.0, 2.0, 3.0, 3.0, 5.0, 4.0, 1.0, 3.0, 7.0 / 3.0, 3.0],
                [4.0, 3.0, 4.0, 4.0, 6.0, 5.0, 2.0, 4.0, 10.0 / 3.0, 4.0],
            ]
        ),
    )
    assert mesh.vertex_count == expected.vertex_count == 8
    assert mesh.triangle_count == expected.triangle_count == 10
    assert mesh.summary(mesh_path)["included_in_metrics"] is False
    assert mesh.summary(mesh_path)["boundary_values_model_evaluated"] is False


def test_interior_cell_transfer_rejects_boundary_only_triangle(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    _, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )
    boundary_only = replace(
        mesh,
        triangles=np.concatenate(
            (mesh.triangles, np.asarray([[0, 1, 2]], dtype=np.int64)),
            axis=0,
        ),
    )

    with pytest.raises(ValueError, match="only boundary vertices"):
        boundary_only.transfer_interior_cell_values(np.asarray([1.0, 2.0, 3.0]))


def test_visualization_mesh_rejects_invalid_cache_contracts(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    mesh_path, mesh = write_visualization_mesh_npz(
        tmp_path / "visualization_mesh.npz",
        geometry_path=geometry_path,
    )

    with pytest.raises(ValueError, match="auxiliary fraction exceeds"):
        replace(mesh, max_auxiliary_fraction=0.1).validate()
    with pytest.raises(ValueError, match="out-of-range vertex index"):
        replace(mesh, triangles=np.asarray([[0, 1, 99]], dtype=np.int64)).validate()
    with pytest.raises(ValueError, match="cannot repeat a vertex"):
        replace(mesh, triangles=np.asarray([[0, 1, 1]], dtype=np.int64)).validate()
    with pytest.raises(ValueError, match="cannot reference auxiliary"):
        replace(
            mesh,
            aux_interp_vertex_index=np.asarray([2, 4, 5, 7], dtype=np.int64),
        ).validate()

    different_geometry = write_geometry_npz(
        tmp_path / "different_geometry.npz",
        hx=np.asarray(0.25),
    )
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_complex_visualization_mesh(mesh_path, geometry_path=different_geometry)
    with pytest.raises(ValueError, match="does not match coords_valid"):
        load_complex_visualization_mesh(
            mesh_path,
            coords_valid=np.asarray(
                [[0.26, 0.25], [0.75, 0.25], [0.25, 0.75]],
                dtype=np.float64,
            ),
        )

    incomplete_path = tmp_path / "incomplete.npz"
    np.savez(incomplete_path, vertices=mesh.vertices)
    with pytest.raises(KeyError, match="missing required keys"):
        load_complex_visualization_mesh(incomplete_path)


def test_auxiliary_stencil_uses_only_mesh_connected_known_vertices() -> None:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.4, 0.3],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int64,
    )
    known_mask = np.asarray([True, True, True, False], dtype=np.bool_)
    auxiliary_mask = ~known_mask

    ptr, indices, weights = MeshAdjacencyInterpolationMixin.build_auxiliary_stencils(
        vertices=vertices,
        triangles=triangles,
        known_mask=known_mask,
        auxiliary_mask=auxiliary_mask,
        h_reference=1.0,
    )

    np.testing.assert_array_equal(ptr, [0, 3])
    assert set(indices.tolist()) == {0, 1, 2}
    assert np.all(known_mask[indices])
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)


def test_visualization_mesh_config_and_cli_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    gmsh_script = tmp_path / "domain.py"
    gmsh_script.write_text("def build_domain(gmsh, context): ...\n")
    out = tmp_path / "mesh.npz"
    config = ComplexVisualizationMeshConfig(
        geometry=geometry_path,
        gmsh_script=gmsh_script,
        out=out,
    )
    assert config.boundary_size_factor == 3.0
    assert config.max_auxiliary_fraction == 1.0e-3

    with pytest.raises(ValueError, match="at least 1.0"):
        replace(config, boundary_size_factor=0.5)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        replace(config, max_auxiliary_fraction=1.1)

    def raise_import_error(name: str) -> None:
        del name
        raise ImportError("gmsh unavailable")

    monkeypatch.setattr(
        "greenonet.complex_visualization_mesh.importlib.import_module",
        raise_import_error,
    )
    with pytest.raises(RuntimeError, match="green_fenicsx"):
        ComplexVisualizationMeshGenerator._load_gmsh()

    args = MakeComplexVisualizationMeshCLI().parser.parse_args(
        [
            "--geometry",
            str(geometry_path),
            "--gmsh-script",
            str(gmsh_script),
            "--out",
            str(out),
        ]
    )
    assert args.boundary_size_factor == 3.0
    assert args.max_auxiliary_fraction == 1.0e-3
    assert args.overwrite is False


def test_visualization_mesh_write_requires_explicit_overwrite(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    gmsh_script = tmp_path / "domain.py"
    gmsh_script.write_text("def build_domain(gmsh, context): ...\n")
    out = tmp_path / "mesh.npz"
    out.write_bytes(b"existing")
    generator = ComplexVisualizationMeshGenerator(
        ComplexVisualizationMeshConfig(
            geometry=geometry_path,
            gmsh_script=gmsh_script,
            out=out,
        )
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        generator.write()
