from __future__ import annotations

import numpy as np
import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from test.complex_fixtures import write_geometry_npz


def test_complex_geometry_loads_disconnected_segments(tmp_path):
    path = write_geometry_npz(tmp_path / "geometry.npz")

    geometry = load_complex_geometry(path)

    assert geometry.coords_valid.shape == (3, 2)
    assert geometry.num_x_segments == 2
    assert geometry.num_y_segments == 3
    assert geometry.x_edges.tolist() == [[0, 1]]
    assert geometry.y_edges.tolist() == [[0, 2]]
    torch.testing.assert_close(geometry.hx, torch.tensor(0.5, dtype=torch.float64))


def test_complex_geometry_rejects_endpoint_valid_points(tmp_path):
    path = write_geometry_npz(
        tmp_path / "bad_geometry.npz",
        x_local_t=np.array([0.0, 0.75, 0.5], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="strictly inside"):
        load_complex_geometry(path)


def test_complex_geometry_rejects_bad_reconstruction_endpoint(tmp_path):
    path = write_geometry_npz(
        tmp_path / "bad_geometry.npz",
        x_recon_valid_index=np.array([0, 0, 1, -1, -1, 2, -1], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="endpoints"):
        load_complex_geometry(path)


def test_complex_geometry_rejects_cross_segment_edges(tmp_path):
    path = write_geometry_npz(
        tmp_path / "bad_geometry.npz",
        x_edges=np.array([[0, 2]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="same segment"):
        load_complex_geometry(path)


def test_complex_geometry_rejects_nonpositive_lengths(tmp_path):
    path = write_geometry_npz(
        tmp_path / "bad_geometry.npz",
        y_segment_length=np.array([1.0, 0.0, 1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="positive"):
        load_complex_geometry(path)
