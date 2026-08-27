from __future__ import annotations

import numpy as np

from greenonet.complex_tangent_topology_analysis import (
    AxialSegmentTopologyAnalyzer,
    AxialTopologyResult,
)
from greenonet.geometry_k_connectivity_visualization import (
    global_reach_fraction,
    pointwise_reach_fraction,
    select_geometry_k,
    select_representative_tail_seed,
)


def _alternating_chain_topology() -> AxialTopologyResult:
    coords = np.column_stack((np.arange(6, dtype=np.float64), np.zeros(6)))
    return AxialSegmentTopologyAnalyzer(
        coords=coords,
        x_segment_id=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        y_segment_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        chunk_size=2,
    ).analyze()


def test_geometry_k_reach_helpers_select_k4_for_chain() -> None:
    topology = _alternating_chain_topology()

    np.testing.assert_allclose(
        pointwise_reach_fraction(topology, 3),
        [5 / 6, 1.0, 1.0, 1.0, 1.0, 5 / 6],
    )
    assert global_reach_fraction(topology, 3) == 34 / 36
    assert global_reach_fraction(topology, 4) == 1.0
    assert (
        select_geometry_k(
            topology,
            global_threshold=0.99,
            tail_quantile=0.05,
            tail_threshold=0.99,
        )
        == 4
    )


def test_geometry_k_reach_saturates_after_full_reach() -> None:
    topology = _alternating_chain_topology()

    np.testing.assert_array_equal(pointwise_reach_fraction(topology, 20), 1.0)
    assert global_reach_fraction(topology, 20) == 1.0


def test_representative_seed_uses_preselection_tail() -> None:
    topology = _alternating_chain_topology()

    assert (
        select_representative_tail_seed(
            topology,
            selection_k=3,
            tail_quantile=0.05,
        )
        == 0
    )
