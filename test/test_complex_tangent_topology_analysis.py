from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from greenonet.complex_tangent_topology_analysis import (
    AxialSegmentTopologyAnalyzer,
    TrainedKComparisonParser,
)


def test_axial_topology_chain_requires_k4_for_end_to_end_reach() -> None:
    coords = np.column_stack((np.arange(6, dtype=np.float64), np.zeros(6)))
    x_segment_id = np.array([0, 1, 1, 2, 2, 3], dtype=np.int64)
    y_segment_id = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    analyzer = AxialSegmentTopologyAnalyzer(
        coords=coords,
        x_segment_id=x_segment_id,
        y_segment_id=y_segment_id,
        chunk_size=2,
    )
    result = analyzer.analyze()

    assert result.point_graph_diameter == 5
    assert result.a_graph_diameter == 3
    assert result.longest_path_point_ids == (0, 1, 2, 3, 4, 5)
    assert result.a_distance_pair_counts[3] == 2
    np.testing.assert_array_equal(result.point_a_eccentricity, [3, 2, 2, 2, 2, 3])

    point_distance, a_distance = analyzer.distances_from_point(0)
    np.testing.assert_array_equal(point_distance, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(a_distance, [0, 1, 1, 2, 2, 3])


def test_axial_topology_point_distance_rejects_invalid_index() -> None:
    analyzer = AxialSegmentTopologyAnalyzer(
        coords=np.array([[0.0, 0.0]], dtype=np.float64),
        x_segment_id=np.array([0], dtype=np.int64),
        y_segment_id=np.array([0], dtype=np.int64),
    )

    with pytest.raises(TypeError, match="integer"):
        analyzer.distances_from_point(True)
    with pytest.raises(IndexError, match=r"\[0, 0\]"):
        analyzer.distances_from_point(1)


def test_trained_k_comparison_parser_reads_quality_and_runtime_tables(
    tmp_path: Path,
) -> None:
    report = tmp_path / "analysis_report.md"
    report.write_text(
        """# Comparison

| run | K | rel_sol mean | p95 | max | rel_u_phi | rel_u_psi | rel_flux | optimized energy | response cost | correction / symmetric pair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coupling8 | 1 | 2.678% | 5.131% | 10.317% | 5.022% | 4.832% | 46.558% | 5.628e-5 | 2.681e-8 | 4.232% |
| coupling9 | 2 | 1.590% | 2.866% | 5.104% | 3.289% | 2.711% | 39.658% | 2.842e-5 | 9.986e-9 | 8.026% |

| K | forward only | ratio to K1 | forward + backward | ratio to K1 |
|---:|---:|---:|---:|---:|
| 1 | 32.800 ms | 1.00x | 141.373 ms | 1.00x |
| 2 | 48.075 ms | 1.47x | 211.807 ms | 1.50x |
""",
        encoding="utf-8",
    )

    quality, runtime = TrainedKComparisonParser(report).parse()

    assert [row.subspace_dimension for row in quality] == [1, 2]
    assert quality[0].rel_sol_mean == pytest.approx(0.02678)
    assert quality[1].rel_u_psi_mean == pytest.approx(0.02711)
    assert quality[1].response_cost_mean == 9.986e-9
    assert runtime[0].forward_ms == 32.8
    assert runtime[1].forward_backward_ms == 211.807
