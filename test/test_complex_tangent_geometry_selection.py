from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_tangent_geometry_selection import (
    POINTWISE_TAIL_QUANTILE,
    AxialSegmentTopologyAnalyzer,
    GeometryTangentDimensionResolver,
    geometry_k_reach_metric,
    select_geometry_k,
)
from greenonet.complex_tangent_projection import (
    SymmetricTangentGreenResponseContextBuilder,
)
from greenonet.config import (
    BalanceProjectionConfig,
    CouplingModelConfig,
    GeometryKSelectionConfig,
    SymmetricTangentGreenResponseProjectionConfig,
)
from greenonet.coupling_artifacts import load_coupling_artifact_configs
from test.complex_fixtures import write_geometry_npz


def _alternating_chain():
    return AxialSegmentTopologyAnalyzer(
        coords=np.column_stack((np.arange(6, dtype=np.float64), np.zeros(6))),
        x_segment_id=np.array([0, 1, 1, 2, 2, 3], dtype=np.int64),
        y_segment_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        chunk_size=2,
    ).analyze()


def _tangent_model_config(*, auto: bool, maximum: int = 8) -> CouplingModelConfig:
    return CouplingModelConfig(
        balance_projection=BalanceProjectionConfig(
            mode="symmetric_tangent_green_response",
            symmetric_tangent_green_response={
                "subspace_dimension": 4,
                "max_subspace_dimension": maximum,
                "geometry_k_selection": {
                    "enabled": auto,
                    "global_reach_threshold": 0.99,
                    "pointwise_tail_reach_threshold": 0.99,
                },
                "eta_strategy": "closed_loop_exact_line_search",
            },
        )
    )


def test_geometry_k_config_is_strict_and_independently_configurable() -> None:
    config = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        {
            "subspace_dimension": 5,
            "max_subspace_dimension": 8,
            "geometry_k_selection": {
                "enabled": True,
                "global_reach_threshold": 0.97,
                "pointwise_tail_reach_threshold": 0.93,
            },
            "eta_strategy": "closed_loop_exact_line_search",
        }
    )

    assert config.subspace_dimension == 5
    assert config.max_subspace_dimension == 8
    assert config.geometry_k_selection.enabled
    assert config.geometry_k_selection.global_reach_threshold == 0.97
    assert config.geometry_k_selection.pointwise_tail_reach_threshold == 0.93
    with pytest.raises(TypeError, match="unknown keys"):
        GeometryKSelectionConfig.from_raw({"enabled": True, "tail_quantile": 0.05})
    with pytest.raises(TypeError, match="must be numeric"):
        GeometryKSelectionConfig(global_reach_threshold=True)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        GeometryKSelectionConfig(pointwise_tail_reach_threshold=0.0)
    with pytest.raises(ValueError, match="must not exceed"):
        SymmetricTangentGreenResponseProjectionConfig(
            subspace_dimension=9,
            max_subspace_dimension=8,
            eta_strategy="closed_loop_exact_line_search",
        )
    with pytest.raises(ValueError, match="requires.*symmetric_tangent_green_response"):
        BalanceProjectionConfig(
            mode="physical_symmetric",
            symmetric_tangent_green_response={
                "subspace_dimension": 4,
                "geometry_k_selection": {"enabled": True},
                "eta_strategy": "closed_loop_exact_line_search",
            },
        )


def test_unresolved_auto_k_fails_before_tangent_context_build() -> None:
    with pytest.raises(ValueError, match="Unresolved geometry_k_selection"):
        SymmetricTangentGreenResponseContextBuilder(
            {
                "subspace_dimension": 4,
                "geometry_k_selection": {"enabled": True},
                "eta_strategy": "closed_loop_exact_line_search",
            }
        )


def test_chain_selects_minimum_k_and_thresholds_change_selection() -> None:
    topology = _alternating_chain()
    strict = select_geometry_k(
        topology,
        config=GeometryKSelectionConfig(enabled=True),
        max_subspace_dimension=8,
    )
    relaxed = select_geometry_k(
        topology,
        config=GeometryKSelectionConfig(
            enabled=True,
            global_reach_threshold=0.94,
            pointwise_tail_reach_threshold=0.83,
        ),
        max_subspace_dimension=8,
    )

    assert POINTWISE_TAIL_QUANTILE == 0.05
    assert strict.selected_subspace_dimension == 4
    assert relaxed.selected_subspace_dimension == 3
    assert geometry_k_reach_metric(topology, 3).global_reach_fraction == 34 / 36


def test_geometry_selection_rejects_disconnected_graph_and_limit() -> None:
    disconnected = AxialSegmentTopologyAnalyzer(
        coords=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        x_segment_id=np.array([0, 1], dtype=np.int64),
        y_segment_id=np.array([0, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="disconnected"):
        disconnected.analyze()

    with pytest.raises(ValueError, match=r"requires K=4.*max_subspace_dimension=3"):
        select_geometry_k(
            _alternating_chain(),
            config=GeometryKSelectionConfig(enabled=True),
            max_subspace_dimension=3,
        )


def test_resolver_bypasses_topology_in_explicit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    geometry = load_complex_geometry(geometry_path)

    def fail_analyze(*_args, **_kwargs):
        raise AssertionError("Explicit K must not analyze geometry topology.")

    monkeypatch.setattr(AxialSegmentTopologyAnalyzer, "analyze", fail_analyze)
    resolution = GeometryTangentDimensionResolver.resolve(
        model_config=_tangent_model_config(auto=False),
        geometry=geometry,
        geometry_path=geometry_path,
    )

    projection = BalanceProjectionConfig.from_raw(
        resolution.model_config.balance_projection
    )
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )
    assert tangent.subspace_dimension == 4
    assert resolution.provenance is not None
    assert resolution.provenance["selection_mode"] == "explicit"


def test_resolver_materializes_auto_k_and_full_provenance(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    geometry = load_complex_geometry(geometry_path)
    resolution = GeometryTangentDimensionResolver.resolve(
        model_config=_tangent_model_config(auto=True),
        geometry=geometry,
        geometry_path=geometry_path,
    )

    projection = BalanceProjectionConfig.from_raw(
        resolution.model_config.balance_projection
    )
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )
    assert tangent.subspace_dimension == 2
    assert not tangent.geometry_k_selection.enabled
    assert resolution.provenance is not None
    assert resolution.provenance["selection_mode"] == "geometry_auto"
    assert resolution.provenance["resolved_subspace_dimension"] == 2
    assert resolution.provenance["geometry_path"] == str(geometry_path.resolve())
    assert len(str(resolution.provenance["geometry_sha256"])) == 64
    assert resolution.provenance["pde_dependent_inputs_used"] is False
    assert resolution.provenance["reference_targets_used"] is False


def test_artifact_config_loader_resolves_original_auto_config(tmp_path: Path) -> None:
    geometry_path = write_geometry_npz(tmp_path / "geometry.npz")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "geometry_mode": "complex",
                    "geometry_path": str(geometry_path),
                },
                "coupling_model": {
                    "balance_projection": {
                        "mode": "symmetric_tangent_green_response",
                        "symmetric_tangent_green_response": {
                            "subspace_dimension": 4,
                            "geometry_k_selection": {"enabled": True},
                            "eta_strategy": "closed_loop_exact_line_search",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    configs = load_coupling_artifact_configs(config_path)
    projection = BalanceProjectionConfig.from_raw(
        configs.coupling_model.balance_projection
    )
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )

    assert tangent.subspace_dimension == 2
    assert not tangent.geometry_k_selection.enabled
    provenance = configs.raw["tangent_subspace_dimension_provenance"]
    assert provenance["selection_mode"] == "geometry_auto"
    assert provenance["resolved_subspace_dimension"] == 2


@pytest.mark.parametrize(
    ("path", "expected_k"),
    [
        (Path("data/geometry/unit_square_h_1_128.npz"), 2),
        (Path("data/geometry/disk_radius_05_1_128.npz"), 2),
        (Path("data/geometry/annulus_02_05_1_128.npz"), 4),
        (Path("data/geometry/pentagram_r05_h00078125.npz"), 4),
    ],
)
def test_canonical_geometry_k_regression(path: Path, expected_k: int) -> None:
    geometry = load_complex_geometry(path, dtype=torch.float64)
    topology = AxialSegmentTopologyAnalyzer.from_geometry(geometry).analyze()
    selection = select_geometry_k(
        topology,
        config=GeometryKSelectionConfig(enabled=True),
        max_subspace_dimension=8,
    )

    assert selection.selected_subspace_dimension == expected_k
