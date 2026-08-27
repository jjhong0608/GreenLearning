from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from greenonet.complex_geometry import ComplexGeometryMetadata
from greenonet.config import (
    BalanceProjectionConfig,
    CouplingModelConfig,
    GeometryKSelectionConfig,
    SymmetricTangentGreenResponseProjectionConfig,
)


POINTWISE_TAIL_QUANTILE = 0.05


@dataclass(frozen=True)
class AxialTopologyResult:
    """Connected axial-segment graph distances for every valid-point pair."""

    coords: np.ndarray
    num_x_segments: int
    num_y_segments: int
    point_graph_diameter: int
    a_graph_diameter: int
    point_distance_pair_counts: tuple[int, ...]
    a_distance_pair_counts: tuple[int, ...]
    point_a_distance_counts: np.ndarray
    point_a_eccentricity: np.ndarray
    longest_path_point_ids: tuple[int, ...]

    @property
    def num_points(self) -> int:
        return int(self.coords.shape[0])


class AxialSegmentTopologyAnalyzer:
    """Analyze tangent reach on the connected axial-segment incidence graph."""

    def __init__(
        self,
        *,
        coords: np.ndarray,
        x_segment_id: np.ndarray,
        y_segment_id: np.ndarray,
        chunk_size: int = 256,
    ) -> None:
        self.coords = np.asarray(coords, dtype=np.float64)
        self.x_segment_id = np.asarray(x_segment_id, dtype=np.int64)
        self.y_segment_id = np.asarray(y_segment_id, dtype=np.int64)
        self.chunk_size = chunk_size
        self._validate()

    @classmethod
    def from_geometry(
        cls,
        geometry: ComplexGeometryMetadata,
        *,
        chunk_size: int = 256,
    ) -> AxialSegmentTopologyAnalyzer:
        return cls(
            coords=geometry.coords_valid.detach().cpu().numpy(),
            x_segment_id=geometry.x_segment_id.detach().cpu().numpy(),
            y_segment_id=geometry.y_segment_id.detach().cpu().numpy(),
            chunk_size=chunk_size,
        )

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        chunk_size: int = 256,
    ) -> AxialSegmentTopologyAnalyzer:
        if not path.is_file():
            raise FileNotFoundError(f"Geometry NPZ does not exist: {path}")
        with np.load(path) as raw:
            required = {"coords_valid", "x_segment_id", "y_segment_id"}
            missing = sorted(required.difference(raw.files))
            if missing:
                raise ValueError(f"Geometry NPZ is missing required arrays: {missing}")
            return cls(
                coords=np.asarray(raw["coords_valid"]),
                x_segment_id=np.asarray(raw["x_segment_id"]),
                y_segment_id=np.asarray(raw["y_segment_id"]),
                chunk_size=chunk_size,
            )

    def _validate(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must have shape (P, 2).")
        num_points = self.coords.shape[0]
        if num_points < 1:
            raise ValueError("At least one valid point is required.")
        for name, values in (
            ("x_segment_id", self.x_segment_id),
            ("y_segment_id", self.y_segment_id),
        ):
            if values.shape != (num_points,):
                raise ValueError(f"{name} must have shape (P,).")
            if np.any(values < 0):
                raise ValueError(f"{name} must be non-negative.")
            unique = np.unique(values)
            np.testing.assert_array_equal(
                unique,
                np.arange(unique.size, dtype=np.int64),
                err_msg=f"{name} must be contiguous from zero.",
            )
        if not np.all(np.isfinite(self.coords)):
            raise ValueError("coords must be finite.")
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")

    def analyze(self) -> AxialTopologyResult:
        num_x_segments = int(self.x_segment_id.max()) + 1
        num_y_segments = int(self.y_segment_id.max()) + 1
        endpoints = np.stack(
            (self.x_segment_id, self.y_segment_id + num_x_segments),
            axis=1,
        )
        num_segments = num_x_segments + num_y_segments
        adjacency = self._build_segment_adjacency(endpoints, num_segments)
        segment_distances = self._all_pairs_segment_distances(adjacency)

        max_possible_a_distance = (num_segments + 1) // 2
        per_point_a_counts = np.zeros(
            (self.coords.shape[0], max_possible_a_distance + 1),
            dtype=np.int64,
        )
        point_pair_counts = np.zeros(num_segments + 1, dtype=np.int64)
        longest_distance = -1
        longest_pair = (0, 0)

        for start in range(0, self.coords.shape[0], self.chunk_size):
            source_endpoints = endpoints[start : start + self.chunk_size]
            point_distance = self._point_distances(
                source_endpoints=source_endpoints,
                all_endpoints=endpoints,
                segment_distances=segment_distances,
            )
            rows = np.arange(
                start,
                min(start + self.chunk_size, self.coords.shape[0]),
            )
            point_distance[np.arange(rows.size), rows] = 0
            values, counts = np.unique(point_distance, return_counts=True)
            point_pair_counts[values] += counts

            local_maximum = int(point_distance.max())
            if local_maximum > longest_distance:
                local_i, point_j = np.argwhere(point_distance == local_maximum)[0]
                longest_distance = local_maximum
                longest_pair = (start + int(local_i), int(point_j))

            a_distance = (point_distance + 1) // 2
            for local_index, point_index in enumerate(rows):
                counts_row = np.bincount(
                    a_distance[local_index],
                    minlength=max_possible_a_distance + 1,
                )
                per_point_a_counts[point_index] = counts_row[
                    : max_possible_a_distance + 1
                ]

        point_pair_counts = point_pair_counts[: longest_distance + 1]
        a_graph_diameter = (longest_distance + 1) // 2
        a_pair_counts = np.zeros(a_graph_diameter + 1, dtype=np.int64)
        for distance, count in enumerate(point_pair_counts):
            a_pair_counts[(distance + 1) // 2] += count
        per_point_a_counts = per_point_a_counts[:, : a_graph_diameter + 1]
        point_a_eccentricity = np.asarray(
            [int(np.flatnonzero(row)[-1]) for row in per_point_a_counts],
            dtype=np.int64,
        )
        longest_path = self._reconstruct_point_path(
            start_point=longest_pair[0],
            end_point=longest_pair[1],
            endpoints=endpoints,
            adjacency=adjacency,
            num_x_segments=num_x_segments,
        )
        if len(longest_path) - 1 != longest_distance:
            raise RuntimeError("Reconstructed longest path has inconsistent length.")

        return AxialTopologyResult(
            coords=self.coords.copy(),
            num_x_segments=num_x_segments,
            num_y_segments=num_y_segments,
            point_graph_diameter=longest_distance,
            a_graph_diameter=a_graph_diameter,
            point_distance_pair_counts=tuple(int(v) for v in point_pair_counts),
            a_distance_pair_counts=tuple(int(v) for v in a_pair_counts),
            point_a_distance_counts=per_point_a_counts,
            point_a_eccentricity=point_a_eccentricity,
            longest_path_point_ids=longest_path,
        )

    def distances_from_point(self, point_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return point-graph and tangent-graph distances from one active point."""
        if isinstance(point_index, bool) or not isinstance(point_index, int):
            raise TypeError("point_index must be an integer.")
        if point_index < 0 or point_index >= self.coords.shape[0]:
            raise IndexError(f"point_index must be in [0, {self.coords.shape[0] - 1}].")

        num_x_segments = int(self.x_segment_id.max()) + 1
        endpoints = np.stack(
            (self.x_segment_id, self.y_segment_id + num_x_segments),
            axis=1,
        )
        num_segments = num_x_segments + int(self.y_segment_id.max()) + 1
        adjacency = self._build_segment_adjacency(endpoints, num_segments)
        segment_distances = self._all_pairs_segment_distances(adjacency)
        point_distance = self._point_distances(
            source_endpoints=endpoints[point_index : point_index + 1],
            all_endpoints=endpoints,
            segment_distances=segment_distances,
        )[0]
        point_distance[point_index] = 0
        return point_distance, (point_distance + 1) // 2

    @staticmethod
    def _build_segment_adjacency(
        endpoints: np.ndarray,
        num_segments: int,
    ) -> tuple[tuple[int, ...], ...]:
        adjacency: list[set[int]] = [set() for _ in range(num_segments)]
        seen_pairs: set[tuple[int, int]] = set()
        for x_segment, y_segment in endpoints:
            pair = int(x_segment), int(y_segment)
            if pair in seen_pairs:
                raise ValueError(
                    "Each x/y connected-segment pair must identify one valid point."
                )
            seen_pairs.add(pair)
            adjacency[pair[0]].add(pair[1])
            adjacency[pair[1]].add(pair[0])
        return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)

    @staticmethod
    def _all_pairs_segment_distances(
        adjacency: tuple[tuple[int, ...], ...],
    ) -> np.ndarray:
        num_segments = len(adjacency)
        infinity = num_segments + 1
        dtype = np.int16 if infinity <= np.iinfo(np.int16).max else np.int32
        distances = np.full((num_segments, num_segments), infinity, dtype=dtype)
        for source in range(num_segments):
            queue: deque[int] = deque((source,))
            distances[source, source] = 0
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if distances[source, neighbor] == infinity:
                        distances[source, neighbor] = distances[source, node] + 1
                        queue.append(neighbor)
        if np.any(distances == infinity):
            raise ValueError(
                "The connected axial-segment incidence graph is disconnected."
            )
        return distances

    @staticmethod
    def _point_distances(
        *,
        source_endpoints: np.ndarray,
        all_endpoints: np.ndarray,
        segment_distances: np.ndarray,
    ) -> np.ndarray:
        distances = np.asarray(
            np.minimum.reduce(
                (
                    segment_distances[
                        source_endpoints[:, 0, None], all_endpoints[None, :, 0]
                    ],
                    segment_distances[
                        source_endpoints[:, 0, None], all_endpoints[None, :, 1]
                    ],
                    segment_distances[
                        source_endpoints[:, 1, None], all_endpoints[None, :, 0]
                    ],
                    segment_distances[
                        source_endpoints[:, 1, None], all_endpoints[None, :, 1]
                    ],
                )
            ),
            dtype=segment_distances.dtype,
        )
        return distances + 1

    @staticmethod
    def _reconstruct_point_path(
        *,
        start_point: int,
        end_point: int,
        endpoints: np.ndarray,
        adjacency: tuple[tuple[int, ...], ...],
        num_x_segments: int,
    ) -> tuple[int, ...]:
        if start_point == end_point:
            return (start_point,)
        best_segment_path: tuple[int, ...] | None = None
        end_segments = {int(value) for value in endpoints[end_point]}
        for start_segment_raw in endpoints[start_point]:
            start_segment = int(start_segment_raw)
            queue: deque[int] = deque((start_segment,))
            parent: dict[int, int | None] = {start_segment: None}
            reached: int | None = None
            while queue:
                node = queue.popleft()
                if node in end_segments:
                    reached = node
                    break
                for neighbor in adjacency[node]:
                    if neighbor not in parent:
                        parent[neighbor] = node
                        queue.append(neighbor)
            if reached is None:
                continue
            reverse_path: list[int] = []
            path_node: int | None = reached
            while path_node is not None:
                reverse_path.append(path_node)
                path_node = parent[path_node]
            candidate = tuple(reversed(reverse_path))
            if best_segment_path is None or len(candidate) < len(best_segment_path):
                best_segment_path = candidate
        if best_segment_path is None:
            raise RuntimeError("Could not reconstruct a path between valid points.")

        edge_to_point = {
            (int(x_segment), int(y_segment)): point_index
            for point_index, (x_segment, y_segment) in enumerate(endpoints)
        }
        point_path = [start_point]
        for left, right in zip(
            best_segment_path[:-1],
            best_segment_path[1:],
            strict=True,
        ):
            pair = (left, right) if left < num_x_segments else (right, left)
            point_path.append(edge_to_point[pair])
        point_path.append(end_point)
        return tuple(point_path)


def pointwise_reach_fraction(topology: AxialTopologyResult, k: int) -> np.ndarray:
    """Fraction of active points structurally reachable from every point by K."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1:
        raise ValueError("k must be positive.")
    width = min(k, topology.point_a_distance_counts.shape[1])
    return np.asarray(
        topology.point_a_distance_counts[:, :width].sum(axis=1)
        / float(topology.num_points),
        dtype=np.float64,
    )


def global_reach_fraction(topology: AxialTopologyResult, k: int) -> float:
    """Ordered point-pair fraction structurally reachable by K."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 1:
        raise ValueError("k must be positive.")
    width = min(k, len(topology.a_distance_pair_counts))
    return float(sum(topology.a_distance_pair_counts[:width])) / float(
        topology.num_points**2
    )


@dataclass(frozen=True)
class GeometryKReachMetric:
    subspace_dimension: int
    global_reach_fraction: float
    pointwise_tail_reach_fraction: float


@dataclass(frozen=True)
class GeometryKSelectionResult:
    selected_subspace_dimension: int
    configured_max_subspace_dimension: int
    global_reach_threshold: float
    pointwise_tail_reach_threshold: float
    pointwise_tail_quantile: float
    selected_metric: GeometryKReachMetric
    metric_at_configured_max: GeometryKReachMetric
    topology: AxialTopologyResult
    setup_seconds: float


def geometry_k_reach_metric(
    topology: AxialTopologyResult,
    k: int,
) -> GeometryKReachMetric:
    pointwise = pointwise_reach_fraction(topology, k)
    return GeometryKReachMetric(
        subspace_dimension=k,
        global_reach_fraction=global_reach_fraction(topology, k),
        pointwise_tail_reach_fraction=float(
            np.quantile(pointwise, POINTWISE_TAIL_QUANTILE)
        ),
    )


def select_geometry_k(
    topology: AxialTopologyResult,
    *,
    config: GeometryKSelectionConfig,
    max_subspace_dimension: int,
    setup_seconds: float = 0.0,
) -> GeometryKSelectionResult:
    """Select the first K satisfying both fixed geometry-only reach criteria."""
    selected_metric: GeometryKReachMetric | None = None
    for k in range(1, topology.a_graph_diameter + 2):
        metric = geometry_k_reach_metric(topology, k)
        if (
            metric.global_reach_fraction >= config.global_reach_threshold
            and metric.pointwise_tail_reach_fraction
            >= config.pointwise_tail_reach_threshold
        ):
            selected_metric = metric
            break
    if selected_metric is None:
        raise RuntimeError("No K satisfies the reach rule despite full graph reach.")
    metric_at_max = geometry_k_reach_metric(topology, max_subspace_dimension)
    if selected_metric.subspace_dimension > max_subspace_dimension:
        raise ValueError(
            "Geometry-only tangent selection requires "
            f"K={selected_metric.subspace_dimension}, which exceeds "
            f"max_subspace_dimension={max_subspace_dimension}. At the configured "
            f"limit C_global={metric_at_max.global_reach_fraction:.12f} and "
            "Q_0.05(C_i)="
            f"{metric_at_max.pointwise_tail_reach_fraction:.12f}; "
            f"point_graph_diameter={topology.point_graph_diameter}, "
            f"A_graph_diameter={topology.a_graph_diameter}."
        )
    return GeometryKSelectionResult(
        selected_subspace_dimension=selected_metric.subspace_dimension,
        configured_max_subspace_dimension=max_subspace_dimension,
        global_reach_threshold=float(config.global_reach_threshold),
        pointwise_tail_reach_threshold=float(config.pointwise_tail_reach_threshold),
        pointwise_tail_quantile=POINTWISE_TAIL_QUANTILE,
        selected_metric=selected_metric,
        metric_at_configured_max=metric_at_max,
        topology=topology,
        setup_seconds=setup_seconds,
    )


@dataclass(frozen=True)
class TangentDimensionResolution:
    model_config: CouplingModelConfig
    provenance: dict[str, object] | None


class GeometryTangentDimensionResolver:
    """Resolve optional geometry-only K before any tangent runtime is built."""

    @staticmethod
    def resolve(
        *,
        model_config: CouplingModelConfig,
        geometry: ComplexGeometryMetadata,
        geometry_path: Path,
        chunk_size: int = 256,
    ) -> TangentDimensionResolution:
        projection = BalanceProjectionConfig.from_raw(model_config.balance_projection)
        tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
            projection.symmetric_tangent_green_response
        )
        geometry_k_selection = GeometryKSelectionConfig.from_raw(
            tangent.geometry_k_selection
        )
        if projection.mode != "symmetric_tangent_green_response":
            return TangentDimensionResolution(
                model_config=model_config, provenance=None
            )
        if not geometry_k_selection.enabled:
            return TangentDimensionResolution(
                model_config=model_config,
                provenance={
                    "selection_mode": "explicit",
                    "configured_subspace_dimension": tangent.subspace_dimension,
                    "resolved_subspace_dimension": tangent.subspace_dimension,
                    "max_subspace_dimension": tangent.max_subspace_dimension,
                    "geometry_k_selection_enabled": False,
                },
            )

        started = time.perf_counter()
        topology = AxialSegmentTopologyAnalyzer.from_geometry(
            geometry,
            chunk_size=chunk_size,
        ).analyze()
        elapsed = time.perf_counter() - started
        selection = select_geometry_k(
            topology,
            config=geometry_k_selection,
            max_subspace_dimension=tangent.max_subspace_dimension,
            setup_seconds=elapsed,
        )
        resolved_tangent = replace(
            tangent,
            subspace_dimension=selection.selected_subspace_dimension,
            geometry_k_selection=replace(
                geometry_k_selection,
                enabled=False,
            ),
        )
        resolved_projection = replace(
            projection,
            symmetric_tangent_green_response=resolved_tangent,
        )
        resolved_model = replace(
            model_config,
            balance_projection=resolved_projection,
        )
        selected = selection.selected_metric
        at_max = selection.metric_at_configured_max
        provenance: dict[str, object] = {
            "selection_mode": "geometry_auto",
            "configured_subspace_dimension": tangent.subspace_dimension,
            "resolved_subspace_dimension": selection.selected_subspace_dimension,
            "max_subspace_dimension": tangent.max_subspace_dimension,
            "geometry_k_selection_enabled": True,
            "global_reach_threshold": selection.global_reach_threshold,
            "pointwise_tail_reach_threshold": (
                selection.pointwise_tail_reach_threshold
            ),
            "pointwise_tail_quantile": selection.pointwise_tail_quantile,
            "selected_global_reach_fraction": selected.global_reach_fraction,
            "selected_pointwise_tail_reach_fraction": (
                selected.pointwise_tail_reach_fraction
            ),
            "max_global_reach_fraction": at_max.global_reach_fraction,
            "max_pointwise_tail_reach_fraction": (at_max.pointwise_tail_reach_fraction),
            "geometry_path": str(geometry_path.resolve()),
            "geometry_sha256": _sha256_file(geometry_path),
            "num_points": topology.num_points,
            "num_x_segments": topology.num_x_segments,
            "num_y_segments": topology.num_y_segments,
            "point_graph_diameter": topology.point_graph_diameter,
            "a_graph_diameter": topology.a_graph_diameter,
            "selection_setup_seconds": selection.setup_seconds,
            "selector_formula": (
                "min K: C_global(K)>=tau_global and Q_0.05(C_i(K))>=tau_tail"
            ),
            "pde_dependent_inputs_used": False,
            "reference_targets_used": False,
        }
        return TangentDimensionResolution(
            model_config=resolved_model,
            provenance=provenance,
        )


def materialized_tangent_config(config: CouplingModelConfig) -> dict[str, object]:
    """Return the resolved tangent block in JSON-compatible form."""
    projection = BalanceProjectionConfig.from_raw(config.balance_projection)
    tangent = SymmetricTangentGreenResponseProjectionConfig.from_raw(
        projection.symmetric_tangent_green_response
    )
    return asdict(tangent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
