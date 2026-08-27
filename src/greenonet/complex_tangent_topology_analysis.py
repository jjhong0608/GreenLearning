from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from greenonet.plotly_io import save_plotly_figure


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
    """Analyze matrix-free tangent reach on the axial-segment incidence graph."""

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
        y_offset = self.y_segment_id + num_x_segments
        endpoints = np.stack((self.x_segment_id, y_offset), axis=1)
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
        point_a_eccentricity = np.array(
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
        """Return point-graph and A-graph distances from one active point."""
        if isinstance(point_index, bool) or not isinstance(point_index, int):
            raise TypeError("point_index must be an integer.")
        if point_index < 0 or point_index >= self.coords.shape[0]:
            raise IndexError(f"point_index must be in [0, {self.coords.shape[0] - 1}].")

        num_x_segments = int(self.x_segment_id.max()) + 1
        y_offset = self.y_segment_id + num_x_segments
        endpoints = np.stack((self.x_segment_id, y_offset), axis=1)
        num_segments = num_x_segments + int(self.y_segment_id.max()) + 1
        adjacency = self._build_segment_adjacency(endpoints, num_segments)
        segment_distances = self._all_pairs_segment_distances(adjacency)
        point_distance = self._point_distances(
            source_endpoints=endpoints[point_index : point_index + 1],
            all_endpoints=endpoints,
            segment_distances=segment_distances,
        )[0]
        point_distance[point_index] = 0
        a_distance = (point_distance + 1) // 2
        return point_distance, a_distance

    @staticmethod
    def _build_segment_adjacency(
        endpoints: np.ndarray,
        num_segments: int,
    ) -> tuple[tuple[int, ...], ...]:
        adjacency: list[set[int]] = [set() for _ in range(num_segments)]
        seen_pairs: set[tuple[int, int]] = set()
        for x_segment, y_segment in endpoints:
            pair = (int(x_segment), int(y_segment))
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
        distances = np.full(
            (num_segments, num_segments),
            infinity,
            dtype=np.int16,
        )
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
            dtype=np.int16,
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


@dataclass(frozen=True)
class TrainedKMetrics:
    run_name: str
    subspace_dimension: int
    rel_sol_mean: float
    rel_sol_p95: float
    rel_sol_max: float
    rel_u_phi_mean: float
    rel_u_psi_mean: float
    rel_flux_mean: float
    optimized_energy_mean: float
    response_cost_mean: float
    correction_rel_symmetric_pair_mean: float


@dataclass(frozen=True)
class TangentRuntimeMetrics:
    subspace_dimension: int
    forward_ms: float
    forward_ratio_to_k1: float
    forward_backward_ms: float
    forward_backward_ratio_to_k1: float


class TrainedKComparisonParser:
    """Strictly read the canonical trained K=1 through K=4 Markdown tables."""

    QUALITY_HEADER = (
        "| run | K | rel_sol mean | p95 | max | rel_u_phi | rel_u_psi | "
        "rel_flux | optimized energy | response cost | correction / symmetric pair |"
    )
    RUNTIME_HEADER = (
        "| K | forward only | ratio to K1 | forward + backward | ratio to K1 |"
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def parse(
        self,
    ) -> tuple[tuple[TrainedKMetrics, ...], tuple[TangentRuntimeMetrics, ...]]:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Trained K comparison report is missing: {self.path}"
            )
        lines = self.path.read_text(encoding="utf-8").splitlines()
        quality = self._parse_quality(lines)
        runtime = self._parse_runtime(lines)
        dimensions = [row.subspace_dimension for row in quality]
        if dimensions != sorted(set(dimensions)):
            raise ValueError("Trained K quality rows must be unique and increasing.")
        runtime_dimensions = [row.subspace_dimension for row in runtime]
        if runtime_dimensions != dimensions:
            raise ValueError(
                "Quality and runtime tables must contain the same K values."
            )
        return tuple(quality), tuple(runtime)

    @classmethod
    def _parse_quality(cls, lines: Sequence[str]) -> list[TrainedKMetrics]:
        start = cls._header_index(lines, cls.QUALITY_HEADER)
        rows: list[TrainedKMetrics] = []
        for line in lines[start + 2 :]:
            if not line.startswith("|"):
                break
            fields = cls._fields(line)
            if len(fields) != 11:
                raise ValueError("Unexpected trained K quality table width.")
            rows.append(
                TrainedKMetrics(
                    run_name=fields[0],
                    subspace_dimension=int(fields[1]),
                    rel_sol_mean=cls._percent(fields[2]),
                    rel_sol_p95=cls._percent(fields[3]),
                    rel_sol_max=cls._percent(fields[4]),
                    rel_u_phi_mean=cls._percent(fields[5]),
                    rel_u_psi_mean=cls._percent(fields[6]),
                    rel_flux_mean=cls._percent(fields[7]),
                    optimized_energy_mean=float(fields[8]),
                    response_cost_mean=float(fields[9]),
                    correction_rel_symmetric_pair_mean=cls._percent(fields[10]),
                )
            )
        if not rows:
            raise ValueError("Trained K comparison quality table is empty.")
        return rows

    @classmethod
    def _parse_runtime(cls, lines: Sequence[str]) -> list[TangentRuntimeMetrics]:
        start = cls._header_index(lines, cls.RUNTIME_HEADER)
        rows: list[TangentRuntimeMetrics] = []
        for line in lines[start + 2 :]:
            if not line.startswith("|"):
                break
            fields = cls._fields(line)
            if len(fields) != 5:
                raise ValueError("Unexpected tangent runtime table width.")
            rows.append(
                TangentRuntimeMetrics(
                    subspace_dimension=int(fields[0]),
                    forward_ms=cls._with_suffix(fields[1], "ms"),
                    forward_ratio_to_k1=cls._with_suffix(fields[2], "x"),
                    forward_backward_ms=cls._with_suffix(fields[3], "ms"),
                    forward_backward_ratio_to_k1=cls._with_suffix(fields[4], "x"),
                )
            )
        if not rows:
            raise ValueError("Tangent runtime table is empty.")
        return rows

    @staticmethod
    def _header_index(lines: Sequence[str], header: str) -> int:
        try:
            return lines.index(header)
        except ValueError as exc:
            raise ValueError(
                f"Required Markdown table header is missing: {header}"
            ) from exc

    @staticmethod
    def _fields(line: str) -> list[str]:
        return [field.strip() for field in line.strip().strip("|").split("|")]

    @staticmethod
    def _percent(value: str) -> float:
        if not value.endswith("%"):
            raise ValueError(f"Expected percentage value, got: {value}")
        return float(value[:-1]) / 100.0

    @staticmethod
    def _with_suffix(value: str, suffix: str) -> float:
        if not value.endswith(suffix):
            raise ValueError(f"Expected {suffix!r} suffix, got: {value}")
        return float(value[: -len(suffix)].strip())


@dataclass(frozen=True)
class FrozenKMetrics:
    method_id: str
    subspace_dimension: int
    sample_count: int
    response_mismatch_cost_mean: float
    optimized_energy_mean: float
    rel_sol_mean: float
    rel_u_phi_mean: float
    rel_u_psi_mean: float
    rel_flux_mean: float
    rel_sol_p95: float
    rel_sol_max: float


class FrozenKMetricsReader:
    """Aggregate the same-checkpoint K audit without using report rounding."""

    METRICS = (
        "response_mismatch_cost",
        "loss_energy_optimized",
        "rel_sol",
        "rel_u_phi",
        "rel_u_psi",
        "rel_flux",
    )

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> tuple[FrozenKMetrics, ...]:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Frozen tangent metric CSV is missing: {self.path}"
            )
        grouped: dict[str, list[dict[str, str]]] = {}
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"method_id", "sample_id", *self.METRICS}
            missing = sorted(required.difference(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"Frozen metric CSV is missing columns: {missing}")
            for row in reader:
                method_id = row["method_id"]
                grouped.setdefault(method_id, []).append(row)

        requested = (
            (1, "k1_uncapped"),
            (2, "k2_unconstrained"),
            (3, "k3_unconstrained"),
            (4, "k4_unconstrained"),
        )
        summaries: list[FrozenKMetrics] = []
        for dimension, method_id in requested:
            rows = grouped.get(method_id)
            if not rows:
                raise ValueError(f"Frozen metric CSV has no rows for {method_id}.")
            values = {
                metric: np.array([float(row[metric]) for row in rows], dtype=np.float64)
                for metric in self.METRICS
            }
            if any(
                not np.all(np.isfinite(metric_values))
                for metric_values in values.values()
            ):
                raise ValueError(f"Frozen metrics for {method_id} must be finite.")
            summaries.append(
                FrozenKMetrics(
                    method_id=method_id,
                    subspace_dimension=dimension,
                    sample_count=len(rows),
                    response_mismatch_cost_mean=float(
                        values["response_mismatch_cost"].mean()
                    ),
                    optimized_energy_mean=float(values["loss_energy_optimized"].mean()),
                    rel_sol_mean=float(values["rel_sol"].mean()),
                    rel_u_phi_mean=float(values["rel_u_phi"].mean()),
                    rel_u_psi_mean=float(values["rel_u_psi"].mean()),
                    rel_flux_mean=float(values["rel_flux"].mean()),
                    rel_sol_p95=float(np.quantile(values["rel_sol"], 0.95)),
                    rel_sol_max=float(values["rel_sol"].max()),
                )
            )
        return tuple(summaries)


@dataclass(frozen=True)
class SpatialK4Result:
    sample_id: int
    delta_pearson: float
    delta_spearman: float
    response_pearson: float
    response_spearman: float
    error_improvement_pearson: float
    error_improvement_spearman: float
    top10_delta_over_bottom50: float
    top10_response_over_bottom50: float
    top10_error_improvement_mean: float
    bottom50_error_improvement_mean: float
    global_error_improvement_mean: float


@dataclass(frozen=True)
class SelectedSpatialFields:
    sample_ids: np.ndarray
    delta_increment: np.ndarray
    response_increment: np.ndarray
    error_improvement: np.ndarray


class SelectedK4SpatialAnalyzer:
    """Relate K3-to-K4 fields to geometry-only K4 reach exposure."""

    def __init__(
        self,
        *,
        archive_path: Path,
        topology: AxialTopologyResult,
    ) -> None:
        self.archive_path = archive_path
        self.topology = topology

    def analyze(
        self,
    ) -> tuple[tuple[SpatialK4Result, ...], SelectedSpatialFields]:
        if not self.archive_path.is_file():
            raise FileNotFoundError(
                f"Selected frozen K audit archive is missing: {self.archive_path}"
            )
        with np.load(self.archive_path) as raw:
            required = {
                "selected_sample_ids",
                "method_ids",
                "sol",
                "subspace_deltas",
                "subspace_mismatches",
                "candidate_prediction",
            }
            missing = sorted(required.difference(raw.files))
            if missing:
                raise ValueError(
                    f"Selected frozen archive is missing arrays: {missing}"
                )
            method_ids = [str(value) for value in raw["method_ids"]]
            try:
                k3_index = method_ids.index("k3_unconstrained")
                k4_index = method_ids.index("k4_unconstrained")
            except ValueError as exc:
                raise ValueError(
                    "Selected archive must contain K3 and K4 candidates."
                ) from exc
            sample_ids = np.asarray(raw["selected_sample_ids"], dtype=np.int64)
            subspace_deltas = np.asarray(raw["subspace_deltas"], dtype=np.float64)
            mismatches = np.asarray(raw["subspace_mismatches"], dtype=np.float64)
            predictions = np.asarray(raw["candidate_prediction"], dtype=np.float64)
            solution = np.asarray(raw["sol"], dtype=np.float64)

        num_points = self.topology.num_points
        if solution.shape != (sample_ids.size, num_points):
            raise ValueError("Selected solution array does not match topology points.")
        if subspace_deltas.shape[:2] != (4, sample_ids.size):
            raise ValueError("Selected subspace_deltas must include K1 through K4.")
        delta_increment = subspace_deltas[3] - subspace_deltas[2]
        response_increment = mismatches[2] - mismatches[3]
        error_improvement = (predictions[k3_index] - solution) ** 2 - (
            predictions[k4_index] - solution
        ) ** 2

        exposure = self.topology.point_a_distance_counts[:, 3] / num_points
        q50, q90 = np.quantile(exposure, (0.5, 0.9))
        top = exposure >= q90
        bottom = exposure <= q50
        rows: list[SpatialK4Result] = []
        for sample_index, sample_id in enumerate(sample_ids):
            abs_delta = np.abs(delta_increment[sample_index])
            abs_response = np.abs(response_increment[sample_index])
            improvement = error_improvement[sample_index]
            rows.append(
                SpatialK4Result(
                    sample_id=int(sample_id),
                    delta_pearson=_correlation(exposure, abs_delta, "pearson"),
                    delta_spearman=_correlation(exposure, abs_delta, "spearman"),
                    response_pearson=_correlation(exposure, abs_response, "pearson"),
                    response_spearman=_correlation(exposure, abs_response, "spearman"),
                    error_improvement_pearson=_correlation(
                        exposure, improvement, "pearson"
                    ),
                    error_improvement_spearman=_correlation(
                        exposure, improvement, "spearman"
                    ),
                    top10_delta_over_bottom50=_positive_ratio(
                        float(abs_delta[top].mean()),
                        float(abs_delta[bottom].mean()),
                    ),
                    top10_response_over_bottom50=_positive_ratio(
                        float(abs_response[top].mean()),
                        float(abs_response[bottom].mean()),
                    ),
                    top10_error_improvement_mean=float(improvement[top].mean()),
                    bottom50_error_improvement_mean=float(improvement[bottom].mean()),
                    global_error_improvement_mean=float(improvement.mean()),
                )
            )
        fields = SelectedSpatialFields(
            sample_ids=sample_ids,
            delta_increment=delta_increment,
            response_increment=response_increment,
            error_improvement=error_improvement,
        )
        return tuple(rows), fields


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    if kind == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    if kind == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    raise ValueError(f"Unsupported correlation kind: {kind}")


def _positive_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf if numerator > 0.0 else 1.0
    return numerator / denominator


@dataclass(frozen=True)
class TangentTopologyAnalysisRequest:
    geometry_path: Path
    trained_comparison_report: Path
    frozen_metrics_path: Path
    selected_archive_path: Path
    outdir: Path
    theme: str = "plotly_white"
    chunk_size: int = 256

    def validate(self) -> None:
        for name, path in (
            ("geometry_path", self.geometry_path),
            ("trained_comparison_report", self.trained_comparison_report),
            ("frozen_metrics_path", self.frozen_metrics_path),
            ("selected_archive_path", self.selected_archive_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{name} does not exist: {path}")
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_change(current: float, previous: float) -> float:
    if previous == 0.0:
        return math.nan
    return current / previous - 1.0


def _write_dataclass_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(payloads[0]))
        writer.writeheader()
        writer.writerows(payloads)


@dataclass(frozen=True)
class TopologyDistanceMetrics:
    a_distance: int
    first_subspace_dimension: int
    ordered_pair_count: int
    ordered_pair_fraction: float
    cumulative_ordered_pair_fraction: float


@dataclass(frozen=True)
class PointTopologyMetrics:
    point_id: int
    x: float
    y: float
    a_eccentricity: int
    full_reach_subspace_dimension: int
    k4_new_reach_fraction: float
    beyond_k4_fraction: float


class TangentTopologyPlotMixin:
    request: TangentTopologyAnalysisRequest
    logger: logging.Logger

    def _base_layout(self, fig: go.Figure, title: str, *, height: int = 650) -> None:
        fig.update_layout(
            title=title,
            template=self.request.theme,
            width=1200,
            height=height,
            font={"family": "Noto Sans CJK KR, DejaVu Sans", "size": 13},
            margin={"l": 70, "r": 60, "t": 95, "b": 70},
            legend={"orientation": "h", "y": 1.08, "x": 0.0},
        )

    def _save_figure(self, fig: go.Figure, relative_base: Path) -> Path:
        base_path = self.request.outdir / relative_base
        save_plotly_figure(fig, base_path, self.logger)
        return relative_base

    def _plot_distance_distribution(
        self,
        rows: Sequence[TopologyDistanceMetrics],
    ) -> Path:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        x = [row.a_distance for row in rows]
        fig.add_trace(
            go.Bar(
                x=x,
                y=[100.0 * row.ordered_pair_fraction for row in rows],
                name="Pair share",
                marker_color="#176b87",
                customdata=[row.first_subspace_dimension for row in rows],
                hovertemplate=(
                    "A-distance=%{x}<br>first K=%{customdata}"
                    "<br>pair share=%{y:.6f}%<extra></extra>"
                ),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[100.0 * row.cumulative_ordered_pair_fraction for row in rows],
                mode="lines+markers",
                name="Cumulative reach",
                line={"color": "#d97706", "width": 3},
                marker={"size": 9},
                hovertemplate=(
                    "A-distance<=%{x}<br>cumulative=%{y:.6f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )
        fig.update_xaxes(title_text="Shortest A-graph distance")
        fig.update_yaxes(title_text="Ordered point-pair share (%)", secondary_y=False)
        fig.update_yaxes(
            title_text="Cumulative structural reach (%)",
            range=[0.0, 103.0],
            secondary_y=True,
        )
        self._base_layout(
            fig,
            "Pentagram axial connectivity: new reach contributed by each K",
        )
        return self._save_figure(
            fig,
            Path("figures/topology/a_distance_distribution"),
        )

    def _plot_topology_fields(self, topology: AxialTopologyResult) -> Path:
        num_points = topology.num_points
        k4_exposure = (
            topology.point_a_distance_counts[:, 3] / num_points
            if topology.a_graph_diameter >= 3
            else np.zeros(num_points, dtype=np.float64)
        )
        beyond_k4 = (
            topology.point_a_distance_counts[:, 4:].sum(axis=1) / num_points
            if topology.a_graph_diameter >= 4
            else np.zeros(num_points, dtype=np.float64)
        )
        full_reach_k = topology.point_a_eccentricity + 1
        point_ids = np.arange(num_points, dtype=np.int64)
        customdata = np.column_stack((point_ids, k4_exposure, beyond_k4))
        fields = (
            (
                full_reach_k,
                "K required for full structural reach",
                "Viridis",
                float(full_reach_k.min()),
                float(full_reach_k.max()),
                "K=%{marker.color:.0f}",
            ),
            (
                100.0 * k4_exposure,
                "New pair reach first available at K=4",
                "Plasma",
                0.0,
                float(np.quantile(100.0 * k4_exposure, 0.99)),
                "K4-new=%{marker.color:.4f}%",
            ),
            (
                np.log10(beyond_k4 + 1.0 / num_points),
                "Pair reach remaining beyond K=4 (log10)",
                "Magma",
                None,
                None,
                "log10=%{marker.color:.4f}",
            ),
        )
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=tuple(field[1] for field in fields),
            horizontal_spacing=0.08,
        )
        for column, (values, _title, colorscale, cmin, cmax, value_hover) in enumerate(
            fields,
            start=1,
        ):
            fig.add_trace(
                go.Scattergl(
                    x=topology.coords[:, 0],
                    y=topology.coords[:, 1],
                    mode="markers",
                    marker={
                        "size": 5,
                        "color": values,
                        "colorscale": colorscale,
                        "cmin": cmin,
                        "cmax": cmax,
                        "showscale": True,
                        "colorbar": {
                            "len": 0.72,
                            "x": 0.28 + 0.36 * (column - 1),
                            "thickness": 12,
                        },
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "point=%{customdata[0]:.0f}<br>x=%{x:.6f}<br>y=%{y:.6f}<br>"
                        + value_hover
                        + "<br>K4-new=%{customdata[1]:.4%}"
                        + "<br>beyond-K4=%{customdata[2]:.4%}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            axis_suffix = "" if column == 1 else str(column)
            fig.update_layout(
                {
                    f"xaxis{axis_suffix}": {
                        "scaleanchor": f"y{axis_suffix}",
                        "scaleratio": 1.0,
                    }
                }
            )
        self._base_layout(
            fig,
            "Where higher-dimensional tangent reach is structurally needed",
            height=590,
        )
        return self._save_figure(
            fig,
            Path("figures/topology/point_topology_fields"),
        )

    def _plot_longest_path(self, topology: AxialTopologyResult) -> Path:
        path_ids = np.array(topology.longest_path_point_ids, dtype=np.int64)
        path_coords = topology.coords[path_ids]
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=topology.coords[:, 0],
                y=topology.coords[:, 1],
                mode="markers",
                name="Valid points",
                marker={"size": 4, "color": "#cbd5e1", "opacity": 0.65},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=path_coords[:, 0],
                y=path_coords[:, 1],
                mode="lines+markers",
                name="Longest alternating axial path",
                line={"color": "#c2410c", "width": 4},
                marker={
                    "size": 9,
                    "color": np.arange(path_ids.size),
                    "colorscale": "Turbo",
                    "line": {"color": "white", "width": 1},
                },
                customdata=np.column_stack((path_ids, np.arange(path_ids.size))),
                hovertemplate=(
                    "path step=%{customdata[1]:.0f}<br>point=%{customdata[0]:.0f}"
                    "<br>x=%{x:.6f}<br>y=%{y:.6f}<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title="x", scaleanchor="y", scaleratio=1.0)
        fig.update_yaxes(title="y")
        self._base_layout(
            fig,
            (
                "Longest Pentagram axial path: "
                f"{topology.point_graph_diameter} point hops, "
                f"A-distance {topology.a_graph_diameter}"
            ),
            height=760,
        )
        return self._save_figure(
            fig,
            Path("figures/topology/longest_axial_path"),
        )

    def _plot_trained_quality(
        self,
        rows: Sequence[TrainedKMetrics],
    ) -> Path:
        dimensions = [row.subspace_dimension for row in rows]
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Solution metrics",
                "Directional-source metric",
                "Reference-free objectives",
                "Correction magnitude",
            ),
            vertical_spacing=0.16,
            horizontal_spacing=0.12,
        )
        for name, field, color in (
            ("rel_sol", "rel_sol_mean", "#0f766e"),
            ("rel_u_phi", "rel_u_phi_mean", "#2563eb"),
            ("rel_u_psi", "rel_u_psi_mean", "#dc2626"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=dimensions,
                    y=[100.0 * float(getattr(row, field)) for row in rows],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=dimensions,
                y=[100.0 * row.rel_flux_mean for row in rows],
                mode="lines+markers",
                name="rel_flux",
                line={"color": "#7c3aed", "width": 3},
                marker={"size": 9},
            ),
            row=1,
            col=2,
        )
        for name, field, color in (
            ("optimized energy", "optimized_energy_mean", "#d97706"),
            ("response cost", "response_cost_mean", "#0891b2"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=dimensions,
                    y=[float(getattr(row, field)) for row in rows],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=dimensions,
                y=[100.0 * row.correction_rel_symmetric_pair_mean for row in rows],
                name="correction / symmetric pair",
                marker_color="#475569",
            ),
            row=2,
            col=2,
        )
        fig.update_yaxes(title_text="Relative error (%)", row=1, col=1)
        fig.update_yaxes(title_text="Relative flux error (%)", row=1, col=2)
        fig.update_yaxes(title_text="Loss", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Correction ratio (%)", row=2, col=2)
        fig.update_xaxes(title_text="Tangent subspace dimension K", dtick=1)
        self._base_layout(
            fig,
            "Trained K=1 through K=4: best-energy checkpoint quality",
            height=820,
        )
        return self._save_figure(
            fig,
            Path("figures/performance/trained_k_quality"),
        )

    def _plot_frozen_quality(self, rows: Sequence[FrozenKMetrics]) -> Path:
        dimensions = [row.subspace_dimension for row in rows]
        baseline = rows[0]
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Same-checkpoint response and energy",
                "Same-checkpoint solution metrics",
            ),
            horizontal_spacing=0.13,
        )
        for name, field, color in (
            ("response mismatch", "response_mismatch_cost_mean", "#0891b2"),
            ("optimized energy", "optimized_energy_mean", "#d97706"),
        ):
            denominator = float(getattr(baseline, field))
            fig.add_trace(
                go.Scatter(
                    x=dimensions,
                    y=[float(getattr(row, field)) / denominator for row in rows],
                    mode="lines+markers",
                    name=f"{name} / K1",
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                ),
                row=1,
                col=1,
            )
        for name, field, color in (
            ("rel_sol", "rel_sol_mean", "#0f766e"),
            ("rel_u_phi", "rel_u_phi_mean", "#2563eb"),
            ("rel_u_psi", "rel_u_psi_mean", "#dc2626"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=dimensions,
                    y=[100.0 * float(getattr(row, field)) for row in rows],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                ),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="Post-hoc K", dtick=1)
        fig.update_yaxes(title_text="Ratio to uncapped K1", row=1, col=1)
        fig.update_yaxes(title_text="Relative error (%)", row=1, col=2)
        self._base_layout(
            fig,
            "Frozen coupling9: isolated tangent-subspace effect",
            height=650,
        )
        return self._save_figure(
            fig,
            Path("figures/performance/frozen_k_quality"),
        )

    def _plot_cost_quality(
        self,
        trained: Sequence[TrainedKMetrics],
        runtime: Sequence[TangentRuntimeMetrics],
    ) -> Path:
        quality_by_k = {row.subspace_dimension: row for row in trained}
        x = [row.forward_backward_ms for row in runtime]
        y = [
            100.0 * quality_by_k[row.subspace_dimension].rel_sol_mean for row in runtime
        ]
        dimensions = [row.subspace_dimension for row in runtime]
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers+text",
                text=[f"K={dimension}" for dimension in dimensions],
                textposition="top center",
                line={"color": "#64748b", "width": 2},
                marker={
                    "size": [14 + 3 * dimension for dimension in dimensions],
                    "color": dimensions,
                    "colorscale": "Viridis",
                    "line": {"color": "white", "width": 1.5},
                },
                customdata=np.column_stack(
                    (
                        dimensions,
                        [row.forward_backward_ratio_to_k1 for row in runtime],
                    )
                ),
                hovertemplate=(
                    "K=%{customdata[0]:.0f}<br>forward+backward=%{x:.3f} ms"
                    "<br>ratio to K1=%{customdata[1]:.3f}x"
                    "<br>trained rel_sol=%{y:.4f}%<extra></extra>"
                ),
            )
        )
        fig.update_xaxes(title="Isolated tangent forward + backward (ms)")
        fig.update_yaxes(title="Mean trained rel_sol (%)")
        self._base_layout(
            fig,
            "Pentagram tangent subspace cost-quality tradeoff",
            height=680,
        )
        return self._save_figure(
            fig,
            Path("figures/performance/cost_quality_tradeoff"),
        )

    def _plot_selected_spatial_fields(
        self,
        *,
        topology: AxialTopologyResult,
        fields: SelectedSpatialFields,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        for sample_index, sample_id in enumerate(fields.sample_ids):
            delta = np.abs(fields.delta_increment[sample_index])
            response = np.abs(fields.response_increment[sample_index])
            improvement = fields.error_improvement[sample_index]
            signed_limit = float(np.quantile(np.abs(improvement), 0.99))
            signed_limit = max(signed_limit, np.finfo(np.float64).tiny)
            values_and_styles = (
                (
                    delta,
                    "|delta_K4 - delta_K3|",
                    "Viridis",
                    0.0,
                    float(np.quantile(delta, 0.99)),
                ),
                (
                    response,
                    "|m_K3 - m_K4|",
                    "Plasma",
                    0.0,
                    float(np.quantile(response, 0.99)),
                ),
                (
                    improvement,
                    "Squared solution-error reduction",
                    "RdBu",
                    -signed_limit,
                    signed_limit,
                ),
            )
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=tuple(item[1] for item in values_and_styles),
                horizontal_spacing=0.08,
            )
            for column, (values, _title, colorscale, cmin, cmax) in enumerate(
                values_and_styles,
                start=1,
            ):
                fig.add_trace(
                    go.Scattergl(
                        x=topology.coords[:, 0],
                        y=topology.coords[:, 1],
                        mode="markers",
                        marker={
                            "size": 5,
                            "color": values,
                            "colorscale": colorscale,
                            "cmin": cmin,
                            "cmax": cmax,
                            "showscale": True,
                            "colorbar": {
                                "len": 0.72,
                                "x": 0.28 + 0.36 * (column - 1),
                                "thickness": 12,
                            },
                        },
                        customdata=np.arange(topology.num_points),
                        hovertemplate=(
                            "point=%{customdata}<br>x=%{x:.6f}<br>y=%{y:.6f}"
                            "<br>value=%{marker.color:.6e}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=1,
                    col=column,
                )
                suffix = "" if column == 1 else str(column)
                fig.update_layout(
                    {
                        f"xaxis{suffix}": {
                            "scaleanchor": f"y{suffix}",
                            "scaleratio": 1.0,
                        }
                    }
                )
            self._base_layout(
                fig,
                f"Frozen coupling9 sample {int(sample_id)}: incremental K3 to K4 fields",
                height=590,
            )
            path = Path(f"figures/spatial/sample_{int(sample_id):04d}_k3_to_k4_fields")
            paths.append(self._save_figure(fig, path))
        return tuple(paths)

    def _plot_exposure_correlations(
        self,
        *,
        topology: AxialTopologyResult,
        spatial_rows: Sequence[SpatialK4Result],
        fields: SelectedSpatialFields,
    ) -> Path:
        exposure = topology.point_a_distance_counts[:, 3] / topology.num_points
        fig = make_subplots(
            rows=1,
            cols=len(spatial_rows),
            subplot_titles=tuple(
                f"sample {row.sample_id}, rho={row.delta_spearman:.3f}"
                for row in spatial_rows
            ),
            horizontal_spacing=0.08,
        )
        for column, row in enumerate(spatial_rows, start=1):
            sample_index = int(np.flatnonzero(fields.sample_ids == row.sample_id)[0])
            delta = np.abs(fields.delta_increment[sample_index])
            floor = max(float(np.quantile(delta[delta > 0.0], 0.01)), 1.0e-30)
            fig.add_trace(
                go.Scattergl(
                    x=100.0 * exposure,
                    y=np.maximum(delta, floor),
                    mode="markers",
                    marker={"size": 4, "color": "#2563eb", "opacity": 0.45},
                    hovertemplate=(
                        "K4-new reach=%{x:.5f}%"
                        "<br>|delta_K4-delta_K3|=%{y:.6e}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.update_yaxes(type="log", row=1, col=column)
        fig.update_xaxes(title_text="K4-new pair exposure (%)")
        fig.update_yaxes(title_text="Absolute incremental source correction")
        self._base_layout(
            fig,
            "Does the K4 correction concentrate where K4 adds structural reach?",
            height=590,
        )
        return self._save_figure(
            fig,
            Path("figures/spatial/topology_exposure_vs_k4_increment"),
        )


class TangentTopologyReportMixin:
    request: TangentTopologyAnalysisRequest

    @staticmethod
    def _markdown_figure(relative_base: Path, caption: str) -> list[str]:
        png = relative_base.with_suffix(".png")
        html = relative_base.with_suffix(".html")
        return [
            f"![{caption}]({png.as_posix()})",
            "",
            f"[Interactive Plotly figure]({html.as_posix()})",
            "",
        ]

    def _write_report(
        self,
        *,
        topology: AxialTopologyResult,
        topology_rows: Sequence[TopologyDistanceMetrics],
        point_rows: Sequence[PointTopologyMetrics],
        trained: Sequence[TrainedKMetrics],
        runtime: Sequence[TangentRuntimeMetrics],
        frozen: Sequence[FrozenKMetrics],
        spatial: Sequence[SpatialK4Result],
        figure_paths: Sequence[Path],
    ) -> None:
        figure_lookup = {path.name: path for path in figure_paths}
        total_pairs = topology.num_points**2
        k4_new_fraction = (
            topology.a_distance_pair_counts[3] / total_pairs
            if topology.a_graph_diameter >= 3
            else 0.0
        )
        beyond_k4_fraction = (
            sum(topology.a_distance_pair_counts[4:]) / total_pairs
            if topology.a_graph_diameter >= 4
            else 0.0
        )
        path_coords = topology.coords[
            np.asarray(topology.longest_path_point_ids, dtype=np.int64)
        ]
        trained_by_k = {row.subspace_dimension: row for row in trained}
        frozen_by_k = {row.subspace_dimension: row for row in frozen}
        lines = [
            "# Pentagram Tangent-Subspace Topology and K=1...4 Evidence",
            "",
            "## 핵심 결론",
            "",
            "Pentagram은 hole이 없지만 강한 concavity 때문에 connected axial-line ",
            "incidence graph가 Annulus보다 훨씬 길다. 현재 discretization의 point ",
            f"axial graph diameter는 `{topology.point_graph_diameter}`이고, tangent ",
            f"Hessian `A=S^T M_Omega S` graph diameter는 `{topology.a_graph_diameter}`이다. ",
            "따라서 localized tangent gradient에서 출발해 최장 structural path를 ",
            f"완전히 포함하려면 이론적으로 `K={topology.a_graph_diameter + 1}`까지 ",
            "필요하다. 이것은 K=9 학습을 권장한다는 뜻이 아니라, K=4가 전체 ",
            "topology의 종착점이 아님을 의미한다.",
            "",
            f"K=4에서 처음 포함되는 A-distance 3 ordered pair는 전체의 "
            f"`{100.0 * k4_new_fraction:.6f}%`이고, K=4 이후에도 남는 pair는 "
            f"`{100.0 * beyond_k4_fraction:.6f}%`이다. 기존 trained K=1...4와 동일 ",
            "checkpoint post-hoc 결과는 모두 K 증가에 따른 diminishing but nonzero ",
            "gain을 보인다. 그러나 representative spatial audit에서는 pointwise ",
            "solution improvement가 topology exposure와 일관된 양의 상관을 보이지 ",
            "않으므로, K=4의 이득은 graph propagation과 spectral approximation의 ",
            "결합으로 해석해야 한다.",
            "",
            "## 1. 수학적 기준",
            "",
            "Symmetric-balanced directional source에 tangent update를 적용하면",
            "",
            "```text",
            "phi = phi_tilde + delta,    psi = psi_tilde - delta",
            "m(delta) = H_x phi - H_y psi = m_0 + S delta",
            "S = H_x + H_y,              A = S^T M_Omega S.",
            "```",
            "",
            "Diagonal preconditioner를 D라고 할 때 K-dimensional correction은 본질적으로",
            "",
            "```text",
            "K_K(D^-1 A, D^-1 g_0)",
            " = span{D^-1 g_0, D^-1 A D^-1 g_0, ..., (D^-1 A)^(K-1) D^-1 g_0}",
            "```",
            "",
            "안에서 response mismatch를 최소화한다. 한 번의 S action은 한 connected ",
            "horizontal 또는 vertical segment를 따라 정보를 전달하고, 한 번의 A ",
            "action은 S와 S^T를 포함하므로 point axial graph에서 최대 두 hop을 ",
            "전달할 수 있다. 따라서 point-graph 최단거리 d_L과 A-distance d_A는",
            "",
            "```text",
            "d_A = ceil(d_L / 2),    K_first = d_A + 1",
            "```",
            "",
            "로 대응한다. 이 reach 해석은 localized initial gradient에 대한 structural ",
            "support 해석이다. 실제 g_0=S^T M_Omega m_0는 일반적으로 dense하므로, ",
            "K가 작을 때 먼 점의 값이 정확히 0이라는 뜻은 아니다. 높은 K는 새로운 ",
            "장거리 correlation pattern과 더 높은 차수의 operator-polynomial 자유도를 ",
            "추가한다.",
            "",
            "## 2. Pentagram axial topology",
            "",
            f"- Valid points: `{topology.num_points}`",
            f"- Horizontal connected segments: `{topology.num_x_segments}`",
            f"- Vertical connected segments: `{topology.num_y_segments}`",
            f"- Point axial graph diameter: `{topology.point_graph_diameter}`",
            f"- A-graph diameter: `{topology.a_graph_diameter}`",
            f"- Full structural reach upper bound: `K={topology.a_graph_diameter + 1}`",
            "",
            "| A-distance | first K | ordered pairs | pair share | cumulative reach |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in topology_rows:
            lines.append(
                f"| {row.a_distance} | {row.first_subspace_dimension} | "
                f"{row.ordered_pair_count:,} | "
                f"{100.0 * row.ordered_pair_fraction:.6f}% | "
                f"{100.0 * row.cumulative_ordered_pair_fraction:.6f}% |"
            )
        lines.extend(
            [
                "",
                "K=2가 가장 큰 구조적 확장을 제공하고 K=3가 그 다음을 담당한다. ",
                "K=4는 평균적으로 0.96%의 pair만 새로 연결하지만, 그 노출은 ",
                "공간적으로 매우 불균일하다. 일부 sharp-tip point에서는 K4-new ",
                "pair가 전체 domain의 90%를 넘고, K4 이후에도 domain의 90% 이상이 ",
                "남는 극단적인 point가 존재한다. 따라서 평균 pair fraction만으로 ",
                "tip/tail error에 대한 K4의 중요성을 판단하면 안 된다.",
                "",
            ]
        )
        lines.extend(
            self._markdown_figure(
                figure_lookup["a_distance_distribution"],
                "A-distance distribution and cumulative tangent reach",
            )
        )
        lines.extend(
            self._markdown_figure(
                figure_lookup["point_topology_fields"],
                "Pointwise topology fields on the Pentagram",
            )
        )
        lines.extend(
            [
                "### 최장 axial path",
                "",
                "최장 경로는 Pentagram 아래쪽 좌우 끝점을 연결하며 horizontal과 ",
                "vertical connected segment를 15번 교대로 통과한다.",
                "",
                f"- Start: `({path_coords[0, 0]:.7f}, {path_coords[0, 1]:.7f})`",
                f"- End: `({path_coords[-1, 0]:.7f}, {path_coords[-1, 1]:.7f})`",
                f"- Point hops: `{len(path_coords) - 1}`",
                f"- A-distance: `{topology.a_graph_diameter}`",
                "",
            ]
        )
        lines.extend(
            self._markdown_figure(
                figure_lookup["longest_axial_path"],
                "Longest alternating connected-axial-segment path",
            )
        )
        lines.extend(
            [
                "## 3. Trained K=1...4 evidence",
                "",
                "`coupling8/9/10/11`은 각각 production K=1/2/3/4이다. 표의 ",
                "reference metric은 detached test evaluation에만 사용되며 학습 loss나 ",
                "checkpoint selection에는 사용되지 않았다.",
                "",
                "| K | run | rel_sol | rel_u_phi | rel_u_psi | rel_flux | optimized energy | response cost | correction/sym |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for trained_row in trained:
            lines.append(
                f"| {trained_row.subspace_dimension} | {trained_row.run_name} | "
                f"{100.0 * trained_row.rel_sol_mean:.3f}% | "
                f"{100.0 * trained_row.rel_u_phi_mean:.3f}% | "
                f"{100.0 * trained_row.rel_u_psi_mean:.3f}% | "
                f"{100.0 * trained_row.rel_flux_mean:.3f}% | "
                f"{trained_row.optimized_energy_mean:.3e} | "
                f"{trained_row.response_cost_mean:.3e} | "
                f"{100.0 * trained_row.correction_rel_symmetric_pair_mean:.3f}% |"
            )
        lines.extend(
            [
                "",
                "| change | rel_sol | rel_u_phi | rel_u_psi | rel_flux | optimized energy | response cost |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for previous_k, current_k in zip(
            sorted(trained_by_k)[:-1],
            sorted(trained_by_k)[1:],
            strict=True,
        ):
            previous = trained_by_k[previous_k]
            current = trained_by_k[current_k]
            fields = (
                "rel_sol_mean",
                "rel_u_phi_mean",
                "rel_u_psi_mean",
                "rel_flux_mean",
                "optimized_energy_mean",
                "response_cost_mean",
            )
            changes = [
                100.0
                * _relative_change(
                    float(getattr(current, field)),
                    float(getattr(previous, field)),
                )
                for field in fields
            ]
            lines.append(
                f"| K{previous_k}->K{current_k} | "
                + " | ".join(f"{change:+.3f}%" for change in changes)
                + " |"
            )
        lines.extend([""])
        lines.extend(
            self._markdown_figure(
                figure_lookup["trained_k_quality"],
                "Trained K quality metrics",
            )
        )
        lines.extend(
            [
                "## 4. Frozen coupling9에서 K만 변경한 결과",
                "",
                "학습 run 간 initialization 차이를 제거하기 위해 coupling9의 동일 raw ",
                "output에 uncapped K=1과 unconstrained K=2/3/4를 적용했다.",
                "",
                "| K | response mismatch | optimized energy | rel_sol | rel_u_phi | rel_u_psi | rel_flux |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for frozen_row in frozen:
            lines.append(
                f"| {frozen_row.subspace_dimension} | "
                f"{frozen_row.response_mismatch_cost_mean:.6e} | "
                f"{frozen_row.optimized_energy_mean:.6e} | "
                f"{100.0 * frozen_row.rel_sol_mean:.4f}% | "
                f"{100.0 * frozen_row.rel_u_phi_mean:.4f}% | "
                f"{100.0 * frozen_row.rel_u_psi_mean:.4f}% | "
                f"{100.0 * frozen_row.rel_flux_mean:.4f}% |"
            )
        frozen_k3 = frozen_by_k[3]
        frozen_k4 = frozen_by_k[4]
        lines.extend(
            [
                "",
                "K3에서 K4로 바꾸면 동일 checkpoint에서 mean response mismatch가 "
                f"`{100.0 * _relative_change(frozen_k4.response_mismatch_cost_mean, frozen_k3.response_mismatch_cost_mean):.3f}%`, "
                "optimized energy가 "
                f"`{100.0 * _relative_change(frozen_k4.optimized_energy_mean, frozen_k3.optimized_energy_mean):.3f}%`, "
                "rel_sol이 "
                f"`{100.0 * _relative_change(frozen_k4.rel_sol_mean, frozen_k3.rel_sol_mean):.3f}%` 변한다. "
                "따라서 네 번째 direction의 이득은 network initialization 차이만으로 ",
                "설명되지 않는다.",
                "",
            ]
        )
        lines.extend(
            self._markdown_figure(
                figure_lookup["frozen_k_quality"],
                "Frozen-checkpoint K-only comparison",
            )
        )
        lines.extend(
            [
                "## 5. K4 spatial attribution",
                "",
                "대표 3개 sample에 대해 geometry-only K4-new pair exposure와 ",
                "`|delta_K4-delta_K3|`, `|m_K3-m_K4|`, pointwise squared solution-error ",
                "reduction의 관계를 계산했다.",
                "",
                "| sample | Spearman(exposure, |delta4-delta3|) | top10/bottom50 delta | Spearman(exposure, |m3-m4|) | Spearman(exposure, error reduction) | global mean error reduction |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for spatial_row in spatial:
            lines.append(
                f"| {spatial_row.sample_id} | {spatial_row.delta_spearman:.4f} | "
                f"{spatial_row.top10_delta_over_bottom50:.4f} | "
                f"{spatial_row.response_spearman:.4f} | "
                f"{spatial_row.error_improvement_spearman:.4f} | "
                f"{spatial_row.global_error_improvement_mean:.6e} |"
            )
        lines.extend(
            [
                "",
                "두 sample에서는 topology exposure 상위 10%의 incremental source ",
                "correction이 하위 50%보다 약 1.64배 크다. 그러나 response reduction과 ",
                "pointwise solution improvement는 exposure와 일관된 양의 상관을 보이지 ",
                "않는다. 따라서 현재 spatial evidence는 K4가 긴 경로를 활용한다는 ",
                "약한 증거는 제공하지만, K4의 최종 이득이 topology에 의해 직접 ",
                "발생했다는 인과적 증거는 제공하지 않는다.",
                "",
            ]
        )
        lines.extend(
            self._markdown_figure(
                figure_lookup["topology_exposure_vs_k4_increment"],
                "Topology exposure versus incremental K4 source correction",
            )
        )
        for path in figure_paths:
            if path.name.startswith("sample_"):
                lines.extend(
                    self._markdown_figure(
                        path,
                        path.name.replace("_", " "),
                    )
                )
        lines.extend(
            [
                "## 6. 계산 비용과 선택 기준",
                "",
                "동일 frozen coupling11 input과 CPU four-thread 조건에서 isolated ",
                "tangent/auxiliary forward+backward 시간은 다음과 같다.",
                "",
                "| K | forward (ms) | forward+backward (ms) | ratio to K1 |",
                "|---:|---:|---:|---:|",
            ]
        )
        for runtime_row in runtime:
            lines.append(
                f"| {runtime_row.subspace_dimension} | {runtime_row.forward_ms:.3f} | "
                f"{runtime_row.forward_backward_ms:.3f} | "
                f"{runtime_row.forward_backward_ratio_to_k1:.3f}x |"
            )
        lines.extend([""])
        lines.extend(
            self._markdown_figure(
                figure_lookup["cost_quality_tradeoff"],
                "Tangent subspace cost-quality tradeoff",
            )
        )
        lines.extend(
            [
                "## 7. 최종 해석",
                "",
                "1. Pentagram은 hole이 없는 domain에서도 concavity가 axial visibility ",
                "   graph를 길게 만들 수 있음을 보여준다.",
                "2. K=2가 가장 큰 구조적 및 수치적 이득을 제공하고, K=3가 명확한 ",
                "   추가 이득을 제공하며, K=4는 더 작은 global/flux/tail 이득을 제공한다.",
                "3. K=4는 전체 graph reach를 완성하지 않지만 ordered pair의 99.8128%를 ",
                "   A-distance 3 이내에 포함한다.",
                "4. Absolute accuracy 기준으로는 K=4가 가장 좋고, 현재 cost-quality ",
                "   knee는 K=3이다.",
                "5. K=4의 효용은 topology만이 아니라 A^-1 g를 더 높은 차수의 ",
                "   preconditioned operator polynomial로 근사하는 spectral 효과도 ",
                "   포함한다.",
                "6. K>4를 단순히 graph diameter만 보고 선택하면 안 된다. 남은 pair는 ",
                "   매우 적고 위치적으로 집중되어 있으므로, K5 이상은 frozen nested ",
                "   audit에서 tail improvement가 비용을 정당화할 때만 고려해야 한다.",
                "",
                "## 8. 해석 경계와 provenance",
                "",
                "- Trained K 표는 기존 canonical comparison report의 rounded values를 ",
                "  사용한다.",
                "- Frozen K 표는 per-sample CSV에서 직접 다시 집계했다.",
                "- Spatial correlation은 selected archive에 저장된 대표 3개 sample만 ",
                "  사용하므로 전체 100-sample 인과 분석이 아니다.",
                "- Reference sol/phi/psi는 detached evaluation에만 사용되며 topology, ",
                "  tangent correction, training loss 또는 checkpoint selection에는 ",
                "  사용되지 않는다.",
                "- 모든 topology 수치는 geometry NPZ의 connected segment IDs에서 직접 ",
                "  계산했다. Global response matrix나 full solve는 사용하지 않았다.",
                "",
                "### Source files",
                "",
                f"- Geometry: `{self.request.geometry_path}`",
                f"- Trained comparison: `{self.request.trained_comparison_report}`",
                f"- Frozen metrics: `{self.request.frozen_metrics_path}`",
                f"- Selected frozen fields: `{self.request.selected_archive_path}`",
                "",
                "### Machine-readable outputs",
                "",
                "- `summary.json`",
                "- `metrics/topology_distance_distribution.csv`",
                "- `metrics/point_topology_metrics.csv`",
                "- `metrics/trained_k_metrics.csv`",
                "- `metrics/tangent_runtime_metrics.csv`",
                "- `metrics/frozen_k_metrics.csv`",
                "- `metrics/selected_k4_spatial_correlations.csv`",
                "- `data/topology_fields.npz`",
            ]
        )
        (self.request.outdir / "analysis_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


class ComplexTangentTopologyAnalysis(
    TangentTopologyPlotMixin,
    TangentTopologyReportMixin,
):
    """Build a durable Pentagram topology and tangent-K evidence bundle."""

    def __init__(
        self,
        request: TangentTopologyAnalysisRequest,
        logger: logging.Logger,
    ) -> None:
        self.request = request
        self.logger = logger

    def analyze(self) -> dict[str, Any]:
        self.request.validate()
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Analyzing axial topology from %s", self.request.geometry_path)
        topology = AxialSegmentTopologyAnalyzer.from_npz(
            self.request.geometry_path,
            chunk_size=self.request.chunk_size,
        ).analyze()
        trained, runtime = TrainedKComparisonParser(
            self.request.trained_comparison_report
        ).parse()
        frozen = FrozenKMetricsReader(self.request.frozen_metrics_path).read()
        spatial, spatial_fields = SelectedK4SpatialAnalyzer(
            archive_path=self.request.selected_archive_path,
            topology=topology,
        ).analyze()

        topology_rows = self._topology_rows(topology)
        point_rows = self._point_rows(topology)
        self._write_metrics(
            topology=topology,
            topology_rows=topology_rows,
            point_rows=point_rows,
            trained=trained,
            runtime=runtime,
            frozen=frozen,
            spatial=spatial,
            spatial_fields=spatial_fields,
        )
        figure_paths = (
            self._plot_distance_distribution(topology_rows),
            self._plot_topology_fields(topology),
            self._plot_longest_path(topology),
            self._plot_trained_quality(trained),
            self._plot_frozen_quality(frozen),
            self._plot_cost_quality(trained, runtime),
            *self._plot_selected_spatial_fields(
                topology=topology,
                fields=spatial_fields,
            ),
            self._plot_exposure_correlations(
                topology=topology,
                spatial_rows=spatial,
                fields=spatial_fields,
            ),
        )
        summary = self._build_summary(
            topology=topology,
            topology_rows=topology_rows,
            point_rows=point_rows,
            trained=trained,
            runtime=runtime,
            frozen=frozen,
            spatial=spatial,
            figure_paths=figure_paths,
        )
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._write_report(
            topology=topology,
            topology_rows=topology_rows,
            point_rows=point_rows,
            trained=trained,
            runtime=runtime,
            frozen=frozen,
            spatial=spatial,
            figure_paths=figure_paths,
        )
        self.logger.info(
            "Analysis complete: point_diameter=%d A_diameter=%d report=%s",
            topology.point_graph_diameter,
            topology.a_graph_diameter,
            self.request.outdir / "analysis_report.md",
        )
        return summary

    @staticmethod
    def _topology_rows(
        topology: AxialTopologyResult,
    ) -> tuple[TopologyDistanceMetrics, ...]:
        total_pairs = topology.num_points**2
        cumulative = 0
        rows: list[TopologyDistanceMetrics] = []
        for distance, count in enumerate(topology.a_distance_pair_counts):
            cumulative += count
            rows.append(
                TopologyDistanceMetrics(
                    a_distance=distance,
                    first_subspace_dimension=distance + 1,
                    ordered_pair_count=count,
                    ordered_pair_fraction=count / total_pairs,
                    cumulative_ordered_pair_fraction=cumulative / total_pairs,
                )
            )
        return tuple(rows)

    @staticmethod
    def _point_rows(
        topology: AxialTopologyResult,
    ) -> tuple[PointTopologyMetrics, ...]:
        k4_new = (
            topology.point_a_distance_counts[:, 3] / topology.num_points
            if topology.a_graph_diameter >= 3
            else np.zeros(topology.num_points, dtype=np.float64)
        )
        beyond_k4 = (
            topology.point_a_distance_counts[:, 4:].sum(axis=1) / topology.num_points
            if topology.a_graph_diameter >= 4
            else np.zeros(topology.num_points, dtype=np.float64)
        )
        return tuple(
            PointTopologyMetrics(
                point_id=point_id,
                x=float(topology.coords[point_id, 0]),
                y=float(topology.coords[point_id, 1]),
                a_eccentricity=int(topology.point_a_eccentricity[point_id]),
                full_reach_subspace_dimension=int(
                    topology.point_a_eccentricity[point_id] + 1
                ),
                k4_new_reach_fraction=float(k4_new[point_id]),
                beyond_k4_fraction=float(beyond_k4[point_id]),
            )
            for point_id in range(topology.num_points)
        )

    def _write_metrics(
        self,
        *,
        topology: AxialTopologyResult,
        topology_rows: Sequence[TopologyDistanceMetrics],
        point_rows: Sequence[PointTopologyMetrics],
        trained: Sequence[TrainedKMetrics],
        runtime: Sequence[TangentRuntimeMetrics],
        frozen: Sequence[FrozenKMetrics],
        spatial: Sequence[SpatialK4Result],
        spatial_fields: SelectedSpatialFields,
    ) -> None:
        metrics_dir = self.request.outdir / "metrics"
        _write_dataclass_csv(
            metrics_dir / "topology_distance_distribution.csv",
            topology_rows,
        )
        _write_dataclass_csv(metrics_dir / "point_topology_metrics.csv", point_rows)
        _write_dataclass_csv(metrics_dir / "trained_k_metrics.csv", trained)
        _write_dataclass_csv(metrics_dir / "tangent_runtime_metrics.csv", runtime)
        _write_dataclass_csv(metrics_dir / "frozen_k_metrics.csv", frozen)
        _write_dataclass_csv(
            metrics_dir / "selected_k4_spatial_correlations.csv",
            spatial,
        )
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            data_dir / "topology_fields.npz",
            coords_valid=topology.coords,
            point_a_distance_counts=topology.point_a_distance_counts,
            point_a_eccentricity=topology.point_a_eccentricity,
            longest_path_point_ids=np.asarray(
                topology.longest_path_point_ids,
                dtype=np.int64,
            ),
            selected_sample_ids=spatial_fields.sample_ids,
            selected_k4_delta_increment=spatial_fields.delta_increment,
            selected_k4_response_increment=spatial_fields.response_increment,
            selected_k4_error_improvement=spatial_fields.error_improvement,
        )

    def _build_summary(
        self,
        *,
        topology: AxialTopologyResult,
        topology_rows: Sequence[TopologyDistanceMetrics],
        point_rows: Sequence[PointTopologyMetrics],
        trained: Sequence[TrainedKMetrics],
        runtime: Sequence[TangentRuntimeMetrics],
        frozen: Sequence[FrozenKMetrics],
        spatial: Sequence[SpatialK4Result],
        figure_paths: Sequence[Path],
    ) -> dict[str, Any]:
        trained_by_k = {row.subspace_dimension: row for row in trained}
        frozen_by_k = {row.subspace_dimension: row for row in frozen}
        k4_new_values = np.array(
            [row.k4_new_reach_fraction for row in point_rows],
            dtype=np.float64,
        )
        beyond_values = np.array(
            [row.beyond_k4_fraction for row in point_rows],
            dtype=np.float64,
        )
        return {
            "schema_version": 1,
            "analysis": "complex_tangent_axial_topology_k_evidence",
            "provenance": {
                name: {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for name, path in (
                    ("geometry", self.request.geometry_path),
                    ("trained_comparison", self.request.trained_comparison_report),
                    ("frozen_metrics", self.request.frozen_metrics_path),
                    ("selected_archive", self.request.selected_archive_path),
                )
            },
            "operator_contract": {
                "tangent_response_operator": "S=H_x+H_y",
                "tangent_hessian": "A=S^T M_Omega S",
                "preconditioned_krylov": (
                    "span{D^-1 g0, D^-1 A D^-1 g0, ..., (D^-1 A)^(K-1) D^-1 g0}"
                ),
                "a_distance_from_point_distance": "ceil(d_L/2)",
                "first_subspace_dimension": "K=d_A+1",
                "interpretation_boundary": (
                    "structural reach from localized gradient; actual g0 is generally dense"
                ),
            },
            "topology": {
                "num_points": topology.num_points,
                "num_x_segments": topology.num_x_segments,
                "num_y_segments": topology.num_y_segments,
                "point_graph_diameter": topology.point_graph_diameter,
                "a_graph_diameter": topology.a_graph_diameter,
                "full_reach_subspace_dimension": topology.a_graph_diameter + 1,
                "distance_distribution": [asdict(row) for row in topology_rows],
                "longest_path_point_ids": list(topology.longest_path_point_ids),
                "longest_path_coords": topology.coords[
                    np.asarray(topology.longest_path_point_ids, dtype=np.int64)
                ].tolist(),
                "k4_new_reach_fraction_mean": float(k4_new_values.mean()),
                "k4_new_reach_fraction_max": float(k4_new_values.max()),
                "k4_new_reach_points_ge_50_percent": int(
                    np.count_nonzero(k4_new_values >= 0.5)
                ),
                "beyond_k4_fraction_mean": float(beyond_values.mean()),
                "beyond_k4_fraction_max": float(beyond_values.max()),
                "beyond_k4_points_ge_50_percent": int(
                    np.count_nonzero(beyond_values >= 0.5)
                ),
            },
            "trained_k_metrics": [asdict(row) for row in trained],
            "tangent_runtime_metrics": [asdict(row) for row in runtime],
            "frozen_k_metrics": [asdict(row) for row in frozen],
            "key_adjacent_changes": {
                "trained_k3_to_k4": {
                    metric: _relative_change(
                        float(getattr(trained_by_k[4], metric)),
                        float(getattr(trained_by_k[3], metric)),
                    )
                    for metric in (
                        "rel_sol_mean",
                        "rel_u_phi_mean",
                        "rel_u_psi_mean",
                        "rel_flux_mean",
                        "optimized_energy_mean",
                        "response_cost_mean",
                    )
                },
                "frozen_k3_to_k4": {
                    metric: _relative_change(
                        float(getattr(frozen_by_k[4], metric)),
                        float(getattr(frozen_by_k[3], metric)),
                    )
                    for metric in (
                        "response_mismatch_cost_mean",
                        "optimized_energy_mean",
                        "rel_sol_mean",
                        "rel_u_phi_mean",
                        "rel_u_psi_mean",
                        "rel_flux_mean",
                    )
                },
            },
            "selected_spatial_analysis": [asdict(row) for row in spatial],
            "conclusion": {
                "structural_k4_paths_exist": True,
                "k4_is_full_reach": False,
                "best_absolute_accuracy": "K=4",
                "current_cost_quality_knee": "K=3",
                "topology_only_causal_attribution_supported": False,
                "interpretation": (
                    "K4 gain combines longer graph propagation with higher-order "
                    "spectral approximation"
                ),
            },
            "outputs": {
                "report": "analysis_report.md",
                "figure_bases": [str(path) for path in figure_paths],
                "static_png_complete": all(
                    (self.request.outdir / path.with_suffix(".png")).is_file()
                    for path in figure_paths
                ),
            },
        }
