from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from greenonet.complex_tangent_geometry_selection import (
    AxialSegmentTopologyAnalyzer,
    AxialTopologyResult,
)
from greenonet.plotly_io import save_plotly_figure


@dataclass(frozen=True)
class GeometryReachSpec:
    """One fixed active-point geometry included in the K-reach comparison."""

    slug: str
    label: str
    path: Path

    def validate(self) -> None:
        if not self.slug or not self.slug.replace("_", "").isalnum():
            raise ValueError("Geometry slug must be non-empty and alphanumeric.")
        if not self.label.strip():
            raise ValueError("Geometry label must be non-empty.")
        if not self.path.is_file():
            raise FileNotFoundError(f"Geometry NPZ does not exist: {self.path}")


@dataclass(frozen=True)
class GeometryKConnectivityRequest:
    """Configuration for structural K=1 through K=max_k visualization."""

    geometries: tuple[GeometryReachSpec, ...]
    outdir: Path
    max_k: int = 4
    global_threshold: float = 0.99
    tail_quantile: float = 0.05
    tail_threshold: float = 0.99
    chunk_size: int = 256
    theme: str = "plotly_white"

    def validate(self) -> None:
        if not self.geometries:
            raise ValueError("At least one geometry is required.")
        slugs = [spec.slug for spec in self.geometries]
        if len(slugs) != len(set(slugs)):
            raise ValueError("Geometry slugs must be unique.")
        for spec in self.geometries:
            spec.validate()
        if isinstance(self.max_k, bool) or not isinstance(self.max_k, int):
            raise TypeError("max_k must be an integer.")
        if self.max_k < 1 or self.max_k > 8:
            raise ValueError("max_k must be in [1, 8].")
        if isinstance(self.chunk_size, bool) or not isinstance(self.chunk_size, int):
            raise TypeError("chunk_size must be an integer.")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        for name, value in (
            ("global_threshold", self.global_threshold),
            ("tail_quantile", self.tail_quantile),
            ("tail_threshold", self.tail_threshold),
        ):
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")


@dataclass(frozen=True)
class GeometryKReachMetric:
    """Reach statistics for one geometry and one tangent dimension."""

    domain: str
    subspace_dimension: int
    global_reach_fraction: float
    point_reach_min: float
    point_reach_q01: float
    point_reach_q05: float
    point_reach_median: float
    representative_seed_reach_fraction: float
    representative_seed_new_shell_fraction: float


@dataclass(frozen=True)
class GeometryKReachResult:
    """Topology, selected seed, and K shells for one geometry."""

    spec: GeometryReachSpec
    topology: AxialTopologyResult
    selected_geometry_k: int
    seed_selection_k: int
    seed_point_id: int
    point_distance_from_seed: np.ndarray
    a_distance_from_seed: np.ndarray
    first_reach_k: np.ndarray
    metrics: tuple[GeometryKReachMetric, ...]

    @property
    def seed_coordinate(self) -> tuple[float, float]:
        coordinate = self.topology.coords[self.seed_point_id]
        return float(coordinate[0]), float(coordinate[1])


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


def select_geometry_k(
    topology: AxialTopologyResult,
    *,
    global_threshold: float,
    tail_quantile: float,
    tail_threshold: float,
) -> int:
    """Select the first K satisfying the fixed geometry-only reach rule."""
    for k in range(1, topology.a_graph_diameter + 2):
        point_reach = pointwise_reach_fraction(topology, k)
        if (
            global_reach_fraction(topology, k) >= global_threshold
            and float(np.quantile(point_reach, tail_quantile)) >= tail_threshold
        ):
            return k
    raise RuntimeError("No K satisfies the reach rule despite full graph reach.")


def select_representative_tail_seed(
    topology: AxialTopologyResult,
    *,
    selection_k: int,
    tail_quantile: float,
) -> int:
    """Choose a deterministic difficult point at one fixed structural K."""
    reach = pointwise_reach_fraction(topology, selection_k)
    target = float(np.quantile(reach, tail_quantile))
    difference = np.abs(reach - target)
    candidates = np.flatnonzero(
        np.isclose(difference, difference.min(), rtol=0.0, atol=1e-15)
    )
    center = topology.coords.mean(axis=0)
    radius_squared = np.sum((topology.coords[candidates] - center) ** 2, axis=1)
    maximum_radius = float(radius_squared.max())
    radial_candidates = candidates[
        np.isclose(radius_squared, maximum_radius, rtol=0.0, atol=1e-15)
    ]
    return int(radial_candidates.min())


class GeometryKConnectivityPlotMixin:
    """Plotly output for cumulative K reach and first-reach shells."""

    request: GeometryKConnectivityRequest
    logger: logging.Logger

    SHELL_COLORS = (
        "#b91c1c",
        "#f59e0b",
        "#0f766e",
        "#2563eb",
        "#7c3aed",
        "#db2777",
        "#65a30d",
        "#0891b2",
    )

    def _save_figure(self, figure: go.Figure, relative_base: Path) -> Path:
        save_plotly_figure(
            figure,
            self.request.outdir / relative_base,
            logger=self.logger,
        )
        return relative_base

    def _plot_seed_reach(self, result: GeometryKReachResult) -> Path:
        subplot_titles = [
            (
                f"K={metric.subspace_dimension}: "
                f"{100.0 * metric.representative_seed_reach_fraction:.3f}% reached"
            )
            for metric in result.metrics
        ]
        figure = make_subplots(
            rows=1,
            cols=self.request.max_k,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.035,
        )
        coords = result.topology.coords
        point_ids = np.arange(result.topology.num_points, dtype=np.int64)
        marker_size = 3.0 if result.topology.num_points > 10_000 else 4.0

        for column, k in enumerate(range(1, self.request.max_k + 1), start=1):
            unreachable = result.first_reach_k > k
            figure.add_trace(
                go.Scattergl(
                    x=coords[unreachable, 0],
                    y=coords[unreachable, 1],
                    mode="markers",
                    name="Not reached yet",
                    legendgroup="unreached",
                    showlegend=column == 1,
                    marker={"size": marker_size, "color": "#d7dde5", "opacity": 0.7},
                    customdata=np.column_stack(
                        (
                            point_ids[unreachable],
                            result.first_reach_k[unreachable],
                        )
                    ),
                    hovertemplate=(
                        "point=%{customdata[0]}<br>x=%{x:.6f}<br>y=%{y:.6f}"
                        "<br>first K=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            for shell in range(1, k + 1):
                selected = result.first_reach_k == shell
                figure.add_trace(
                    go.Scattergl(
                        x=coords[selected, 0],
                        y=coords[selected, 1],
                        mode="markers",
                        name=f"First reached at K={shell}",
                        legendgroup=f"shell-{shell}",
                        showlegend=column == shell,
                        marker={
                            "size": marker_size + 0.8,
                            "color": self.SHELL_COLORS[shell - 1],
                            "opacity": 0.9,
                        },
                        customdata=np.column_stack(
                            (
                                point_ids[selected],
                                result.point_distance_from_seed[selected],
                                result.a_distance_from_seed[selected],
                            )
                        ),
                        hovertemplate=(
                            "point=%{customdata[0]}<br>x=%{x:.6f}<br>y=%{y:.6f}"
                            "<br>point hops=%{customdata[1]}"
                            "<br>A-distance=%{customdata[2]}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=column,
                )
            seed_x, seed_y = result.seed_coordinate
            figure.add_trace(
                go.Scatter(
                    x=[seed_x],
                    y=[seed_y],
                    mode="markers",
                    name="Representative seed",
                    legendgroup="seed",
                    showlegend=column == 1,
                    marker={
                        "symbol": "star",
                        "size": 14,
                        "color": "#111827",
                        "line": {"color": "white", "width": 1.5},
                    },
                    hovertemplate=(
                        f"seed point={result.seed_point_id}<br>"
                        "x=%{x:.6f}<br>y=%{y:.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            suffix = "" if column == 1 else str(column)
            figure.update_xaxes(
                title_text="x",
                constrain="domain",
                row=1,
                col=column,
            )
            figure.update_yaxes(
                title_text="y" if column == 1 else None,
                scaleanchor=f"x{suffix}",
                scaleratio=1.0,
                constrain="domain",
                row=1,
                col=column,
            )

        seed_x, seed_y = result.seed_coordinate
        figure.update_layout(
            title=(
                f"{result.spec.label}: structural first-reach shells from a "
                f"K={result.seed_selection_k} lower-5% tail seed "
                f"({seed_x:.5f}, {seed_y:.5f})"
            ),
            template=self.request.theme,
            width=1580,
            height=560,
            font={"family": "Noto Sans CJK KR, DejaVu Sans", "size": 12},
            margin={"l": 55, "r": 35, "t": 110, "b": 65},
            legend={"orientation": "h", "x": 0.0, "y": 1.13},
        )
        return self._save_figure(
            figure,
            Path(f"figures/{result.spec.slug}_representative_seed_k1_k4"),
        )

    def _plot_all_domain_reach(
        self,
        results: Sequence[GeometryKReachResult],
    ) -> Path:
        titles = [
            f"{result.spec.label}, K={k}"
            for result in results
            for k in range(1, self.request.max_k + 1)
        ]
        figure = make_subplots(
            rows=len(results),
            cols=self.request.max_k,
            subplot_titles=titles,
            horizontal_spacing=0.025,
            vertical_spacing=0.06,
        )
        for row, result in enumerate(results, start=1):
            coords = result.topology.coords
            point_ids = np.arange(result.topology.num_points, dtype=np.int64)
            marker_size = 2.3 if result.topology.num_points > 10_000 else 3.5
            for column, k in enumerate(range(1, self.request.max_k + 1), start=1):
                reach = pointwise_reach_fraction(result.topology, k)
                show_scale = row == 1 and column == self.request.max_k
                figure.add_trace(
                    go.Scattergl(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        mode="markers",
                        marker={
                            "size": marker_size,
                            "color": 100.0 * reach,
                            "colorscale": "Cividis",
                            "cmin": 0.0,
                            "cmax": 100.0,
                            "showscale": show_scale,
                            "colorbar": (
                                {
                                    "title": {"text": "Reach (%)", "side": "right"},
                                    "x": 1.01,
                                    "len": 0.88,
                                }
                                if show_scale
                                else None
                            ),
                        },
                        customdata=np.column_stack((point_ids, reach)),
                        hovertemplate=(
                            "point=%{customdata[0]}<br>x=%{x:.6f}<br>y=%{y:.6f}"
                            "<br>reachable=%{customdata[1]:.6%}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=row,
                    col=column,
                )
                axis_number = (row - 1) * self.request.max_k + column
                suffix = "" if axis_number == 1 else str(axis_number)
                figure.update_xaxes(
                    constrain="domain",
                    row=row,
                    col=column,
                )
                figure.update_yaxes(
                    scaleanchor=f"x{suffix}",
                    scaleratio=1.0,
                    constrain="domain",
                    row=row,
                    col=column,
                )
        figure.update_layout(
            title=(
                "Pointwise cumulative structural reach: fraction of the domain "
                "reachable by K"
            ),
            template=self.request.theme,
            width=1600,
            height=330 * len(results) + 130,
            font={"family": "Noto Sans CJK KR, DejaVu Sans", "size": 11},
            margin={"l": 45, "r": 90, "t": 105, "b": 45},
        )
        return self._save_figure(
            figure,
            Path("figures/all_domains_pointwise_reach_k1_k4"),
        )


class GeometryKConnectivityVisualization(GeometryKConnectivityPlotMixin):
    """Analyze and visualize geometry-only K connectivity without PDE data."""

    def __init__(
        self,
        request: GeometryKConnectivityRequest,
        logger: logging.Logger,
    ) -> None:
        request.validate()
        self.request = request
        self.logger = logger

    def analyze(self) -> dict[str, object]:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        results = tuple(
            self._analyze_geometry(spec) for spec in self.request.geometries
        )
        figure_paths = [self._plot_seed_reach(result) for result in results]
        figure_paths.append(self._plot_all_domain_reach(results))
        self._write_metrics(results)
        self._write_arrays(results)
        summary = self._build_summary(results, figure_paths)
        self._write_report(results, figure_paths)
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.logger.info(
            "Saved K connectivity visualization for %d geometries to %s",
            len(results),
            self.request.outdir,
        )
        return summary

    def _analyze_geometry(self, spec: GeometryReachSpec) -> GeometryKReachResult:
        self.logger.info("Analyzing structural K reach for %s", spec.path)
        analyzer = AxialSegmentTopologyAnalyzer.from_npz(
            spec.path,
            chunk_size=self.request.chunk_size,
        )
        topology = analyzer.analyze()
        selected_k = select_geometry_k(
            topology,
            global_threshold=self.request.global_threshold,
            tail_quantile=self.request.tail_quantile,
            tail_threshold=self.request.tail_threshold,
        )
        seed_selection_k = max(1, selected_k - 1)
        seed_point_id = select_representative_tail_seed(
            topology,
            selection_k=seed_selection_k,
            tail_quantile=self.request.tail_quantile,
        )
        point_distance, a_distance = analyzer.distances_from_point(seed_point_id)
        first_reach_k = a_distance.astype(np.int64) + 1
        metrics = tuple(
            self._build_metric(
                spec=spec,
                topology=topology,
                seed_point_id=seed_point_id,
                first_reach_k=first_reach_k,
                k=k,
            )
            for k in range(1, self.request.max_k + 1)
        )
        for metric in metrics:
            expected = pointwise_reach_fraction(
                topology,
                metric.subspace_dimension,
            )[seed_point_id]
            if not math.isclose(
                metric.representative_seed_reach_fraction,
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise RuntimeError("Seed reach does not match all-point count audit.")
        return GeometryKReachResult(
            spec=spec,
            topology=topology,
            selected_geometry_k=selected_k,
            seed_selection_k=seed_selection_k,
            seed_point_id=seed_point_id,
            point_distance_from_seed=point_distance,
            a_distance_from_seed=a_distance,
            first_reach_k=first_reach_k,
            metrics=metrics,
        )

    def _build_metric(
        self,
        *,
        spec: GeometryReachSpec,
        topology: AxialTopologyResult,
        seed_point_id: int,
        first_reach_k: np.ndarray,
        k: int,
    ) -> GeometryKReachMetric:
        reach = pointwise_reach_fraction(topology, k)
        return GeometryKReachMetric(
            domain=spec.slug,
            subspace_dimension=k,
            global_reach_fraction=global_reach_fraction(topology, k),
            point_reach_min=float(reach.min()),
            point_reach_q01=float(np.quantile(reach, 0.01)),
            point_reach_q05=float(np.quantile(reach, 0.05)),
            point_reach_median=float(np.quantile(reach, 0.5)),
            representative_seed_reach_fraction=float(reach[seed_point_id]),
            representative_seed_new_shell_fraction=float(np.mean(first_reach_k == k)),
        )

    def _write_metrics(self, results: Sequence[GeometryKReachResult]) -> None:
        path = self.request.outdir / "metrics" / "per_domain_k_reach.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(metric) for result in results for metric in result.metrics]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_arrays(self, results: Sequence[GeometryKReachResult]) -> None:
        data_dir = self.request.outdir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for result in results:
            np.savez_compressed(
                data_dir / f"{result.spec.slug}_k_reach.npz",
                coords_valid=result.topology.coords,
                point_a_distance_counts=result.topology.point_a_distance_counts,
                point_a_eccentricity=result.topology.point_a_eccentricity,
                seed_point_id=np.array(result.seed_point_id, dtype=np.int64),
                point_distance_from_seed=result.point_distance_from_seed,
                a_distance_from_seed=result.a_distance_from_seed,
                first_reach_k=result.first_reach_k,
            )

    def _build_summary(
        self,
        results: Sequence[GeometryKReachResult],
        figure_paths: Sequence[Path],
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "semantics": {
                "point_graph": (
                    "active points share an edge when they lie on the same connected "
                    "horizontal or vertical axial segment"
                ),
                "a_distance": "ceil(point_graph_distance / 2)",
                "first_reach_k": "a_distance + 1 for a localized source-coordinate seed",
                "actual_gradient_caveat": (
                    "production g0 is generally dense; the figure is structural reach, "
                    "not a literal zero/nonzero gradient map"
                ),
            },
            "selection_rule": {
                "global_threshold": self.request.global_threshold,
                "tail_quantile": self.request.tail_quantile,
                "tail_threshold": self.request.tail_threshold,
            },
            "max_visualized_k": self.request.max_k,
            "geometries": [
                {
                    "slug": result.spec.slug,
                    "label": result.spec.label,
                    "path": str(result.spec.path),
                    "sha256": self._sha256(result.spec.path),
                    "num_points": result.topology.num_points,
                    "num_x_segments": result.topology.num_x_segments,
                    "num_y_segments": result.topology.num_y_segments,
                    "point_graph_diameter": result.topology.point_graph_diameter,
                    "a_graph_diameter": result.topology.a_graph_diameter,
                    "selected_geometry_k": result.selected_geometry_k,
                    "seed_selection_k": result.seed_selection_k,
                    "seed_point_id": result.seed_point_id,
                    "seed_coordinate": result.seed_coordinate,
                    "metrics": [asdict(metric) for metric in result.metrics],
                }
                for result in results
            ],
            "figures": [str(path) for path in figure_paths],
        }

    def _write_report(
        self,
        results: Sequence[GeometryKReachResult],
        figure_paths: Sequence[Path],
    ) -> None:
        del figure_paths
        lines = [
            "# Geometry-Only K=1...4 Structural Connectivity",
            "",
            "## 해석 대상",
            "",
            "이 자료는 한 active point에 localized source-coordinate gradient가 있다고 ",
            "가정했을 때, tangent Krylov subspace의 K가 커지며 어떤 point가 구조적으로 ",
            "처음 연결될 수 있는지를 보여준다.",
            "",
            "```text",
            "d_A(i,j) = ceil(d_L(i,j) / 2)",
            "K_first(i,j) = d_A(i,j) + 1",
            "```",
            "",
            "실제 production gradient `g0=S^T M_Omega m0`는 일반적으로 dense하다. ",
            "따라서 회색 point가 실제 계산에서 정확히 영향이 0이라는 뜻은 아니다. ",
            "그림은 K가 추가하는 correlation pattern의 structural reach를 설명한다.",
            "",
            "## Geometry 요약",
            "",
            (
                "| domain | points | point diameter | A diameter | selected K | "
                "seed selection K | seed |"
            ),
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for result in results:
            seed_x, seed_y = result.seed_coordinate
            lines.append(
                f"| {result.spec.label} | {result.topology.num_points:,} | "
                f"{result.topology.point_graph_diameter} | "
                f"{result.topology.a_graph_diameter} | "
                f"{result.selected_geometry_k} | "
                f"{result.seed_selection_k} | "
                f"{result.seed_point_id} ({seed_x:.6f}, {seed_y:.6f}) |"
            )
        lines.extend(
            [
                "",
                "## 전체 domain의 pointwise cumulative reach",
                "",
                "각 point의 색은 그 point를 seed로 보았을 때 K-step 안에서 도달 가능한 ",
                "domain 비율 `C_i(K)`이다.",
                "",
                "![All-domain pointwise reach](figures/all_domains_pointwise_reach_k1_k4.png)",
                "",
                "[Interactive Plotly figure](figures/all_domains_pointwise_reach_k1_k4.html)",
                "",
            ]
        )
        for result in results:
            lines.extend(
                [
                    f"## {result.spec.label}: representative tail seed",
                    "",
                    (
                        f"대표 seed는 geometry-only selected K 직전인 "
                        f"K={result.seed_selection_k}에서 pointwise reach의 하위 5% "
                        "값에 가장 가까운 point 중 중심에서 가장 먼 점이다."
                    ),
                    (
                        "이 seed 그림은 한 point의 누적 shell을 설명하기 위한 예시이며, "
                        "전체 하위 5% quantile 판정은 위의 all-domain map과 CSV를 따른다."
                    ),
                    "",
                    "| K | cumulative seed reach | newly reached shell | global reach |",
                    "|---:|---:|---:|---:|",
                ]
            )
            for metric in result.metrics:
                lines.append(
                    f"| {metric.subspace_dimension} | "
                    f"{100.0 * metric.representative_seed_reach_fraction:.6f}% | "
                    f"{100.0 * metric.representative_seed_new_shell_fraction:.6f}% | "
                    f"{100.0 * metric.global_reach_fraction:.6f}% |"
                )
            lines.extend(
                [
                    "",
                    (
                        f"![{result.spec.label} K reach]"
                        f"(figures/{result.spec.slug}_representative_seed_k1_k4.png)"
                    ),
                    "",
                    (
                        f"[Interactive Plotly figure]"
                        f"(figures/{result.spec.slug}_representative_seed_k1_k4.html)"
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "## 핵심 관찰",
                "",
                "1. Square와 Disk는 K=2에서 모든 active point pair가 구조적으로 연결된다.",
                "2. Annulus는 hole이 axial segment를 분리하므로 K=3에서도 tail reach가 ",
                "   부족하며 K=4에서 완전 연결된다.",
                "3. Pentagram은 K=4가 global/tail 99% rule을 만족하지만 극단적인 tip ",
                "   point의 full reach에는 더 큰 K가 필요하다.",
                "4. 이 그림은 geometry-only K 선택의 설명 자료이며 PDE별 numerical ",
                "   optimality를 주장하지 않는다.",
                "",
                "## Machine-readable outputs",
                "",
                "- `summary.json`",
                "- `metrics/per_domain_k_reach.csv`",
                "- `data/<domain>_k_reach.npz`",
            ]
        )
        (self.request.outdir / "analysis_report.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
