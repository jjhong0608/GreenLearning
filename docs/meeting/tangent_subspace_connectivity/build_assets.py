from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import html
import json
import logging
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from rich.logging import RichHandler


PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
PRESENTATION_ROOT: Final = Path(__file__).resolve().parent
DEFAULT_OUTDIR: Final = PRESENTATION_ROOT / "assets"
BUILDER_VERSION: Final = 3


@dataclass(frozen=True)
class MeetingEvidencePaths:
    annulus_html: Path = Path(
        "docs/meeting/annulus_transition_error/assets/"
        "annulus_transition_sample47_error_matrix.html"
    )
    annulus_manifest: Path = Path(
        "docs/meeting/annulus_transition_error/assets/manifest.json"
    )
    geometry_root: Path = Path("checkpoints/geometry_k_connectivity_visualization")
    pentagram_root: Path = Path("checkpoints/pentagram/tangent_topology_k_analysis")
    unit_square_root: Path = Path(
        "checkpoints/numerical_examples/unit_square_poisson/training_size_analysis"
    )
    pentagram_artifact_root: Path = Path(
        "checkpoints/pentagram/coupling11/artifacts_best_energy"
    )
    pentagram_config: Path = Path("checkpoints/pentagram/coupling11/config_used.json")
    pentagram_coefficient: Path = Path("coefficients/CDR_pentagram.py")
    unit_square_artifact_root: Path = Path(
        "checkpoints/numerical_examples/unit_square_poisson/"
        "coupling_train4800_seed0/artifacts_best_energy"
    )
    unit_square_config: Path = Path(
        "checkpoints/numerical_examples/unit_square_poisson/"
        "coupling_train4800_seed0/config_used.json"
    )
    unit_square_coefficient: Path = Path(
        "numerical_examples/unit_square/coefficients.py"
    )

    def resolved(self, project_root: Path) -> MeetingEvidencePaths:
        return MeetingEvidencePaths(
            annulus_html=self._resolve(project_root, self.annulus_html),
            annulus_manifest=self._resolve(project_root, self.annulus_manifest),
            geometry_root=self._resolve(project_root, self.geometry_root),
            pentagram_root=self._resolve(project_root, self.pentagram_root),
            unit_square_root=self._resolve(project_root, self.unit_square_root),
            pentagram_artifact_root=self._resolve(
                project_root, self.pentagram_artifact_root
            ),
            pentagram_config=self._resolve(project_root, self.pentagram_config),
            pentagram_coefficient=self._resolve(
                project_root, self.pentagram_coefficient
            ),
            unit_square_artifact_root=self._resolve(
                project_root, self.unit_square_artifact_root
            ),
            unit_square_config=self._resolve(project_root, self.unit_square_config),
            unit_square_coefficient=self._resolve(
                project_root, self.unit_square_coefficient
            ),
        )

    @staticmethod
    def _resolve(project_root: Path, path: Path) -> Path:
        return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class MeetingAssetConfig:
    project_root: Path = PROJECT_ROOT
    outdir: Path = DEFAULT_OUTDIR
    evidence: MeetingEvidencePaths = MeetingEvidencePaths()
    overwrite: bool = False

    @property
    def resolved_evidence(self) -> MeetingEvidencePaths:
        return self.evidence.resolved(self.project_root)


class TangentSubspaceMeetingAssetBuilder:
    """Build an offline deck bundle from frozen diagnostic artifacts only."""

    def __init__(self, config: MeetingAssetConfig) -> None:
        self.config = config
        self.paths = config.resolved_evidence
        self.logger = self._build_logger(config.outdir.parent / "build_assets.log")
        self._sources: set[Path] = set()

    @staticmethod
    def _build_logger(log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(
            f"tangent_subspace_meeting_assets.{log_path.resolve()}"
        )
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)
        logger.addHandler(rich_handler)

        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        return logger

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def _source(self, path: Path) -> Path:
        if not path.is_file():
            raise FileNotFoundError(f"Frozen presentation source is missing: {path}")
        self._sources.add(path)
        return path

    def _prepare_outdir(self) -> None:
        manifest_path = self.config.outdir / "manifest.json"
        if manifest_path.exists() and not self.config.overwrite:
            raise FileExistsError(
                f"Output already exists: {manifest_path}; pass --overwrite"
            )
        self.config.outdir.mkdir(parents=True, exist_ok=True)

    def _write_figure(self, source: Path, output_name: str) -> Path:
        source = self._source(source)
        figure = pio.from_json(source.read_text(encoding="utf-8"))
        self._prepare_figure_for_deck(figure, output_name)
        return self._write_figure_object(figure, output_name)

    def _write_figure_object(self, figure: Any, output_name: str) -> Path:
        output = self.config.outdir / output_name
        pio.write_html(
            figure,
            output,
            include_plotlyjs="directory",
            full_html=True,
            config={
                "responsive": True,
                "displaylogo": False,
                "displayModeBar": False,
            },
            default_width="100vw",
            default_height="100vh",
            div_id=f"figure-{output.stem}",
        )
        return output

    def _write_static_mesh_grid(
        self,
        *,
        panels: list[tuple[Path, str]],
        rows: int,
        cols: int,
        output_name: str,
    ) -> tuple[Path, list[Path]]:
        if len(panels) != rows * cols:
            raise ValueError("Static mesh panels must fill the requested grid.")

        sources: list[Path] = []
        cards: list[str] = []
        for json_path, title in panels:
            json_source = self._source(json_path)
            png_source = self._source(json_path.with_suffix(".png"))
            sources.extend((json_source, png_source))
            encoded = base64.b64encode(png_source.read_bytes()).decode("ascii")
            title_markup = f"<h2>{html.escape(title)}</h2>" if title else ""
            cards.append(
                '<article class="mesh-panel">'
                f"{title_markup}"
                '<div class="mesh-image-shell">'
                f'<img src="data:image/png;base64,{encoded}" '
                f'alt="{html.escape(title or output_name)}">'
                "</div></article>"
            )

        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="asset-kind" content="static-mesh-grid">
<style>
html, body {{ width: 100%; height: 100%; margin: 0; overflow: hidden; background: #fff; }}
body {{ font-family: Aptos, 'Helvetica Neue', sans-serif; color: #233139; }}
.mesh-grid {{
  box-sizing: border-box;
  display: grid;
  grid-template-columns: repeat({cols}, minmax(0, 1fr));
  grid-template-rows: repeat({rows}, minmax(0, 1fr));
  gap: 8px 10px;
  width: 100%;
  height: 100%;
  padding: 2px;
}}
.mesh-panel {{
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  align-items: stretch;
}}
.mesh-panel h2 {{
  flex: 0 0 auto;
  margin: 0 0 2px;
  text-align: center;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.15;
}}
.mesh-image-shell {{ flex: 1 1 auto; min-width: 0; min-height: 0; }}
.mesh-panel img {{
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
}}
</style>
</head>
<body data-asset-kind="static-mesh-grid">
<main class="mesh-grid">{"".join(cards)}</main>
</body>
</html>
"""
        output = self.config.outdir / output_name
        output.write_text(document, encoding="utf-8")
        return output, sources

    def _read_plotly_figure(self, path: Path) -> Any:
        source = self._source(path)
        return pio.from_json(source.read_text(encoding="utf-8"))

    @staticmethod
    def _linear_quantile(values: list[float], probability: float) -> float:
        if not values:
            raise ValueError("Cannot compute a quantile from an empty sequence.")
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

    @classmethod
    def _distribution(cls, values: list[float]) -> dict[str, float]:
        if not values:
            raise ValueError("Cannot summarize an empty distribution.")
        return {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "q25": cls._linear_quantile(values, 0.25),
            "median": cls._linear_quantile(values, 0.5),
            "q75": cls._linear_quantile(values, 0.75),
            "p90": cls._linear_quantile(values, 0.9),
            "p95": cls._linear_quantile(values, 0.95),
            "max": max(values),
        }

    @staticmethod
    def _prepare_figure_for_deck(figure: Any, output_name: str) -> None:
        """Remove source-report chrome that competes with the slide context."""
        figure.update_layout(
            autosize=True,
            width=None,
            height=None,
            title=None,
            font={"size": 14},
            hovermode="closest",
        )
        figure.update_xaxes(title_text=None, automargin=True)
        figure.update_yaxes(title_text=None, automargin=True)

        if output_name.startswith("geometry_"):
            figure.update_layout(
                showlegend=False,
                margin={"l": 34, "r": 20, "t": 44, "b": 34},
            )
            for annotation in figure.layout.annotations or ():
                annotation.update(font={"size": 14})
            return

        if output_name == "pentagram_cost_quality_tradeoff.html":
            figure.update_layout(
                showlegend=False,
                margin={"l": 58, "r": 24, "t": 24, "b": 42},
            )
            return

        if output_name == "pentagram_trained_k_quality.html":
            figure.update_layout(
                legend={
                    "orientation": "h",
                    "x": 0.5,
                    "xanchor": "center",
                    "y": -0.08,
                    "yanchor": "top",
                    "font": {"size": 11},
                    "entrywidth": 125,
                    "entrywidthmode": "pixels",
                },
                margin={"l": 58, "r": 24, "t": 28, "b": 76},
            )
            return

        figure.update_layout(
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.02,
                "yanchor": "bottom",
                "font": {"size": 13},
            },
            margin={"l": 58, "r": 24, "t": 62, "b": 42},
        )

    def _copy_annulus_asset(self) -> Path:
        source = self._source(self.paths.annulus_html)
        output = self.config.outdir / "annulus_transition_sample47_error_matrix.html"
        shutil.copyfile(source, output)
        return output

    @staticmethod
    def _assert_close(actual: float, expected: float, label: str) -> None:
        tolerance = 1e-12 * max(1.0, abs(expected))
        if abs(actual - expected) > tolerance:
            raise ValueError(
                f"Frozen metric drift for {label}: expected {expected}, got {actual}"
            )

    def _geometry_contract(self) -> dict[str, Any]:
        summary_path = self._source(self.paths.geometry_root / "summary.json")
        summary = self._read_json(summary_path)
        by_slug = {item["slug"]: item for item in summary["geometries"]}
        selected_expected = {"square": 2, "disk": 2, "annulus": 4, "pentagram": 4}
        selected = {
            slug: int(by_slug[slug]["selected_geometry_k"])
            for slug in selected_expected
        }
        if selected != selected_expected:
            raise ValueError(f"Geometry K contract drifted: {selected}")

        reach = {
            slug: [
                100.0 * float(metric["representative_seed_reach_fraction"])
                for metric in by_slug[slug]["metrics"]
            ]
            for slug in selected_expected
        }
        expected_reach = {
            "annulus": [
                0.009269558769002595,
                31.21060437523174,
                98.26659251019652,
                100.0,
            ],
            "pentagram": [
                0.021872265966754156,
                83.61767279090113,
                98.81889763779527,
                99.86876640419947,
            ],
        }
        for slug, expected_values in expected_reach.items():
            for index, expected in enumerate(expected_values):
                self._assert_close(reach[slug][index], expected, f"{slug}.K{index + 1}")
        return {
            "selected_k": selected,
            "representative_reach_percent": reach,
            "semantics": summary["semantics"],
            "selection_rule": summary["selection_rule"],
        }

    def _pentagram_contract(self) -> dict[str, Any]:
        metrics_path = self._source(
            self.paths.pentagram_root / "metrics" / "trained_k_metrics.csv"
        )
        runtime_path = self._source(
            self.paths.pentagram_root / "metrics" / "tangent_runtime_metrics.csv"
        )
        metrics = sorted(
            self._read_csv(metrics_path), key=lambda row: int(row["subspace_dimension"])
        )
        runtime = sorted(
            self._read_csv(runtime_path), key=lambda row: int(row["subspace_dimension"])
        )
        contract = {
            "subspace_dimension": [int(row["subspace_dimension"]) for row in metrics],
            "rel_sol_percent": [100.0 * float(row["rel_sol_mean"]) for row in metrics],
            "rel_u_phi_percent": [
                100.0 * float(row["rel_u_phi_mean"]) for row in metrics
            ],
            "rel_u_psi_percent": [
                100.0 * float(row["rel_u_psi_mean"]) for row in metrics
            ],
            "rel_flux_percent": [
                100.0 * float(row["rel_flux_mean"]) for row in metrics
            ],
            "forward_backward_ms": [
                float(row["forward_backward_ms"]) for row in runtime
            ],
            "evidence_caveat": (
                "Separate trained runs; engineering trend, not a same-initialization "
                "causal ablation."
            ),
        }
        expected = {
            "rel_sol_percent": [2.678, 1.590, 1.234, 1.112],
            "rel_u_phi_percent": [5.022, 3.289, 2.552, 2.394],
            "rel_u_psi_percent": [4.832, 2.711, 2.120, 1.865],
            "rel_flux_percent": [46.558, 39.658, 35.261, 31.930],
            "forward_backward_ms": [141.373, 211.807, 282.183, 361.773],
        }
        for key, values in expected.items():
            for index, expected_value in enumerate(values):
                self._assert_close(
                    contract[key][index], expected_value, f"{key}[{index}]"
                )
        return contract

    def _unit_square_contract(self) -> dict[str, Any]:
        summary_path = self._source(
            self.paths.unit_square_root / "dataset_size_summary.csv"
        )
        adjacent_path = self._source(
            self.paths.unit_square_root / "adjacent_comparisons.csv"
        )
        rows = sorted(
            self._read_csv(summary_path), key=lambda row: int(row["train_size"])
        )
        adjacent = [
            row
            for row in self._read_csv(adjacent_path)
            if row["metric"] == "rel_sol_mean"
        ]
        contract = {
            "num_train": [int(row["train_size"]) for row in rows],
            "rel_sol_percent": [100.0 * float(row["rel_sol_mean"]) for row in rows],
            "rel_flux_percent": [100.0 * float(row["rel_flux_mean"]) for row in rows],
            "selected_num_train": 4800,
            "fixed_optimizer_steps": 2400,
            "seed_count": 4,
            "paired_seed_improvements": [
                int(row["improved_seed_count"]) for row in adjacent
            ],
        }
        expected = [
            0.42547671154041755,
            0.39311790743094077,
            0.3695945505358992,
            0.3504999895637964,
        ]
        for index, expected_value in enumerate(expected):
            self._assert_close(
                contract["rel_sol_percent"][index],
                expected_value,
                f"unit_square.rel_sol[{index}]",
            )
        return contract

    def _problem_contract(
        self,
        *,
        artifact_root: Path,
        config_path: Path,
        coefficient_path: Path,
        representative_sample: int,
        expected_rel_sol_mean: float,
        expected_rel_flux_mean: float,
    ) -> dict[str, Any]:
        summary_path = self._source(artifact_root / "summary.json")
        metrics_path = self._source(
            artifact_root / "metrics" / "per_sample_metrics.csv"
        )
        config_path = self._source(config_path)
        coefficient_path = self._source(coefficient_path)
        summary = self._read_json(summary_path)
        config = self._read_json(config_path)
        rows = self._read_csv(metrics_path)
        if len(rows) != 100:
            raise ValueError(
                f"Expected a 100-sample frozen test set at {metrics_path}, "
                f"found {len(rows)} rows."
            )
        selected_roles = {
            key: int(value) for key, value in summary["selected_sample_roles"].items()
        }
        if selected_roles.get("q50") != representative_sample:
            raise ValueError(
                "Representative sample drifted: "
                f"expected q50={representative_sample}, got {selected_roles.get('q50')}"
            )
        representative = next(
            row for row in rows if int(row["sample_id"]) == representative_sample
        )
        metrics = {
            key: [float(row[key]) for row in rows]
            for key in (
                "rel_sol",
                "rel_flux",
                "loss_energy_consistency",
                "tangent_response_mismatch_ratio",
            )
        }
        distributions = {
            key: self._distribution(values) for key, values in metrics.items()
        }
        self._assert_close(
            distributions["rel_sol"]["mean"],
            expected_rel_sol_mean,
            f"{artifact_root.name}.rel_sol_mean",
        )
        self._assert_close(
            distributions["rel_flux"]["mean"],
            expected_rel_flux_mean,
            f"{artifact_root.name}.rel_flux_mean",
        )
        projection = config["coupling_model"]["balance_projection"]
        source_config = config["dataset"]["coupling_source"]["indexed_gp"]
        return {
            "artifact_summary": self._relative(summary_path),
            "config": self._relative(config_path),
            "coefficient_source": self._relative(coefficient_path),
            "test_sample_count": len(rows),
            "representative_sample": representative_sample,
            "representative_sample_role": "q50_rel_sol",
            "representative_metrics": {
                key: float(representative[key])
                for key in (
                    "rel_sol",
                    "rel_flux",
                    "loss_energy_consistency",
                    "tangent_response_mismatch_ratio",
                )
            },
            "distributions": distributions,
            "mesh_figure_fields": summary["mesh_figure_fields"],
            "coefficient_mesh_figure_fields": summary["coefficient_mesh_figure_fields"],
            "source_config": {
                "num_train": int(source_config["num_train"]),
                "num_valid": int(source_config["num_valid"]),
                "seed": int(source_config["seed"]),
                "lengthscale": float(source_config["lengthscale"]),
                "amplitude": float(source_config["amplitude"]),
                "mean": float(source_config["mean"]),
            },
            "projection_mode": projection["mode"],
            "subspace_dimension": int(
                projection["symmetric_tangent_green_response"]["subspace_dimension"]
            ),
            "coefficient_terms": config["coupling_model"]["coefficient_terms"],
        }

    def _benchmark_contracts(self) -> dict[str, dict[str, Any]]:
        return {
            "pentagram": self._problem_contract(
                artifact_root=self.paths.pentagram_artifact_root,
                config_path=self.paths.pentagram_config,
                coefficient_path=self.paths.pentagram_coefficient,
                representative_sample=79,
                expected_rel_sol_mean=0.011122212653144637,
                expected_rel_flux_mean=0.3192977279196811,
            ),
            "unit_square": self._problem_contract(
                artifact_root=self.paths.unit_square_artifact_root,
                config_path=self.paths.unit_square_config,
                coefficient_path=self.paths.unit_square_coefficient,
                representative_sample=11,
                expected_rel_sol_mean=0.0034930418293645615,
                expected_rel_flux_mean=0.030186603381894286,
            ),
        }

    @staticmethod
    def _sample_mesh_path(root: Path, field: str, sample_id: int) -> Path:
        return (
            root
            / "figures"
            / "mesh"
            / field
            / (f"sample_{sample_id:04d}_sample_{sample_id:06d}_{field}_mesh.json")
        )

    @staticmethod
    def _coefficient_mesh_path(root: Path, field: str) -> Path:
        return root / "figures" / "coefficients" / "mesh" / f"{field}_mesh.json"

    @staticmethod
    def _mesh_trace(figure: Any) -> Any:
        return next(trace for trace in figure.data if trace.type == "mesh3d")

    @classmethod
    def _mesh_scale(cls, figures: list[Any]) -> tuple[float, float, Any]:
        meshes = [cls._mesh_trace(figure) for figure in figures]
        minima = [
            float(mesh.cmin)
            if mesh.cmin is not None
            else min(float(value) for value in mesh.intensity)
            for mesh in meshes
        ]
        maxima = [
            float(mesh.cmax)
            if mesh.cmax is not None
            else max(float(value) for value in mesh.intensity)
            for mesh in meshes
        ]
        colorscale = meshes[0].colorscale or "Viridis"
        return min(minima), max(maxima), colorscale

    @staticmethod
    def _scene_key(index: int) -> str:
        return "scene" if index == 1 else f"scene{index}"

    def _add_mesh_panel(
        self,
        target: Any,
        source: Any,
        *,
        row: int,
        col: int,
        scene_index: int,
        coloraxis: str,
        show_scale: bool,
    ) -> None:
        for trace in source.data:
            is_mesh = trace.type == "mesh3d"
            is_boundary = trace.type == "scatter3d" and trace.name == "Domain boundary"
            if not is_mesh and not is_boundary:
                continue
            copied = copy.deepcopy(trace)
            copied.showlegend = False
            if is_mesh:
                copied.update(
                    coloraxis=coloraxis,
                    showscale=show_scale,
                    hovertemplate=(
                        "x=%{x:.5f}<br>y=%{y:.5f}<br>value=%{intensity:.5e}"
                        "<extra></extra>"
                    ),
                )
            target.add_trace(copied, row=row, col=col)

        source_scene = source.layout.scene.to_plotly_json()
        source_scene.pop("domain", None)
        source_scene.setdefault("xaxis", {}).update(
            {"title": {"text": ""}, "tickfont": {"size": 9}}
        )
        source_scene.setdefault("yaxis", {}).update(
            {"title": {"text": ""}, "tickfont": {"size": 9}}
        )
        source_scene.setdefault("zaxis", {}).update({"visible": False})
        target.layout[self._scene_key(scene_index)].update(source_scene)

    def _mesh_grid(
        self,
        *,
        panels: list[tuple[Path, str, str, bool]],
        rows: int,
        cols: int,
        colorbar_specs: dict[str, dict[str, Any]],
        height: int,
    ) -> tuple[Any, list[Path]]:
        if len(panels) != rows * cols:
            raise ValueError("Mesh grid panels must fill the requested subplot grid.")
        figures = [self._read_plotly_figure(path) for path, _, _, _ in panels]
        figure = make_subplots(
            rows=rows,
            cols=cols,
            specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
            subplot_titles=[title for _, title, _, _ in panels],
            horizontal_spacing=0.035,
            vertical_spacing=0.075,
        )
        for index, (source, panel) in enumerate(zip(figures, panels, strict=True), 1):
            _, _, coloraxis, show_scale = panel
            row = (index - 1) // cols + 1
            col = (index - 1) % cols + 1
            self._add_mesh_panel(
                figure,
                source,
                row=row,
                col=col,
                scene_index=index,
                coloraxis=coloraxis,
                show_scale=show_scale,
            )
        figure.update_layout(
            template="plotly_white",
            autosize=True,
            width=None,
            height=height,
            showlegend=False,
            margin={"l": 8, "r": 54, "t": 34, "b": 8},
            font={"size": 12, "color": "#233139"},
        )
        for annotation in figure.layout.annotations or ():
            annotation.update(font={"size": 13, "color": "#233139"})
        for coloraxis, spec in colorbar_specs.items():
            figure.update_layout(**{coloraxis: spec})
        return figure, [path for path, _, _, _ in panels]

    def _coloraxis_spec(
        self,
        figures: list[Any],
        *,
        title: str,
        x: float,
        y: float,
        length: float,
    ) -> dict[str, Any]:
        cmin, cmax, colorscale = self._mesh_scale(figures)
        return {
            "cmin": cmin,
            "cmax": cmax,
            "colorscale": colorscale,
            "colorbar": {
                "title": {"text": title, "side": "right"},
                "x": x,
                "y": y,
                "len": length,
                "thickness": 9,
                "tickfont": {"size": 9},
                "outlinewidth": 0,
            },
        }

    def _asset_sources(self) -> list[tuple[Path, str, list[str]]]:
        geometry_figures = self.paths.geometry_root / "figures"
        pentagram_figures = self.paths.pentagram_root / "figures" / "performance"
        unit_figures = self.paths.unit_square_root / "figures"
        return [
            (
                geometry_figures / "all_domains_pointwise_reach_k1_k4.json",
                "geometry_all_domains_k_reach.html",
                ["C_i(K)", "K=1..4", "square/disk/annulus/pentagram"],
            ),
            (
                geometry_figures / "square_representative_seed_k1_k4.json",
                "geometry_square_k_reach.html",
                ["representative_seed_reach_fraction", "square", "K=1..4"],
            ),
            (
                geometry_figures / "disk_representative_seed_k1_k4.json",
                "geometry_disk_k_reach.html",
                ["representative_seed_reach_fraction", "disk", "K=1..4"],
            ),
            (
                geometry_figures / "annulus_representative_seed_k1_k4.json",
                "geometry_annulus_k_reach.html",
                ["representative_seed_reach_fraction", "annulus", "K=1..4"],
            ),
            (
                geometry_figures / "pentagram_representative_seed_k1_k4.json",
                "geometry_pentagram_k_reach.html",
                ["representative_seed_reach_fraction", "pentagram", "K=1..4"],
            ),
            (
                pentagram_figures / "trained_k_quality.json",
                "pentagram_trained_k_quality.html",
                ["rel_sol", "rel_u_phi", "rel_u_psi", "rel_flux", "K=1..4"],
            ),
            (
                pentagram_figures / "cost_quality_tradeoff.json",
                "pentagram_cost_quality_tradeoff.html",
                ["forward_backward_ms", "rel_sol", "K=1..4"],
            ),
            (
                unit_figures / "test_metrics_by_train_size.json",
                "unit_square_training_size.html",
                ["rel_sol_mean", "rel_flux_mean", "num_train", "seed_count=4"],
            ),
            (
                unit_figures / "paired_seed_errors.json",
                "unit_square_paired_seed_errors.html",
                ["paired_seed", "rel_sol", "num_train"],
            ),
        ]

    def _distribution_figure(
        self,
        *,
        artifact_root: Path,
        representative_sample: int,
        label: str,
    ) -> tuple[Any, Path]:
        metrics_path = self._source(
            artifact_root / "metrics" / "per_sample_metrics.csv"
        )
        rows = self._read_csv(metrics_path)
        representative = next(
            row for row in rows if int(row["sample_id"]) == representative_sample
        )
        figure = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Relative solution error",
                "Relative flux error",
                "Canonical energy consistency",
                "Post/pre response mismatch",
            ),
            horizontal_spacing=0.12,
            vertical_spacing=0.16,
        )
        line_color = "#147A7E"
        marker_color = "#D75B3F"
        for col, key in enumerate(("rel_sol", "rel_flux"), 1):
            values = sorted(100.0 * float(row[key]) for row in rows)
            cumulative = [
                100.0 * (index + 1) / len(values) for index in range(len(values))
            ]
            sample_value = 100.0 * float(representative[key])
            sample_cdf = (
                100.0 * sum(value <= sample_value for value in values) / len(values)
            )
            figure.add_trace(
                go.Scatter(
                    x=values,
                    y=cumulative,
                    mode="lines",
                    line={"color": line_color, "width": 3},
                    name="Test ECDF",
                    showlegend=col == 1,
                    hovertemplate="error=%{x:.4f}%<br>samples≤error=%{y:.1f}%<extra></extra>",
                ),
                row=1,
                col=col,
            )
            figure.add_trace(
                go.Scatter(
                    x=[sample_value],
                    y=[sample_cdf],
                    mode="markers",
                    marker={
                        "color": marker_color,
                        "size": 11,
                        "line": {"color": "#FFFFFF", "width": 1.5},
                    },
                    name=f"Shown sample {representative_sample}",
                    showlegend=col == 1,
                    hovertemplate=(
                        f"{label} sample {representative_sample}<br>"
                        "error=%{x:.4f}%<br>ECDF=%{y:.1f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

        energy = [float(row["loss_energy_consistency"]) for row in rows]
        mismatch = [
            100.0 * float(row["tangent_response_mismatch_ratio"]) for row in rows
        ]
        for col, values, hover_suffix in (
            (1, energy, ""),
            (2, mismatch, "%"),
        ):
            figure.add_trace(
                go.Box(
                    x=values,
                    orientation="h",
                    boxpoints="all",
                    jitter=0.32,
                    pointpos=0,
                    fillcolor="rgba(20,122,126,0.18)",
                    line={"color": line_color, "width": 2},
                    marker={"color": line_color, "size": 5, "opacity": 0.55},
                    showlegend=False,
                    hovertemplate=f"value=%{{x:.5g}}{hover_suffix}<extra></extra>",
                ),
                row=2,
                col=col,
            )

        figure.update_xaxes(title_text="error (%)", row=1, col=1)
        figure.update_xaxes(title_text="error (%)", row=1, col=2)
        figure.update_yaxes(title_text="test samples (%)", range=[0, 101], row=1, col=1)
        figure.update_yaxes(title_text="test samples (%)", range=[0, 101], row=1, col=2)
        figure.update_xaxes(
            title_text="energy",
            title_standoff=8,
            type="log",
            automargin=True,
            row=2,
            col=1,
        )
        figure.update_xaxes(
            title_text="remaining mismatch (%)",
            title_standoff=8,
            automargin=True,
            row=2,
            col=2,
        )
        figure.update_yaxes(showticklabels=False, row=2, col=1)
        figure.update_yaxes(showticklabels=False, row=2, col=2)
        figure.update_layout(
            template="plotly_white",
            autosize=True,
            width=None,
            height=None,
            margin={"l": 58, "r": 18, "t": 50, "b": 72},
            font={"size": 12, "color": "#233139"},
            legend={
                "orientation": "h",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.08,
                "yanchor": "bottom",
                "font": {"size": 11},
            },
        )
        for annotation in figure.layout.annotations or ():
            annotation.update(font={"size": 13, "color": "#233139"})
        return figure, metrics_path

    def _benchmark_assets(
        self, contracts: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        del contracts
        assets: dict[str, dict[str, Any]] = {}

        pentagram_coefficients = [
            self._coefficient_mesh_path(self.paths.pentagram_artifact_root, field)
            for field in ("diffusion_a", "convection_magnitude", "reaction_c")
        ]
        output, sources = self._write_static_mesh_grid(
            panels=[
                (pentagram_coefficients[0], "Diffusion  a(x,y)"),
                (pentagram_coefficients[1], "Convection magnitude  |b(x,y)|"),
                (pentagram_coefficients[2], "Reaction  c(x,y)"),
            ],
            rows=1,
            cols=3,
            output_name="pentagram_problem_coefficients.html",
        )
        assets[output.name] = {
            "source_files": [self._relative(path) for path in sources],
            "metric_keys": ["a(x,y)", "|b(x,y)|", "c(x,y)", "R=0.5"],
        }

        benchmark_specs = (
            (
                "pentagram",
                self.paths.pentagram_artifact_root,
                79,
                "Pentagram",
            ),
            (
                "unit_square",
                self.paths.unit_square_artifact_root,
                11,
                "Unit square",
            ),
        )
        for slug, root, sample_id, label in benchmark_specs:
            if slug == "unit_square":
                coefficient_source = self._coefficient_mesh_path(root, "diffusion_a")
                output_name = "unit_square_problem_coefficient.html"
                output, sources = self._write_static_mesh_grid(
                    panels=[(coefficient_source, "Constant field  a=1")],
                    rows=1,
                    cols=1,
                    output_name=output_name,
                )
                assets[output.name] = {
                    "source_files": [self._relative(path) for path in sources],
                    "metric_keys": ["a(x,y)=1", "b(x,y)=0", "c(x,y)=0"],
                }

            rhs_source = self._sample_mesh_path(root, "rhs", sample_id)
            rhs_output_name = f"{slug}_sample{sample_id}_rhs.html"
            rhs_output, rhs_sources = self._write_static_mesh_grid(
                panels=[(rhs_source, "")],
                rows=1,
                cols=1,
                output_name=rhs_output_name,
            )
            assets[rhs_output.name] = {
                "source_files": [self._relative(path) for path in rhs_sources],
                "metric_keys": [f"sample_id={sample_id}", "rhs", "q50_rel_sol"],
            }

            directional_paths = [
                self._sample_mesh_path(root, field, sample_id)
                for field in (
                    "target_phi",
                    "phi",
                    "phi_error",
                    "target_psi",
                    "psi",
                    "psi_error",
                )
            ]
            directional_output, directional_sources = self._write_static_mesh_grid(
                panels=[
                    (directional_paths[0], "Reference  φ*"),
                    (directional_paths[1], "Prediction  φ̂"),
                    (directional_paths[2], "Error  φ̂ - φ*"),
                    (directional_paths[3], "Reference  ψ*"),
                    (directional_paths[4], "Prediction  ψ̂"),
                    (directional_paths[5], "Error  ψ̂ - ψ*"),
                ],
                rows=2,
                cols=3,
                output_name=f"{slug}_sample{sample_id}_directional.html",
            )
            assets[directional_output.name] = {
                "source_files": [self._relative(path) for path in directional_sources],
                "metric_keys": [
                    f"sample_id={sample_id}",
                    "target_phi",
                    "phi",
                    "phi_error",
                    "target_psi",
                    "psi",
                    "psi_error",
                ],
            }

            solution_paths = [
                self._sample_mesh_path(root, field, sample_id)
                for field in ("sol", "u_pred", "u_pred_error")
            ]
            solution_output, solution_sources = self._write_static_mesh_grid(
                panels=[
                    (solution_paths[0], "Reference  u*"),
                    (solution_paths[1], "Prediction  û"),
                    (solution_paths[2], "Signed error  û - u*"),
                ],
                rows=1,
                cols=3,
                output_name=f"{slug}_sample{sample_id}_solution.html",
            )
            assets[solution_output.name] = {
                "source_files": [self._relative(path) for path in solution_sources],
                "metric_keys": [
                    f"sample_id={sample_id}",
                    "sol",
                    "u_pred",
                    "u_pred_error",
                ],
            }

            distribution_figure, metrics_path = self._distribution_figure(
                artifact_root=root,
                representative_sample=sample_id,
                label=label,
            )
            distribution_name = f"{slug}_test_distribution.html"
            distribution_output = self._write_figure_object(
                distribution_figure, distribution_name
            )
            assets[distribution_output.name] = {
                "source_files": [self._relative(metrics_path)],
                "metric_keys": [
                    "rel_sol",
                    "rel_flux",
                    "loss_energy_consistency",
                    "tangent_response_mismatch_ratio",
                ],
            }
        return assets

    def run(self) -> dict[str, Any]:
        self._prepare_outdir()
        geometry = self._geometry_contract()
        pentagram = self._pentagram_contract()
        unit_square = self._unit_square_contract()
        benchmarks = self._benchmark_contracts()
        self._source(self.paths.annulus_manifest)

        assets: dict[str, dict[str, Any]] = {}
        annulus_output = self._copy_annulus_asset()
        assets[annulus_output.name] = {
            "source_files": [
                self._relative(self.paths.annulus_html),
                self._relative(self.paths.annulus_manifest),
            ],
            "metric_keys": [
                "sample_id=47",
                "phi_error",
                "psi_error",
                "u_phi_error",
                "u_pred_error",
                "u_psi_error",
            ],
        }
        for source, output_name, metric_keys in self._asset_sources():
            output = self._write_figure(source, output_name)
            assets[output_name] = {
                "source_files": [self._relative(source)],
                "metric_keys": metric_keys,
            }
        assets.update(self._benchmark_assets(benchmarks))

        for name, metadata in assets.items():
            output = self.config.outdir / name
            metadata["generated_sha256"] = self._sha256(output)
            metadata["generated_size_bytes"] = output.stat().st_size

        source_provenance = {
            self._relative(path): {
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(self._sources)
        }
        manifest = {
            "builder_version": BUILDER_VERSION,
            "offline_plotly": True,
            "model_inference_used": False,
            "assets": assets,
            "source_provenance": source_provenance,
            "geometry_contract": geometry,
            "pentagram_contract": pentagram,
            "unit_square_contract": unit_square,
            "benchmark_contracts": benchmarks,
            "plotly_bundle": {
                "path": "plotly.min.js",
                "sha256": self._sha256(self.config.outdir / "plotly.min.js"),
                "size_bytes": (self.config.outdir / "plotly.min.js").stat().st_size,
            },
        }
        manifest_path = self.config.outdir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.logger.info(
            "Saved %d offline assets to %s", len(assets), self.config.outdir
        )
        return manifest

    def _relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.config.project_root.resolve()))
        except ValueError:
            return str(path.resolve())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build frozen offline assets for the tangent-subspace meeting deck."
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    TangentSubspaceMeetingAssetBuilder(
        MeetingAssetConfig(outdir=args.outdir, overwrite=args.overwrite)
    ).run()


if __name__ == "__main__":
    main()
