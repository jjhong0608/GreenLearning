from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import length_jump_balanced_edge_energy_loss
from greenonet.config import ComplexLengthJumpBalanceConfig


@dataclass(frozen=True)
class ComplexEnergyRefinementRequest:
    """Inputs and acceptance bounds for a persistent-jump refinement audit."""

    geometries: tuple[Path, ...]
    outdir: Path
    jump_axis: Literal["x", "y"] = "y"
    jump_coordinate: float | None = None
    exponent_min: float = -1.25
    exponent_max: float = -0.75
    scaled_energy_relative_spread_max: float = 0.35
    fail_on_violation: bool = True

    def __post_init__(self) -> None:
        if len(self.geometries) < 3:
            raise ValueError("Refinement audit requires at least three geometries.")
        if self.jump_axis not in {"x", "y"}:
            raise ValueError("jump_axis must be 'x' or 'y'.")
        for name, value in (
            ("exponent_min", self.exponent_min),
            ("exponent_max", self.exponent_max),
            (
                "scaled_energy_relative_spread_max",
                self.scaled_energy_relative_spread_max,
            ),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.exponent_min >= self.exponent_max:
            raise ValueError("exponent_min must be smaller than exponent_max.")
        if self.scaled_energy_relative_spread_max < 0.0:
            raise ValueError("scaled_energy_relative_spread_max must be non-negative.")


@dataclass(frozen=True)
class RefinementMetric:
    geometry_path: str
    step_size: float
    num_points: int
    jump_coordinate: float
    bulk_energy: float
    boundary_energy: float
    canonical_energy: float
    h_times_bulk_energy: float
    h_times_canonical_energy: float


class ComplexEnergyRefinementReportMixin:
    request: ComplexEnergyRefinementRequest

    @staticmethod
    def _markdown_report(summary: dict[str, Any]) -> str:
        rows = [
            (
                f"| `{Path(row['geometry_path']).name}` | {row['step_size']:.8g} | "
                f"{row['num_points']} | {row['bulk_energy']:.8e} | "
                f"{row['boundary_energy']:.8e} | "
                f"{row['canonical_energy']:.8e} | "
                f"{row['h_times_canonical_energy']:.8e} |"
            )
            for row in summary["metrics"]
        ]
        return "\n".join(
            (
                "# Complex Energy Grid-Refinement Audit",
                "",
                (
                    f"- Persistent jump: `{summary['jump_axis']} >= "
                    f"{summary['jump_coordinate']:.8g}` on a fixed interior patch"
                ),
                (
                    "- Expected canonical-energy scaling: "
                    f"`E_h = O(h^{summary['expected_log_slope']})`"
                ),
                (
                    "- Measured canonical log-log slope: "
                    f"`{summary['canonical_energy_log_slope']:.8f}`"
                ),
                (
                    "- Relative spread of `h * E_h`: "
                    f"`{summary['canonical_h_times_energy_relative_spread']:.8f}`"
                ),
                f"- Acceptance passed: `{summary['acceptance']['passed']}`",
                "",
                "## Refinement Metrics",
                "",
                (
                    "| Geometry | h | Points | Bulk energy | Boundary energy | "
                    "Canonical energy | h * canonical |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|",
                *rows,
                "",
                "## Interpretation",
                "",
                (
                    "The synthetic residual is zero near every physical boundary and "
                    "has fixed-amplitude jumps on a grid-independent interior patch. "
                    "Therefore boundary anchoring does not manufacture the observed "
                    "inverse-h growth; the bulk difference quotient detects persistent "
                    "interface jumps."
                ),
                "",
            )
        )

    def _write_outputs(self, summary: dict[str, Any]) -> None:
        self.request.outdir.mkdir(parents=True, exist_ok=True)
        (self.request.outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metrics = summary["metrics"]
        with (self.request.outdir / "refinement_metrics.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        (self.request.outdir / "analysis_report.md").write_text(
            self._markdown_report(summary),
            encoding="utf-8",
        )


class ComplexEnergyRefinementAnalyzer(ComplexEnergyRefinementReportMixin):
    """Audit whether a persistent internal jump receives inverse-h energy."""

    def __init__(
        self,
        request: ComplexEnergyRefinementRequest,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.request = request
        self.logger = logger or logging.getLogger(__name__)
        missing = [path for path in request.geometries if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Geometry file does not exist: {missing[0]}")

    def run(self) -> dict[str, Any]:
        metadata = [self._annulus_metadata(path) for path in self.request.geometries]
        inner_radius = metadata[0][0]
        outer_radius = metadata[0][1]
        for current_inner, current_outer in metadata[1:]:
            if not (
                math.isclose(current_inner, inner_radius, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(
                    current_outer,
                    outer_radius,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("All refinement geometries must use the same annulus.")
        jump_coordinate = (
            inner_radius
            if self.request.jump_coordinate is None
            else float(self.request.jump_coordinate)
        )
        metrics = tuple(
            sorted(
                (
                    self._measure_geometry(
                        path,
                        inner_radius=inner_radius,
                        outer_radius=outer_radius,
                        jump_coordinate=jump_coordinate,
                    )
                    for path in self.request.geometries
                ),
                key=lambda item: item.step_size,
                reverse=True,
            )
        )
        step_sizes = np.asarray(
            [item.step_size for item in metrics],
            dtype=np.float64,
        )
        bulk_values = np.asarray(
            [item.bulk_energy for item in metrics],
            dtype=np.float64,
        )
        canonical_values = np.asarray(
            [item.canonical_energy for item in metrics],
            dtype=np.float64,
        )
        bulk_slope = self._log_slope(step_sizes, bulk_values)
        canonical_slope = self._log_slope(step_sizes, canonical_values)
        scaled = step_sizes * canonical_values
        scaled_spread = self._relative_spread(scaled)
        slope_passed = (
            self.request.exponent_min <= canonical_slope <= self.request.exponent_max
        )
        spread_passed = scaled_spread <= self.request.scaled_energy_relative_spread_max
        passed = slope_passed and spread_passed
        summary: dict[str, Any] = {
            "schema_version": 1,
            "geometries": [str(path.resolve()) for path in self.request.geometries],
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "jump_axis": self.request.jump_axis,
            "jump_coordinate": jump_coordinate,
            "expected_log_slope": -1.0,
            "bulk_energy_log_slope": bulk_slope,
            "canonical_energy_log_slope": canonical_slope,
            "canonical_h_times_energy_relative_spread": scaled_spread,
            "acceptance": {
                "passed": passed,
                "slope_passed": slope_passed,
                "spread_passed": spread_passed,
                "exponent_min": self.request.exponent_min,
                "exponent_max": self.request.exponent_max,
                "scaled_energy_relative_spread_max": (
                    self.request.scaled_energy_relative_spread_max
                ),
            },
            "metrics": [asdict(item) for item in metrics],
        }
        self._write_outputs(summary)
        self.logger.info(
            "Refinement audit slope=%.6f hE_spread=%.6f passed=%s",
            canonical_slope,
            scaled_spread,
            passed,
        )
        if self.request.fail_on_violation and not passed:
            raise RuntimeError(
                "Persistent-jump refinement audit failed: "
                f"slope={canonical_slope:.6f}, hE_spread={scaled_spread:.6f}."
            )
        return summary

    def _measure_geometry(
        self,
        path: Path,
        *,
        inner_radius: float,
        outer_radius: float,
        jump_coordinate: float,
    ) -> RefinementMetric:
        geometry = load_complex_geometry(path)
        coords = geometry.coords_valid
        if self.request.jump_axis == "x":
            transverse = coords[:, 0]
            primary = coords[:, 1]
        else:
            transverse = coords[:, 1]
            primary = coords[:, 0]
        primary_extent_squared = outer_radius**2 - jump_coordinate**2
        if primary_extent_squared <= 0.0:
            raise ValueError("jump_coordinate must lie strictly inside the annulus.")
        primary_extent = math.sqrt(primary_extent_squared)
        band_width = 0.2 * (outer_radius - inner_radius)
        absolute_primary = primary.abs()
        patch = (
            (transverse >= jump_coordinate)
            & (transverse <= jump_coordinate + band_width)
            & (absolute_primary >= 0.35 * primary_extent)
            & (absolute_primary <= 0.75 * primary_extent)
        )
        residual = patch.to(coords.dtype).unsqueeze(0)
        energy = length_jump_balanced_edge_energy_loss(
            u_phi_valid=residual,
            u_psi_valid=torch.zeros_like(residual),
            a_valid=torch.ones_like(residual),
            geometry=geometry,
            config=ComplexLengthJumpBalanceConfig(enabled=False),
        )
        step_size = float(torch.maximum(geometry.hx, geometry.hy).item())
        return RefinementMetric(
            geometry_path=str(path.resolve()),
            step_size=step_size,
            num_points=geometry.num_points,
            jump_coordinate=jump_coordinate,
            bulk_energy=float(energy.bulk_unweighted.item()),
            boundary_energy=float(energy.boundary.item()),
            canonical_energy=float(energy.unweighted.item()),
            h_times_bulk_energy=step_size * float(energy.bulk_unweighted.item()),
            h_times_canonical_energy=step_size * float(energy.unweighted.item()),
        )

    @staticmethod
    def _annulus_metadata(path: Path) -> tuple[float, float]:
        with np.load(path, allow_pickle=False) as raw:
            missing = {"inner_radius", "outer_radius"} - set(raw.files)
            if missing:
                raise ValueError(
                    f"{path} is missing annulus metadata: {', '.join(sorted(missing))}."
                )
            inner_radius = float(np.asarray(raw["inner_radius"]).item())
            outer_radius = float(np.asarray(raw["outer_radius"]).item())
        if not (
            math.isfinite(inner_radius)
            and math.isfinite(outer_radius)
            and 0.0 < inner_radius < outer_radius
        ):
            raise ValueError(f"{path} has invalid annulus radii.")
        return inner_radius, outer_radius

    @staticmethod
    def _log_slope(step_sizes: np.ndarray, energies: np.ndarray) -> float:
        if np.any(step_sizes <= 0.0) or np.any(energies <= 0.0):
            raise ValueError("Refinement step sizes and energies must be positive.")
        return float(np.polyfit(np.log(step_sizes), np.log(energies), deg=1)[0])

    @staticmethod
    def _relative_spread(values: np.ndarray) -> float:
        mean = float(np.mean(values))
        if mean <= 0.0:
            raise ValueError("Scaled refinement energy must have positive mean.")
        return float((np.max(values) - np.min(values)) / mean)


def audit_complex_energy_refinement(
    request: ComplexEnergyRefinementRequest,
    *,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    return ComplexEnergyRefinementAnalyzer(request, logger=logger).run()
