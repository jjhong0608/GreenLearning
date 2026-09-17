"""Summarize the right-scaling pilot without discarding nonconverged cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


class RightScalingReport:
    def run(self, root: Path) -> None:
        provenance = json.loads((root / "input_hashes.json").read_text())
        executed_cli = Path("cli/audit_right_scaled_solver.py").resolve()
        archive = root / "executed_audit.py"
        if not archive.exists():
            raise ValueError("Archive the executed audit CLI before reporting")
        for filename, digest in provenance.items():
            path = Path(filename)
            verified_path = archive if path.resolve() == executed_cli else path
            if _sha256(verified_path) != digest:
                raise ValueError(f"Provenance changed: {filename}")
        metrics = read(root / "per_sample.csv")
        timings = read(root / "timing.csv")
        native_timings = read(root / "native_prediction_timing.csv")
        stops = read(root / "stops.csv")
        if (
            len(metrics) != 70
            or len(timings) != 70
            or len(stops) != 40
            or len(native_timings) != 10
        ):
            raise ValueError("Incomplete two-problem five-source experiment")
        summary: list[dict[str, Any]] = []
        for example in ("unit_square", "disk"):
            methods = sorted(
                {r["condition"] for r in metrics if r["example"] == example}
            )
            for method in methods:
                selected = [
                    r
                    for r in metrics
                    if r["example"] == example and r["condition"] == method
                ]
                measured = [
                    float(r["seconds"])
                    for r in timings
                    if r["example"] == example and r["method"] == method
                ]
                common_wrapper_seconds = float(np.median(measured))
                if method == "learned":
                    measured = [
                        float(r["seconds"])
                        for r in native_timings
                        if r["example"] == example
                    ]
                status = [
                    r
                    for r in stops
                    if r["example"] == example and r["method"] == method
                ]
                assert len(selected) == len(measured) == 5
                errors = [float(r["rel_sol"]) * 100 for r in selected]
                assert max(float(r["balance_max_abs"]) for r in selected) < 1e-11
                summary.append(
                    dict(
                        example=example,
                        method=method,
                        mean_error_percent=float(np.mean(errors)),
                        p95_error_percent=float(np.quantile(errors, 0.95)),
                        max_error_percent=max(errors),
                        mean_j=float(
                            np.mean([0.5 * float(r["response_cost"]) for r in selected])
                        ),
                        max_balance_abs=max(
                            float(r["balance_max_abs"]) for r in selected
                        ),
                        median_seconds=float(np.median(measured)),
                        common_wrapper_median_seconds=common_wrapper_seconds,
                        timing_path="native_response_reuse"
                        if method == "learned"
                        else "common_cached_reconstruction",
                        min_seconds=min(measured),
                        max_seconds=max(measured),
                        max_relative_residual=max(
                            float(r["relative_equation_residual"]) for r in selected
                        ),
                        max_relative_d_to_lu=max(
                            float(r["relative_d_to_lu"]) for r in selected
                        ),
                        converged=sum(
                            int(r["status"])
                            in ((0,) if method.startswith("gmres") else (0, 1, 2, 4, 5))
                            for r in status
                        )
                        if status
                        else None,
                        min_iterations=min(int(r["iterations"]) for r in status)
                        if status
                        else None,
                        max_iterations=max(int(r["iterations"]) for r in status)
                        if status
                        else None,
                    )
                )
        baseline = read(
            Path(
                "docs/analysis/paper_source_initialization_audit_v2/learned_baseline_per_sample.csv"
            )
        )
        lookup = {
            (r["example"], r["sample_id"]): r for r in baseline if r["seed"] == "0"
        }
        for row in metrics:
            if row["condition"] == "learned":
                np.testing.assert_allclose(
                    float(row["rel_sol"]),
                    float(lookup[row["example"], row["sample_id"]]["rel_sol"]),
                    rtol=1e-8,
                    atol=1e-12,
                )
        _write_csv(root / "summary.csv", summary)
        lines = [
            "# Right scaling pilot tables",
            "",
            "Five sources per problem, seed0. Errors are percent; times are median seconds per batch5. Not full-test or matched-accuracy results.",
            "Learned timing uses original production tangent-response reuse; common-wrapper timing is retained separately in summary.csv.",
            "",
            "|Problem|Method|Mean error|P95|Max|Median seconds|Converged|Iterations min/max|",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for row in summary:
            lines.append(
                f"|{row['example']}|{row['method']}|{row['mean_error_percent']:.8f}|{row['p95_error_percent']:.8f}|{row['max_error_percent']:.8f}|{row['median_seconds']:.5f}|{row['converged']}|{row['min_iterations']}/{row['max_iterations']}|"
            )
        (root / "tables.md").write_text("\n".join(lines) + "\n")
        _json(
            root / "report_verification.json",
            dict(
                native_metrics_reproduced=True,
                metric_rows=len(metrics),
                timing_rows=len(timings),
                native_timing_rows=len(native_timings),
                stop_rows=len(stops),
                full_test=False,
                nonconvergence_retained=True,
                numerical_input_hashes_verified=len(provenance),
                executed_audit_archive_present=True,
            ),
        )
        _json(
            root / "report_input_hashes.json",
            {
                str(p): _sha256(p)
                for p in (
                    root / "per_sample.csv",
                    root / "timing.csv",
                    root / "stops.csv",
                    root / "native_prediction_timing.csv",
                    Path(
                        "docs/analysis/paper_source_initialization_audit_v2/learned_baseline_per_sample.csv"
                    ),
                )
            },
        )
        _json(
            root / "execution_hashes.json",
            {
                "note": "Executed audit snapshot retained before type-only cleanup; separate from numerical-input hashes.",
                "executed_audit": _sha256(archive),
                "current_audit": _sha256(executed_cli),
                "block_system_helper": _sha256(
                    Path("cli/audit_coupling_solver_pilot.py")
                ),
                "native_timer": _sha256(Path("cli/time_right_scaled_native.py")),
                "reporter": _sha256(Path(__file__)),
            },
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    RightScalingReport().run(parser.parse_args().outdir)
