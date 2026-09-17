"""Validate and summarize the retrospective GMRES accuracy-budget pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cli.audit_gmres_accuracy_budget import first_match
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv


def read(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


class BudgetReport:
    def run(self, root: Path) -> None:
        hashes = json.loads((root / "execution_inputs.json").read_text())
        for path, digest in hashes.items():
            if _sha256(Path(path)) != digest:
                raise ValueError(f"Changed input: {path}")
        rows = read(root / "per_sample.csv")
        raw_curves = read(root / "accuracy_curve.csv")
        matches = read(root / "matches.csv")
        timing = read(root / "timing.csv")
        if len(rows) != 1070 or len(raw_curves) != 214 or len(matches) != 2:
            raise ValueError("Incomplete two-example coverage")
        old = read(Path("docs/analysis/right_scaled_solver_pilot/per_sample.csv"))
        summary: list[dict[str, Any]] = []
        time_curve: list[dict[str, Any]] = []
        for match in matches:
            example = match["example"]
            curve: list[dict[str, Any]] = []
            for raw in [r for r in raw_curves if r["example"] == example]:
                budget = int(raw["budget"])
                selected = [
                    r
                    for r in rows
                    if r["example"] == example and int(r["budget"]) == budget
                ]
                assert len(selected) == 5
                errors = np.array([float(r["rel_sol"]) for r in selected])
                assert max(float(r["balance_max"]) for r in selected) < 1e-11
                assert max(int(r["iterations"]) for r in selected) <= budget
                np.testing.assert_allclose(
                    [float(raw[k]) for k in ("mean", "p95", "maximum")],
                    [errors.mean(), np.quantile(errors, 0.95), errors.max()],
                    rtol=1e-13,
                )
                curve.append(
                    dict(
                        budget=budget,
                        mean=float(raw["mean"]),
                        p95=float(raw["p95"]),
                        maximum=float(raw["maximum"]),
                    )
                )
                if budget == 700:
                    previous = [
                        float(r["rel_sol"])
                        for r in old
                        if r["example"] == example and r["condition"] == "gmres_scaled"
                    ]
                    np.testing.assert_allclose(errors, previous, rtol=1e-8, atol=1e-12)
            target_mean, target_p95 = (
                float(match["target_mean"]),
                float(match["target_p95"]),
            )
            native_times = [
                float(r["seconds"])
                for r in timing
                if r["example"] == example and int(r["budget"]) == -1
            ]
            assert len(native_times) == 5
            native_seconds = float(np.median(native_times))
            for budget in sorted(
                {int(r["budget"]) for r in timing if r["example"] == example}
            ):
                selected_times = [
                    r
                    for r in timing
                    if r["example"] == example and int(r["budget"]) == budget
                ]
                assert sorted(int(r["repeat"]) for r in selected_times) == list(
                    range(5)
                )
                seconds = [float(r["seconds"]) for r in selected_times]
                item = (
                    dict(mean=target_mean, p95=target_p95)
                    if budget == -1
                    else next(r for r in curve if r["budget"] == budget)
                )
                time_curve.append(
                    dict(
                        example=example,
                        budget=budget,
                        mean_percent=100 * item["mean"],
                        p95_percent=100 * item["p95"],
                        median_seconds=float(np.median(seconds)),
                        min_seconds=min(seconds),
                        max_seconds=max(seconds),
                    )
                )
            for require_tail, name in ((False, "mean"), (True, "mean+p95")):
                matched_budget = first_match(
                    curve, target_mean, target_p95, require_tail
                )
                recorded = match["mean_p95_budget" if require_tail else "mean_budget"]
                assert (None if not recorded else int(recorded)) == matched_budget
                entry: dict[str, Any] = dict(
                    example=example,
                    criterion=name,
                    target_mean_percent=100 * target_mean,
                    target_p95_percent=100 * target_p95,
                    native_seconds=native_seconds,
                    budget=matched_budget,
                    reached=matched_budget is not None,
                )
                if matched_budget is not None:
                    accuracy = next(r for r in curve if r["budget"] == matched_budget)
                    cost = next(
                        r
                        for r in time_curve
                        if r["example"] == example and r["budget"] == matched_budget
                    )
                    entry.update(
                        gmres_mean_percent=100 * accuracy["mean"],
                        gmres_p95_percent=100 * accuracy["p95"],
                        gmres_max_percent=100 * accuracy["maximum"],
                        gmres_seconds=cost["median_seconds"],
                        gmres_over_native_time=cost["median_seconds"] / native_seconds,
                    )
                summary.append(entry)
        _write_csv(root / "matched_accuracy_summary.csv", summary)
        _write_csv(root / "accuracy_time_curve.csv", time_curve)
        self.figure(root, time_curve)
        _json(
            root / "report_verification.json",
            dict(
                input_hashes_verified=len(hashes),
                per_sample_rows=len(rows),
                curve_rows=len(raw_curves),
                timing_rows=len(timing),
                matched_rows=len(summary),
                baseline_700_reproduced=True,
                selection="retrospective reference-selected crossing, not deployment stopping",
            ),
        )
        _json(
            root / "report_inputs.json",
            {
                str(p): _sha256(p)
                for p in [
                    Path(__file__),
                    root / "per_sample.csv",
                    root / "accuracy_curve.csv",
                    root / "timing.csv",
                    root / "matches.csv",
                ]
            },
        )
        print(json.dumps(summary, indent=2))

    def figure(self, root: Path, rows: list[dict[str, Any]]) -> None:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Unit square: mean",
                "Unit square: P95",
                "Disk: mean",
                "Disk: P95",
            ),
            vertical_spacing=0.18,
        )
        for row_index, example in enumerate(("unit_square", "disk"), 1):
            selected = [r for r in rows if r["example"] == example]
            for col, metric in enumerate(("mean_percent", "p95_percent"), 1):
                for native, label, color, symbol in (
                    (False, "Scaled GMRES", "#087e8b", "circle"),
                    (True, "Learned native K2", "#b83d62", "diamond"),
                ):
                    values = [r for r in selected if (r["budget"] == -1) == native]
                    fig.add_trace(
                        go.Scatter(
                            x=[r["median_seconds"] for r in values],
                            y=[r[metric] for r in values],
                            mode="markers",
                            name=label,
                            legendgroup=label,
                            showlegend=row_index == 1 and col == 1,
                            marker=dict(
                                color=color, symbol=symbol, size=10 if native else 7
                            ),
                            text=[
                                "native K2" if native else f"budget={r['budget']}"
                                for r in values
                            ],
                            hovertemplate="%{text}<br>%{x:.5f}s / batch5<br>%{y:.6f}%<extra></extra>",
                        ),
                        row=row_index,
                        col=col,
                    )
                fig.update_xaxes(
                    type="log",
                    title_text="Seconds / batch5 (log)",
                    tickvals=[0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 3],
                    ticktext=["0.03", "0.05", "0.1", "0.2", "0.5", "1", "2", "3"],
                    range=[np.log10(0.025), np.log10(3)],
                    row=row_index,
                    col=col,
                )
                fig.update_yaxes(
                    type="log",
                    title_text="Relative solution error % (log)",
                    tickvals=[0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                    ticktext=["0.03", "0.1", "0.3", "1", "3", "10", "30", "100"],
                    range=[np.log10(0.03), np.log10(120)],
                    row=row_index,
                    col=col,
                )
        fig.update_layout(
            template="plotly_white",
            width=1200,
            height=850,
            title=dict(
                text="Accuracy versus online time<br><sup>GPU1, seed0, first5 test sources; independent median of 5 repeats; not full-test</sup>",
                x=0.03,
            ),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=160, l=85, r=30, b=65),
            font=dict(size=13),
        )
        fig.write_image(str(root / "accuracy_time.png"), scale=1.5)
        fig.write_json(str(root / "accuracy_time.plotly.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    BudgetReport().run(parser.parse_args().outdir)
