"""Coverage checks and source-backed presentation of the initialization audit."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from greenonet.complex_frozen_tangent_csv import _write_csv
from greenonet.source_initialization_metrics import match_accuracy, summarize


def verify_coverage(
    rows: list[dict[str, Any]],
    timings: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    stage: str,
    max_k: int,
    repeats: int,
) -> dict[str, Any]:
    expected: set[tuple[str, str, int, int]] = set()
    shared = {}
    for run in runs:
        shared[run["fingerprint"]] = run
        if stage in {"all", "initialization"}:
            expected.update(
                (run["run_id"], "learned", run["native_k"], i)
                for i in range(run["sample_count"])
            )
    for fingerprint, run in shared.items():
        if stage in {"all", "reference"}:
            expected.update(
                (fingerprint, condition, 0, i)
                for condition in ("reference_raw", "reference_balanced")
                for i in range(run["sample_count"])
            )
        if stage in {"all", "initialization"}:
            expected.update(
                (fingerprint, "equal_split", k, i)
                for k in range(max_k + 1)
                for i in range(run["sample_count"])
            )
    actual = [
        (
            r["run_id"] if r["condition"] == "learned" else r["fingerprint"],
            r["condition"],
            r["evaluation_k"],
            r["sample_id"],
        )
        for r in rows
    ]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated sample/K coverage.")
    expected_timing: set[tuple[str, str, int, int]] = set()
    if stage in {"all", "initialization"}:
        summaries = summarize(rows)
        targets = [r for r in summaries if r["condition"] == "learned"]
        recalculated = [
            match_accuracy(
                t,
                [
                    r
                    for r in summaries
                    if r["condition"] == "equal_split"
                    and r["fingerprint"] == t["fingerprint"]
                ],
            )
            for t in targets
        ]
        if sorted(matches, key=lambda r: r["run_id"]) != sorted(
            recalculated, key=lambda r: r["run_id"]
        ):
            raise ValueError(
                "Stored matches do not reproduce from complete sample rows."
            )
        for run in runs:
            expected_timing.update(
                (run["run_id"], "learned", run["native_k"], repeat)
                for repeat in range(repeats)
            )
        for fingerprint, run in shared.items():
            ks = {run["native_k"], max_k}
            ks.update(
                m[key]
                for m in matches
                if m["fingerprint"] == fingerprint
                for key in ("mean_k", "mean_p95_k")
                if m[key] is not None
            )
            expected_timing.update(
                (fingerprint, "equal_split", k, repeat)
                for k in ks
                for repeat in range(repeats)
            )
    actual_timing = [
        (
            r["run_id"] if r["condition"] == "learned" else r["fingerprint"],
            r["condition"],
            r["evaluation_k"],
            r["repeat"],
        )
        for r in timings
    ]
    if (
        len(set(actual_timing)) != len(actual_timing)
        or set(actual_timing) != expected_timing
    ):
        raise ValueError("Incomplete or duplicated independent timing coverage.")
    return dict(
        expected_sample_rows=len(expected),
        actual_sample_rows=len(actual),
        timing_rows=len(actual_timing),
        unique_operator_groups=len(shared),
        learned_runs=sum(r["condition"] == "learned" for r in summarize(rows)),
        matches_recomputed=stage in {"all", "initialization"},
    )


class SourceInitializationReport:
    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir

    def write(
        self,
        rows: list[dict[str, Any]],
        timings: list[dict[str, Any]],
        matches: list[dict[str, Any]],
    ) -> None:
        summaries = summarize(rows)
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in timings:
            groups[
                (
                    row["example"],
                    row["fingerprint"],
                    row["run_id"],
                    row["condition"],
                    row["evaluation_k"],
                )
            ].append(row)
        timing_summary = []
        for keys, values in groups.items():
            record = dict(
                zip(
                    ("example", "fingerprint", "run_id", "condition", "evaluation_k"),
                    keys,
                    strict=True,
                )
            )
            seconds = np.array([v["seconds"] for v in values])
            record.update(
                seconds_median=float(np.median(seconds)),
                seconds_min=float(seconds.min()),
                seconds_max=float(seconds.max()),
                repeats=len(values),
                seconds_per_sample_median=float(np.median(seconds))
                / values[0]["sample_count"],
                peak_allocated_mib=max(v["peak_allocated_mib"] for v in values),
                peak_reserved_mib=max(v["peak_reserved_mib"] for v in values),
            )
            timing_summary.append(record)
        _write_csv(self.outdir / "timing_summary.csv", timing_summary)
        text = [
            "# Reference source and learned initialization audit",
            "",
            "All weights are frozen. Native learned K is fixed; only physical phi=psi=f/2 is extended. The latter still uses GreenNet.",
            "CSV errors are fractions; tables and error axes below are percentages. P95 uses linear interpolation across test samples.",
            "",
        ]
        for example in sorted({r["example"] for r in rows}):
            selected = [r for r in summaries if r["example"] == example]
            text.extend(
                [
                    f"## {example}",
                    "",
                    "### Reference reconstruction",
                    "",
                    "| Condition | Weak mean (%) | P95 (%) | Maximum (%) | Mean balance relative norm |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            references = [
                r for r in selected if r["condition"].startswith("reference_")
            ]
            for r in references:
                text.append(
                    f"| {r['condition']} | {100 * r['rel_sol_mean']:.6f} | {100 * r['rel_sol_p95']:.6f} | {100 * r['rel_sol_max']:.6f} | {r.get('balance_relative_mean', 'NA')} |"
                )
            curve = sorted(
                [r for r in selected if r["condition"] == "equal_split"],
                key=lambda r: r["evaluation_k"],
            )
            learned = [r for r in selected if r["condition"] == "learned"]
            figure = go.Figure()
            for metric in ("mean", "p95", "max"):
                figure.add_scatter(
                    x=[r["evaluation_k"] for r in curve],
                    y=[100 * r[f"rel_sol_{metric}"] for r in curve],
                    name=f"f/2 {metric}",
                )
            text.extend(
                [
                    "",
                    "### Matched accuracy",
                    "",
                    "| Run | Native K | Learned mean (%) | P95 (%) | Maximum (%) | First mean K | First mean+P95 K |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for m in [m for m in matches if m["example"] == example]:
                text.append(
                    f"| {m['run_id']} | {m['training_k']} | {100 * m['learned_mean']:.6f} | {100 * m['learned_p95']:.6f} | {100 * m['learned_max']:.6f} | {m['mean_k'] if m['mean_k'] is not None else 'not reached'} | {m['mean_p95_k'] if m['mean_p95_k'] is not None else 'not reached'} |"
                )
                figure.add_scatter(
                    x=[0, curve[-1]["evaluation_k"]],
                    y=[100 * m["learned_mean"]] * 2,
                    mode="lines",
                    line=dict(dash="dot", width=1.5),
                    name=f"Learned seed {m['seed']} (native K{m['training_k']})",
                )
                for key in ("mean", "mean_p95"):
                    if m[f"{key}_k"] is not None:
                        figure.add_scatter(
                            x=[m[f"{key}_k"]],
                            y=[100 * m[f"{key}_mean"]],
                            mode="markers",
                            name=f"Seed {m['seed']} {key} match K{m[f'{key}_k']}",
                        )
            figure.update_layout(
                template="plotly_white",
                title=f"{example}: frozen initialization",
                xaxis_title="Correction dimension K",
                yaxis_title="Weak relative L2 error (%) [log scale]",
                yaxis_type="log",
            )
            figure.write_html(
                self.outdir / f"{example}_accuracy_k.html", include_plotlyjs=True
            )
            time_figure = go.Figure()
            text.extend(
                [
                    "",
                    "### Online prediction timing",
                    "",
                    "| Condition | Run | K | Full-test median (s) | Range (s) |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for t in [t for t in timing_summary if t["example"] == example]:
                candidates = [
                    r
                    for r in learned + curve
                    if r["condition"] == t["condition"]
                    and r["evaluation_k"] == t["evaluation_k"]
                    and r["fingerprint"] == t["fingerprint"]
                    and (t["condition"] == "equal_split" or r["run_id"] == t["run_id"])
                ]
                if len(candidates) != 1:
                    raise ValueError("Timing must map to exactly one accuracy summary.")
                time_figure.add_scatter(
                    x=[t["seconds_median"]],
                    y=[100 * candidates[0]["rel_sol_mean"]],
                    mode="markers",
                    name=f"{t['run_id']} {t['condition']} K{t['evaluation_k']}",
                )
                text.append(
                    f"| {t['condition']} | {t['run_id']} | {t['evaluation_k']} | {t['seconds_median']:.4f} | {t['seconds_min']:.4f} - {t['seconds_max']:.4f} |"
                )
            time_figure.update_layout(
                template="plotly_white",
                title=example,
                xaxis_title="Full-test prediction time (s)",
                yaxis_title="Mean weak relative L2 error (%) [log scale]",
                yaxis_type="log",
            )
            time_figure.write_html(
                self.outdir / f"{example}_accuracy_time.html", include_plotlyjs=True
            )
            ref_figure = go.Figure()
            for metric, label in (
                ("rel_u_phi_mean", "u_phi"),
                ("rel_u_psi_mean", "u_psi"),
                ("rel_sol_equal_mean_mean", "Equal mean"),
                ("rel_sol_mean", "Weak"),
            ):
                ref_figure.add_bar(
                    x=[r["condition"] for r in references],
                    y=[100 * r[metric] for r in references],
                    name=label,
                )
            ref_figure.update_layout(
                template="plotly_white",
                title=example,
                yaxis_title="Mean relative L2 error (%)",
                barmode="group",
            )
            ref_figure.write_html(
                self.outdir / f"{example}_reference.html", include_plotlyjs=True
            )
        text.extend(
            [
                "",
                "## Interpretation boundaries",
                "",
                "The reference sources are projected directional derivatives of a finite-element solution; satisfying the weak PDE does not ensure an exact physical source balance at axial points. Raw imbalance is a diagnostic, not a failed run.",
                "The symmetric pair correction has weighted norm ||phi_ref+psi_ref-f||/sqrt(2). Under matching source definitions this is a lower bound on source-pair error, not on solution error. Neither raw nor balanced reference-source reconstruction is a GreenNet error floor.",
                "First-crossing K is selected retrospectively using this test set. It is not a reference-free deployment stopping rule; later worsening is retained in matches.csv and the complete curves.",
                "A smaller correction dimension does not establish an online speedup. Use measured matched-accuracy time, including neural inference, physical scaling, balance projection, tangent correction and weak reconstruction. Loading, data transfer, context preparation, metrics and training are excluded. Training cost is disclosed but total-cost superiority is not claimed.",
                "One equal-split curve per operator fingerprint is shared across seed-specific targets; these repetitions are not independent samples. No average across different examples is reported.",
                "Timing uses three full-test warmups and five measurements by default, synchronized CUDA calls and reversing condition order. See request.json for actual settings, timing.csv for raw repeats and memory, preflight.json for generation contracts, and verification.json for coverage and unchanged hashes.",
                "",
            ]
        )
        (self.outdir / "report.md").write_text("\n".join(text))
