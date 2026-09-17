"""Aggregate weak Green diagnostics without pooling seed-dependent tails."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_weak_closure import build_directional_weak_context


class WeakResidualReport:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self) -> None:
        with (self.root / "per_sample.csv").open() as f:
            rows = list(csv.DictReader(f))
        groups: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(
            list
        )
        for r in rows:
            groups[r["example"], r["method"], int(r["seed"]), int(r["k"])].append(r)
        keys = [
            key
            for key in rows[0]
            if key not in {"example", "method", "seed", "k", "sample_id", "run_id"}
        ]
        summaries: list[dict[str, Any]] = []
        for (example, method, seed, k), values in sorted(groups.items()):
            assert len(values) == (100 if example == "unit_square" else 50)
            assert len({r["sample_id"] for r in values}) == len(values)
            row: dict[str, Any] = dict(example=example, method=method, seed=seed, k=k)
            for key in keys:
                x = np.array([float(r[key]) for r in values])
                for stat, value in (
                    ("mean", x.mean()),
                    ("p95", np.quantile(x, 0.95)),
                    ("max", x.max()),
                ):
                    row[f"{key}_{stat}"] = float(value)
            summaries.append(row)
        _write_csv(self.root / "per_seed_summary.csv", summaries)
        agg: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for r in summaries:
            agg[r["example"], r["method"], r["k"]].append(r)
        curves: list[dict[str, Any]] = []
        for (example, method, k), values in sorted(agg.items()):
            assert len(values) == 4
            row = dict(example=example, method=method, k=k)
            for key in keys:
                for stat in ("mean", "p95", "max"):
                    name = f"{key}_{stat}"
                    row[name] = float(np.mean([r[name] for r in values]))
            curves.append(row)
        _write_csv(self.root / "curve.csv", curves)
        paired = []
        for example in ("unit_square", "disk"):
            for k in range(65):
                for a, b in (("LL", "LR"), ("LR", "RR"), ("LL", "RR")):
                    for seed in range(4):
                        left = {r["sample_id"]: r for r in groups[example, a, seed, k]}
                        right = {r["sample_id"]: r for r in groups[example, b, seed, k]}
                        row = dict(
                            example=example,
                            k=k,
                            seed=seed,
                            before=a,
                            after=b,
                            samples=len(left),
                        )
                        for key in (
                            "own",
                            "phi_full",
                            "psi_full",
                            "weak_full",
                            "rel_sol",
                            "J",
                        ):
                            row[f"{key}_decreased"] = sum(
                                float(right[s][key]) < float(left[s][key]) for s in left
                            )
                        paired.append(row)
        _write_csv(self.root / "paired.csv", paired)
        trends = []
        for example in ("unit_square", "disk"):
            for method in ("LL", "LR", "RR"):
                selected = sorted(
                    [
                        r
                        for r in curves
                        if r["example"] == example and r["method"] == method
                    ],
                    key=lambda r: r["k"],
                )
                for key in (
                    "J",
                    "rel_sol",
                    "own",
                    "weak_full",
                    "equal_full",
                    "phi_full",
                    "psi_full",
                ):
                    x = np.array([r[f"{key}_mean"] for r in selected])
                    trends.append(
                        dict(
                            example=example,
                            method=method,
                            metric=key,
                            minimum_k=int(x.argmin()),
                            minimum=float(x.min()),
                            increases=[
                                int(i + 1) for i in np.flatnonzero(np.diff(x) > 0)
                            ],
                        )
                    )
        _json(self.root / "trends.json", trends)
        correlations = []
        for (example, method, seed, k), values in sorted(groups.items()):
            if k not in (0, 2, 16, 32, 64):
                continue
            target = np.array([float(r["rel_sol"]) for r in values])
            for metric in ("J", "own", "cross", "weak_full", "equal_full"):
                x = np.array([float(r[metric]) for r in values])
                ranks = []
                for array in (x, target):
                    _, inverse, counts = np.unique(
                        array, return_inverse=True, return_counts=True
                    )
                    average = np.cumsum(counts) - 0.5 * (counts - 1)
                    ranks.append(average[inverse])
                correlation = (
                    float(np.corrcoef(ranks)[0, 1])
                    if all(np.std(r) > 0 for r in ranks)
                    else None
                )
                correlations.append(
                    dict(
                        example=example,
                        method=method,
                        seed=seed,
                        k=k,
                        metric=metric,
                        spearman=correlation,
                    )
                )
        _write_csv(self.root / "sample_correlations.csv", correlations)
        boundary_rows = []
        specifications = json.loads((self.root / "preflight.json").read_text())
        for spec in specifications:
            if spec["example"] not in {"unit_square", "disk"}:
                continue
            geometry = load_complex_geometry(Path(spec["paths"]["geometry"]))
            context = build_directional_weak_context(
                geometry,
                load_coefficient_functions(Path(spec["paths"]["coefficients"])),
            )
            masks = {}
            for name, axis in (("phi_x", context.x), ("psi_y", context.y)):
                index = axis.element_valid_index.numpy()
                touching = index[(index < 0).any(1)]
                mask = np.zeros(geometry.num_points, dtype=bool)
                mask[touching[touching >= 0]] = True
                masks[name] = (mask, axis.nodal_mass.numpy())
            for path in sorted(
                (self.root / "raw").glob(f"{spec['run_id']}_batch*.npz")
            ):
                with np.load(path) as data:
                    for method in ("LL", "LR", "RR"):
                        total = np.zeros(len(data["sample_ids"]))
                        boundary = total.copy()
                        for name, (mask, mass) in masks.items():
                            energy = data[f"{method}_k64_{name}"] ** 2 / mass
                            total += energy.sum(1)
                            boundary += energy[:, mask].sum(1)
                        for i, sid in enumerate(data["sample_ids"]):
                            boundary_rows.append(
                                dict(
                                    example=spec["example"],
                                    run_id=spec["run_id"],
                                    method=method,
                                    sample_id=int(sid),
                                    k=64,
                                    own_boundary_energy_fraction=float(
                                        boundary[i] / total[i]
                                    ),
                                )
                            )
        _write_csv(self.root / "boundary_contribution.csv", boundary_rows)
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[
                f"{e}: {m}"
                for m in (
                    "solution relative error",
                    "own-axis indicator",
                    "final weak indicator",
                    "consistency J (operator-dependent)",
                )
                for e in ("Unit square", "Disk")
            ],
        )
        colors = {"LL": "#a44444", "LR": "#94751c", "RR": "#137d96"}
        for j, example in enumerate(("unit_square", "disk"), 1):
            for i, metric in enumerate(("rel_sol", "own", "weak_full", "J"), 1):
                for method in ("LL", "LR", "RR"):
                    selected = [
                        r
                        for r in curves
                        if r["example"] == example and r["method"] == method
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=[r["k"] for r in selected],
                            y=[r[f"{metric}_mean"] for r in selected],
                            name=method,
                            legendgroup=method,
                            showlegend=i == j == 1,
                            line=dict(color=colors[method]),
                        ),
                        row=i,
                        col=j,
                    )
                fig.update_yaxes(type="log", row=i, col=j)
                fig.update_xaxes(title_text="K", row=i, col=j)
        fig.update_layout(
            height=1300,
            template="plotly_white",
            title="Mean of four seed means; indicators are not solution errors",
        )
        fig.write_html(self.root / "curves.html", include_plotlyjs=True)
        lines = [
            "# Selected metrics",
            "",
            "All values are fractions, not percent. Tails are averaged per seed.",
            "",
            "|Example|K|Method|Solution|Own|Phi full|Psi full|Weak full|J|",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in curves:
            if r["k"] in (0, 2, 8, 16, 32, 48, 64):
                lines.append(
                    f"|{r['example']}|{r['k']}|{r['method']}|"
                    + "|".join(
                        f"{r[f'{m}_mean']:.8g}"
                        for m in (
                            "rel_sol",
                            "own",
                            "phi_full",
                            "psi_full",
                            "weak_full",
                            "J",
                        )
                    )
                    + "|"
                )
        (self.root / "tables.md").write_text("\n".join(lines) + "\n")
        _json(
            self.root / "report_verification.json",
            dict(
                rows=len(rows),
                groups=len(groups),
                curves=len(curves),
                full_coverage=len(groups) == 1560,
                input_sha256=_sha256(self.root / "per_sample.csv"),
            ),
        )
        print("Report written:", len(rows), "rows,", len(curves), "curve points")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=Path("docs/analysis/weak_green_residual_audit")
    )
    WeakResidualReport(parser.parse_args().outdir).run()
