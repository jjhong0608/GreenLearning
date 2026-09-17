"""Report all predeclared mixed-energy weights without test-set selection."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.logging import RichHandler

from greenonet.complex_frozen_tangent_csv import _write_csv


class MixedEnergyReport:
    metrics = (
        "rel_sol",
        "rel_equal",
        "rel_phi",
        "rel_psi",
        "C",
        "E",
        "J",
        "energy",
        "correction_norm",
    )

    def __init__(self, out: Path) -> None:
        self.out = out
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(show_path=True, omit_repeated_times=False),
            logging.FileHandler(out / "report.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)

    def run(self) -> None:
        with (self.out / "per_sample.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 48000:
            raise ValueError("Incomplete experiment")
        groups: dict[tuple[str, str, str, int, str], list[dict[str, str]]] = {}
        lookup = {}
        for r in rows:
            key = (
                r["example"],
                r["initialization"],
                r["seed"],
                r["sample_id"],
                r["k"],
                r["objective"],
            )
            if key in lookup:
                raise ValueError(f"Duplicate key {key}")
            lookup[key] = r
            for seed in (r["seed"], "pooled"):
                groups.setdefault(
                    (
                        r["example"],
                        r["initialization"],
                        seed,
                        int(r["k"]),
                        r["objective"],
                    ),
                    [],
                ).append(r)
        summary: list[dict[str, Any]] = []
        paired = []
        for (example, init, seed, k, method), selected in sorted(groups.items()):
            row: dict[str, Any] = dict(
                example=example,
                initialization=init,
                seed=seed,
                k=k,
                objective=method,
                count=len(selected),
            )
            for metric in self.metrics:
                vals = np.array([float(r[metric]) for r in selected])
                if not np.isfinite(vals).all():
                    raise ValueError(f"Nonfinite {metric}")
                for name, value in (
                    ("mean", vals.mean()),
                    ("median", np.median(vals)),
                    ("p95", np.quantile(vals, 0.95)),
                    ("max", vals.max()),
                ):
                    row[f"{metric}_{name}"] = float(value)
                if method != "l2":
                    base = np.array(
                        [
                            float(
                                lookup[
                                    (
                                        example,
                                        init,
                                        r["seed"],
                                        r["sample_id"],
                                        str(k),
                                        "l2",
                                    )
                                ][metric]
                            )
                            for r in selected
                        ]
                    )
                    paired.append(
                        dict(
                            example=example,
                            initialization=init,
                            seed=seed,
                            k=k,
                            objective=method,
                            metric=metric,
                            count=len(vals),
                            improved=int((vals < base - 1e-12).sum()),
                            worsened=int((vals > base + 1e-12).sum()),
                            mean_delta=float((vals - base).mean()),
                            ratio_of_means=float(vals.mean() / base.mean())
                            if base.mean()
                            else 1.0,
                        )
                    )
            summary.append(row)
        _write_csv(self.out / "summary.csv", summary)
        _write_csv(self.out / "paired.csv", paired)
        pooled = {
            (r["example"], r["initialization"], r["k"], r["objective"]): r
            for r in summary
            if r["seed"] == "pooled"
        }
        methods = ("l2", "mix_0.01", "mix_0.1", "mix_1", "energy")
        lines = [
            "# All predeclared results",
            "",
            "Errors are relative L2 percentages. No weight is selected.",
            "",
            "## Mean production weak solution error",
            "",
            "|Problem|Initialization|K|L2|lambda=.01|lambda=.1|lambda=1|Energy|",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for example in ("unit_square", "disk"):
            for init in ("learned", "equal_split"):
                for k in (0, 1, 2, 4, 8, 16, 32, 64):
                    values = [
                        f"{100 * pooled[example, init, k, m]['rel_sol_mean']:.7f}"
                        for m in methods
                    ]
                    lines.append("|" + "|".join([example, init, str(k), *values]) + "|")
        lines += [
            "",
            "## K64 pooled tails and equal-mean candidate",
            "",
            "Pooled seeds repeat the same sources; these are not independent source samples.",
            "",
            "|Problem|Initialization|Objective|Weak P95 (%)|Weak max (%)|Equal mean (%)|",
            "|---|---|---|---:|---:|---:|",
        ]
        for (example, init, k, method), r in sorted(pooled.items()):
            if k == 64:
                values = [
                    f"{100 * r[key]:.7f}"
                    for key in ("rel_sol_p95", "rel_sol_max", "rel_equal_mean")
                ]
                lines.append("|" + "|".join([example, init, method, *values]) + "|")
        (self.out / "tables.md").write_text("\n".join(lines) + "\n")
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Square learned", "Square f/2", "Disk learned", "Disk f/2"),
        )
        colors = {
            "l2": "#156c9c",
            "mix_0.01": "#309447",
            "mix_0.1": "#d69916",
            "mix_1": "#bd3d62",
            "energy": "#60549b",
        }
        for ri, example in enumerate(("unit_square", "disk"), 1):
            for ci, init in enumerate(("learned", "equal_split"), 1):
                for method in ("l2", "mix_0.01", "mix_0.1", "mix_1", "energy"):
                    curve = [
                        r
                        for r in summary
                        if r["example"] == example
                        and r["initialization"] == init
                        and r["seed"] == "pooled"
                        and r["objective"] == method
                    ]
                    fig.add_trace(
                        go.Scatter(
                            x=[r["k"] for r in curve],
                            y=[100 * r["rel_sol_mean"] for r in curve],
                            name=method,
                            legendgroup=method,
                            line=dict(color=colors[method]),
                            showlegend=ri == ci == 1,
                            mode="lines+markers",
                        ),
                        row=ri,
                        col=ci,
                    )
        fig.update_yaxes(type="log", title_text="Mean solution error (%)")
        fig.update_xaxes(title_text="K")
        fig.update_layout(
            template="plotly_white",
            height=800,
            title="Fixed-space objective comparison; no test-selected weight",
        )
        if len(fig.data) != 20 or any(len(trace.x) != 8 for trace in fig.data):
            raise ValueError("Incomplete comparison curves")
        fig.write_html(self.out / "curves.html", include_plotlyjs=True)
        self.logger.info(
            "Verified %d rows; generated %d summaries", len(rows), len(summary)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    MixedEnergyReport(parser.parse_args().outdir).run()
