"""Aggregate paired fixed-space objective audit; never select objectives by labels."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from greenonet.complex_frozen_tangent_csv import _write_csv


class FixedSpaceReport:
    metrics = (
        "rel_sol",
        "rel_equal",
        "rel_phi",
        "rel_psi",
        "J",
        "weak_indicator",
        "C",
        "W",
        "joint",
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
        if len(rows) != 14400:
            raise ValueError("Expected the complete 14400-row experiment")
        groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
        lookup = {}
        for row in rows:
            key = (
                row["example"],
                row["seed"],
                row["sample_id"],
                row["k"],
                row["objective"],
            )
            if key in lookup:
                raise ValueError(f"Duplicate sample key: {key}")
            lookup[key] = row
            for seed in (row["seed"], "pooled"):
                groups.setdefault(
                    (row["example"], seed, row["k"], row["objective"]), []
                ).append(row)
        aggregate = []
        paired = []
        for (example, seed, k, objective), selected in sorted(groups.items()):
            summary: dict[str, str | int | float] = dict(
                example=example,
                seed=seed,
                k=k,
                objective=objective,
                count=len(selected),
            )
            for metric in self.metrics:
                values = np.array([float(r[metric]) for r in selected])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {metric}")
                summary.update(
                    {
                        f"{metric}_{stat}": float(value)
                        for stat, value in (
                            ("mean", values.mean()),
                            ("median", np.median(values)),
                            ("p95", np.quantile(values, 0.95)),
                            ("max", values.max()),
                        )
                    }
                )
                if objective != "consistency":
                    base = np.array(
                        [
                            float(
                                lookup[
                                    (
                                        example,
                                        r["seed"],
                                        r["sample_id"],
                                        k,
                                        "consistency",
                                    )
                                ][metric]
                            )
                            for r in selected
                        ]
                    )
                    paired.append(
                        dict(
                            example=example,
                            seed=seed,
                            k=k,
                            objective=objective,
                            metric=metric,
                            count=len(values),
                            improved=int((values < base - 1e-12).sum()),
                            worsened=int((values > base + 1e-12).sum()),
                            mean_delta=float((values - base).mean()),
                            ratio_of_means=float(values.mean() / base.mean())
                            if base.mean()
                            else 1.0,
                        )
                    )
            aggregate.append(summary)
        _write_csv(self.out / "summary.csv", aggregate)
        _write_csv(self.out / "paired.csv", paired)
        self.logger.info(
            "Aggregated %d rows into %d summaries", len(rows), len(aggregate)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    FixedSpaceReport(parser.parse_args().outdir).run()
