"""Scalar-only, sample-paired tables for frozen tangent sweeps."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

CsvRow = dict[str, Any]
ERROR_METRICS = (
    "response_cost",
    "rel_sol",
    "rel_sol_equal_mean",
    "rel_u_phi",
    "rel_u_psi",
    "rel_flux",
    "canonical_bulk_energy",
    "canonical_boundary_energy",
    "loss_energy_optimized",
)
CONTEXT_COLUMNS = (
    "run_id",
    "seed",
    "training_k",
    "evaluation_k",
    "baseline_k",
    "global_reach",
    "lower_5pct_reach",
    "minimum_reach",
    "full_reach_k",
)


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def add_paired_metrics(row: CsvRow, baseline: CsvRow) -> None:
    for metric in ERROR_METRICS:
        if metric not in row:
            continue
        row[f"{metric}_baseline"] = baseline[metric]
        value = ratio(row[metric], baseline[metric])
        row[f"{metric}_ratio_baseline"] = value
        row[f"{metric}_gain_baseline"] = None if value is None else 1.0 - value
    value = ratio(row["response_cost"], row["response_cost_previous"])
    row["response_cost_ratio_previous"] = value
    row["response_cost_gain_previous"] = None if value is None else 1.0 - value


def aggregate_samples(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[tuple[str, int], list[CsvRow]] = defaultdict(list)
    for row in rows:
        groups[(row["run_id"], row["evaluation_k"])].append(row)
    result: list[CsvRow] = []
    for group in groups.values():
        first = group[0]
        output = {key: first[key] for key in CONTEXT_COLUMNS if key in first}
        output["sample_count"] = len(group)
        excluded = {*CONTEXT_COLUMNS, "sample_id", "file_stem"}
        for key in sorted(set().union(*(row.keys() for row in group)) - excluded):
            values = [row.get(key) for row in group]
            present = [value for value in values if value is not None]
            if present and not all(
                isinstance(value, (int, float)) for value in present
            ):
                continue
            data = np.asarray(present, dtype=np.float64)
            if not np.isfinite(data).all():
                raise ValueError(f"Non-finite scalar metric: {key}")
            output[f"{key}_valid_count"] = len(present)
            for suffix in ("mean", "median", "p95", "max"):
                output[f"{key}_{suffix}"] = None
            if data.size:
                output.update(
                    {
                        f"{key}_mean": float(data.mean()),
                        f"{key}_median": float(np.median(data)),
                        f"{key}_p95": float(np.quantile(data, 0.95)),
                        f"{key}_max": float(data.max()),
                    }
                )
        for metric in ERROR_METRICS:
            if f"{metric}_baseline_mean" not in output:
                continue
            value = ratio(output[f"{metric}_mean"], output[f"{metric}_baseline_mean"])
            output[f"{metric}_ratio_baseline"] = value
            output[f"{metric}_gain_baseline"] = None if value is None else 1.0 - value
            output[f"{metric}_improved_fraction"] = float(
                np.mean([row[metric] < row[f"{metric}_baseline"] for row in group])
            )
        value = ratio(
            output["response_cost_mean"], output["response_cost_previous_mean"]
        )
        output["response_cost_ratio_previous"] = value
        output["response_cost_gain_previous"] = None if value is None else 1.0 - value
        result.append(output)
    return result


def aggregate_seeds(rows: Sequence[CsvRow]) -> list[CsvRow]:
    groups: dict[int, list[CsvRow]] = defaultdict(list)
    for row in rows:
        groups[row["evaluation_k"]].append(row)
    result: list[CsvRow] = []
    for k, group in sorted(groups.items()):
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError(
                "Duplicate seeds cannot be aggregated as independent runs."
            )
        output: CsvRow = {"evaluation_k": k, "num_seeds": len(group)}
        keys = set().union(*(row.keys() for row in group)) - {
            "run_id",
            "seed",
            "evaluation_k",
        }
        for key in sorted(keys):
            values = [row.get(key) for row in group]
            if any(value is None for value in values):
                output[f"{key}_seed_mean"] = None
                output[f"{key}_seed_std"] = None
            elif all(isinstance(value, (int, float)) for value in values):
                data = np.asarray(values, dtype=np.float64)
                output[f"{key}_seed_mean"] = float(data.mean())
                output[f"{key}_seed_std"] = (
                    float(data.std(ddof=1)) if len(data) > 1 else None
                )
        result.append(output)
    return result
