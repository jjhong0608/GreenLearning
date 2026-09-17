"""Pure metrics for the read-only reference/source-initialization audit."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch


def symmetric_source_balance(
    pair: torch.Tensor, rhs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if pair.ndim != 3 or pair.shape[1] != 2 or pair[:, 0].shape != rhs.shape:
        raise ValueError("Require pair [batch, 2, points] and rhs [batch, points].")
    if not torch.isfinite(pair).all() or not torch.isfinite(rhs).all():
        raise ValueError("Non-finite physical source.")
    residual = pair.sum(dim=1) - rhs
    balanced = pair - residual.unsqueeze(1) / 2
    torch.testing.assert_close(balanced.sum(dim=1), rhs, rtol=1e-12, atol=1e-12)
    return balanced, residual


def physical_equal_split(rhs: torch.Tensor) -> torch.Tensor:
    return torch.stack((rhs / 2, rhs / 2), dim=1)


def relative_norm(
    error: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> tuple[float | None, str]:
    denominator = float(target.norm())
    if denominator <= eps:
        return None, "target_norm_at_or_below_eps"
    return float(error.norm()) / denominator, "defined"


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    keys = ("example", "fingerprint", "run_id", "seed", "condition", "evaluation_k")
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for identity, samples in groups.items():
        summary = dict(zip(keys, identity, strict=True))
        summary["sample_count"] = len(samples)
        for metric in (
            "rel_sol",
            "rel_sol_equal_mean",
            "rel_u_phi",
            "rel_u_psi",
            "response_cost",
            "loss_energy_optimized",
            "balance_l2",
            "balance_relative",
            "balance_max_abs",
            "correction_pair_l2",
            "effective_dimension",
        ):
            values = [row[metric] for row in samples if row.get(metric) is not None]
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError(f"Non-finite {metric} in summary.")
            summary.update(
                {
                    f"{metric}_mean": float(array.mean()),
                    f"{metric}_p95": float(np.quantile(array, 0.95, method="linear")),
                    f"{metric}_max": float(array.max()),
                    f"{metric}_count": len(array),
                }
            )
        result.append(summary)
    return result


def match_accuracy(
    target: dict[str, Any], curve: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(curve, key=lambda row: row["evaluation_k"])
    ks = [row["evaluation_k"] for row in ordered]
    if not ks or ks != list(range(max(ks) + 1)):
        raise ValueError("Matching requires a complete, unique K0..Kmax curve.")
    for row in [target, *ordered]:
        for metric in ("rel_sol_mean", "rel_sol_p95", "rel_sol_max"):
            if not np.isfinite(row[metric]):
                raise ValueError(f"Invalid {metric}; this is not nonattainment.")
    result = {key: target[key] for key in ("example", "run_id", "seed", "fingerprint")}
    result.update(training_k=target["evaluation_k"], max_k=max(ks))
    for metric in ("mean", "p95", "max"):
        result[f"learned_{metric}"] = target[f"rel_sol_{metric}"]
    for criterion in ("mean", "mean_p95"):
        accepted = [
            row
            for row in ordered
            if row["rel_sol_mean"] <= target["rel_sol_mean"]
            and (criterion == "mean" or row["rel_sol_p95"] <= target["rel_sol_p95"])
        ]
        result[f"{criterion}_status"] = "reached" if accepted else "not_reached"
        result[f"{criterion}_k"] = accepted[0]["evaluation_k"] if accepted else None
        for metric in ("mean", "p95", "max"):
            result[f"{criterion}_{metric}"] = (
                accepted[0][f"rel_sol_{metric}"] if accepted else None
            )
        result[f"{criterion}_later_nonattainment"] = bool(accepted) and any(
            row["evaluation_k"] > accepted[0]["evaluation_k"] and row not in accepted
            for row in ordered
        )
    return result
