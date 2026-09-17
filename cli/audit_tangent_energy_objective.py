"""Isolated matrix-free bulk-energy versus L2 tangent objective experiment."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import plotly.graph_objects as go
from rich.logging import RichHandler
from greenonet.config import ComplexCanonicalEnergyConfig

from audit_equal_split_initialization import (
    InitializationAudit,
    base,
    digest,
    write_csv,
)


def summarize(outdir, matched_dir=None):
    paths = {name: outdir / name for name in ("l2", "energy")}
    if matched_dir is not None:
        paths["energy_matched"] = matched_dir
    elif (outdir / "energy_matched").is_dir():
        paths["energy_matched"] = outdir / "energy_matched"
    rows = []
    for objective, path in paths.items():
        with (path / "per_sample.csv").open() as handle:
            rows.extend(dict(r, objective=objective) for r in csv.DictReader(handle))
    lookup = {
        (r["objective"], r["initialization"], int(r["K"]), int(r["sample_id"])): r
        for r in rows
    }
    assert len(lookup) == len(rows)
    max_k = max(int(r["K"]) for r in rows)
    samples = sorted({int(r["sample_id"]) for r in rows})
    assert len(rows) == 2 * len(paths) * (max_k + 1) * len(samples)
    for mode in ("learned", "equal_split"):
        for sample in samples:
            for objective in paths:
                for metric in ("energy", "rel_sol", "response_cost"):
                    assert (
                        lookup[("l2", mode, 0, sample)][metric]
                        == lookup[(objective, mode, 0, sample)][metric]
                    )
                metric = "response_cost" if objective == "l2" else "energy"
                costs = np.array(
                    [
                        float(lookup[(objective, mode, k, sample)][metric])
                        for k in range(max_k + 1)
                    ]
                )
                assert np.isfinite(costs).all()
                assert np.all(np.diff(costs) <= 1e-10 * costs[0] + 1e-24)
    comparison = []
    for objective, path in paths.items():
        with (path / "by_k.csv").open() as handle:
            comparison.extend(
                dict(r, objective=objective) for r in csv.DictReader(handle)
            )
    write_csv(outdir / "comparison.csv", comparison)
    validation = dict(
        sample_count=len(samples),
        rows=len(rows),
        max_k=max_k,
        same_initial_values=True,
        per_sample_target_monotonicity=True,
    )
    (outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    figure = go.Figure()
    for objective in paths:
        for mode in ("learned", "equal_split"):
            selected = [
                r
                for r in comparison
                if r["objective"] == objective and r["initialization"] == mode
            ]
            figure.add_scatter(
                x=[int(r["K"]) for r in selected],
                y=[100 * float(r["rel_sol_mean"]) for r in selected],
                name=f"{objective} / {mode}",
            )
    figure.update_layout(
        template="plotly_white",
        xaxis_title="Evaluation K",
        yaxis_title="Mean weak relative L2 error (%)",
        yaxis_type="log",
    )
    figure.write_html(outdir / "comparison.html", include_plotlyjs=True)
    return validation


class ResponseMetric:
    """Apply the exact production interior-edge energy without assembling Q."""

    def __init__(self, geometry, a_valid, objective):
        self.objective = objective
        self.mass = geometry.hx * geometry.hy
        self.edges = []
        for edges, spacing in (
            (geometry.x_edges, geometry.hx),
            (geometry.y_edges, geometry.hy),
        ):
            left, right = edges[:, 0], edges[:, 1]
            weight = (
                self.mass * (a_valid[:, left] + a_valid[:, right]) / (2 * spacing**2)
            )
            self.edges.append((left, right, weight))

    def apply(self, value):
        if self.objective == "l2":
            return self.mass * value
        result = torch.zeros_like(value)
        for left, right, weight in self.edges:
            flux = weight * (value[:, right] - value[:, left])
            result.index_add_(1, left, -flux)
            result.index_add_(1, right, flux)
        return result

    def inner(self, x, y):
        if self.objective == "l2":
            return self.mass * (x * y).sum(-1)
        return sum(
            (weight * (x[:, right] - x[:, left]) * (y[:, right] - y[:, left])).sum(-1)
            for left, right, weight in self.edges
        )


def metric_subspace(context, mismatch, metric, k, eps=1e-12):
    """Two-pass metric MGS; paired normalization and relative independence."""
    residual = mismatch.clone()
    delta = torch.zeros_like(mismatch)
    directions, responses, deltas, activities = [], [], [], []
    previous_cost = initial_cost = metric.inner(residual, residual)
    for _ in range(k):
        gradient = context.tangent_gradient(metric.apply(residual) / context.point_mass)
        source = gradient / context.denominator
        scale = source.abs().amax(-1).clamp_min(torch.finfo(source.dtype).tiny)
        source = source / scale[:, None]
        response = context.response_operator.forward_pair(
            torch.stack((source, source), 1)
        ).sum(1)
        cost = metric.inner(response, response)
        valid = cost > torch.finfo(cost.dtype).tiny
        norm = torch.where(valid, cost, 1.0).sqrt()
        source, response = source / norm[:, None], response / norm[:, None]
        before = metric.inner(response, response)
        for _pass in range(2):
            for old_source, old_response in zip(directions, responses, strict=True):
                cross = metric.inner(response, old_response)
                source = source - cross[:, None] * old_source
                response = response - cross[:, None] * old_response
        after = metric.inner(response, response)
        active = valid & (after > eps * before)
        norm = torch.where(active, after, 1.0).sqrt()
        source = torch.where(active[:, None], source / norm[:, None], 0.0)
        response = torch.where(active[:, None], response / norm[:, None], 0.0)
        denom = torch.where(active, metric.inner(response, response) * (1 + eps), 1.0)
        coefficient = metric.inner(residual, response) / denom
        delta = delta - coefficient[:, None] * source
        residual = residual - coefficient[:, None] * response
        cost = metric.inner(residual, residual)
        if torch.any(cost > previous_cost + 1e-10 * initial_cost + 1e-24):
            raise RuntimeError("Target objective increased")
        if not torch.isfinite(delta).all() or not torch.isfinite(cost).all():
            raise RuntimeError("Nonfinite correction")
        previous_cost = cost
        directions.append(source)
        responses.append(response)
        deltas.append(delta.clone())
        activities.append(active)
    return SimpleNamespace(
        deltas=torch.stack(deltas),
        direction_active=torch.stack(activities),
        directions=torch.stack(directions),
        responses=torch.stack(responses),
    )


def energy_separable_diagonal(context, metric, chunk_size=64):
    """Stream columns and retain diag(Hx^T Q Hx + Hy^T Q Hy) only."""
    points = context.denominator.numel()
    base = torch.empty_like(context.denominator)
    for start in range(0, points, chunk_size):
        stop = min(start + chunk_size, points)
        source = base.new_zeros((stop - start, points))
        source[
            torch.arange(stop - start, device=source.device),
            torch.arange(start, stop, device=source.device),
        ] = 1.0
        pair = context.response_operator.forward_pair(torch.stack((source, source), 1))
        base[start:stop] = metric.inner(pair[:, 0], pair[:, 0]) + metric.inner(
            pair[:, 1], pair[:, 1]
        )
    damping = (context.relative_lambda + context.denominator_relative_eps) * base.mean()
    denominator = base + damping
    if not torch.isfinite(denominator).all() or not (denominator > 0).all():
        raise RuntimeError("Energy denominator must be finite and positive")
    return base, denominator


class EnergyObjectiveAudit(InitializationAudit):
    def prepare(self, batch, mode):
        self.metric = ResponseMetric(batch.geometry, batch.a_valid, self.objective)
        if self.objective == "energy_matched":
            if not hasattr(self, "energy_denominator"):
                self.sync()
                start = time.perf_counter()
                self.fixed_a = batch.a_valid[:1].clone()
                column_metric = ResponseMetric(batch.geometry, self.fixed_a, "energy")
                self.energy_base, self.energy_denominator = energy_separable_diagonal(
                    self.tangent_context, column_metric
                )
                self.sync()
                self.energy_setup_seconds = time.perf_counter() - start
                self.solver_context = replace(
                    self.tangent_context, denominator=self.energy_denominator
                )
                np.savez(
                    self.request.outdir / "energy_preconditioner.npz",
                    base=self.energy_base.cpu().numpy(),
                    denominator=self.energy_denominator.cpu().numpy(),
                )
                self.logger.info(
                    "Energy diagonal setup %.6f seconds", self.energy_setup_seconds
                )
            torch.testing.assert_close(
                batch.a_valid, self.fixed_a.expand_as(batch.a_valid), rtol=0, atol=0
            )
        return super().prepare(batch, mode)

    def subspace(self, prepared, k):
        if not k:
            return None
        return metric_subspace(
            getattr(self, "solver_context", self.tangent_context),
            prepared.mismatch,
            self.metric,
            k,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=("l2", "energy", "energy_matched"),
        default=["l2", "energy"],
    )
    args = parser.parse_args()
    if args.max_k < 2:
        parser.error("max-k must be >=2")
    args.outdir.mkdir(parents=True, exist_ok=False)
    config_path = args.run_dir / "config_used.json"
    config = json.loads(config_path.read_text())
    if (
        ComplexCanonicalEnergyConfig.from_raw(
            config["coupling_training"].get("canonical_energy")
        ).boundary_weight
        != 0
    ):
        raise ValueError("This bulk-only audit requires boundary_weight=0")
    context_path = args.run_dir / "tangent_response_context.safetensors"
    config["coupling_training"]["tangent_context_checkpoint"].update(
        save_after_build=False, path=str(context_path.resolve())
    )
    config["coupling_training"]["compile"] = {"enabled": False}
    effective = args.outdir / "evaluation_config.json"
    effective.write_text(json.dumps(config, indent=2))
    checkpoint = args.run_dir / "complex_coupling_model_best_energy.safetensors"
    green = Path(config["pipeline"]["green_pretrained_path"])
    protected = [config_path, checkpoint, green, context_path]
    hashes = {str(p): digest(p) for p in protected}
    logger = logging.getLogger("EnergyObjectiveAudit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        RichHandler(show_path=False),
        logging.FileHandler(args.outdir / "run.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    torch.set_num_threads(4)
    metadata = dict(
        protected_sha256=hashes,
        device=args.device,
        training=False,
        normalization="paired metric normalization, two-pass MGS, squared independence 1e-12",
        preconditioner="see per-objective metadata; same relative damping coefficients",
        energy="production bulk energy; boundary weight zero; no mass regularization",
    )
    for objective in args.objectives:
        out = args.outdir / objective
        out.mkdir()
        request = base.TangentSubspaceAuditRequest(
            config=effective,
            coupling_checkpoint=checkpoint,
            green_checkpoint=green,
            outdir=out,
            device=args.device,
            batch_size=5,
            max_subspace_dimension=args.max_k,
            tangent_context=context_path,
        )
        audit = EnergyObjectiveAudit(request, logger=logger)
        audit.objective = objective
        start = time.perf_counter()
        logger.info("Starting objective %s", objective)
        audit.execute(3)
        metadata[objective] = dict(
            elapsed_seconds=time.perf_counter() - start,
            operator_equivalence_max_abs=audit._operator_equivalence_max_abs,
            context_build_count=audit._context_build_count,
            energy_diagonal_setup_seconds=getattr(audit, "energy_setup_seconds", 0.0),
            preconditioner="energy-separable"
            if objective == "energy_matched"
            else "original L2-separable",
        )
        del audit
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert hashes == {str(p): digest(p) for p in protected}
    (args.outdir / "provenance.json").write_text(json.dumps(metadata, indent=2))
    if (args.outdir / "l2").exists() and (args.outdir / "energy").exists():
        summarize(args.outdir)


if __name__ == "__main__":
    main()
