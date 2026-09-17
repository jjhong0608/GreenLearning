"""Separate frozen direction-space construction from coefficient objective."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import torch
import numpy as np
import plotly.graph_objects as go
from rich.logging import RichHandler

from audit_tangent_energy_objective import (
    EnergyObjectiveAudit,
    ResponseMetric,
    base,
    digest,
)
from audit_equal_split_initialization import write_csv


def summarize(outdir):
    lookup, summary = {}, []
    for basis in ("l2", "energy_matched"):
        for fit in ("l2", "energy"):
            path = outdir / f"basis_{basis}__fit_{fit}"
            with (path / "per_sample.csv").open() as handle:
                for row in csv.DictReader(handle):
                    key = (basis, fit, int(row["K"]), int(row["sample_id"]))
                    assert key not in lookup
                    lookup[key] = row
            with (path / "by_k.csv").open() as handle:
                summary.extend(
                    dict(row, basis=basis, fit=fit) for row in csv.DictReader(handle)
                )
    max_k = max(key[2] for key in lookup)
    samples = sorted({key[3] for key in lookup})
    assert len(lookup) == 4 * (max_k + 1) * len(samples)
    worst_dominance_violation = 0.0
    for basis in ("l2", "energy_matched"):
        for sample in samples:
            for fit, metric in (("l2", "response_cost"), ("energy", "energy")):
                other = "energy" if fit == "l2" else "l2"
                chosen = np.array(
                    [
                        float(lookup[(basis, fit, k, sample)][metric])
                        for k in range(max_k + 1)
                    ]
                )
                alternative = np.array(
                    [
                        float(lookup[(basis, other, k, sample)][metric])
                        for k in range(max_k + 1)
                    ]
                )
                assert np.isfinite(chosen).all()
                violation = np.max(chosen - alternative) / (chosen[0] + 1e-30)
                worst_dominance_violation = max(
                    worst_dominance_violation, float(violation)
                )
                assert violation <= 1e-10
                assert np.all(np.diff(chosen) <= 1e-10 * chosen[0] + 1e-24)
    write_csv(outdir / "comparison.csv", summary)
    validation = dict(
        sample_count=len(samples),
        rows=len(lookup),
        max_k=max_k,
        same_space_objective_dominance=True,
        worst_positive_violation_relative_to_initial=worst_dominance_violation,
        nested_objective_monotonicity=True,
    )
    (outdir / "validation.json").write_text(json.dumps(validation, indent=2))
    fig = go.Figure()
    for basis in ("l2", "energy_matched"):
        for fit in ("l2", "energy"):
            selected = [r for r in summary if r["basis"] == basis and r["fit"] == fit]
            fig.add_scatter(
                x=[int(r["K"]) for r in selected],
                y=[100 * float(r["rel_sol_mean"]) for r in selected],
                name=f"basis {basis}, fit {fit}",
            )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="K",
        yaxis_title="Mean weak relative L2 error (%)",
        yaxis_type="log",
    )
    fig.write_html(outdir / "comparison.html", include_plotlyjs=True)
    return validation


class RefitMetric(Protocol):
    def inner(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor: ...


def refit_subspace(
    mismatch: torch.Tensor,
    directions: torch.Tensor,
    responses: torch.Tensor,
    metric: RefitMetric,
    eps: float = 1e-12,
) -> SimpleNamespace:
    """Nested metric QR using only vector operations, no Gram system solve."""
    sources, vectors, deltas, activities = [], [], [], []
    delta = torch.zeros_like(directions[0])
    for source, response in zip(directions, responses, strict=True):
        initial = metric.inner(response, response)
        valid = initial > torch.finfo(initial.dtype).tiny
        norm = torch.where(valid, initial, 1.0).sqrt()
        source, response = source / norm[:, None], response / norm[:, None]
        for _pass in range(2):
            for old_source, old_response in zip(sources, vectors, strict=True):
                cross = metric.inner(response, old_response)
                source = source - cross[:, None] * old_source
                response = response - cross[:, None] * old_response
        remaining = metric.inner(response, response)
        active = valid & (remaining > eps)
        norm = torch.where(active, remaining, 1.0).sqrt()
        source = torch.where(active[:, None], source / norm[:, None], 0.0)
        response = torch.where(active[:, None], response / norm[:, None], 0.0)
        denominator = torch.where(active, metric.inner(response, response), 1.0)
        coefficient = metric.inner(mismatch, response) / denominator
        delta = delta - coefficient[:, None] * source
        if not torch.isfinite(delta).all():
            raise RuntimeError("Nonfinite refit")
        sources.append(source)
        vectors.append(response)
        deltas.append(delta.clone())
        activities.append(active)
    return SimpleNamespace(
        deltas=torch.stack(deltas), direction_active=torch.stack(activities)
    )


class FixedSpaceAudit(EnergyObjectiveAudit):
    initializations = ("equal_split",)

    def prepare(self, batch, mode):
        prepared = super().prepare(batch, mode)
        self.fit_metric = ResponseMetric(
            batch.geometry, batch.a_valid, self.fit_objective
        )
        return prepared

    def subspace(self, prepared, k):
        if not k:
            return None
        accuracy = (
            k == self.request.max_subspace_dimension
            and len(self.basis_hashes) < self.audit_batch_count
        )
        if accuracy and not self.first_fit:
            cached = self.basis_cache[len(self.basis_hashes)]
            basis = SimpleNamespace(
                directions=cached.directions.to(prepared.mismatch.device),
                responses=cached.responses.to(prepared.mismatch.device),
            )
        else:
            basis = super().subspace(prepared, k)
        if accuracy:
            if self.first_fit:
                self.basis_cache.append(
                    SimpleNamespace(
                        directions=basis.directions.cpu(),
                        responses=basis.responses.cpu(),
                    )
                )
            self.basis_hashes.append(
                hashlib.sha256(basis.directions.cpu().numpy().tobytes()).hexdigest()
            )
        return refit_subspace(
            prepared.mismatch, basis.directions, basis.responses, self.fit_metric
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-k", type=int, default=64)
    args = parser.parse_args()
    if args.max_k < 2:
        parser.error("max-k must be >=2")
    args.outdir.mkdir(parents=True, exist_ok=False)
    original = args.run_dir / "config_used.json"
    config = json.loads(original.read_text())
    context = args.run_dir / "tangent_response_context.safetensors"
    config["coupling_training"]["tangent_context_checkpoint"].update(
        save_after_build=False, path=str(context.resolve())
    )
    config["coupling_training"]["compile"] = {"enabled": False}
    effective = args.outdir / "evaluation_config.json"
    effective.write_text(json.dumps(config, indent=2))
    checkpoint = args.run_dir / "complex_coupling_model_best_energy.safetensors"
    green = Path(config["pipeline"]["green_pretrained_path"])
    protected = [original, context, checkpoint, green]
    hashes = {str(p): digest(p) for p in protected}
    logger = logging.getLogger("FixedSpaceAudit")
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
        initialization="physical f/2",
        training=False,
        refit="two-pass metric QR, exact scalar projection, independence eps=1e-12",
    )
    for construction in ("l2", "energy_matched"):
        basis_cache = []
        for fit in ("l2", "energy"):
            name = f"basis_{construction}__fit_{fit}"
            out = args.outdir / name
            out.mkdir()
            request = base.TangentSubspaceAuditRequest(
                config=effective,
                coupling_checkpoint=checkpoint,
                green_checkpoint=green,
                outdir=out,
                device=args.device,
                batch_size=5,
                max_subspace_dimension=args.max_k,
                tangent_context=context,
            )
            audit = FixedSpaceAudit(request, logger=logger)
            audit.objective, audit.fit_objective = construction, fit
            audit.basis_hashes = []
            audit.basis_cache = basis_cache
            audit.first_fit = fit == "l2"
            logger.info("Starting %s", name)
            audit.execute(3)
            metadata[name] = dict(
                basis_sha256_by_batch=audit.basis_hashes,
                context_build_count=audit._context_build_count,
                operator_equivalence_max_abs=audit._operator_equivalence_max_abs,
                energy_setup_seconds=getattr(audit, "energy_setup_seconds", 0.0),
            )
            del audit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    for construction in ("l2", "energy_matched"):
        assert (
            metadata[f"basis_{construction}__fit_l2"]["basis_sha256_by_batch"]
            == metadata[f"basis_{construction}__fit_energy"]["basis_sha256_by_batch"]
        )
    assert hashes == {str(p): digest(p) for p in protected}
    (args.outdir / "provenance.json").write_text(json.dumps(metadata, indent=2))
    summarize(args.outdir)


if __name__ == "__main__":
    main()
