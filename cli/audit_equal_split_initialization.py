"""Frozen learned-versus-f/2 initialization audit; no training or model mutation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from greenonet import complex_tangent_subspace_audit as base
from greenonet.config import ComplexCanonicalEnergyConfig


def equal_split(rhs: torch.Tensor) -> torch.Tensor:
    return torch.stack((0.5 * rhs, 0.5 * rhs), dim=1)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_results(outdir: Path, artifact: Path) -> dict:
    with (outdir / "per_sample.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with artifact.open() as handle:
        reference = {int(r["sample_id"]): r for r in csv.DictReader(handle)}
    baseline = {
        int(r["sample_id"]): r
        for r in rows
        if r["initialization"] == "learned" and int(r["K"]) == 10
    }
    if baseline.keys() != reference.keys():
        raise ValueError("Audit must cover the complete saved artifact test set")
    differences = {}
    for actual, saved in (
        ("rel_sol", "rel_sol"),
        ("rel_equal", "rel_sol_equal_mean"),
        ("response_cost", "tangent_response_cost_k10"),
        ("energy", "loss_energy_optimized"),
    ):
        a = np.array([float(baseline[i][actual]) for i in sorted(baseline)])
        b = np.array([float(reference[i][saved]) for i in sorted(baseline)])
        np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-12)
        differences[actual] = float(np.max(np.abs(a - b)))
    max_k = max(int(r["K"]) for r in rows)
    for mode in ("learned", "equal_split"):
        for sample in reference:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["initialization"] == mode and int(r["sample_id"]) == sample
                ),
                key=lambda r: int(r["K"]),
            )
            assert [int(r["K"]) for r in selected] == list(range(max_k + 1))
            cost = np.array([float(r["response_cost"]) for r in selected])
            assert np.all(np.diff(cost) <= 1e-10 * cost[0] + 1e-24)
    result = dict(
        full_test_sample_count=len(reference),
        row_count=len(rows),
        max_k=max_k,
        baseline_max_absolute_differences=differences,
        response_cost_monotone_for_every_sample=True,
        artifact_sha256=digest(artifact),
    )
    (outdir / "validation.json").write_text(json.dumps(result, indent=2))
    return result


class InitializationAudit(base.ComplexTangentSubspaceAudit):
    """Reuse frozen loading/context contracts and the production tangent helper."""

    initializations = ("learned", "equal_split")

    def sync(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def prepare(self, batch, mode):
        if mode == "learned":
            return base.prepare_tangent_audit_batch(
                model=self._coupling_model, context=self.tangent_context, batch=batch
            )
        if mode != "equal_split":
            raise ValueError(f"Unknown initialization: {mode}")
        physical = equal_split(batch.rhs_valid)
        solution = self.response_operator.forward_pair(physical)
        mismatch = solution[:, 0] - solution[:, 1]
        return base.PreparedTangentBatch(
            raw_physical=physical,
            symmetric_physical=physical,
            mismatch=mismatch,
            gradient=self.tangent_context.tangent_gradient(mismatch),
        )

    def subspace(self, prepared, k):
        if k == 0:
            return None
        return base.matrix_free_krylov_subspace_audit(
            context=self.tangent_context,
            mismatch=prepared.mismatch,
            gradient=prepared.gradient,
            max_dimension=max(2, k),
            relative_eps=self.request.subspace_relative_eps,
            monotonicity_relative_tol=self.request.monotonicity_relative_tol,
        )

    def candidate(self, batch, prepared, result, k):
        delta = torch.zeros_like(prepared.gradient) if k == 0 else result.deltas[k - 1]
        physical = torch.stack(
            (
                prepared.symmetric_physical[:, 0] + delta,
                prepared.symmetric_physical[:, 1] - delta,
            ),
            dim=1,
        )
        torch.testing.assert_close(
            physical.sum(1), batch.rhs_valid, rtol=1e-12, atol=1e-12
        )
        solution = self.response_operator.forward_pair(physical)
        cross = self._cross_axis_reconstructor.reconstruct(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            projected_physical=physical,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        return delta, solution, cross

    @torch.no_grad()
    def execute(self, timing_repeats: int) -> None:
        out = self.request.outdir
        self._configs = base.load_coupling_artifact_configs(self.request.config)
        self._device = torch.device(self.request.device)
        self.geometry = base.load_complex_geometry(
            self._configs.dataset.geometry_path, dtype=self._configs.dataset.dtype
        )
        dataset = base.ComplexCouplingDataset(
            self._configs.dataset.test_path,
            self.geometry,
            base.load_coefficient_functions(
                self._configs.dataset.coefficient_functions_path
            ),
            branch_input_dim=self._configs.coupling_model.branch_input_dim,
            dtype=self._configs.dataset.dtype,
            coefficient_terms=self._configs.coupling_model.coefficient_terms,
            integration_rule=self._configs.coupling_training.integration_rule,
        )
        self._load_models()
        self._cross_axis_reconstructor = base.ComplexCrossAxisReconstructor(
            self._configs.coupling_model.cross_axis_reconstruction
        )
        self.boundary_context = base.build_boundary_energy_context(self.geometry)
        canonical = ComplexCanonicalEnergyConfig.from_raw(
            self._configs.coupling_training.canonical_energy
        )
        rows = []
        loader = DataLoader(
            dataset,
            batch_size=self.request.batch_size,
            shuffle=False,
            collate_fn=base.complex_coupling_collate_fn,
        )
        first = None
        self.audit_batch_count = len(loader)
        for batch_index, batch in enumerate(loader):
            batch = batch.to(self._device)
            if first is None:
                first = batch
            self._initialize_context(batch)
            for mode in self.initializations:
                prepared = self.prepare(batch, mode)
                result = self.subspace(prepared, self.request.max_subspace_dimension)
                for k in range(self.request.max_subspace_dimension + 1):
                    delta, solution, cross = self.candidate(batch, prepared, result, k)
                    energy = base.canonical_complex_energy_loss(
                        u_phi_valid=solution[:, 0],
                        u_psi_valid=solution[:, 1],
                        a_valid=batch.a_valid,
                        geometry=batch.geometry,
                        boundary_context=self.boundary_context,
                    )
                    values = {}
                    for name, prediction in (
                        ("rel_sol", cross.u_pred_valid),
                        ("rel_equal", cross.u_equal_mean_valid),
                        ("rel_u_phi", solution[:, 0]),
                        ("rel_u_psi", solution[:, 1]),
                    ):
                        values[name] = (prediction - batch.sol_valid).norm(
                            dim=1
                        ) / batch.sol_valid.norm(dim=1).clamp_min(1e-30)
                    values["response_cost"] = self.tangent_context.point_mass * (
                        solution[:, 0] - solution[:, 1]
                    ).square().sum(1)
                    values["energy"] = (
                        energy.bulk_per_sample
                        + canonical.boundary_weight * energy.boundary_per_sample
                    )
                    values["correction_l2"] = (
                        self.tangent_context.point_mass * delta.square().sum(1)
                    ).sqrt()
                    values["active_directions"] = (
                        torch.zeros_like(values["energy"])
                        if k == 0
                        else result.direction_active[:k].sum(0)
                    )
                    for i, sample in enumerate(batch.sample_indices.tolist()):
                        row = dict(initialization=mode, K=k, sample_id=sample)
                        for name, value in values.items():
                            if not torch.isfinite(value).all():
                                raise RuntimeError(f"Nonfinite {name}")
                            row[name] = float(value[i])
                        rows.append(row)
            self.logger.info(
                "Evaluated batch %d/%d, configured initializations through K%d",
                batch_index + 1,
                len(loader),
                self.request.max_subspace_dimension,
            )
            write_csv(out / "per_sample.csv", rows)
        aggregates = []
        for mode in self.initializations:
            for k in range(self.request.max_subspace_dimension + 1):
                selected = [
                    r for r in rows if r["initialization"] == mode and r["K"] == k
                ]
                row = dict(initialization=mode, K=k, sample_count=len(selected))
                for metric in values:
                    a = np.array([r[metric] for r in selected])
                    for stat, value in (
                        ("mean", a.mean()),
                        ("p95", np.quantile(a, 0.95)),
                        ("max", a.max()),
                    ):
                        row[f"{metric}_{stat}"] = float(value)
                aggregates.append(row)
        write_csv(out / "by_k.csv", aggregates)
        timing = []
        for k in (0, 2, 4, 9, 10, 16, 32, 64):
            if k > self.request.max_subspace_dimension:
                continue
            for repeat in range(timing_repeats + 1):
                for mode in (
                    self.initializations
                    if repeat % 2 == 0
                    else tuple(reversed(self.initializations))
                ):
                    self.sync()
                    start = time.perf_counter()
                    prepared = self.prepare(first, mode)
                    result = self.subspace(prepared, k)
                    self.candidate(first, prepared, result, k)
                    self.sync()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        timing.append(
                            dict(
                                initialization=mode,
                                K=k,
                                repeat=repeat,
                                batch_size=self.request.batch_size,
                                seconds=elapsed,
                            )
                        )
            self.logger.info("Independent prediction timing K%d completed", k)
        write_csv(out / "timing.csv", timing)
        figure = go.Figure()
        for mode in self.initializations:
            selected = [r for r in aggregates if r["initialization"] == mode]
            figure.add_scatter(
                x=[r["K"] for r in selected],
                y=[100 * r["rel_sol_mean"] for r in selected],
                name=mode,
            )
        figure.update_layout(
            template="plotly_white",
            xaxis_title="Evaluation K",
            yaxis_title="Mean weak relative L2 error (%)",
            yaxis_type="log",
        )
        figure.write_html(out / "accuracy.html", include_plotlyjs=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=3)
    args = parser.parse_args()
    if args.max_k < 2 or args.batch_size < 1 or args.timing_repeats < 1:
        parser.error("Require max-k>=2, batch-size>=1 and timing-repeats>=1")
    args.outdir.mkdir(parents=True, exist_ok=False)
    config_path = args.run_dir / "config_used.json"
    config = json.loads(config_path.read_text())
    original_context = args.run_dir / "tangent_response_context.safetensors"
    if not original_context.is_file():
        raise FileNotFoundError(original_context)
    config["coupling_training"]["tangent_context_checkpoint"]["save_after_build"] = (
        False
    )
    config["coupling_training"]["tangent_context_checkpoint"]["path"] = str(
        original_context.resolve()
    )
    config["coupling_training"]["compile"] = {"enabled": False}
    effective = args.outdir / "evaluation_config.json"
    effective.write_text(json.dumps(config, indent=2))
    checkpoint = args.run_dir / "complex_coupling_model_best_energy.safetensors"
    green = Path(config["pipeline"]["green_pretrained_path"])
    protected = [config_path, checkpoint, green, original_context]
    hashes = {str(p): digest(p) for p in protected}
    logger = logging.getLogger("InitializationAudit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        RichHandler(show_path=True, omit_repeated_times=False),
        logging.FileHandler(args.outdir / "run.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    torch.set_num_threads(4)
    request = base.TangentSubspaceAuditRequest(
        config=effective,
        coupling_checkpoint=checkpoint,
        green_checkpoint=green,
        outdir=args.outdir,
        device=args.device,
        batch_size=args.batch_size,
        max_subspace_dimension=args.max_k,
        tangent_context=original_context,
    )
    audit = InitializationAudit(request, logger=logger)
    start = time.perf_counter()
    audit.execute(args.timing_repeats)
    if args.max_k >= 10:
        validate_results(
            args.outdir,
            args.run_dir / "artifacts_best_energy/metrics/per_sample_metrics.csv",
        )
    if hashes != {str(p): digest(p) for p in protected}:
        raise RuntimeError("Protected input changed")
    metadata = dict(
        protected_sha256=hashes,
        device=args.device,
        torch_version=torch.__version__,
        elapsed_seconds=time.perf_counter() - start,
        context_build_count=audit._context_build_count,
        operator_equivalence_max_abs=audit._operator_equivalence_max_abs,
        training=False,
        normalization=audit.tangent_context.direction_normalization,
        relative_eps=request.subspace_relative_eps,
        timing="Independent warmed full prediction calls, first test batch, no data loading/context setup; K1 excluded from timing because helper seeds K2.",
    )
    (args.outdir / "provenance.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
