"""Read-only full-test compatibility audit of paper Examples 1, 3 and 4."""

import argparse
import gc
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from cli.audit_normalized_tangent import normalized_step
from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_coupling_data import (
    ComplexCouplingDataset,
    complex_coupling_collate_fn,
)
from greenonet.complex_cross_axis_reconstruction import ComplexCrossAxisReconstructor
from greenonet.complex_geometry import load_complex_geometry
from greenonet.complex_losses import (
    build_boundary_energy_context,
    canonical_complex_energy_loss,
)
from greenonet.complex_tangent_subspace_audit import (
    ComplexTangentSubspaceAudit,
    TangentSubspaceAuditRequest,
)
from greenonet.complex_tangent_projection import matrix_free_krylov_subspace_step
from greenonet.config import TangentContextCheckpointConfig
from greenonet.coupling_artifacts import load_coupling_artifact_configs
from greenonet.unit_square_trunk_audit import digest, local_path, read_csv, write_csv


class PaperCompatibilityAudit:
    def __init__(self, out, logger):
        self.out, self.logger = out, logger
        self.contexts = {}
        self.hashes = {}
        self.posthoc_k = None
        self.production_normalized = False

    def normalized_step(self, **kwargs):
        if not self.production_normalized:
            return normalized_step(**kwargs)
        kwargs["context"] = replace(
            kwargs["context"], direction_normalization="response"
        )
        return matrix_free_krylov_subspace_step(**kwargs)

    def fingerprint(self, path):
        path = Path(path)
        if str(path) not in self.hashes:
            self.hashes[str(path)] = digest(path)
        return self.hashes[str(path)]

    @torch.no_grad()
    def run_one(self, config_path, example):
        run = config_path.parent
        out = self.out / example / run.name
        if (out / "summary.json").exists():
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        payload = json.loads(config_path.read_text())
        projection = payload["coupling_model"]["balance_projection"]
        checkpoint = run / "complex_coupling_model_best_energy.safetensors"
        self.fingerprint(config_path)
        self.fingerprint(checkpoint)
        baseline_path = run / "artifacts_best_energy/metrics/per_sample_metrics.csv"
        baseline = {r["file_stem"]: r for r in read_csv(baseline_path)}
        self.fingerprint(baseline_path)
        if projection["mode"] == "physical_symmetric":
            result = dict(
                example=example,
                run=str(run),
                k=0,
                samples=len(baseline),
                status="bypass_no_tangent_not_reexecuted",
            )
            (out / "summary.json").write_text(json.dumps(result, indent=2))
            return result
        tangent = projection["symmetric_tangent_green_response"]
        trained_k = int(tangent["subspace_dimension"])
        k = self.posthoc_k or trained_k
        if tangent.get("eta_cap_enabled", True) and k == 1:
            raise ValueError("Capped K1 needs an explicit compatibility protocol")
        green = local_path(payload["pipeline"]["green_pretrained_path"])
        data = payload["dataset"]
        geometry_path = local_path(data["geometry_path"])
        coefficients = local_path(data["coefficient_functions_path"])
        test_path = local_path(data["test_path"])
        audit = ComplexTangentSubspaceAudit(
            TangentSubspaceAuditRequest(
                config=config_path,
                coupling_checkpoint=checkpoint,
                green_checkpoint=green,
                geometry=geometry_path,
                coefficients=coefficients,
                test_path=test_path,
                outdir=out,
                device="cpu",
                max_subspace_dimension=max(k, 2),
            ),
            logger=self.logger,
        )
        audit._configs = load_coupling_artifact_configs(config_path)
        audit._configs = replace(
            audit._configs,
            coupling_training=replace(
                audit._configs.coupling_training,
                tangent_context_checkpoint=TangentContextCheckpointConfig(
                    enabled=False
                ),
            ),
        )
        audit._device = torch.device("cpu")
        audit.geometry = load_complex_geometry(geometry_path, dtype=torch.float64)
        audit._load_models()
        dataset = ComplexCouplingDataset(
            test_path,
            audit.geometry,
            load_coefficient_functions(coefficients),
            branch_input_dim=audit._configs.coupling_model.branch_input_dim,
            dtype=torch.float64,
            coefficient_terms=audit._configs.coupling_model.coefficient_terms,
            integration_rule=audit._configs.coupling_training.integration_rule,
        )
        for path in sorted(test_path.glob("*.npz")):
            self.fingerprint(path)
        key = tuple(
            self.fingerprint(p) for p in (green, geometry_path, coefficients)
        ) + (
            json.dumps(
                {
                    name: value
                    for name, value in tangent.items()
                    if name not in ("subspace_dimension", "geometry_k_selection")
                },
                sort_keys=True,
            ),
        )
        reconstructor = ComplexCrossAxisReconstructor(
            audit._configs.coupling_model.cross_axis_reconstruction
        )
        boundary = build_boundary_energy_context(audit.geometry)
        rows, stages, backward = [], [], []
        started = time.perf_counter()
        for batch_index, batch in enumerate(
            DataLoader(
                dataset,
                batch_size=10,
                shuffle=False,
                collate_fn=complex_coupling_collate_fn,
            )
        ):
            if batch_index == 0:
                if key not in self.contexts:
                    audit._initialize_context(batch)
                    self.contexts[key] = (
                        audit.tangent_context,
                        batch.x_green_branch.clone(),
                        batch.y_green_branch.clone(),
                    )
                context, x, y = self.contexts[key]
                if not torch.equal(x, batch.x_green_branch) or not torch.equal(
                    y, batch.y_green_branch
                ):
                    raise ValueError("Cached operator inputs differ")
                audit.tangent_context = replace(context, subspace_dimension=k)
                audit.response_operator = context.response_operator
                audit._verify_operator_equivalence(batch)
            context = audit.tangent_context
            prepared = audit._prepare_batch(batch)
            start = time.perf_counter()
            if k == 1:
                legacy = context.tangent_step(
                    mismatch=prepared.mismatch, gradient=prepared.gradient
                )
                old_delta = legacy.delta
                old_costs = (
                    context.point_mass
                    * (
                        prepared.mismatch
                        + context.response_operator.forward_pair(
                            torch.stack((old_delta, old_delta), 1)
                        ).sum(1)
                    )
                    .square()
                    .sum(-1)
                )[None]
                old_activity = None
            else:
                legacy = matrix_free_krylov_subspace_step(
                    context=context,
                    mismatch=prepared.mismatch,
                    gradient=prepared.gradient,
                    max_dimension=k,
                    relative_eps=context.line_search_relative_eps,
                    monotonicity_relative_tol=1e-10,
                )
                old_delta, old_costs, old_activity = (
                    legacy.deltas[-1],
                    legacy.costs,
                    legacy.direction_active,
                )
            old_seconds = time.perf_counter() - start
            start = time.perf_counter()
            new = self.normalized_step(
                context=context,
                mismatch=prepared.mismatch,
                gradient=prepared.gradient,
                max_dimension=k,
                relative_eps=1e-12,
            )
            new_seconds = time.perf_counter() - start
            physical = torch.stack(
                [
                    prepared.symmetric_physical + torch.stack((delta, -delta), 1)
                    for delta in (old_delta, new.deltas[-1])
                ]
            )
            flat = physical.flatten(0, 1)
            solution = context.response_operator.forward_pair(flat)
            cross = reconstructor.reconstruct(
                u_phi_valid=solution[:, 0],
                u_psi_valid=solution[:, 1],
                projected_physical=flat,
                geometry=batch.geometry,
                weak_context=batch.weak_context,
            )
            energy = canonical_complex_energy_loss(
                u_phi_valid=solution[:, 0],
                u_psi_valid=solution[:, 1],
                a_valid=batch.a_valid.repeat(2, 1),
                geometry=batch.geometry,
                boundary_context=boundary,
            )
            b = batch.rhs_valid.shape[0]
            fields = {
                "u_phi": solution[:, 0].reshape(2, b, -1),
                "u_psi": solution[:, 1].reshape(2, b, -1),
                "equal": cross.u_equal_mean_valid.reshape(2, b, -1),
                "weak": cross.u_pred_valid.reshape(2, b, -1),
            }
            if not all(torch.isfinite(v).all() for v in fields.values()):
                raise RuntimeError("Nonfinite predicted field")
            for i, stem in enumerate(batch.file_stems):
                row = dict(
                    example=example,
                    run=run.name,
                    k=k,
                    sample_id=int(batch.sample_indices[i]),
                    file_stem=stem,
                )
                for name, field in fields.items():
                    denom = batch.sol_valid[i].norm().clamp_min(1e-12)
                    for j, label in enumerate(("old", "new")):
                        row[f"{label}_rel_{name}"] = float(
                            (field[j, i] - batch.sol_valid[i]).norm() / denom
                        )
                    row[f"max_abs_change_{name}"] = float(
                        (field[1, i] - field[0, i]).abs().max()
                    )
                    if example == "example3":
                        xy = batch.geometry.coords_valid
                        mask = (xy.abs() - 0.2).abs().amin(1) <= 2 / 128
                        for j, label in enumerate(("old", "new")):
                            row[f"{label}_transition_rms_{name}"] = float(
                                (field[j, i, mask] - batch.sol_valid[i, mask])
                                .square()
                                .mean()
                                .sqrt()
                            )
                row["baseline_rel_sol_abs_error"] = abs(
                    row["old_rel_weak"] - float(baseline[stem]["rel_sol"])
                )
                row["baseline_equal_abs_error"] = abs(
                    row["old_rel_equal"] - float(baseline[stem]["rel_sol_equal_mean"])
                )
                row["old_energy"] = float(energy.bulk_per_sample[i])
                row["new_energy"] = float(energy.bulk_per_sample[b + i])
                row["old_cost"] = float(old_costs[-1, i])
                row["new_cost"] = float(new.costs[-1, i])
                row["balance_max_abs"] = float(
                    (physical[:, i].sum(1) - batch.rhs_valid[i]).abs().max()
                )
                row["old_projection_seconds_per_sample"] = old_seconds / b
                row["new_projection_seconds_per_sample"] = new_seconds / b
                rows.append(row)
                for j in range(k):
                    stages.append(
                        dict(
                            run=run.name,
                            sample_id=int(batch.sample_indices[i]),
                            stage=j + 1,
                            old_cost=float(old_costs[j, i]),
                            new_cost=float(new.costs[j, i]),
                            old_active=int(old_activity[j, i])
                            if old_activity is not None
                            else "not_exposed",
                            new_active=int(new.direction_active[j, i]),
                        )
                    )
            if batch_index == 0:
                with torch.enable_grad():
                    derivatives = []
                    for normalized in (False, True):
                        m = prepared.mismatch[:1].detach().clone().requires_grad_(True)
                        g = context.tangent_gradient(m)
                        if normalized:
                            delta = self.normalized_step(
                                context=context,
                                mismatch=m,
                                gradient=g,
                                max_dimension=k,
                                relative_eps=1e-12,
                            ).deltas[-1]
                        else:
                            delta = context.tangent_step(mismatch=m, gradient=g).delta
                        probe = torch.linspace(
                            0.5, 1.5, delta.shape[-1], dtype=delta.dtype
                        )
                        derivatives.append(
                            torch.autograd.grad((delta * probe).sum(), m)[0]
                        )
                    if not all(torch.isfinite(d).all() for d in derivatives):
                        raise RuntimeError("Nonfinite compatibility backward")
                    backward.append(
                        dict(
                            max_abs=float(
                                (derivatives[1] - derivatives[0]).abs().max()
                            ),
                            relative_l2=float(
                                (derivatives[1] - derivatives[0]).norm()
                                / derivatives[0].norm().clamp_min(1e-30)
                            ),
                        )
                    )
            if (batch_index + 1) % 5 == 0:
                self.logger.info("%s: %d/%d samples", run.name, len(rows), len(dataset))
        if set(baseline) != {r["file_stem"] for r in rows}:
            raise ValueError("Test coverage differs from saved artifacts")
        write_csv(out / "per_sample.csv", rows)
        write_csv(out / "stages.csv", stages)
        result = dict(
            example=example,
            run=str(run),
            k=k,
            trained_k=trained_k,
            baseline_is_same_k=k == trained_k,
            normalization_implementation="production"
            if self.production_normalized
            else "prototype",
            samples=len(rows),
            seconds=time.perf_counter() - started,
            backward=backward,
            baseline_max_abs=max(
                max(r["baseline_rel_sol_abs_error"], r["baseline_equal_abs_error"])
                for r in rows
            ),
            field_max_abs=max(r[f"max_abs_change_{f}"] for r in rows for f in fields),
            relative_error_max_abs=max(
                abs(r[f"new_rel_{f}"] - r[f"old_rel_{f}"]) for r in rows for f in fields
            ),
            activity_changes=sum(
                r["old_active"] != "not_exposed" and r["old_active"] != r["new_active"]
                for r in stages
            ),
            balance_max_abs=max(r["balance_max_abs"] for r in rows),
        )
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        self.logger.info("Finished %s: %s", run.name, result)
        del audit
        gc.collect()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--example", choices=["example1", "example3", "example4"])
    parser.add_argument("--posthoc-k", type=int, choices=[64])
    parser.add_argument("--production-normalized", action="store_true")
    parser.add_argument("--max-runs-per-example", type=int)
    args = parser.parse_args()
    if args.max_runs_per_example is not None and args.max_runs_per_example < 1:
        parser.error("--max-runs-per-example must be positive")
    torch.set_num_threads(4)
    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("paper_compatibility")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in (
        RichHandler(show_path=False),
        logging.FileHandler(args.outdir / "audit.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    audit = PaperCompatibilityAudit(args.outdir, logger)
    audit.posthoc_k = args.posthoc_k
    audit.production_normalized = args.production_normalized
    patterns = {
        "example1": "unit_square/coupling/*/config_used.json",
        "example3": "annulus/coupling_sequential/*/config_used.json",
        "example4": "pentagram/*/seed*/*/config_used.json",
    }
    results = []
    for example, pattern in patterns.items():
        if args.example and example != args.example:
            continue
        evaluated = 0
        for path in sorted(Path("checkpoints/numerical_examples").glob(pattern)):
            if (
                args.max_runs_per_example is not None
                and evaluated >= args.max_runs_per_example
            ):
                break
            if args.posthoc_k:
                if example != "example4" or "pentagram_k10_" not in path.parent.name:
                    continue
            results.append(audit.run_one(path, example))
            evaluated += int(results[-1]["k"] > 0)
            (args.outdir / "runs.json").write_text(json.dumps(results, indent=2))
            (args.outdir / "input_hashes.json").write_text(
                json.dumps(audit.hashes, indent=2)
            )


if __name__ == "__main__":
    main()
