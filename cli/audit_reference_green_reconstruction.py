"""Compare frozen physical sources under learned/reference diffusion kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_axial_response_operator import (
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_coupling_data import ComplexCouplingBatch
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.reference_green_reconstruction import DiffusionReferenceBuilder
from greenonet.source_initialization_metrics import symmetric_source_balance


FACTORS = (1, 4, 16, 64, 256)
METRICS = (
    "J",
    "relative_mismatch",
    "rel_sol",
    "rel_sol_equal_mean",
    "rel_u_phi",
    "rel_u_psi",
    "loss_energy_optimized",
    "balance_relative",
    "balance_max_abs",
    "field_change_x",
    "field_change_y",
    "field_change_weak",
    "refinement_change_x",
    "refinement_change_y",
    "refinement_change_weak",
)


class FixedSourceSession(SourceRunSession):
    evaluation_operator: FrozenBidirectionalResponseOperator | None = None

    def fields(
        self, batch: ComplexCouplingBatch, pair: torch.Tensor
    ) -> tuple[torch.Tensor, ComplexCrossAxisReconstructionResult]:
        if self.evaluation_operator is None:
            return super().fields(batch, pair)
        solution = self.evaluation_operator.forward_pair(pair)
        cross = self.cross_axis.reconstruct(
            u_phi_valid=solution[:, 0],
            u_psi_valid=solution[:, 1],
            projected_physical=pair,
            geometry=batch.geometry,
            weak_context=batch.weak_context,
        )
        if (
            not torch.isfinite(solution).all()
            or not torch.isfinite(cross.u_pred_valid).all()
        ):
            raise RuntimeError("Nonfinite reference reconstruction")
        return solution, cross


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "example",
        "run_id",
        "seed",
        "source_condition",
        "evaluation_k",
        "kernel",
        "factor",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, samples in groups.items():
        result = dict(zip(keys, key, strict=True), sample_count=len(samples))
        for metric in METRICS:
            values = np.asarray([r[metric] for r in samples], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Undefined metric: {metric}")
            result.update(
                {
                    f"{metric}_mean": float(values.mean()),
                    f"{metric}_p95": float(np.quantile(values, 0.95)),
                    f"{metric}_max": float(values.max()),
                }
            )
        output.append(result)
    return output


class FixedSourceAudit:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = args.outdir.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "sources").mkdir()
        self.logger = logging.getLogger("reference_green_reconstruction")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(
                rich_tracebacks=True, show_path=True, omit_repeated_times=False
            ),
            logging.FileHandler(self.out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)
        self.rows: list[dict[str, Any]] = []
        self.block_rows: list[dict[str, Any]] = []
        self.cache: dict[str, dict[int, FrozenBidirectionalResponseOperator]] = {}
        self.baseline = {
            (r["run_id"], int(r["evaluation_k"]), int(r["sample_id"])): r
            for r in read_csv(args.learned_baseline / "per_sample.csv")
        }
        self.reference = {
            (r["example"], r["condition"], int(r["sample_id"])): r
            for r in read_csv(args.reference_baseline / "per_sample.csv")
            if r["condition"] in {"reference_raw", "reference_balanced"}
        }
        self.best = {
            r["run_id"]: int(r["evaluation_k"])
            for r in read_csv(args.learned_baseline / "minima.csv")
            if r["selected_by"] == "mean"
        }

    def operators(
        self, session: FixedSourceSession
    ) -> dict[int, FrozenBidirectionalResponseOperator]:
        key = session.spec.fingerprint
        if key in self.cache:
            return self.cache[key]
        builder = DiffusionReferenceBuilder(
            session.batches[0].geometry,
            load_coefficient_functions(session.spec.coefficients),
            session.device,
        )
        operators = {}
        previous = None
        for factor in FACTORS:
            operator = builder.build(factor)
            for axis in ("x", "y"):
                learned = getattr(session.context.response_operator, axis)
                reference = getattr(operator, axis)
                for i, (old, new) in enumerate(
                    zip(learned.blocks, reference.blocks, strict=True)
                ):
                    torch.testing.assert_close(old.valid_indices, new.valid_indices)
                    change = 0.0
                    if previous is not None:
                        prev = getattr(previous, axis).blocks[i].matrix
                        change = float((new.matrix - prev).norm() / new.matrix.norm())
                    self.block_rows.append(
                        dict(
                            example=session.run.example,
                            factor=factor,
                            axis=axis,
                            block=i,
                            learned_relative_difference=float(
                                (old.matrix - new.matrix).norm() / new.matrix.norm()
                            ),
                            previous_factor_relative_difference=change,
                        )
                    )
            operators[factor] = operator
            previous = operator
            self.logger.info(
                "%s reference factor=%d built", session.run.example, factor
            )
        self.cache[key] = operators
        return operators

    def evaluate_pair(
        self,
        session: FixedSourceSession,
        batch: ComplexCouplingBatch,
        pair: torch.Tensor,
        condition: str,
        k: int,
        operators: dict[int, FrozenBidirectionalResponseOperator],
    ) -> None:
        frozen = pair.clone()
        session.evaluation_operator = None
        learned, lc = session.fields(batch, pair)
        initial_fields = torch.stack((learned[:, 0], learned[:, 1], lc.u_pred_valid), 1)
        previous_fields = initial_fields
        previous_factor = 0
        norm = batch.sol_valid.norm(dim=1)
        if torch.any(norm <= 1e-12):
            raise ValueError("Undefined relative error for near-zero reference")
        for factor in (0, *FACTORS):
            session.evaluation_operator = operators[factor] if factor else None
            records = session.metrics(batch, pair, condition, k, pair)
            sol, cross = session.fields(batch, pair)
            fields = torch.stack((sol[:, 0], sol[:, 1], cross.u_pred_valid), 1)
            difference = (fields - initial_fields).norm(dim=2) / norm[:, None]
            refinement = (
                (fields - previous_fields).norm(dim=2) / norm[:, None]
                if previous_factor
                else torch.zeros_like(difference)
            )
            for i, row in enumerate(records):
                if factor == 0:
                    old = (
                        self.baseline[session.spec.run_id, k, row["sample_id"]]
                        if condition == "learned"
                        else self.reference[
                            session.run.example, condition, row["sample_id"]
                        ]
                    )
                    for name in (
                        "response_cost",
                        "rel_sol",
                        "rel_sol_equal_mean",
                        "rel_u_phi",
                        "rel_u_psi",
                    ):
                        np.testing.assert_allclose(
                            row[name], float(old[name]), rtol=1e-8, atol=1e-12
                        )
                    if row["fingerprint"] != old["fingerprint"]:
                        raise ValueError("Baseline operator fingerprint mismatch")
                row.update(
                    source_condition=condition,
                    kernel="reference" if factor else "learned",
                    factor=factor,
                    previous_factor=previous_factor,
                    J=row["response_cost"] / 2,
                    relative_mismatch=float((sol[i, 0] - sol[i, 1]).norm() / norm[i]),
                    source_sha256=hashlib.sha256(
                        pair[i].cpu().numpy().tobytes()
                    ).hexdigest(),
                    weak_error_improved_vs_learned=row["rel_sol"]
                    < float((lc.u_pred_valid[i] - batch.sol_valid[i]).norm() / norm[i]),
                )
                for j, name in enumerate(("x", "y", "weak")):
                    row[f"field_change_{name}"] = float(difference[i, j])
                    row[f"refinement_change_{name}"] = float(refinement[i, j])
                self.rows.append(row)
            previous_fields, previous_factor = fields, factor
            torch.testing.assert_close(pair, frozen, rtol=0, atol=0)
        session.evaluation_operator = None

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                self.args.manifest,
                self.out,
                batch_size=self.args.batch_size,
            )
        )
        audit.preflight()
        for base, names in (
            (self.args.learned_baseline, ("per_sample.csv", "minima.csv")),
            (self.args.reference_baseline, ("per_sample.csv",)),
        ):
            for name in names:
                path = (base / name).resolve()
                audit.hashes[str(path)] = _sha256(path)
        for path in [Path(__file__), *Path("src/greenonet").glob("*.py")]:
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "request.json",
            dict(
                factors=FACTORS,
                device="cuda:1",
                dtype="float64",
                batch_size=self.args.batch_size,
                run_id_filter=self.args.run_id,
                max_batches=self.args.max_batches,
                source_conditions="reference raw/balanced; learned K0/K2/prior mean-minimum/K64",
                source_quadrature="unchanged production weights; same evaluation nodes",
                weak_blend="same reconstruction rule; reliability weights recomputed",
                reoptimization=False,
            ),
        )
        _json(self.out / "environment.json", audit.runs[0].audit._environment())
        for run in audit.runs:
            if run.example not in {"unit_square", "disk"}:
                continue
            if self.args.run_id and run.spec.run_id not in self.args.run_id:
                continue
            audit._gpu_idle()
            run.audit.logger = self.logger
            self.logger.info("Starting %s", run.spec.run_id)
            session = FixedSourceSession(run)
            try:
                operators = self.operators(session)
                ks = sorted({0, 2, self.best[run.spec.run_id], 64})
                for index, batch in enumerate(session.batches):
                    if self.args.max_batches and index >= self.args.max_batches:
                        break
                    prepared = session._prepare(batch)
                    self.logger.info(
                        "%s batch %d: reproducing learned K64 sources",
                        run.spec.run_id,
                        index,
                    )
                    result = session._krylov(prepared, 64)
                    pairs = []
                    for k in ks:
                        pair = prepared.symmetric_physical
                        if k:
                            delta = result.deltas[k - 1]
                            pair = pair + torch.stack((delta, -delta), 1)
                        pairs.append(("learned", k, pair))
                    if run.spec.seed == 0:
                        balanced, _ = symmetric_source_balance(
                            batch.flux_valid, batch.rhs_valid
                        )
                        pairs.extend(
                            (
                                ("reference_raw", 0, batch.flux_valid),
                                ("reference_balanced", 0, balanced),
                            )
                        )
                    np.savez_compressed(
                        self.out
                        / "sources"
                        / f"{run.spec.run_id}_batch{index:03d}.npz",
                        allow_pickle=False,
                        sample_ids=batch.sample_indices.cpu().numpy(),
                        **{
                            f"{condition}_k{k}": pair.cpu().numpy()
                            for condition, k, pair in pairs
                        },
                    )
                    for condition, k, pair in pairs:
                        self.evaluate_pair(
                            session, batch, pair, condition, k, operators
                        )
                    _write_csv(self.out / "per_sample.csv", self.rows)
                    _write_csv(self.out / "summary.csv", summarize(self.rows))
                    _write_csv(self.out / "operator_comparison.csv", self.block_rows)
                    self.logger.info(
                        "%s batch %d/%d complete",
                        run.spec.run_id,
                        index + 1,
                        len(session.batches),
                    )
            finally:
                session.close()
        for path, expected in audit.hashes.items():
            if _sha256(Path(path)) != expected:
                raise RuntimeError(f"Input modified during evaluation: {path}")
        final = [r for r in self.rows if r["factor"] == FACTORS[-1]]
        convergence = max(
            r[f"refinement_change_{axis}"] for r in final for axis in ("x", "y", "weak")
        )
        _json(
            self.out / "verification.json",
            dict(
                input_hashes_unchanged=True,
                fixed_sources_bitwise_verified=True,
                learned_baseline_reproduced=True,
                rows=len(self.rows),
                full_test=not self.args.max_batches,
                final_refinement_field_change_max=convergence,
                convergence_threshold_relative_to_reference_solution=1e-7,
                convergence_passed=convergence <= 1e-7,
            ),
        )
        self.logger.info(
            "Finished %d rows; max normalized refinement change %.6e",
            len(self.rows),
            convergence,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/paper_source_initialization_audit.json"),
    )
    parser.add_argument(
        "--learned-baseline",
        type=Path,
        default=Path("docs/analysis/learned_k_extension_square_disk_v2"),
    )
    parser.add_argument(
        "--reference-baseline",
        type=Path,
        default=Path("docs/analysis/paper_source_initialization_audit_v2"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--max-batches", type=int)
    FixedSourceAudit(parser.parse_args()).run()


if __name__ == "__main__":
    main()
