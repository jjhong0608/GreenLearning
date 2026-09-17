"""Reoptimize archived learned initial sources with a reference Green operator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.logging import RichHandler

from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_axial_response_operator import (
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.complex_tangent_projection import (
    KrylovSubspaceStepResult,
    SymmetricTangentGreenResponseContext,
    matrix_free_krylov_subspace_step,
)
from greenonet.config import SymmetricTangentGreenResponseProjectionConfig
from greenonet.reference_green_reconstruction import DiffusionReferenceBuilder
from greenonet.source_initialization_metrics import summarize


def reference_context(
    operator: FrozenBidirectionalResponseOperator,
    mass: torch.Tensor | float,
    config: SymmetricTangentGreenResponseProjectionConfig,
    max_k: int,
) -> SymmetricTangentGreenResponseContext:
    return SymmetricTangentGreenResponseContext.from_response_operator(
        response_operator=operator,
        point_mass=mass,
        config=replace(config, subspace_dimension=max_k, max_subspace_dimension=max_k),
    )


def optimize_sources(
    context: SymmetricTangentGreenResponseContext,
    pair: torch.Tensor,
    max_k: int,
) -> KrylovSubspaceStepResult:
    response = context.response_operator.forward_pair(pair)
    mismatch = response[:, 0] - response[:, 1]
    return matrix_free_krylov_subspace_step(
        context=context,
        mismatch=mismatch,
        gradient=context.tangent_gradient(mismatch),
        max_dimension=max_k,
        relative_eps=context.line_search_relative_eps,
        monotonicity_relative_tol=1e-10,
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


class ReferenceReoptimizationAudit:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out = args.outdir.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "sources").mkdir()
        self.logger = logging.getLogger("reference_reoptimization")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(
                console=Console(width=140),
                rich_tracebacks=True,
                show_path=False,
                omit_repeated_times=False,
            ),
            logging.FileHandler(self.out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)
        self.rows: list[dict[str, Any]] = []
        self.verification: list[dict[str, Any]] = []
        self.context_rows: list[dict[str, Any]] = []
        self.baseline = {
            (r["run_id"], int(r["evaluation_k"]), int(r["sample_id"])): r
            for r in read_csv(args.fixed_baseline / "per_sample.csv")
            if r["source_condition"] == "learned" and r["factor"] == "256"
        }
        self.initial_hashes = {
            (r["run_id"], int(r["sample_id"])): r["source_sha256"]
            for r in read_csv(args.fixed_baseline / "per_sample.csv")
            if r["source_condition"] == "learned"
            and r["factor"] == "0"
            and r["evaluation_k"] == "0"
        }
        self.operator_cache: dict[str, FrozenBidirectionalResponseOperator] = {}

    def archived_initial(self, run_id: str) -> dict[int, np.ndarray]:
        pairs = {}
        for path in sorted(
            (self.args.fixed_baseline / "sources").glob(f"{run_id}_batch*.npz")
        ):
            with np.load(path, allow_pickle=False) as data:
                array = data["learned_k0"]
                for i, sample in enumerate(data["sample_ids"]):
                    if int(sample) in pairs:
                        raise ValueError("Duplicate archived initial source")
                    value = array[i].copy()
                    digest = hashlib.sha256(value.tobytes()).hexdigest()
                    if digest != self.initial_hashes[run_id, int(sample)]:
                        raise ValueError("Archived initial source hash differs")
                    pairs[int(sample)] = value
        if not pairs:
            raise ValueError(f"No archived source for {run_id}")
        return pairs

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
        for path in [
            Path(__file__),
            *Path("src/greenonet").glob("*.py"),
            self.args.fixed_baseline / "per_sample.csv",
            self.args.fixed_baseline / "independent_gauss_verification.json",
            *sorted((self.args.fixed_baseline / "sources").glob("*.npz")),
        ]:
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "request.json",
            dict(
                initial="archived learned balance-projected K0; bitwise source identity",
                operator_for_optimization="reference factor256",
                operator_for_reconstruction="reference factor256",
                max_k=64,
                full_curve=True,
                preconditioner="rebuilt from reference response with unchanged config",
                source_quadrature="unchanged production weights and L^2 scaling",
                fixed_baseline=str(self.args.fixed_baseline),
                run_ids=self.args.run_id,
                max_batches=self.args.max_batches,
                device="cuda:1",
                dtype="float64",
            ),
        )
        _json(self.out / "environment.json", audit.runs[0].audit._environment())
        for run in audit.runs:
            if run.example not in {"unit_square", "disk"}:
                continue
            if self.args.run_id and run.spec.run_id not in self.args.run_id:
                continue
            audit._gpu_idle()
            self.logger.info("Starting %s", run.spec.run_id)
            run.audit.logger = self.logger
            session = SourceRunSession(run)
            try:
                archived = self.archived_initial(run.spec.run_id)
                if len(archived) != len(session.dataset):
                    raise ValueError("Archived source coverage differs")
                original_context = session.context
                key = run.spec.fingerprint
                builder = DiffusionReferenceBuilder(
                    session.batches[0].geometry,
                    load_coefficient_functions(run.spec.coefficients),
                    session.device,
                )
                if key not in self.operator_cache:
                    self.operator_cache[key] = builder.build(256)
                operator = self.operator_cache[key]
                session.context = reference_context(
                    operator, original_context.point_mass, session.tangent, 64
                )
                self.context_rows.append(
                    dict(
                        run_id=run.spec.run_id,
                        original_tangent_settings=asdict(session.tangent),
                        reference_context_statistics=session.context.statistics(),
                        denominator_relative_change=float(
                            (
                                session.context.denominator
                                - original_context.denominator
                            ).norm()
                            / original_context.denominator.norm()
                        ),
                        response_operator_changed=session.context.response_operator
                        is not original_context.response_operator,
                    )
                )
                seen = []
                for index, batch in enumerate(session.batches):
                    if self.args.max_batches and index >= self.args.max_batches:
                        break
                    ids = batch.sample_indices.tolist()
                    initial = torch.from_numpy(np.stack([archived[i] for i in ids])).to(
                        session.device
                    )
                    # The archived pair is the control. Fresh network evaluation is
                    # checked only; no learned correction is used as the new start.
                    prepared = session._prepare(batch)
                    torch.testing.assert_close(
                        initial, prepared.symmetric_physical, rtol=1e-12, atol=1e-12
                    )
                    frozen = initial.clone()
                    result = optimize_sources(session.context, initial, 64)
                    response = operator.forward_pair(initial)
                    initial_cost = session.context.point_mass * (
                        response[:, 0] - response[:, 1]
                    ).square().sum(1)
                    previous = torch.cat((initial_cost[None], result.costs[:-1]))
                    if torch.any(
                        result.costs
                        > previous
                        + 1e-10 * initial_cost[None]
                        + torch.finfo(torch.float64).tiny
                    ):
                        raise RuntimeError("Nonmonotone reference objective")
                    if index == 0:
                        prefix = optimize_sources(session.context, initial, 2)
                        torch.testing.assert_close(
                            prefix.final_delta, result.deltas[1], rtol=1e-8, atol=1e-12
                        )
                    norm = batch.sol_valid.norm(dim=1)
                    saved = {
                        "initial": initial.cpu().numpy(),
                        "sample_ids": batch.sample_indices.cpu().numpy(),
                    }
                    for k in range(65):
                        pair = (
                            initial
                            if k == 0
                            else initial
                            + torch.stack(
                                (result.deltas[k - 1], -result.deltas[k - 1]), 1
                            )
                        )
                        rows = session.metrics(
                            batch, pair, "learned", k, initial, result if k else None
                        )
                        for i, row in enumerate(rows):
                            if k == 0:
                                old = self.baseline[
                                    run.spec.run_id, 0, row["sample_id"]
                                ]
                                for metric in (
                                    "response_cost",
                                    "rel_sol",
                                    "rel_sol_equal_mean",
                                    "rel_u_phi",
                                    "rel_u_psi",
                                ):
                                    np.testing.assert_allclose(
                                        row[metric],
                                        float(old[metric]),
                                        rtol=1e-8,
                                        atol=1e-12,
                                    )
                            else:
                                np.testing.assert_allclose(
                                    row["response_cost"],
                                    float(result.costs[k - 1, i]),
                                    rtol=1e-7,
                                    atol=1e-18,
                                )
                            row.update(
                                optimization_operator="reference",
                                reconstruction_operator="reference",
                                reference_factor=256,
                                J=row["response_cost"] / 2,
                                relative_mismatch=float(
                                    torch.sqrt(
                                        torch.as_tensor(
                                            row["response_cost"],
                                            dtype=torch.float64,
                                            device=session.device,
                                        )
                                        / session.context.point_mass
                                    )
                                    / norm[i]
                                ),
                                initial_source_sha256=self.initial_hashes[
                                    run.spec.run_id, row["sample_id"]
                                ],
                            )
                            self.rows.append(row)
                        if k in (2, 4, 8, 16, 32, 48, 64):
                            saved[f"reference_optimized_k{k}"] = pair.cpu().numpy()
                    torch.testing.assert_close(initial, frozen, rtol=0, atol=0)
                    np.savez_compressed(
                        self.out
                        / "sources"
                        / f"{run.spec.run_id}_batch{index:03d}.npz",
                        allow_pickle=False,
                        **saved,
                    )
                    seen.extend(ids)
                    _write_csv(self.out / "per_sample.csv", self.rows)
                    stats = summarize(self.rows)
                    for row in stats:
                        for name in ("mean", "p95", "max"):
                            row[f"J_{name}"] = row[f"response_cost_{name}"] / 2
                    _write_csv(self.out / "summary.csv", stats)
                    _json(self.out / "contexts.json", self.context_rows)
                    self.logger.info(
                        "%s batch %d/%d complete (K0..64)",
                        run.spec.run_id,
                        index + 1,
                        len(session.batches),
                    )
                self.verification.append(
                    dict(
                        run_id=run.spec.run_id,
                        samples=len(seen),
                        initial_source_hashes_verified=True,
                        initial_network_reproduction=True,
                        reference_K0_reproduces_fixed_source_audit=True,
                        reference_context_rebuilt=True,
                        objective_monotone=True,
                        direct_response_matches_solver_costs=True,
                        independent_K2_prefix_verified=True,
                    )
                )
                _json(self.out / "verification_runs.json", self.verification)
            finally:
                session.close()
        for path, expected in audit.hashes.items():
            if _sha256(Path(path)) != expected:
                raise RuntimeError(f"Input changed: {path}")
        _json(
            self.out / "verification.json",
            dict(
                input_hashes_unchanged=True,
                rows=len(self.rows),
                runs=len(self.verification),
                full_test=not self.args.max_batches,
                reference_reoptimization=True,
            ),
        )
        self.logger.info(
            "Finished %d rows, %d runs", len(self.rows), len(self.verification)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/paper_source_initialization_audit.json"),
    )
    parser.add_argument(
        "--fixed-baseline",
        type=Path,
        default=Path("docs/analysis/reference_green_fixed_sources"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--run-id", action="append")
    parser.add_argument("--max-batches", type=int)
    ReferenceReoptimizationAudit(parser.parse_args()).run()


if __name__ == "__main__":
    main()
