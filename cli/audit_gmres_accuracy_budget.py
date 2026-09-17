"""Frozen scaled-GMRES accuracy budgets and independently measured latency."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.dlpack import from_dlpack

from cli.audit_right_scaled_solver import GpuBlocks, right_scale
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)


def restart_for_budget(budget: int) -> int:
    if budget < 1 or (budget > 100 and budget % 100):
        raise ValueError("Use 1..100 or multiples of 100")
    return min(100, budget)


def first_match(
    rows: list[dict[str, Any]], mean: float, p95: float, require_tail: bool
) -> int | None:
    return next(
        (
            int(r["budget"])
            for r in sorted(rows, key=lambda r: r["budget"])
            if r["mean"] <= mean and (not require_tail or r["p95"] <= p95)
        ),
        None,
    )


class AccuracyBudgetAudit:
    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=False)
        self.out = out
        self.cp = importlib.import_module("cupy")
        self.log = logging.getLogger("gmres_accuracy_budget")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        for handler in (
            RichHandler(show_path=True, omit_repeated_times=False),
            logging.FileHandler(out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.log.addHandler(handler)
        self.rows: list[dict[str, Any]] = []
        self.curves: list[dict[str, Any]] = []
        self.timings: list[dict[str, Any]] = []
        self.matches: list[dict[str, Any]] = []

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                Path("configs/paper_source_initialization_audit.json"),
                self.out,
                batch_size=5,
            )
        )
        previous = json.loads(
            Path(
                "docs/analysis/right_scaled_solver_pilot/input_hashes.json"
            ).read_text()
        )
        audit.preflight()
        for path, digest in audit.hashes.items():
            if previous[path] != digest:
                raise ValueError(f"Changed numerical input: {path}")
        code = [
            Path(__file__),
            Path("cli/audit_right_scaled_solver.py"),
            Path("/tmp/greennet-cupy-pilot/cupyx/scipy/sparse/linalg/_iterative.py"),
        ]
        hashes = {**audit.hashes, **{str(p.resolve()): _sha256(p) for p in code}}
        _json(self.out / "execution_inputs.json", hashes)
        _json(
            self.out / "protocol.json",
            dict(
                device="cuda:1",
                dtype="float64",
                batch_size=5,
                seed=0,
                samples="first5, identical to right_scaled_solver_pilot",
                budgets=list(range(101)) + list(range(200, 701, 100)),
                scaling="unchanged D**(-1/2)",
                initial="phi=psi=f/2",
                rtol=1e-11,
                atol=0,
                restart="min(100,budget); no restart before first100",
                warmup=3,
                repeats=5,
                scope="prepared GPU inputs -> final production weak reconstruction",
                matching="retrospective mean and mean+p95; reference never enters solver",
                timing_budgets="0,1,2,4,8,16,32,64,100,200,400,700 and first crossings",
                caveat="standard single-RHS solves sequential within batch5; not block GMRES",
            ),
        )
        for run in audit.runs:
            if run.spec.seed != 0 or run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            session = SourceRunSession(run)
            try:
                self.evaluate(session)
            finally:
                session.close()
                self.cp.get_default_memory_pool().free_all_blocks()
        for path, digest in hashes.items():
            if _sha256(Path(path)) != digest:
                raise ValueError(f"Input changed during experiment: {path}")
        _json(
            self.out / "verification.json",
            dict(
                input_hashes_unchanged=True,
                rows=len(self.rows),
                timing_rows=len(self.timings),
                full_test=False,
            ),
        )

    def evaluate(self, session: SourceRunSession) -> None:
        cp = self.cp
        batch = session.batches[0]
        f = batch.rhs_valid
        example = session.run.example
        original = session.context.response_operator
        gpu = GpuBlocks(original)
        p = cp.asarray(right_scale(session.context.denominator.cpu().numpy()))
        operator = gpu.operator(p)

        def predict(
            budget: int, record: bool = False
        ) -> tuple[Any, Any, list[dict[str, int]]]:
            rhs = 0.5 * (original.y.forward(f) - original.x.forward(f))
            stops = []
            columns = []
            for i, b in enumerate(cp.from_dlpack(rhs)):
                if budget == 0:
                    z, status, iterations = cp.zeros_like(b), -1, 0
                else:
                    history: list[Any] = []
                    z, status = gpu.sla.gmres(
                        operator,
                        b,
                        rtol=1e-11,
                        atol=0,
                        restart=restart_for_budget(budget),
                        maxiter=budget,
                        callback=history.append,
                        callback_type="pr_norm",
                    )
                    iterations = len(history) * restart_for_budget(budget)
                columns.append(p * z)
                if record:
                    stops.append(dict(status=int(status), iterations=iterations))
            delta = from_dlpack(cp.stack(columns))
            pair = torch.stack((f / 2 + delta, f / 2 - delta), 1)
            return pair, session.fields(batch, pair)[1], stops

        native = session._prediction_forward(batch, session.run.native_k)
        assert isinstance(native, ComplexCrossAxisReconstructionResult)
        native_errors = (
            (
                (native.u_pred_valid - batch.sol_valid).norm(dim=-1)
                / batch.sol_valid.norm(dim=-1)
            )
            .cpu()
            .numpy()
        )
        with Path(
            "docs/analysis/right_scaled_solver_pilot/per_sample.csv"
        ).open() as stream:
            old = [
                r
                for r in csv.DictReader(stream)
                if r["example"] == example and r["condition"] == "learned"
            ]
        np.testing.assert_allclose(
            native_errors, [float(r["rel_sol"]) for r in old], rtol=1e-8, atol=1e-12
        )
        target_mean, target_p95 = (
            float(native_errors.mean()),
            float(np.quantile(native_errors, 0.95)),
        )
        curve = []
        for budget in list(range(101)) + list(range(200, 701, 100)):
            pair, result, stops = predict(budget, True)
            errors = (
                (
                    (result.u_pred_valid - batch.sol_valid).norm(dim=-1)
                    / batch.sol_valid.norm(dim=-1)
                )
                .cpu()
                .numpy()
            )
            balance = float((pair.sum(1) - f).abs().max())
            if balance > 1e-11 or not np.isfinite(errors).all():
                raise ValueError("Invalid fields/balance")
            response = original.forward_pair(pair)
            rhs = 0.5 * (original.y.forward(f) - original.x.forward(f))
            residuals = (response[:, 0] - response[:, 1]).norm(dim=-1) / rhs.norm(
                dim=-1
            )
            for i, error in enumerate(errors):
                self.rows.append(
                    dict(
                        example=example,
                        budget=budget,
                        sample_id=int(batch.sample_indices[i]),
                        rel_sol=float(error),
                        relative_residual=float(residuals[i]),
                        balance_max=balance,
                        **stops[i],
                    )
                )
            item = dict(
                example=example,
                budget=budget,
                mean=float(errors.mean()),
                p95=float(np.quantile(errors, 0.95)),
                maximum=float(errors.max()),
            )
            curve.append(item)
            self.curves.append(item)
            if budget % 10 == 0:
                self.log.info(
                    "%s budget%d mean%%=%.6f p95%%=%.6f",
                    example,
                    budget,
                    100 * item["mean"],
                    100 * item["p95"],
                )
        _write_csv(self.out / "per_sample.csv", self.rows)
        _write_csv(self.out / "accuracy_curve.csv", self.curves)
        mean_budget = first_match(curve, target_mean, target_p95, False)
        tail_budget = first_match(curve, target_mean, target_p95, True)
        self.matches.append(
            dict(
                example=example,
                target_mean=target_mean,
                target_p95=target_p95,
                mean_budget=mean_budget,
                mean_p95_budget=tail_budget,
            )
        )
        _write_csv(self.out / "matches.csv", self.matches)
        budgets = sorted(
            {0, 1, 2, 4, 8, 16, 32, 64, 100, 200, 400, 700}
            | {b for b in (mean_budget, tail_budget) if b is not None}
        )
        conditions = [-1] + budgets
        self.log.info(
            "%s crossings mean=%s mean+p95=%s", example, mean_budget, tail_budget
        )
        for repeat in range(-3, 5):
            for budget in conditions[:: (-1 if repeat % 2 else 1)]:
                torch.cuda.synchronize(1)
                start = time.perf_counter()
                if budget == -1:
                    session._prediction_forward(batch, session.run.native_k)
                else:
                    predict(budget)
                torch.cuda.synchronize(1)
                elapsed = time.perf_counter() - start
                if repeat >= 0:
                    self.timings.append(
                        dict(
                            example=example,
                            budget=budget,
                            repeat=repeat,
                            seconds=elapsed,
                            batch_size=5,
                        )
                    )
                    _write_csv(self.out / "timing.csv", self.timings)
            self.log.info("%s timing repeat%d complete", example, repeat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    cp = importlib.import_module("cupy")
    with (
        cp.cuda.Device(1),
        cp.cuda.ExternalStream(torch.cuda.current_stream(1).cuda_stream),
    ):
        AccuracyBudgetAudit(args.outdir).run()
