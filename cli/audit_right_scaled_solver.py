"""GPU1 batch5 controlled right-scaling pilot; production models remain frozen."""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler
from torch.utils.dlpack import from_dlpack

from cli.audit_coupling_solver_pilot import BlockSystem
from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.complex_projection import apply_complex_balance_projection


def right_scale(denominator: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("Right scaling requires finite positive D")
    return np.asarray(1 / np.sqrt(denominator))


class GpuBlocks:
    """Padded segment batches, no global matrix in iterative products."""

    def __init__(self, response: Any) -> None:
        self.cp = importlib.import_module("cupy")
        self.sla = importlib.import_module("cupyx.scipy.sparse.linalg")
        self.n = response.point_count
        self.axes = []
        for axis in (response.x, response.y):
            width = max(len(b.valid_indices) for b in axis.blocks)
            idx = torch.zeros(
                (len(axis.blocks), width), device="cuda:1", dtype=torch.long
            )
            mask = torch.zeros_like(idx, dtype=torch.bool)
            matrices = torch.zeros(
                (len(axis.blocks), width, width), device="cuda:1", dtype=torch.float64
            )
            for i, block in enumerate(axis.blocks):
                n = len(block.valid_indices)
                idx[i, :n] = block.valid_indices
                mask[i, :n] = True
                matrices[i, :n, :n] = block.matrix
            self.axes.append(
                tuple(self.cp.from_dlpack(t) for t in (idx, mask, matrices))
            )

    def axis(self, v: Any, axis: int, transpose: bool = False) -> Any:
        idx, mask, mat = self.axes[axis]
        product = self.cp.matmul(
            mat.swapaxes(1, 2) if transpose else mat, v[idx][..., None]
        )[..., 0]
        out = self.cp.empty(self.n, dtype=self.cp.float64)
        out[idx[mask]] = product[mask]
        return out

    def forward(self, v: Any) -> Any:
        return self.axis(v, 0) + self.axis(v, 1)

    def adjoint(self, v: Any) -> Any:
        return self.axis(v, 0, True) + self.axis(v, 1, True)

    def operator(self, p: Any) -> Any:
        return self.sla.LinearOperator(
            (self.n, self.n),
            matvec=lambda z: self.forward(p * z),
            rmatvec=lambda v: p * self.adjoint(v),
            dtype=self.cp.float64,
        )


class RightScaledAudit:
    def __init__(self, out: Path) -> None:
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.cp = importlib.import_module("cupy")
        self.log = logging.getLogger("right_scaled")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        for handler in (
            RichHandler(
                rich_tracebacks=True, show_path=False, omit_repeated_times=False
            ),
            logging.FileHandler(self.out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.log.addHandler(handler)
        self.metrics: list[dict[str, Any]] = []
        self.timings: list[dict[str, Any]] = []
        self.stops: list[dict[str, Any]] = []

    def sync(self) -> None:
        torch.cuda.synchronize(1)

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
        audit.preflight()
        for path in (Path(__file__), Path("cli/audit_coupling_solver_pilot.py")):
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "protocol.json",
            dict(
                batch_size=5,
                device="cuda:1",
                dtype="float64",
                samples="first5 seed0 Square/Disk; pilot not full-test",
                warmup=3,
                repeats=5,
                start="prepared GPU inputs; compute neural initialization or RHS inside timing",
                end="cached Green + production weak reconstruction",
                reconstruction="shared cached response for all methods; compare values to archived production output; historical timing not reused",
                preconditioner="P=D**(-1/2) from unchanged production context including existing damping; identity control",
                iterations="independent RHS solved sequentially within batch5; not block Krylov",
                cupy=self.cp.__version__,
                torch=torch.__version__,
                limits="GMRES restart100/maxiter10000, rtol1e-11; LSMR atol/btol1e-13,maxiter10000,conlim1e12",
                scope="no solver-specific stronger preconditioner or matched-accuracy iteration selection",
            ),
        )
        for run in audit.runs:
            if run.spec.seed != 0 or run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            self.log.info("Preparing %s", run.example)
            session = SourceRunSession(run)
            try:
                self.evaluate(session)
            finally:
                session.close()
                self.cp.get_default_memory_pool().free_all_blocks()
        for filename, digest in audit.hashes.items():
            if _sha256(Path(filename)) != digest:
                raise RuntimeError(f"Input changed: {filename}")
        _json(
            self.out / "verification.json",
            dict(
                input_hashes_unchanged=True,
                rows=len(self.metrics),
                timing_rows=len(self.timings),
                full_test=False,
            ),
        )

    def evaluate(self, session: SourceRunSession) -> None:
        cp = self.cp
        example = session.run.example
        batch = session.batches[0]
        f = batch.rhs_valid
        self.sync()
        start = time.perf_counter()
        gpu = GpuBlocks(session.context.response_operator)
        p = cp.asarray(right_scale(session.context.denominator.cpu().numpy()))
        operators = {
            "identity": gpu.operator(cp.ones_like(p)),
            "scaled": gpu.operator(p),
        }
        self.sync()
        scaling_setup = time.perf_counter() - start
        random = np.random.default_rng(12).normal(size=gpu.n)
        v = torch.tensor(random[None], device="cuda:1")
        original = session.context.response_operator
        np.testing.assert_allclose(
            cp.asnumpy(gpu.forward(cp.asarray(random))),
            (original.x.forward(v) + original.y.forward(v))[0].cpu().numpy(),
            rtol=1e-10,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            cp.asnumpy(gpu.adjoint(cp.asarray(random))),
            (original.x.adjoint(v) + original.y.adjoint(v))[0].cpu().numpy(),
            rtol=1e-10,
            atol=1e-13,
        )
        self.sync()
        start = time.perf_counter()
        matrix = torch.tensor(BlockSystem(original).matrix.toarray(), device="cuda:1")
        self.sync()
        assembly = time.perf_counter() - start
        start = time.perf_counter()
        lu, piv = torch.linalg.lu_factor(matrix)
        self.sync()
        lu_setup = time.perf_counter() - start
        start = time.perf_counter()
        q, r = torch.linalg.qr(matrix)
        self.sync()
        qr_setup = time.perf_counter() - start
        if torch.any(
            r.diagonal().abs()
            <= torch.finfo(torch.float64).eps
            * gpu.n
            * torch.linalg.norm(r, ord=float("inf"))
        ):
            raise ValueError("QR rank screen failed")
        del matrix
        _json(
            self.out / f"{example}_setup.json",
            dict(
                padded_blocks_and_scaling_seconds=scaling_setup,
                assembly_transfer_seconds=assembly,
                lu_factor_seconds=lu_setup,
                qr_factor_seconds=qr_setup,
                p_min=float(p.min()),
                p_max=float(p.max()),
            ),
        )
        methods = [
            "learned",
            "lu",
            "qr",
            "gmres_identity",
            "gmres_scaled",
            "lsmr_identity",
            "lsmr_scaled",
        ]

        def predict(method: str, record: bool = False) -> tuple[torch.Tensor, Any]:
            if method == "learned":
                raw, _ = session.model.forward_with_fusion_diagnostics(
                    geometry=batch.geometry,
                    x_source_branch=batch.x_source_branch,
                    y_source_branch=batch.y_source_branch,
                    x_source_amplitude=batch.x_source_amplitude,
                    y_source_amplitude=batch.y_source_amplitude,
                    x_coefficient_branch=batch.x_coefficient_branch,
                    y_coefficient_branch=batch.y_coefficient_branch,
                    rhs_phys=f,
                )
                projected = apply_complex_balance_projection(
                    raw_response=raw,
                    rhs_phys=f,
                    geometry=batch.geometry,
                    config=session.projection_by_k[session.run.native_k],
                    symmetric_tangent_context=session.context_by_k[
                        session.run.native_k
                    ],
                )
                pair = projected.projected_physical
            else:
                rhs = 0.5 * (original.y.forward(f) - original.x.forward(f))
                if method == "lu":
                    delta = torch.linalg.lu_solve(lu, piv, rhs.T).T
                elif method == "qr":
                    delta = torch.linalg.solve_triangular(r, q.T @ rhs.T, upper=True).T
                else:
                    solver, scaling = method.split("_")
                    scale = p if scaling == "scaled" else cp.ones_like(p)
                    columns = []
                    for i, b in enumerate(cp.from_dlpack(rhs)):
                        if solver == "gmres":
                            history: list[Any] = []
                            z, status = gpu.sla.gmres(
                                operators[scaling],
                                b,
                                rtol=1e-11,
                                atol=0,
                                restart=100,
                                maxiter=10000,
                                callback=history.append,
                                callback_type="pr_norm",
                            )
                            iterations = len(history) * 100
                        else:
                            fit = gpu.sla.lsmr(
                                operators[scaling],
                                b,
                                atol=1e-13,
                                btol=1e-13,
                                conlim=1e12,
                                maxiter=10000,
                            )
                            z, status, iterations = fit[:3]
                        columns.append(scale * z)
                        if record:
                            self.stops.append(
                                dict(
                                    example=example,
                                    method=method,
                                    sample_id=int(batch.sample_indices[i]),
                                    status=int(status),
                                    iterations=int(iterations),
                                )
                            )
                    delta = from_dlpack(cp.stack(columns))
                pair = torch.stack((f / 2 + delta, f / 2 - delta), 1)
            return pair, session.fields(batch, pair)[1]

        cached = predict("learned")[1].u_pred_valid
        production_result = session._prediction_forward(batch, session.run.native_k)
        assert isinstance(production_result, ComplexCrossAxisReconstructionResult)
        production = production_result.u_pred_valid
        torch.testing.assert_close(cached, production, rtol=1e-8, atol=1e-12)
        lu_pair, _ = predict("lu")
        for method in methods:
            self.log.info("%s diagnostic %s", example, method)
            pair, _ = predict(method, True)
            rows = session.metrics(
                batch,
                pair,
                method,
                session.run.native_k if method == "learned" else 0,
                torch.stack((f / 2, f / 2), 1),
            )
            rhs = 0.5 * (original.y.forward(f) - original.x.forward(f))
            response = original.forward_pair(pair)
            for i, row in enumerate(rows):
                row.update(
                    relative_equation_residual=float(
                        (response[i, 0] - response[i, 1]).norm() / rhs[i].norm()
                    ),
                    relative_d_to_lu=float(
                        (pair[i, 0] - lu_pair[i, 0]).norm()
                        / (lu_pair[i, 0] - f[i] / 2).norm()
                    ),
                )
            self.metrics.extend(rows)
            _write_csv(self.out / "per_sample.csv", self.metrics)
            _write_csv(self.out / "stops.csv", self.stops)
        for repeat in range(-3, 5):
            for method in methods[:: (-1 if repeat % 2 else 1)]:
                self.sync()
                start = time.perf_counter()
                predict(method)
                self.sync()
                elapsed = time.perf_counter() - start
                if repeat >= 0:
                    self.timings.append(
                        dict(
                            example=example,
                            method=method,
                            repeat=repeat,
                            batch_size=5,
                            seconds=elapsed,
                        )
                    )
                    _write_csv(self.out / "timing.csv", self.timings)
                self.log.info("%s %s repeat%d %.4fs", example, method, repeat, elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    cp = importlib.import_module("cupy")
    with (
        cp.cuda.Device(1),
        cp.cuda.ExternalStream(torch.cuda.current_stream(1).cuda_stream),
    ):
        RightScaledAudit(args.outdir).run()
