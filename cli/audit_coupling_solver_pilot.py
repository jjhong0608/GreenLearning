"""Fixed learned-operator structure audit and three-source numerical solver pilot."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import scipy.sparse as sp
import torch
from rich.logging import RichHandler
from scipy.sparse.csgraph import structural_rank
from scipy.sparse.linalg import LinearOperator, gmres, lsmr

from greenonet.complex_axial_response_operator import (
    FrozenBidirectionalResponseOperator,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)


class BlockSystem:
    """Independent CPU block products; assembled CSC is used only for direct paths."""

    def __init__(self, response: FrozenBidirectionalResponseOperator) -> None:
        self.n = response.point_count
        self.blocks = [
            [
                (b.valid_indices.cpu().numpy(), b.matrix.cpu().numpy())
                for b in axis.blocks
            ]
            for axis in (response.x, response.y)
        ]
        rows, cols, data = [], [], []
        for blocks in self.blocks:
            for idx, mat in blocks:
                rows.append(np.repeat(idx, len(idx)))
                cols.append(np.tile(idx, len(idx)))
                data.append(mat.ravel())
        self.matrix = sp.coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n),
        ).tocsc()
        self.matrix.eliminate_zeros()
        self.operator = LinearOperator(
            (self.n, self.n),
            matvec=self.forward,
            rmatvec=self.adjoint,
            dtype=np.float64,
        )

    def axis(self, v: np.ndarray, axis: int, transpose: bool = False) -> np.ndarray:
        out = np.zeros(self.n)
        for idx, mat in self.blocks[axis]:
            out[idx] = (mat.T if transpose else mat) @ np.asarray(v).reshape(-1)[idx]
        return out

    def forward(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(self.axis(v, 0) + self.axis(v, 1))

    def adjoint(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(self.axis(v, 0, True) + self.axis(v, 1, True))


def direct_solutions(
    matrix: np.ndarray, rhs: np.ndarray, device: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    a = torch.as_tensor(matrix, device=device, dtype=torch.float64)
    b = torch.as_tensor(rhs, device=device, dtype=torch.float64)

    def sync() -> None:
        if a.is_cuda:
            torch.cuda.synchronize(a.device)

    sync()
    start = time.perf_counter()
    q, r = torch.linalg.qr(a)
    sync()
    qr_setup = time.perf_counter() - start
    diag = r.diagonal().abs()
    threshold = (
        torch.finfo(a.dtype).eps * len(matrix) * torch.linalg.norm(r, ord=float("inf"))
    )
    # This is a screening test, not a rank-revealing SVD certificate.
    if bool((diag <= threshold).any()):
        raise ValueError("QR rank screen failed: requires rank-revealing fallback")
    start = time.perf_counter()
    qr = torch.linalg.solve_triangular(r, q.T @ b, upper=True)
    sync()
    qr_solve = time.perf_counter() - start
    del q
    start = time.perf_counter()
    lu, piv, status = torch.linalg.lu_factor_ex(a)
    sync()
    lu_setup = time.perf_counter() - start
    if int(status) != 0:
        raise ValueError(f"LU factorization failed: {int(status)}")
    start = time.perf_counter()
    x = torch.linalg.lu_solve(lu, piv, b)
    sync()
    lu_solve = time.perf_counter() - start
    return {"lu": x.cpu().numpy(), "qr": qr.cpu().numpy()}, dict(
        qr_full_rank_screen=True,
        qr_diagonal_min=float(diag.min()),
        qr_rank_screen_threshold=float(threshold),
        lu_status=int(status),
        qr_setup_seconds=qr_setup,
        lu_setup_seconds=lu_setup,
        qr_three_rhs_seconds=qr_solve,
        lu_three_rhs_seconds=lu_solve,
        direct_device=device,
        rank_certificate="unpivoted QR screen only",
    )


class SolverPilot:
    def __init__(self, out: Path) -> None:
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.log = logging.getLogger("solver_pilot")
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        for handler in (
            RichHandler(
                rich_tracebacks=True, show_path=True, omit_repeated_times=False
            ),
            logging.FileHandler(self.out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.log.addHandler(handler)
        self.structures: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                Path("configs/paper_source_initialization_audit.json"),
                self.out,
                batch_size=3,
            )
        )
        audit.preflight()
        audit.hashes[str(Path(__file__).resolve())] = _sha256(Path(__file__))
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "protocol.json",
            dict(
                samples="first batch: 3 sources, seed0 for each problem; no outcome selection",
                structure_examples=["unit_square", "disk", "annulus", "pentagram"],
                solver_examples=["unit_square", "disk"],
                direct="GPU1 dense LU and QR, reusable factors, batched RHS",
                iterative="CPU4 independent segment products, unpreconditioned, zero initial d",
                gmres="rtol=1e-11, atol=0, restart=100, maxiter=100 cycles",
                lsmr="atol=btol=1e-13, conlim=1e12, maxiter=10000",
                timing="single pilot measurements only; CPU/GPU mixed, NOT a speed comparison",
                mass="positive scalar hx*hy; scaling objective by mass does not change minimizer",
                torch=torch.__version__,
                scipy=scipy.__version__,
            ),
        )
        for run in audit.runs:
            if run.spec.seed != 0:
                continue
            audit._gpu_idle()
            self.log.info("Preparing %s", run.example)
            session = SourceRunSession(run)
            try:
                operator = session.context.response_operator
                sys = BlockSystem(operator)
                mat = sys.matrix
                rng = np.random.default_rng(20260917)
                v, w = rng.normal(size=(2, sys.n))
                v_t = torch.tensor(v[None], device=session.device)
                forward = (
                    (operator.x.forward(v_t) + operator.y.forward(v_t))[0].cpu().numpy()
                )
                adjoint = (
                    (operator.x.adjoint(v_t) + operator.y.adjoint(v_t))[0].cpu().numpy()
                )
                np.testing.assert_allclose(
                    sys.forward(v), forward, rtol=1e-10, atol=1e-13
                )
                np.testing.assert_allclose(
                    sys.adjoint(v), adjoint, rtol=1e-10, atol=1e-13
                )
                np.testing.assert_allclose(mat @ v, forward, rtol=1e-10, atol=1e-13)
                np.testing.assert_allclose(mat.T @ v, adjoint, rtol=1e-10, atol=1e-13)
                norm = float(sp.linalg.norm(mat))
                row = dict(
                    example=run.example,
                    n=sys.n,
                    nnz=mat.nnz,
                    density=mat.nnz / sys.n**2,
                    dense_gib=8 * sys.n**2 / 2**30,
                    csc_mib=(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes)
                    / 2**20,
                    qr_minimum_a_q_r_gib=24 * sys.n**2 / 2**30,
                    zero_rows=int(np.sum(np.asarray(abs(mat).sum(1)).ravel() == 0)),
                    zero_columns=int(np.sum(np.asarray(abs(mat).sum(0)).ravel() == 0)),
                    structural_rank=structural_rank(mat),
                    asymmetry_fro=float(sp.linalg.norm(mat - mat.T)) / norm,
                    point_mass=float(session.context.point_mass),
                    block_entries=sum(m.size for axis in sys.blocks for _, m in axis),
                    forward_relative_error=float(
                        np.linalg.norm(sys.forward(v) - forward)
                        / np.linalg.norm(forward)
                    ),
                    adjoint_dot_relative_error=float(
                        abs(w @ sys.forward(v) - v @ sys.adjoint(w))
                        / max(abs(w @ sys.forward(v)), 1e-30)
                    ),
                    boundary_contract="valid-point unknowns only; endpoint valid_index=-1 excluded by production builder",
                )
                self.structures.append(row)
                _write_csv(self.out / "structure.csv", self.structures)
                self.log.info(
                    "Structure %s: N=%d, nnz=%d, dense %.3f GiB",
                    run.example,
                    sys.n,
                    mat.nnz,
                    row["dense_gib"],
                )
                if run.example not in {"unit_square", "disk"}:
                    continue
                batch = session.batches[0]
                f = batch.rhs_valid.cpu().numpy()
                rhs = np.stack(
                    [0.5 * (sys.axis(x, 1) - sys.axis(x, 0)) for x in f], axis=1
                )
                self.log.info("Starting reusable QR and LU: %s", run.example)
                solutions, setup = direct_solutions(mat.toarray(), rhs, "cuda:1")
                _json(self.out / f"{run.example}_direct.json", setup)
                self.log.info("Direct factorizations complete: %s", setup)
                baseline = session._prepare(batch)
                result = session._krylov(baseline, run.native_k)
                native_pair, _, _ = session._candidate(
                    batch, baseline, result, run.native_k
                )
                self.rows.extend(
                    session.metrics(
                        batch,
                        native_pair,
                        "learned",
                        run.native_k,
                        baseline.symmetric_physical,
                    )
                )
                for method in ("gmres", "lsmr"):
                    columns = []
                    for i, b in enumerate(rhs.T):
                        start = time.perf_counter()
                        history: list[float] = []
                        if method == "gmres":
                            x, status = gmres(
                                sys.operator,
                                b,
                                rtol=1e-11,
                                atol=0,
                                restart=100,
                                maxiter=100,
                                callback=history.append,
                                callback_type="pr_norm",
                            )
                            iterations = len(history)
                            extra = dict(residual_history=history)
                        else:
                            fit = lsmr(
                                sys.operator,
                                b,
                                atol=1e-13,
                                btol=1e-13,
                                conlim=1e12,
                                maxiter=10000,
                            )
                            x, status, iterations = fit[:3]
                            extra = dict(
                                normr=fit[3], normar=fit[4], norma=fit[5], conda=fit[6]
                            )
                        _json(
                            self.out / f"{run.example}_{method}_sample{i}.json",
                            dict(
                                status=int(status),
                                iterations=int(iterations),
                                seconds=time.perf_counter() - start,
                                **extra,
                            ),
                        )
                        columns.append(x)
                        self.log.info(
                            "%s %s sample%d status%d iterations%d",
                            run.example,
                            method,
                            i,
                            status,
                            iterations,
                        )
                    solutions[method] = np.stack(columns, axis=1)
                for method, delta in solutions.items():
                    pair_np = np.stack((f / 2 + delta.T, f / 2 - delta.T), axis=1)
                    pair = torch.tensor(pair_np, device=session.device)
                    rows = session.metrics(
                        batch,
                        pair,
                        method,
                        0,
                        torch.tensor(
                            np.stack((f / 2, f / 2), 1), device=session.device
                        ),
                    )
                    for i, r in enumerate(rows):
                        residual = sys.forward(delta[:, i]) - rhs[:, i]
                        r.update(
                            J=r["response_cost"] / 2,
                            equation_relative_residual=float(
                                np.linalg.norm(residual) / np.linalg.norm(rhs[:, i])
                            ),
                            stationarity_scaled=float(
                                np.linalg.norm(sys.adjoint(residual))
                                / (norm * np.linalg.norm(rhs[:, i]))
                            ),
                            delta_relative_to_lu=float(
                                np.linalg.norm(delta[:, i] - solutions["lu"][:, i])
                                / np.linalg.norm(solutions["lu"][:, i])
                            ),
                        )
                    self.rows.extend(rows)
                np.savez_compressed(
                    self.out / f"{run.example}_solutions.npz",
                    sample_ids=batch.sample_indices.cpu().numpy(),
                    rhs=rhs,
                    lu=solutions["lu"],
                    qr=solutions["qr"],
                    gmres=solutions["gmres"],
                    lsmr=solutions["lsmr"],
                )
                _write_csv(self.out / "per_sample.csv", self.rows)
            finally:
                session.close()
        for path, digest in audit.hashes.items():
            if _sha256(Path(path)) != digest:
                raise RuntimeError(f"Input changed: {path}")
        _json(
            self.out / "verification.json",
            dict(
                input_hashes_unchanged=True,
                structure_count=len(self.structures),
                metric_rows=len(self.rows),
                production_operator_actions_verified=True,
                full_benchmark=False,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    SolverPilot(args.outdir).run()
