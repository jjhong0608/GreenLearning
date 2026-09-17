"""Compare objective-only refits in frozen learned tangent spaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler

from audit_fixed_tangent_subspace import refit_subspace
from audit_reference_green_reoptimization import optimize_sources
from weak_pde_fixed_space import InteriorWeakPDE, EuclideanMetric, manufactured_checks
from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.reference_green_reconstruction import DiffusionReferenceBuilder


class ObjectiveAudit:
    def __init__(self, out: Path):
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.logger = logging.getLogger("weak_pde_refit")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(show_path=False, omit_repeated_times=False),
            logging.FileHandler(out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        _json(self.out / "manufactured.json", manufactured_checks())
        request = SourceInitializationRequest(
            Path("configs/paper_source_initialization_audit.json"),
            self.out,
            batch_size=10,
        )
        audit = SourceInitializationAudit(request)
        audit.preflight()
        baseline = Path(
            "docs/analysis/learned_k_extension_square_disk_v2/per_sample.csv"
        )
        with baseline.open() as f:
            old = {
                (r["run_id"], int(r["evaluation_k"]), int(r["sample_id"])): r
                for r in csv.DictReader(f)
            }
        fixed = Path("docs/analysis/reference_green_fixed_sources/sources")
        for path in [
            Path(__file__),
            Path("cli/weak_pde_fixed_space.py"),
            Path("cli/audit_fixed_tangent_subspace.py"),
            Path("cli/audit_reference_green_reoptimization.py"),
            baseline,
            *fixed.glob("*.npz"),
            *Path("docs/analysis/reference_green_reoptimization/sources").glob("*.npz"),
            *Path("src/greenonet").glob("*.py"),
        ]:
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        ks = [0, 1, 2, 4, 8, 16, 32, 64]
        _json(
            self.out / "request.json",
            dict(
                k=ks,
                device="cuda:1",
                dtype="float64",
                green="frozen learned",
                basis="existing learned consistency K64 Krylov prefixes, shared among all objectives",
                candidate="equal mean; production weak blend evaluated separately",
                weak="interior full-support 2D Q1 tests, Gauss3 a, consistent Q1 f load",
                normalization="each objective divided by its own initial squared norm, per sample",
                combined_weight=1,
                objectives=["consistency", "weak_pde", "joint"],
                labels="evaluation only",
                boundaries="tests whose supports require unavailable boundary source values are excluded",
            ),
        )
        rows = []
        provenance = []
        controls = []
        for run in audit.runs:
            if run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            run.audit.logger = self.logger
            self.logger.info("Starting %s", run.spec.run_id)
            session = SourceRunSession(run)
            try:
                initial_by_id = {}
                for path in sorted(fixed.glob(f"{run.spec.run_id}_batch*.npz")):
                    with np.load(path) as data:
                        for sid, pair in zip(
                            data["sample_ids"], data["learned_k0"], strict=True
                        ):
                            initial_by_id[int(sid)] = pair.copy()
                coeff = load_coefficient_functions(run.spec.coefficients)
                geometry = session.batches[0].geometry
                for fun in (coeff.bx_fun, coeff.by_fun, coeff.c_fun):
                    if torch.any(
                        fun(
                            geometry.coords_valid[:, 0], geometry.coords_valid[:, 1]
                        ).abs()
                        > 1e-12
                    ):
                        raise ValueError("This experiment supports diffusion only")
                weak = InteriorWeakPDE.build(
                    geometry.coords_valid, coeff.a_fun, device=session.device
                )
                # A higher-order quadrature control never uses reference labels.
                control = InteriorWeakPDE.build(
                    geometry.coords_valid, coeff.a_fun, order=4, device=session.device
                )
                qdiff = float(
                    (weak.stiffness - control.stiffness).coalesce().values().norm()
                    / control.stiffness.values().norm()
                )
                if qdiff > 1e-6:
                    raise RuntimeError("Gauss3/4 stiffness check failed")
                controls.append(
                    dict(
                        run_id=run.spec.run_id,
                        points=weak.point_count,
                        tests=len(weak.test_indices),
                        cells=weak.cell_count,
                        test_fraction=len(weak.test_indices) / weak.point_count,
                        gauss3_4_relative_stiffness=qdiff,
                    )
                )
                refop = None
                if run.spec.seed == 0:
                    refop = DiffusionReferenceBuilder(
                        geometry, coeff, session.device
                    ).build(256)
                for bi, batch in enumerate(session.batches):
                    initial = torch.from_numpy(
                        np.stack([initial_by_id[int(i)] for i in batch.sample_indices])
                    ).to(session.device)
                    torch.testing.assert_close(
                        initial,
                        session._prepare(batch).symmetric_physical,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                    basis = optimize_sources(session.context, initial, 64)
                    fields0 = session.context.response_operator.forward_pair(initial)
                    c0 = fields0[:, 0] - fields0[:, 1]
                    w0 = weak.residual(fields0.mean(1), batch.rhs_valid)
                    cn = c0.norm(dim=-1)
                    wn = w0.norm(dim=-1)
                    if torch.any(cn <= 1e-30) or torch.any(wn <= 1e-30):
                        raise RuntimeError("Degenerate initial objective")
                    c0 = c0 / cn[:, None]
                    w0 = w0 / wn[:, None]
                    cdesign = basis.response_directions / cn[None, :, None]
                    udesign = 0.5 * (
                        basis.directional_responses[:, :, 0]
                        - basis.directional_responses[:, :, 1]
                    )
                    wdesign = weak.apply(udesign) / wn[None, :, None]
                    fits = {
                        "consistency": refit_subspace(
                            c0, basis.directions, cdesign, EuclideanMetric()
                        ),
                        "weak_pde": refit_subspace(
                            w0, basis.directions, wdesign, EuclideanMetric()
                        ),
                        "joint": refit_subspace(
                            torch.cat((c0, w0), -1),
                            basis.directions,
                            torch.cat((cdesign, wdesign), -1),
                            EuclideanMetric(),
                        ),
                    }
                    provenance.append(
                        dict(
                            run_id=run.spec.run_id,
                            batch=bi,
                            basis_sha256=hashlib.sha256(
                                basis.directions.cpu().numpy().tobytes()
                            ).hexdigest(),
                            source_sha256=hashlib.sha256(
                                initial.cpu().numpy().tobytes()
                            ).hexdigest(),
                            refit_active={
                                key: int(value.direction_active.sum(0).min())
                                for key, value in fits.items()
                            },
                        )
                    )
                    objective_history: dict[str, list[torch.Tensor]] = {
                        key: [] for key in fits
                    }
                    for k in ks:
                        evaluations = {}
                        for objective, fit in fits.items():
                            delta = (
                                fit.deltas[k - 1]
                                if k
                                else torch.zeros_like(initial[:, 0])
                            )
                            pair = initial + torch.stack((delta, -delta), 1)
                            torch.testing.assert_close(
                                pair.sum(1), batch.rhs_valid, rtol=1e-10, atol=1e-10
                            )
                            fields, cross = session.fields(batch, pair)
                            equal = fields.mean(1)
                            c = (
                                ((fields[:, 0] - fields[:, 1]) / cn[:, None])
                                .square()
                                .sum(1)
                            )
                            w = (
                                (weak.residual(equal, batch.rhs_valid) / wn[:, None])
                                .square()
                                .sum(1)
                            )
                            evaluations[objective] = (c, w)
                            objective_history[objective].append(
                                c
                                if objective == "consistency"
                                else w
                                if objective == "weak_pde"
                                else c + w
                            )
                            norm = batch.sol_valid.norm(dim=1)
                            values = dict(
                                C=c,
                                W=w,
                                joint=c + w,
                                J=0.5
                                * session.context.point_mass
                                * (fields[:, 0] - fields[:, 1]).square().sum(1),
                                weak_indicator=weak.residual(
                                    equal, batch.rhs_valid
                                ).norm(dim=1)
                                / weak.apply(batch.rhs_valid, load=True).norm(dim=1),
                                rel_sol=(cross.u_pred_valid - batch.sol_valid).norm(
                                    dim=1
                                )
                                / norm,
                                rel_equal=(equal - batch.sol_valid).norm(dim=1) / norm,
                                rel_phi=(fields[:, 0] - batch.sol_valid).norm(dim=1)
                                / norm,
                                rel_psi=(fields[:, 1] - batch.sol_valid).norm(dim=1)
                                / norm,
                                correction_norm=delta.norm(dim=1),
                                balance_max=(pair.sum(1) - batch.rhs_valid)
                                .abs()
                                .amax(1),
                            )
                            arrays = {
                                key: value.cpu().numpy()
                                for key, value in values.items()
                            }
                            for i, sid in enumerate(batch.sample_indices.tolist()):
                                if objective == "consistency":
                                    previous = old[run.spec.run_id, k, sid]
                                    np.testing.assert_allclose(
                                        arrays["rel_sol"][i],
                                        float(previous["rel_sol"]),
                                        rtol=1e-6,
                                        atol=1e-10,
                                    )
                                rows.append(
                                    dict(
                                        example=run.example,
                                        run_id=run.spec.run_id,
                                        seed=run.spec.seed,
                                        sample_id=sid,
                                        k=k,
                                        objective=objective,
                                        **{
                                            key: float(value[i])
                                            for key, value in arrays.items()
                                        },
                                    )
                                )
                        for objective, (c, w) in evaluations.items():
                            for other, (co, wo) in evaluations.items():
                                lhs = (
                                    c
                                    if objective == "consistency"
                                    else w
                                    if objective == "weak_pde"
                                    else c + w
                                )
                                rhs = (
                                    co
                                    if objective == "consistency"
                                    else wo
                                    if objective == "weak_pde"
                                    else co + wo
                                )
                                if torch.any(lhs > rhs + 1e-8):
                                    raise RuntimeError(
                                        f"Objective dominance failed {objective}/{other}"
                                    )
                    for objective, history in objective_history.items():
                        v = torch.stack(history)
                        if torch.any(v[1:] > v[:-1] + 1e-8):
                            raise RuntimeError(f"Nested minimum failed {objective}")
                    if bi == 0 and refop is not None:
                        # Reference Green diagnostic at fixed archived RR K64 sources.
                        refpairs = {}
                        for path in Path(
                            "docs/analysis/reference_green_reoptimization/sources"
                        ).glob(f"{run.spec.run_id}_batch*.npz"):
                            with np.load(path) as data:
                                for sid, pair in zip(
                                    data["sample_ids"],
                                    data["reference_optimized_k64"],
                                    strict=True,
                                ):
                                    refpairs[int(sid)] = pair
                        rp = torch.from_numpy(
                            np.stack([refpairs[int(i)] for i in batch.sample_indices])
                        ).to(session.device)
                        u = refop.forward_pair(rp).mean(1)
                        controls.append(
                            dict(
                                run_id=run.spec.run_id,
                                reference_green_control_samples=len(
                                    batch.sample_indices
                                ),
                                q3_4_residual_relative=float(
                                    (
                                        weak.residual(u, batch.rhs_valid)
                                        - control.residual(u, batch.rhs_valid)
                                    ).norm()
                                    / control.residual(u, batch.rhs_valid).norm()
                                ),
                            )
                        )
                    self.logger.info(
                        "%s batch %d/%d", run.spec.run_id, bi + 1, len(session.batches)
                    )
                _write_csv(self.out / "per_sample.csv", rows)
                _json(self.out / "basis_provenance.json", provenance)
                _json(self.out / "controls.json", controls)
            finally:
                session.close()
        assert len(rows) == 14400
        for path, digest in audit.hashes.items():
            if _sha256(Path(path)) != digest:
                raise RuntimeError(f"Input changed: {path}")
        _json(
            self.out / "verification.json",
            dict(
                rows=len(rows),
                full_test=True,
                inputs_unchanged=True,
                consistency_baseline_reproduced=True,
                same_basis=True,
                objective_dominance=True,
                nested_minima=True,
            ),
        )
        self.logger.info("Completed %d rows", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    ObjectiveAudit(parser.parse_args().outdir).run()
