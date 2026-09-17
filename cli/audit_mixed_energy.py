"""Frozen same-space normalized L2/energy coefficient refits, without selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from audit_fixed_tangent_subspace import refit_subspace
from audit_reference_green_reoptimization import optimize_sources
from weak_pde_fixed_space import EuclideanMetric
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)


WEIGHTS = {"l2": 0.0, "mix_0.01": 0.01, "mix_0.1": 0.1, "mix_1": 1.0, "energy": None}
KS = (0, 1, 2, 4, 8, 16, 32, 64)


class EnergyFeatures:
    """Square root of the existing interior-edge metric, with no boundary term."""

    def __init__(self, geometry: Any, a: torch.Tensor) -> None:
        self.mass = float(geometry.hx * geometry.hy)
        self.edges = []
        for edges, h in (
            (geometry.x_edges, geometry.hx),
            (geometry.y_edges, geometry.hy),
        ):
            left, right = edges[:, 0], edges[:, 1]
            weight = self.mass * (a[:, left] + a[:, right]) / (2 * h**2)
            self.edges.append((left, right, weight.sqrt()))

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                (values[..., right] - values[..., left]) * root
                for left, right, root in self.edges
            ],
            dim=-1,
        )


def features(c: torch.Tensor, e: torch.Tensor, weight: float | None) -> torch.Tensor:
    if weight is None:
        return e
    if weight == 0:
        return c
    if weight < 0:
        raise ValueError("Negative energy weight")
    return torch.cat((c, weight**0.5 * e), dim=-1)


def objective(c: torch.Tensor, e: torch.Tensor, weight: float | None) -> torch.Tensor:
    return e if weight is None else c + weight * e


class MixedEnergyAudit:
    def __init__(self, out: Path) -> None:
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(
                rich_tracebacks=True, show_path=True, omit_repeated_times=False
            ),
            logging.FileHandler(out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                Path("configs/paper_source_initialization_audit.json"),
                self.out,
                batch_size=10,
            )
        )
        audit.preflight()
        baseline = Path(
            "docs/analysis/learned_k_extension_square_disk_v2/per_sample.csv"
        )
        with baseline.open() as f:
            old = {
                (r["run_id"], int(r["evaluation_k"]), int(r["sample_id"])): float(
                    r["rel_sol"]
                )
                for r in csv.DictReader(f)
            }
        for path in [
            Path(__file__),
            Path("cli/audit_fixed_tangent_subspace.py"),
            Path("cli/weak_pde_fixed_space.py"),
            Path("cli/audit_reference_green_reoptimization.py"),
            baseline,
            *Path("src/greenonet").glob("*.py"),
        ]:
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "request.json",
            dict(
                weights=WEIGHTS,
                k=KS,
                initializations=["learned", "equal_split"],
                examples=["unit_square", "disk"],
                seeds=[0, 1, 2, 3],
                device="cuda:1",
                dtype="float64",
                normalization="C=mass*m^2/(mass*m0^2); E=edge_energy(m)/edge_energy(m0), per sample and initialization",
                basis="same consistency-generated K64 prefixes within each initialization, all objectives",
                selection="none; no reference-solution validation set available; all test curves exploratory",
                labels="evaluation only",
                energy="existing interior-edge bulk seminorm, no boundary term",
            ),
        )
        _json(self.out / "environment.json", audit.runs[0].audit._environment())
        rows: list[dict[str, Any]] = []
        provenance = []
        max_delta_relative = 0.0
        for run in audit.runs:
            if run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            run.audit.logger = self.logger
            self.logger.info("Starting %s", run.spec.run_id)
            session = SourceRunSession(run)
            try:
                for bi, batch in enumerate(session.batches):
                    metric = EnergyFeatures(batch.geometry, batch.a_valid)
                    learned = session._prepare(batch).symmetric_physical
                    for init in ("learned", "equal_split"):
                        initial = (
                            learned
                            if init == "learned"
                            else torch.stack(
                                (batch.rhs_valid / 2, batch.rhs_valid / 2), 1
                            )
                        )
                        basis = optimize_sources(session.context, initial, 64)
                        m0 = (
                            session.context.response_operator.forward_pair(initial)
                            .diff(dim=1)
                            .squeeze(1)
                            .neg()
                        )
                        e0 = metric.apply(m0)
                        cn, en = m0.norm(dim=-1), e0.norm(dim=-1)
                        if torch.any(cn <= 1e-30) or torch.any(en <= 1e-30):
                            raise ValueError(
                                "Degenerate initial norm; cannot normalize this sample"
                            )
                        c0, e0 = m0 / cn[:, None], e0 / en[:, None]
                        cdesign = basis.response_directions / cn[None, :, None]
                        edesign = (
                            metric.apply(basis.response_directions) / en[None, :, None]
                        )
                        fits = {
                            name: refit_subspace(
                                features(c0, e0, w),
                                basis.directions,
                                features(cdesign, edesign, w),
                                EuclideanMetric(),
                            )
                            for name, w in WEIGHTS.items()
                        }
                        provenance.append(
                            dict(
                                run_id=run.spec.run_id,
                                batch=bi,
                                initialization=init,
                                initial_sha256=hashlib.sha256(
                                    initial.cpu().numpy().tobytes()
                                ).hexdigest(),
                                basis_sha256=hashlib.sha256(
                                    basis.directions.cpu().numpy().tobytes()
                                ).hexdigest(),
                                active_min={
                                    name: int(fit.direction_active.sum(0).min())
                                    for name, fit in fits.items()
                                },
                            )
                        )
                        histories: dict[str, list[torch.Tensor]] = {
                            name: [] for name in fits
                        }
                        for k in KS:
                            values = {}
                            for name, fit in fits.items():
                                delta = (
                                    fit.deltas[k - 1]
                                    if k
                                    else torch.zeros_like(initial[:, 0])
                                )
                                if name == "l2" and k:
                                    direct = basis.deltas[k - 1]
                                    relative = (delta - direct).norm(
                                        dim=-1
                                    ) / direct.norm(dim=-1).clamp_min(1e-30)
                                    max_delta_relative = max(
                                        max_delta_relative, float(relative.max())
                                    )
                                    if torch.any(relative > 1e-6):
                                        raise RuntimeError(
                                            f"L2 source norm mismatch {init} K{k}: {relative.max()}"
                                        )
                                    refit_fields = (
                                        session.context.response_operator.forward_pair(
                                            torch.stack((delta, -delta), 1)
                                        )
                                    )
                                    direct_fields = (
                                        session.context.response_operator.forward_pair(
                                            torch.stack((direct, -direct), 1)
                                        )
                                    )
                                    response_relative = (
                                        refit_fields - direct_fields
                                    ).flatten(1).norm(dim=-1) / direct_fields.flatten(
                                        1
                                    ).norm(dim=-1).clamp_min(1e-30)
                                    if torch.any(response_relative > 1e-6):
                                        raise RuntimeError(
                                            f"L2 response norm mismatch {init} K{k}"
                                        )
                                pair = initial + torch.stack((delta, -delta), 1)
                                torch.testing.assert_close(
                                    pair.sum(1), batch.rhs_valid, rtol=1e-10, atol=1e-10
                                )
                                fields, cross = session.fields(batch, pair)
                                m = fields[:, 0] - fields[:, 1]
                                raw_e = metric.apply(m).square().sum(-1)
                                c, e = (
                                    m.square().sum(-1) / cn.square(),
                                    raw_e / en.square(),
                                )
                                values[name] = (c, e)
                                histories[name].append(objective(c, e, WEIGHTS[name]))
                                norm = batch.sol_valid.norm(dim=-1)
                                arrays = {
                                    key: value.cpu().numpy()
                                    for key, value in dict(
                                        C=c,
                                        E=e,
                                        J=0.5 * metric.mass * m.square().sum(-1),
                                        energy=raw_e,
                                        rel_sol=(
                                            cross.u_pred_valid - batch.sol_valid
                                        ).norm(dim=-1)
                                        / norm,
                                        rel_equal=(
                                            fields.mean(1) - batch.sol_valid
                                        ).norm(dim=-1)
                                        / norm,
                                        rel_phi=(fields[:, 0] - batch.sol_valid).norm(
                                            dim=-1
                                        )
                                        / norm,
                                        rel_psi=(fields[:, 1] - batch.sol_valid).norm(
                                            dim=-1
                                        )
                                        / norm,
                                        balance_max=(pair.sum(1) - batch.rhs_valid)
                                        .abs()
                                        .amax(-1),
                                        correction_norm=delta.norm(dim=-1),
                                    ).items()
                                }
                                for i, sid in enumerate(batch.sample_indices.tolist()):
                                    if name == "l2" and init == "learned":
                                        np.testing.assert_allclose(
                                            arrays["rel_sol"][i],
                                            old[run.spec.run_id, k, sid],
                                            rtol=1e-6,
                                            atol=1e-10,
                                        )
                                    rows.append(
                                        dict(
                                            example=run.example,
                                            run_id=run.spec.run_id,
                                            seed=run.spec.seed,
                                            sample_id=sid,
                                            initialization=init,
                                            k=k,
                                            objective=name,
                                            **{
                                                key: float(a[i])
                                                for key, a in arrays.items()
                                            },
                                        )
                                    )
                            for name, (c, e) in values.items():
                                for oc, oe in values.values():
                                    if torch.any(
                                        objective(c, e, WEIGHTS[name])
                                        > objective(oc, oe, WEIGHTS[name]) + 1e-8
                                    ):
                                        raise RuntimeError(f"Dominance failed: {name}")
                        for name, history in histories.items():
                            v = torch.stack(history)
                            if torch.any(v[1:] > v[:-1] + 1e-8):
                                raise RuntimeError(f"Nested minimum failed: {name}")
                    self.logger.info(
                        "%s batch %d/%d done",
                        run.spec.run_id,
                        bi + 1,
                        len(session.batches),
                    )
                _write_csv(self.out / "per_sample.csv", rows)
                _json(self.out / "basis_provenance.json", provenance)
            finally:
                session.close()
        if len(rows) != 48000:
            raise RuntimeError(f"Incomplete evaluation: {len(rows)} rows")
        for name, digest in audit.hashes.items():
            if _sha256(Path(name)) != digest:
                raise RuntimeError(f"Input changed: {name}")
        _json(
            self.out / "verification.json",
            dict(
                rows=len(rows),
                inputs_unchanged=True,
                learned_l2_baseline_reproduced=True,
                equal_split_l2_matches_direct=True,
                same_space=True,
                objective_dominance=True,
                nested_minima=True,
                hyperparameter_selection=False,
                max_l2_delta_relative_difference=max_delta_relative,
            ),
        )
        self.logger.info("Completed %d rows", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    MixedEnergyAudit(parser.parse_args().outdir).run()
