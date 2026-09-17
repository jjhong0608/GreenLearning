"""Diagnose archived square/disk Green comparisons with production weak operators."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from cli.audit_reference_green_reoptimization import optimize_sources, reference_context
from greenonet.coefficients import load_coefficient_functions
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.complex_weak_closure import (
    ComplexDirectionalWeakContext,
    assemble_directional_weak_residuals,
)
from greenonet.reference_green_reconstruction import DiffusionReferenceBuilder


def residual_metrics(
    fields: torch.Tensor,
    prediction: torch.Tensor,
    pair: torch.Tensor,
    rhs: torch.Tensor,
    context: ComplexDirectionalWeakContext,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return unsmoothed signed defects and source-normalized mass indicators."""
    raw = {}
    for name, u in (
        ("phi", fields[:, 0]),
        ("psi", fields[:, 1]),
        ("equal", fields.mean(1)),
        ("weak", prediction),
    ):
        r = assemble_directional_weak_residuals(
            u_valid=u, projected_physical=pair, context=context
        )
        raw[f"{name}_x"] = r.x
        raw[f"{name}_y"] = r.y
        raw[f"{name}_full"] = r.full
    denominator = rhs.square().sum(1) * context.point_area
    if torch.any(denominator <= 0):
        raise ValueError("Relative residual requires nonzero source norm")
    mass = {
        "x": context.x.nodal_mass,
        "y": context.y.nodal_mass,
        "full": context.x.nodal_mass + context.y.nodal_mass,
    }
    metrics = {
        name: torch.sqrt(
            (value.square() / mass[name.split("_")[-1]]).sum(1) / denominator
        )
        for name, value in raw.items()
    }
    metrics["own"] = torch.sqrt(
        0.5 * (metrics["phi_x"].square() + metrics["psi_y"].square())
    )
    metrics["cross"] = torch.sqrt(
        0.5 * (metrics["phi_y"].square() + metrics["psi_x"].square())
    )
    for value in metrics.values():
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite residual metric")
    return metrics, raw


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


class WeakGreenAudit:
    def __init__(self, out: Path) -> None:
        self.out = out
        out.mkdir(parents=True, exist_ok=False)
        (out / "raw").mkdir()
        self.logger = logging.getLogger("weak_green_audit")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        for handler in (
            RichHandler(show_path=False, omit_repeated_times=False),
            logging.FileHandler(out / "run.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            self.logger.addHandler(handler)
        self.rows: list[dict[str, Any]] = []

    @torch.no_grad()
    def run(self) -> None:
        torch.set_num_threads(4)
        base = Path("docs/analysis")
        fixed = base / "reference_green_fixed_sources"
        rr = base / "reference_green_reoptimization"
        ll = base / "learned_k_extension_square_disk_v2"
        previous = {}
        for method, folder in (("LL", ll), ("RR", rr)):
            for row in read_rows(folder / "per_sample.csv"):
                previous[
                    method,
                    row["run_id"],
                    int(row["evaluation_k"]),
                    int(row["sample_id"]),
                ] = row
        for row in read_rows(fixed / "per_sample.csv"):
            if row["factor"] == "256" and row["source_condition"] == "learned":
                previous[
                    "LR", row["run_id"], int(row["evaluation_k"]), int(row["sample_id"])
                ] = row
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                Path("configs/paper_source_initialization_audit.json"),
                self.out,
                batch_size=25,
            )
        )
        audit.preflight()
        for path in [
            Path(__file__),
            Path("cli/audit_reference_green_reoptimization.py"),
            *Path("src/greenonet").glob("*.py"),
            *fixed.glob("sources/*.npz"),
            *(folder / "per_sample.csv" for folder in (ll, rr, fixed)),
        ]:
            audit.hashes[str(path.resolve())] = _sha256(path)
        _json(self.out / "input_hashes.json", audit.hashes)
        _json(
            self.out / "request.json",
            dict(
                methods=["LL", "LR", "RR"],
                k=list(range(65)),
                reference_factor=256,
                device="cuda:1",
                dtype="float64",
                metric="sqrt(sum(r_i^2 / nodal_mass_i) / (hx*hy*sum(f_i^2)))",
                own="sqrt((phi_x^2+psi_y^2)/2)",
                raw_saved_k=[0, 2, 16, 64],
                smoothing=False,
                changes="diagnostic only; existing optimizations replayed, no new objective",
            ),
        )
        checked = 0
        for run in audit.runs:
            if run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            run.audit.logger = self.logger
            self.logger.info("Starting %s", run.spec.run_id)
            session = SourceRunSession(run)
            try:
                initial_by_id = {}
                for path in sorted(
                    (fixed / "sources").glob(f"{run.spec.run_id}_batch*.npz")
                ):
                    with np.load(path) as data:
                        for sid, pair in zip(
                            data["sample_ids"], data["learned_k0"], strict=True
                        ):
                            initial_by_id[int(sid)] = pair.copy()
                if len(initial_by_id) != len(session.dataset):
                    raise ValueError("Incomplete initial-source archive")
                learned = session.context
                operator = DiffusionReferenceBuilder(
                    session.batches[0].geometry,
                    load_coefficient_functions(run.spec.coefficients),
                    session.device,
                ).build(256)
                reference = reference_context(
                    operator, learned.point_mass, session.tangent, 64
                )
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
                    steps = {
                        "L": optimize_sources(learned, initial, 64),
                        "R": optimize_sources(reference, initial, 64),
                    }
                    saved = {"sample_ids": batch.sample_indices.cpu().numpy()}
                    for k in range(65):
                        for method in ("LL", "LR", "RR"):
                            delta = (
                                steps[method[0]].deltas[k - 1]
                                if k
                                else torch.zeros_like(initial[:, 0])
                            )
                            pair = initial + torch.stack((delta, -delta), 1)
                            torch.testing.assert_close(
                                pair.sum(1), batch.rhs_valid, rtol=1e-12, atol=1e-12
                            )
                            session.context = learned if method[1] == "L" else reference
                            fields, cross = session.fields(batch, pair)
                            metrics, raw = residual_metrics(
                                fields,
                                cross.u_pred_valid,
                                pair,
                                batch.rhs_valid,
                                batch.weak_context,
                            )
                            metrics["J"] = (
                                0.5
                                * learned.point_mass
                                * (fields[:, 0] - fields[:, 1]).square().sum(1)
                            )
                            metrics["rel_sol"] = (
                                cross.u_pred_valid - batch.sol_valid
                            ).norm(dim=1) / batch.sol_valid.norm(dim=1)
                            metrics["rel_equal"] = (
                                fields.mean(1) - batch.sol_valid
                            ).norm(dim=1) / batch.sol_valid.norm(dim=1)
                            array = {
                                key: value.cpu().numpy()
                                for key, value in metrics.items()
                            }
                            for i, sid in enumerate(batch.sample_indices.tolist()):
                                old = previous.get((method, run.spec.run_id, k, sid))
                                if old is not None:
                                    for key, value in (
                                        ("rel_sol", array["rel_sol"][i]),
                                        ("response_cost", 2 * array["J"][i]),
                                    ):
                                        np.testing.assert_allclose(
                                            value,
                                            float(old[key]),
                                            rtol=1e-7,
                                            atol=1e-12,
                                        )
                                    checked += 1
                                self.rows.append(
                                    dict(
                                        example=run.example,
                                        run_id=run.spec.run_id,
                                        seed=run.spec.seed,
                                        sample_id=sid,
                                        k=k,
                                        method=method,
                                        **{
                                            key: float(value[i])
                                            for key, value in array.items()
                                        },
                                    )
                                )
                            if k in (0, 2, 16, 64):
                                for key, value in raw.items():
                                    saved[f"{method}_k{k}_{key}"] = value.cpu().numpy()
                    np.savez_compressed(
                        self.out / "raw" / f"{run.spec.run_id}_batch{bi:03d}.npz",
                        **saved,
                    )
                    self.logger.info(
                        "%s batch %d/%d", run.spec.run_id, bi + 1, len(session.batches)
                    )
                    session.context = learned
                _write_csv(self.out / "per_sample.csv", self.rows)
            finally:
                session.close()
        if len(self.rows) != 117000:
            raise ValueError("Incomplete coverage")
        for name, digest in audit.hashes.items():
            if _sha256(Path(name)) != digest:
                raise RuntimeError(f"Input changed: {name}")
        _json(
            self.out / "verification.json",
            dict(
                rows=len(self.rows),
                prior_rows_reproduced=checked,
                input_hashes_unchanged=True,
                full_test=True,
                finite_metrics=True,
                source_balance_preserved=True,
            ),
        )
        self.logger.info(
            "Complete: %d rows; %d archived rows reproduced", len(self.rows), checked
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    WeakGreenAudit(parser.parse_args().outdir).run()
