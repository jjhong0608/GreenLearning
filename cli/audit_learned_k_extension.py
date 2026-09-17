"""Extend frozen square/disk learned initializers without changing training."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler

from greenonet.complex_cross_axis_reconstruction import (
    ComplexCrossAxisReconstructionResult,
)
from greenonet.complex_frozen_tangent_csv import _json, _sha256, _write_csv
from greenonet.complex_source_initialization_audit import (
    SourceInitializationAudit,
    SourceInitializationRequest,
    SourceRunSession,
)
from greenonet.source_initialization_metrics import relative_norm, summarize


class LearnedExtensionSession(SourceRunSession):
    @torch.no_grad()
    def curve(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, batch in enumerate(self.batches):
            prepared = self._prepare(batch)
            result = self._krylov(prepared, self.request.max_k)
            initial_cost = self.context.point_mass * prepared.mismatch.square().sum(1)
            previous = torch.cat((initial_cost[None], result.costs[:-1]))
            tolerance = 1e-10 * torch.maximum(previous, initial_cost[None])
            if torch.any(
                result.costs > previous + tolerance + torch.finfo(torch.float64).tiny
            ):
                raise RuntimeError("Response cost increased beyond tolerance")
            for k in range(self.request.max_k + 1):
                pair = prepared.symmetric_physical
                if k:
                    delta = result.deltas[k - 1]
                    pair = pair + torch.stack((delta, -delta), dim=1)
                rows.extend(
                    self.metrics(
                        batch,
                        pair,
                        "learned",
                        k,
                        prepared.symmetric_physical,
                        result if k else None,
                    )
                )
            self.audit.logger.info(
                "%s batch %d/%d", self.spec.run_id, index + 1, len(self.batches)
            )
        return rows

    @torch.no_grad()
    def verify_curve(self, rows: list[dict[str, Any]], ks: set[int]) -> None:
        lookup = {(r["evaluation_k"], r["sample_id"]): r for r in rows}
        for batch in self.batches:
            prepared = self._prepare(batch)
            for k in sorted(ks):
                if k == 0:
                    pair = prepared.symmetric_physical
                    result = None
                else:
                    # Legacy Krylov requires two slots; K1 production is checked below.
                    result = self._krylov(prepared, max(2, k))
                    pair = self._candidate(batch, prepared, result, k)[0]
                independent = self.metrics(
                    batch, pair, "learned", k, prepared.symmetric_physical, result
                )
                for row in independent:
                    saved = lookup[k, row["sample_id"]]
                    for metric in (
                        "rel_sol",
                        "rel_sol_equal_mean",
                        "rel_u_phi",
                        "rel_u_psi",
                        "response_cost",
                        "loss_energy_optimized",
                    ):
                        np.testing.assert_allclose(
                            row[metric], saved[metric], rtol=1e-8, atol=1e-12
                        )
                if k:
                    production = self._prediction_forward(batch, k)
                    if not isinstance(production, ComplexCrossAxisReconstructionResult):
                        raise TypeError("Unexpected production reconstruction result")
                    for i, sample_id in enumerate(batch.sample_indices.tolist()):
                        actual, _ = relative_norm(
                            production.u_pred_valid[i] - batch.sol_valid[i],
                            batch.sol_valid[i],
                        )
                        if actual is None:
                            raise ValueError("Undefined solution relative error")
                        np.testing.assert_allclose(
                            actual,
                            lookup[k, sample_id]["rel_sol"],
                            rtol=1e-8,
                            atol=1e-12,
                        )
        self.verification["independently_verified_k"] = sorted(ks)


def minimum_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ks = sorted(r["evaluation_k"] for r in rows)
    if ks != list(range(max(ks) + 1)):
        raise ValueError("Expected complete unique K0..Kmax curve")
    result = []
    for metric in ("mean", "p95", "max"):
        if any(not np.isfinite(r[f"rel_sol_{metric}"]) for r in rows):
            raise ValueError("Nonfinite curve")
        best = min(rows, key=lambda r: (r[f"rel_sol_{metric}"], r["evaluation_k"]))
        result.append(dict(best, selected_by=metric))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/paper_source_initialization_audit.json"),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("docs/analysis/paper_source_initialization_audit_v2"),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("learned_k_extension")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in (
        RichHandler(rich_tracebacks=True, show_path=True, omit_repeated_times=False),
        logging.FileHandler(out / "run.log"),
    ):
        handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
        logger.addHandler(handler)
    torch.set_num_threads(4)
    audit = SourceInitializationAudit(SourceInitializationRequest(args.manifest, out))
    audit.preflight()
    audit._gpu_idle()
    with (args.baseline / "summary.csv").open() as f:
        prior = list(csv.DictReader(f))
    for name in (
        "summary.csv",
        "per_sample.csv",
        "input_hashes.json",
        "verification.json",
    ):
        path = (args.baseline / name).resolve()
        audit.hashes[str(path)] = _sha256(path)
    audit.hashes[str(Path(__file__).resolve())] = _sha256(Path(__file__))
    _json(out / "input_hashes.json", audit.hashes)
    _json(out / "environment.json", audit.runs[0].audit._environment())
    _json(
        out / "request.json",
        dict(
            examples=["unit_square", "disk"],
            seeds=list(range(4)),
            k_min=0,
            k_max=64,
            device="cuda:1",
            dtype="float64",
            baseline=args.baseline,
            manifest=args.manifest,
            selection="whole-test statistic; post-hoc, not deployment stopping rule",
        ),
    )
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    minima: list[dict[str, Any]] = []
    verified = []
    for run in audit.runs:
        if run.example not in {"unit_square", "disk"}:
            continue
        run.audit.logger = logger
        audit._gpu_idle()
        logger.info("Starting %s", run.spec.run_id)
        session = LearnedExtensionSession(run)
        try:
            native = session.learned_rows()
            curve = session.curve()
            stats = summarize(curve)
            old = next(
                r
                for r in prior
                if r["run_id"] == run.spec.run_id and r["condition"] == "learned"
            )
            assert old["fingerprint"] == run.spec.fingerprint
            now = next(r for r in stats if r["evaluation_k"] == run.native_k)
            for metric in ("rel_sol_mean", "rel_sol_p95", "rel_sol_max"):
                np.testing.assert_allclose(
                    now[metric], float(old[metric]), rtol=1e-8, atol=1e-12
                )
            indexed = {r["sample_id"]: r for r in native}
            for row in curve:
                if row["evaluation_k"] == run.native_k:
                    np.testing.assert_allclose(
                        row["rel_sol"],
                        indexed[row["sample_id"]]["rel_sol"],
                        rtol=1e-8,
                        atol=1e-12,
                    )
            best = minimum_rows(stats)
            ks = {0, 1, run.native_k, 64} | {r["evaluation_k"] for r in best}
            session.verify_curve(curve, ks)
            rows.extend(curve)
            summaries.extend(stats)
            minima.extend(best)
            verified.append(session.verification | {"run_id": run.spec.run_id})
            _write_csv(out / "per_sample.csv", rows)
            _write_csv(out / "summary.csv", summaries)
            _write_csv(out / "minima.csv", minima)
            _json(out / "verification.json", dict(status="running", runs=verified))
            logger.info(
                "Completed %s; best mean K=%s error=%.8g",
                run.spec.run_id,
                best[0]["evaluation_k"],
                best[0]["rel_sol_mean"],
            )
        finally:
            session.close()
    assert len(verified) == 8 and len(summaries) == 8 * 65 and len(rows) == 39000
    changed = [p for p, h in audit.hashes.items() if _sha256(Path(p)) != h]
    if changed:
        raise RuntimeError(f"Input changed: {changed}")
    _json(
        out / "verification.json",
        dict(
            status="complete",
            runs=verified,
            sample_rows=len(rows),
            summary_rows=len(summaries),
            unchanged_input_files=len(audit.hashes),
        ),
    )
    logger.info("All eight runs complete; inputs unchanged")


if __name__ == "__main__":
    main()
