"""Time the original native prediction path, reusing its tangent response."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
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


class NativePredictionTiming:
    @torch.no_grad()
    def run(self, root: Path) -> None:
        target = root / "native_prediction_timing.csv"
        if target.exists():
            raise FileExistsError(target)
        logger = logging.getLogger("native_prediction_timing")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for handler in (
            RichHandler(show_path=True, omit_repeated_times=False),
            logging.FileHandler(root / "native_prediction_timing.log"),
        ):
            handler.setFormatter(logging.Formatter("%(funcName)s - %(message)s"))
            logger.addHandler(handler)
        torch.set_num_threads(4)
        previous = json.loads((root / "input_hashes.json").read_text())
        preflight_root = root / "native_prediction_preflight"
        preflight_root.mkdir(exist_ok=False)
        audit = SourceInitializationAudit(
            SourceInitializationRequest(
                Path("configs/paper_source_initialization_audit.json"),
                preflight_root,
                batch_size=5,
            )
        )
        audit.preflight()
        for filename, digest in audit.hashes.items():
            if previous[filename] != digest:
                raise ValueError(f"Original numerical input changed: {filename}")
        with (root / "per_sample.csv").open() as stream:
            metrics = {
                (row["example"], int(row["sample_id"])): float(row["rel_sol"])
                for row in csv.DictReader(stream)
                if row["condition"] == "learned"
            }
        rows: list[dict[str, Any]] = []
        for run in audit.runs:
            if run.spec.seed != 0 or run.example not in {"unit_square", "disk"}:
                continue
            audit._gpu_idle()
            session = SourceRunSession(run)
            try:
                batch = session.batches[0]
                result = session._prediction_forward(batch, run.native_k)
                assert isinstance(result, ComplexCrossAxisReconstructionResult)
                errors = (result.u_pred_valid - batch.sol_valid).norm(dim=-1)
                errors /= batch.sol_valid.norm(dim=-1)
                np.testing.assert_allclose(
                    errors.cpu().numpy(),
                    [metrics[run.example, int(i)] for i in batch.sample_indices],
                    rtol=1e-8,
                    atol=1e-12,
                )
                for repeat in range(-3, 5):
                    torch.cuda.synchronize(1)
                    start = time.perf_counter()
                    session._prediction_forward(batch, run.native_k)
                    torch.cuda.synchronize(1)
                    elapsed = time.perf_counter() - start
                    logger.info("%s repeat%d %.6fs", run.example, repeat, elapsed)
                    if repeat >= 0:
                        rows.append(
                            dict(
                                example=run.example,
                                method="learned",
                                repeat=repeat,
                                batch_size=5,
                                seconds=elapsed,
                            )
                        )
                        _write_csv(target, rows)
            finally:
                session.close()
        for filename, digest in audit.hashes.items():
            if _sha256(Path(filename)) != digest:
                raise ValueError(f"Input changed during timing: {filename}")
        _json(
            root / "native_prediction_verification.json",
            dict(
                original_numerical_inputs_verified=len(audit.hashes),
                original_prediction_values_reproduced=True,
                rows=len(rows),
                implementation_sha256=_sha256(Path(__file__)),
                timing_scope="original _prediction_forward including tangent response reuse",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    NativePredictionTiming().run(parser.parse_args().outdir)
