from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_symmetric_tangent_audit import (
    SymmetricTangentAuditRequest,
    run_symmetric_tangent_response_audit,
)


class AuditSymmetricTangentResponseCLI:
    """Sweep fixed matrix-free tangent corrections on a frozen checkpoint."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare the configured complex projection with matrix-free "
                "symmetric-tangent response-gradient corrections."
            )
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--coupling-checkpoint", type=Path, required=True)
        parser.add_argument("--green-checkpoint", type=Path, required=True)
        parser.add_argument("--outdir", type=Path, default=None)
        parser.add_argument("--geometry", type=Path, default=None)
        parser.add_argument("--test-path", type=Path, default=None)
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument("--batch-size", type=self._positive_int, default=10)
        parser.add_argument(
            "--etas",
            type=self._positive_float,
            nargs="+",
            default=(
                0.001,
                0.0025,
                0.005,
                0.0075,
                0.01,
                0.015,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
            ),
        )
        parser.add_argument(
            "--relative-lambdas",
            type=self._nonnegative_float,
            nargs="+",
            default=(0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0),
        )
        parser.add_argument(
            "--transition-log-threshold",
            type=self._nonnegative_float,
            default=math.log(2.0),
        )
        parser.add_argument("--selected-samples", type=int, nargs="*", default=None)
        parser.add_argument(
            "--denominator-relative-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        parser.add_argument(
            "--operator-equivalence-tol",
            type=self._positive_float,
            default=1.0e-10,
        )
        parser.add_argument(
            "--save-generated-data",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.parser = parser

    @staticmethod
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be a positive integer")
        return parsed

    @staticmethod
    def _positive_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise argparse.ArgumentTypeError("value must be finite and positive")
        return parsed

    @staticmethod
    def _nonnegative_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise argparse.ArgumentTypeError("value must be finite and non-negative")
        return parsed

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("AuditSymmetricTangentResponse")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logging.root.handlers.clear()
        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(
            outdir / "audit_symmetric_tangent_response.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        outdir = args.outdir or (
            args.coupling_checkpoint.parent / "symmetric_tangent_response_audit"
        )
        logger = self._build_logger(outdir)
        summary = run_symmetric_tangent_response_audit(
            SymmetricTangentAuditRequest(
                config=args.config,
                coupling_checkpoint=args.coupling_checkpoint,
                green_checkpoint=args.green_checkpoint,
                outdir=outdir,
                geometry=args.geometry,
                test_path=args.test_path,
                coefficients=args.coefficients,
                device=args.device,
                theme=args.theme,
                etas=tuple(args.etas),
                relative_lambdas=tuple(args.relative_lambdas),
                transition_log_threshold=args.transition_log_threshold,
                selected_samples=(
                    None
                    if args.selected_samples is None
                    else tuple(args.selected_samples)
                ),
                batch_size=args.batch_size,
                denominator_relative_eps=args.denominator_relative_eps,
                operator_equivalence_tol=args.operator_equivalence_tol,
                save_generated_data=args.save_generated_data,
            ),
            logger=logger,
        )
        logger.info("Summary written to %s", outdir / "summary.json")
        logger.info(
            "Best tangent response method=%s; best tangent canonical method=%s",
            summary["automated_findings"][
                "lowest_mean_response_mismatch_tangent_method"
            ],
            summary["automated_findings"][
                "lowest_mean_canonical_energy_tangent_method"
            ],
        )


if __name__ == "__main__":
    AuditSymmetricTangentResponseCLI().run()
