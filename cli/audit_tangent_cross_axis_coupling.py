from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_tangent_cross_axis_audit import (
    TangentCrossAxisAuditRequest,
    run_tangent_cross_axis_audit,
)
from greenonet.complex_tangent_preconditioner import (
    TANGENT_PRECONDITIONER_VARIANTS,
)


class AuditTangentCrossAxisCouplingCLI:
    """Run the implicit cross-axis Gram and actual-direction audit."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Estimate implicit cross-axis off-diagonal coupling with Rademacher "
                "probes and measure it on frozen CouplingNet tangent directions."
            )
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--coupling-checkpoint", type=Path, required=True)
        parser.add_argument("--green-checkpoint", type=Path, required=True)
        parser.add_argument("--outdir", type=Path, default=None)
        parser.add_argument("--geometry", type=Path, default=None)
        parser.add_argument("--test-path", type=Path, default=None)
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--tangent-context", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument("--batch-size", type=self._positive_int, default=10)
        parser.add_argument("--probe-count", type=self._positive_int, default=256)
        parser.add_argument("--probe-batch-size", type=self._positive_int, default=16)
        parser.add_argument(
            "--probe-seed", type=self._nonnegative_int, default=20260826
        )
        parser.add_argument("--confidence-z", type=self._positive_float, default=1.96)
        parser.add_argument(
            "--preconditioner-variant",
            choices=TANGENT_PRECONDITIONER_VARIANTS,
            default="separable",
        )
        parser.add_argument(
            "--operator-equivalence-tol",
            type=self._positive_float,
            default=1.0e-10,
        )
        parser.add_argument(
            "--posthoc-tangent-override",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument("--posthoc-eta", type=self._nonnegative_float, default=0.01)
        parser.add_argument(
            "--posthoc-line-search-relative-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        parser.add_argument(
            "--posthoc-relative-lambda",
            type=self._nonnegative_float,
            default=0.01,
        )
        parser.add_argument(
            "--posthoc-denominator-relative-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        parser.add_argument(
            "--posthoc-cross-axis-relative-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        self.parser = parser

    @staticmethod
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be a positive integer")
        return parsed

    @staticmethod
    def _nonnegative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("value must be non-negative")
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
        logger = logging.getLogger("AuditTangentCrossAxisCoupling")
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
            outdir / "audit_tangent_cross_axis_coupling.log",
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
            args.coupling_checkpoint.parent / "tangent_cross_axis_coupling_audit"
        )
        logger = self._build_logger(outdir)
        summary = run_tangent_cross_axis_audit(
            TangentCrossAxisAuditRequest(
                config=args.config,
                coupling_checkpoint=args.coupling_checkpoint,
                green_checkpoint=args.green_checkpoint,
                outdir=outdir,
                geometry=args.geometry,
                test_path=args.test_path,
                coefficients=args.coefficients,
                tangent_context=args.tangent_context,
                device=args.device,
                theme=args.theme,
                batch_size=args.batch_size,
                probe_count=args.probe_count,
                probe_batch_size=args.probe_batch_size,
                probe_seed=args.probe_seed,
                confidence_z=args.confidence_z,
                preconditioner_variant=args.preconditioner_variant,
                operator_equivalence_tol=args.operator_equivalence_tol,
                posthoc_tangent_override=args.posthoc_tangent_override,
                posthoc_eta=args.posthoc_eta,
                posthoc_line_search_relative_eps=(
                    args.posthoc_line_search_relative_eps
                ),
                posthoc_relative_lambda=args.posthoc_relative_lambda,
                posthoc_denominator_relative_eps=(
                    args.posthoc_denominator_relative_eps
                ),
                posthoc_cross_axis_relative_eps=(args.posthoc_cross_axis_relative_eps),
            ),
            logger=logger,
        )
        logger.info("Summary written to %s", outdir / "summary.json")
        logger.info(
            "Estimated R_off,C=%.6f; mean actual-direction cross ratio=%.6f",
            summary["operator_global"]["cross"]["off_diagonal_fraction"],
            summary["sample_direction"]["cross_to_tangent_action_ratio_mean"],
        )


if __name__ == "__main__":
    AuditTangentCrossAxisCouplingCLI().run()
