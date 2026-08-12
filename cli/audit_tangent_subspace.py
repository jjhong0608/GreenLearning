from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_tangent_subspace_audit import (
    TangentSubspaceAuditRequest,
    run_tangent_subspace_audit,
)


class AuditTangentSubspaceCLI:
    """Run a frozen-checkpoint matrix-free K=1 through K=4 comparison."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare the configured-cap K=1 tangent correction with nested "
                "unconstrained matrix-free tangent subspaces through K=4."
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
            "--max-subspace-dimension",
            type=self._subspace_dimension,
            choices=(2, 3, 4),
            default=2,
            help="Largest unconstrained diagnostic tangent subspace (default: 2).",
        )
        parser.add_argument("--selected-samples", type=int, nargs="*", default=None)
        parser.add_argument(
            "--transition-log-threshold",
            type=self._nonnegative_float,
            default=math.log(2.0),
        )
        parser.add_argument(
            "--subspace-relative-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        parser.add_argument(
            "--operator-equivalence-tol",
            type=self._positive_float,
            default=1.0e-10,
        )
        parser.add_argument(
            "--monotonicity-relative-tol",
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
    def _subspace_dimension(value: str) -> int:
        parsed = int(value)
        if parsed not in {2, 3, 4}:
            raise argparse.ArgumentTypeError("value must be 2, 3, or 4")
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
        logger = logging.getLogger("AuditTangentSubspace")
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
            outdir / "audit_tangent_subspace.log",
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
            args.coupling_checkpoint.parent
            / f"tangent_subspace_k1_k{args.max_subspace_dimension}_audit"
        )
        logger = self._build_logger(outdir)
        summary = run_tangent_subspace_audit(
            TangentSubspaceAuditRequest(
                config=args.config,
                coupling_checkpoint=args.coupling_checkpoint,
                green_checkpoint=args.green_checkpoint,
                outdir=outdir,
                geometry=args.geometry,
                test_path=args.test_path,
                coefficients=args.coefficients,
                device=args.device,
                theme=args.theme,
                batch_size=args.batch_size,
                selected_samples=(
                    None
                    if args.selected_samples is None
                    else tuple(args.selected_samples)
                ),
                transition_log_threshold=args.transition_log_threshold,
                subspace_relative_eps=args.subspace_relative_eps,
                operator_equivalence_tol=args.operator_equivalence_tol,
                monotonicity_relative_tol=args.monotonicity_relative_tol,
                max_subspace_dimension=args.max_subspace_dimension,
                save_generated_data=args.save_generated_data,
            ),
            logger=logger,
        )
        logger.info("Summary written to %s", outdir / "summary.json")
        maximum_dimension = args.max_subspace_dimension
        comparison = summary["paired_comparisons"][
            f"k{maximum_dimension}_vs_k1_production"
        ]
        logger.info(
            "K%d vs configured-cap K1: response_change=%.6f rel_sol_change=%.6f",
            maximum_dimension,
            comparison["response_mismatch_cost"]["relative_mean_change"],
            comparison["rel_sol"]["relative_mean_change"],
        )


if __name__ == "__main__":
    AuditTangentSubspaceCLI().run()
