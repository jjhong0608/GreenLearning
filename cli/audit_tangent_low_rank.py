from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_tangent_low_rank_audit import (
    LowRankSpectralAuditRequest,
    run_tangent_low_rank_audit,
)


class AuditTangentLowRankCLI:
    """Run the frozen diagonal versus spectral-low-rank K=1..4 audit."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare a diagonal tangent preconditioner with nested matrix-free "
                "spectral low-rank corrections for K=1 through K=4."
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
            "--ranks",
            type=self._positive_int,
            nargs="+",
            default=(4, 8, 16, 32),
        )
        parser.add_argument("--oversampling", type=self._nonnegative_int, default=16)
        parser.add_argument(
            "--power-iterations",
            type=self._nonnegative_int,
            default=3,
        )
        parser.add_argument("--probe-count", type=self._positive_int, default=32)
        parser.add_argument(
            "--probe-batch-size",
            type=self._positive_int,
            default=16,
        )
        parser.add_argument("--seed", type=self._nonnegative_int, default=1729)
        parser.add_argument(
            "--eigenvalue-relative-floor",
            type=self._positive_float,
            default=1.0e-10,
        )
        parser.add_argument(
            "--complement-scale",
            choices=("unit", "next_ritz"),
            default="next_ritz",
        )
        parser.add_argument(
            "--benchmark-warmup",
            type=self._nonnegative_int,
            default=1,
        )
        parser.add_argument(
            "--benchmark-repeats",
            type=self._positive_int,
            default=3,
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
    def _nonnegative_int(value: str) -> int:
        parsed = int(value)
        if parsed < 0:
            raise argparse.ArgumentTypeError("value must be a non-negative integer")
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
        logger = logging.getLogger("AuditTangentLowRank")
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
            outdir / "audit_tangent_low_rank.log",
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
            args.coupling_checkpoint.parent / "tangent_low_rank_k1_k4_audit"
        )
        logger = self._build_logger(outdir)
        summary = run_tangent_low_rank_audit(
            LowRankSpectralAuditRequest(
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
                selected_samples=(
                    None
                    if args.selected_samples is None
                    else tuple(args.selected_samples)
                ),
                transition_log_threshold=args.transition_log_threshold,
                subspace_relative_eps=args.subspace_relative_eps,
                operator_equivalence_tol=args.operator_equivalence_tol,
                monotonicity_relative_tol=args.monotonicity_relative_tol,
                ranks=tuple(args.ranks),
                oversampling=args.oversampling,
                power_iterations=args.power_iterations,
                probe_count=args.probe_count,
                probe_batch_size=args.probe_batch_size,
                seed=args.seed,
                eigenvalue_relative_floor=args.eigenvalue_relative_floor,
                complement_scale=args.complement_scale,
                benchmark_warmup=args.benchmark_warmup,
                benchmark_repeats=args.benchmark_repeats,
                save_generated_data=args.save_generated_data,
            ),
            logger=logger,
        )
        logger.info("Summary written to %s", outdir / "summary.json")
        logger.info(
            "Best rel_sol method: %s",
            summary["findings"].get("lowest_rel_sol_mean_method", "unavailable"),
        )


if __name__ == "__main__":
    AuditTangentLowRankCLI().run()
