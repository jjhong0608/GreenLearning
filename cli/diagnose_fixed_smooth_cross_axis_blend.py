from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_smooth_blend_diagnostics import (
    FixedSmoothBlendConfig,
    FixedSmoothBlendDiagnosticRequest,
    run_fixed_smooth_cross_axis_blend_diagnostic,
)


class DiagnoseFixedSmoothCrossAxisBlendCLI:
    """Post-hoc fixed smooth blend diagnostic for complex CouplingNet."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare the current equal reconstruction mean with a fixed, "
                "geometry-only smooth cross-axis blend."
            )
        )
        parser.add_argument("--config", type=Path, required=True)
        parser.add_argument("--coupling-checkpoint", type=Path, required=True)
        parser.add_argument("--green-checkpoint", type=Path, required=True)
        parser.add_argument(
            "--outdir",
            type=Path,
            default=None,
            help=(
                "Defaults to a method-specific diagnostic directory under the "
                "coupling checkpoint parent."
            ),
        )
        parser.add_argument("--geometry", type=Path, default=None)
        parser.add_argument("--test-path", type=Path, default=None)
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument("--batch-size", type=self._positive_int, default=10)
        parser.add_argument(
            "--selected-samples",
            type=int,
            nargs="*",
            default=None,
            help=(
                "Explicit sample indices. If omitted, baseline rel_sol "
                "min/q25/q50/q75/max samples are selected."
            ),
        )
        parser.add_argument(
            "--weight-construction",
            choices=("jump_smoothed", "compact_c2_ramp"),
            default="jump_smoothed",
            help=(
                "Use the legacy smoothed length-jump reliability or the compact "
                "C2 topology-distance ramp."
            ),
        )
        parser.add_argument(
            "--alpha",
            type=self._positive_float,
            default=1.0 / math.log(2.0),
            help=(
                "Reliability decay. The default makes a log(2) jump contribute "
                "one exponent before smoothing."
            ),
        )
        parser.add_argument(
            "--smoothing-steps",
            type=self._nonnegative_int,
            default=2,
        )
        parser.add_argument(
            "--smoothing-relaxation",
            type=self._unit_interval_positive_float,
            default=0.5,
        )
        parser.add_argument(
            "--reliability-floor",
            type=self._positive_float,
            default=1.0e-6,
        )
        parser.add_argument(
            "--transition-log-threshold",
            type=self._positive_float,
            default=math.log(2.0),
        )
        parser.add_argument(
            "--transition-dilation-steps",
            type=self._nonnegative_int,
            default=2,
        )
        parser.add_argument(
            "--ramp-gamma",
            type=self._closed_unit_interval_float,
            default=0.5,
            help="Maximum compact-ramp directional preference in [0, 1].",
        )
        parser.add_argument(
            "--ramp-width",
            type=self._positive_float,
            default=None,
            help=(
                "Physical compact-ramp half-width. The default resolves to four "
                "times max(hx, hy)."
            ),
        )
        parser.add_argument(
            "--compact-sweep",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "Evaluate a fixed gamma/width grid using the same frozen "
                "directional reconstructions."
            ),
        )
        parser.add_argument(
            "--sweep-gammas",
            type=self._closed_unit_interval_float,
            nargs="+",
            default=(0.25, 0.5, 0.75, 1.0),
        )
        parser.add_argument(
            "--sweep-width-steps",
            type=self._positive_float,
            nargs="+",
            default=(2.0, 4.0, 6.0, 8.0),
            help="Compact ramp widths measured in max(hx, hy) grid steps.",
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
            raise argparse.ArgumentTypeError("value must be non-negative")
        return parsed

    @staticmethod
    def _positive_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise argparse.ArgumentTypeError("value must be finite and positive")
        return parsed

    @staticmethod
    def _unit_interval_positive_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
            raise argparse.ArgumentTypeError("value must be in (0, 1]")
        return parsed

    @staticmethod
    def _closed_unit_interval_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError("value must be in [0, 1]")
        return parsed

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("DiagnoseFixedSmoothCrossAxisBlend")
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
            outdir / "diagnose_fixed_smooth_cross_axis_blend.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        default_outdir_name = (
            "compact_c2_cross_axis_blend"
            if args.weight_construction == "compact_c2_ramp"
            else "fixed_smooth_cross_axis_blend"
        )
        outdir = args.outdir or (args.coupling_checkpoint.parent / default_outdir_name)
        request = FixedSmoothBlendDiagnosticRequest(
            config=args.config,
            coupling_checkpoint=args.coupling_checkpoint,
            green_checkpoint=args.green_checkpoint,
            outdir=outdir,
            geometry=args.geometry,
            test_path=args.test_path,
            coefficients=args.coefficients,
            device=args.device,
            theme=args.theme,
            selected_samples=(
                None if args.selected_samples is None else tuple(args.selected_samples)
            ),
            batch_size=args.batch_size,
            save_generated_data=bool(args.save_generated_data),
            run_compact_sweep=bool(args.compact_sweep),
            sweep_gammas=tuple(args.sweep_gammas),
            sweep_width_steps=tuple(args.sweep_width_steps),
            blend=FixedSmoothBlendConfig(
                weight_construction=args.weight_construction,
                alpha=args.alpha,
                smoothing_steps=args.smoothing_steps,
                smoothing_relaxation=args.smoothing_relaxation,
                reliability_floor=args.reliability_floor,
                transition_log_threshold=args.transition_log_threshold,
                transition_dilation_steps=args.transition_dilation_steps,
                ramp_gamma=args.ramp_gamma,
                ramp_width=args.ramp_width,
            ),
        )
        logger = self._build_logger(outdir)
        logger.info(
            "Starting fixed smooth blend diagnostic: checkpoint=%s",
            request.coupling_checkpoint,
        )
        summary = run_fixed_smooth_cross_axis_blend_diagnostic(
            request,
            logger=logger,
        )
        logger.info(
            "Outputs written to %s (verdict=%s)",
            outdir,
            summary["aggregate_metrics"]["verdict"],
        )


def main() -> None:
    DiagnoseFixedSmoothCrossAxisBlendCLI().run()


if __name__ == "__main__":
    main()
