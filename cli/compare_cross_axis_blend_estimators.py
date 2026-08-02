from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_mismatch_blend_diagnostics import (
    CrossAxisBlendComparisonRequest,
    MismatchGradientBlendConfig,
    MismatchSeamC2BlendConfig,
    run_cross_axis_blend_estimator_comparison,
)
from greenonet.complex_smooth_blend_diagnostics import FixedSmoothBlendConfig


class CompareCrossAxisBlendEstimatorsCLI:
    """Compare equal-mean, geometry-only, and two mismatch estimators."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Run a frozen-checkpoint post-hoc comparison of equal-mean, "
                "geometry-only compact-ramp, and prediction-only "
                "direct mismatch-gradient and detected-seam C2 reconstruction "
                "estimators."
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
                "Defaults to cross_axis_blend_estimator_comparison under the "
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
                "Explicit sample indices. If omitted, equal-mean rel_sol "
                "min/q25/q50/q75/max samples are selected."
            ),
        )
        parser.add_argument(
            "--geometry-gamma",
            type=self._closed_unit_interval_float,
            default=0.5,
        )
        parser.add_argument(
            "--geometry-ramp-width",
            type=self._positive_float,
            default=None,
            help="Physical compact-ramp half-width; default is 4*max(hx, hy).",
        )
        parser.add_argument(
            "--mismatch-gamma",
            type=self._closed_unit_interval_float,
            default=0.5,
        )
        parser.add_argument(
            "--mismatch-smoothing-steps",
            type=self._nonnegative_int,
            default=2,
        )
        parser.add_argument(
            "--mismatch-smoothing-relaxation",
            type=self._unit_interval_positive_float,
            default=0.5,
        )
        parser.add_argument(
            "--mismatch-activation-lower",
            type=self._nonnegative_float,
            default=0.15,
        )
        parser.add_argument(
            "--mismatch-activation-upper",
            type=self._positive_float,
            default=0.35,
        )
        parser.add_argument(
            "--mismatch-scale-eps",
            type=self._positive_float,
            default=1.0e-12,
        )
        parser.add_argument(
            "--seam-gamma",
            type=self._closed_unit_interval_float,
            default=0.5,
        )
        parser.add_argument(
            "--seam-ramp-width",
            type=self._positive_float,
            default=None,
            help="Physical compact C2 half-width; default is 8*max(hx, hy).",
        )
        parser.add_argument(
            "--seam-max-per-axis",
            type=self._positive_int,
            default=2,
        )
        parser.add_argument(
            "--seam-peak-relative-threshold",
            type=self._unit_interval_positive_float,
            default=0.25,
        )
        parser.add_argument(
            "--seam-profile-smoothing-steps",
            type=self._nonnegative_int,
            default=1,
        )
        parser.add_argument(
            "--seam-minimum-separation",
            type=self._positive_float,
            default=None,
            help="Physical NMS separation; default is four times the seam ramp width.",
        )
        parser.add_argument(
            "--seam-sweep",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Run evaluation-only gamma/width/peak-threshold sensitivity sweep.",
        )
        parser.add_argument(
            "--seam-sweep-gammas",
            type=self._closed_unit_interval_float,
            nargs="+",
            default=(0.2, 0.3, 0.4, 0.5),
        )
        parser.add_argument(
            "--seam-sweep-width-steps",
            type=self._positive_float,
            nargs="+",
            default=(4.0, 6.0, 8.0, 10.0, 12.0),
        )
        parser.add_argument(
            "--seam-sweep-peak-thresholds",
            type=self._unit_interval_positive_float,
            nargs="+",
            default=(0.15, 0.2, 0.25, 0.3),
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
    def _nonnegative_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise argparse.ArgumentTypeError("value must be finite and non-negative")
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
        logger = logging.getLogger("CompareCrossAxisBlendEstimators")
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
            outdir / "compare_cross_axis_blend_estimators.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        if args.mismatch_activation_upper <= args.mismatch_activation_lower:
            self.parser.error(
                "--mismatch-activation-upper must be greater than "
                "--mismatch-activation-lower"
            )
        outdir = args.outdir or (
            args.coupling_checkpoint.parent / "cross_axis_blend_estimator_comparison"
        )
        request = CrossAxisBlendComparisonRequest(
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
            blend=FixedSmoothBlendConfig(
                weight_construction="compact_c2_ramp",
                ramp_gamma=args.geometry_gamma,
                ramp_width=args.geometry_ramp_width,
            ),
            mismatch=MismatchGradientBlendConfig(
                gamma=args.mismatch_gamma,
                smoothing_steps=args.mismatch_smoothing_steps,
                smoothing_relaxation=args.mismatch_smoothing_relaxation,
                activation_lower=args.mismatch_activation_lower,
                activation_upper=args.mismatch_activation_upper,
                scale_eps=args.mismatch_scale_eps,
            ),
            seam_c2=MismatchSeamC2BlendConfig(
                gamma=args.seam_gamma,
                ramp_width=args.seam_ramp_width,
                max_seams_per_axis=args.seam_max_per_axis,
                peak_relative_threshold=args.seam_peak_relative_threshold,
                profile_smoothing_steps=args.seam_profile_smoothing_steps,
                minimum_separation=args.seam_minimum_separation,
                scale_eps=args.mismatch_scale_eps,
            ),
            seam_sweep=bool(args.seam_sweep),
            seam_sweep_gammas=tuple(args.seam_sweep_gammas),
            seam_sweep_width_steps=tuple(args.seam_sweep_width_steps),
            seam_sweep_peak_thresholds=tuple(args.seam_sweep_peak_thresholds),
        )
        logger = self._build_logger(outdir)
        logger.info(
            "Starting cross-axis estimator comparison: checkpoint=%s",
            request.coupling_checkpoint,
        )
        summary = run_cross_axis_blend_estimator_comparison(
            request,
            logger=logger,
        )
        geometry = summary["aggregate_metrics"]["geometry_only"]
        mismatch = summary["aggregate_metrics"]["mismatch_gradient"]
        seam_c2 = summary["aggregate_metrics"]["mismatch_detected_seam_c2"]
        logger.info(
            "Outputs written to %s: geometry_rel_sol=%.6f mismatch_rel_sol=%.6f "
            "seam_c2_rel_sol=%.6f",
            outdir,
            geometry["blend_rel_sol_mean"],
            mismatch["blend_rel_sol_mean"],
            seam_c2["blend_rel_sol_mean"],
        )


def main() -> None:
    CompareCrossAxisBlendEstimatorsCLI().run()


if __name__ == "__main__":
    main()
