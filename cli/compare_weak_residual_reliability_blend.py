from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_mismatch_blend_diagnostics import (
    MismatchGradientBlendConfig,
    MismatchSeamC2BlendConfig,
)
from greenonet.complex_smooth_blend_diagnostics import FixedSmoothBlendConfig
from greenonet.complex_weak_residual_blend_diagnostics import (
    WeakResidualBlendComparisonRequest,
    WeakResidualReliabilityBlendConfig,
    run_weak_residual_blend_comparison,
)


class CompareWeakResidualReliabilityBlendCLI:
    """Run the frozen-checkpoint four-estimator reliability comparison."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare equal mean, geometry C2, mismatch-detected seam C2, "
                "and local weak-residual reliability reconstruction blends."
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
        parser.add_argument("--selected-samples", type=int, nargs="*", default=None)

        parser.add_argument(
            "--geometry-gamma",
            type=self._closed_unit_float,
            default=0.5,
        )
        parser.add_argument(
            "--geometry-ramp-width",
            type=self._positive_float,
            default=None,
        )
        parser.add_argument(
            "--seam-gamma",
            type=self._closed_unit_float,
            default=0.3,
        )
        parser.add_argument(
            "--seam-ramp-width",
            type=self._positive_float,
            default=None,
        )
        parser.add_argument("--seam-max-per-axis", type=self._positive_int, default=2)
        parser.add_argument(
            "--seam-peak-relative-threshold",
            type=self._positive_unit_float,
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
        )

        parser.add_argument(
            "--weak-gamma",
            type=self._closed_unit_float,
            default=0.5,
        )
        parser.add_argument(
            "--weak-smoothing-steps",
            type=self._nonnegative_int,
            default=2,
        )
        parser.add_argument(
            "--weak-smoothing-relaxation",
            type=self._positive_unit_float,
            default=0.5,
        )
        parser.add_argument(
            "--weak-relative-floor",
            type=self._nonnegative_float,
            default=0.1,
        )
        parser.add_argument("--weak-eps", type=self._positive_float, default=1.0e-12)
        parser.add_argument(
            "--weak-sweep",
            action=argparse.BooleanOptionalAction,
            default=False,
        )
        parser.add_argument(
            "--weak-sweep-gammas",
            type=self._closed_unit_float,
            nargs="+",
            default=(0.25, 0.5, 0.75, 1.0),
        )
        parser.add_argument(
            "--weak-sweep-relative-floors",
            type=self._nonnegative_float,
            nargs="+",
            default=(0.01, 0.1, 1.0),
        )
        parser.add_argument(
            "--weak-sweep-smoothing-steps",
            type=self._nonnegative_int,
            nargs="+",
            default=(0, 2, 4),
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
    def _closed_unit_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError("value must be in [0, 1]")
        return parsed

    @staticmethod
    def _positive_unit_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
            raise argparse.ArgumentTypeError("value must be in (0, 1]")
        return parsed

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("CompareWeakResidualReliabilityBlend")
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
            outdir / "compare_weak_residual_reliability_blend.log",
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
            / "weak_residual_reliability_blend_comparison"
        )
        logger = self._build_logger(outdir)
        request = WeakResidualBlendComparisonRequest(
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
            save_generated_data=args.save_generated_data,
            blend=FixedSmoothBlendConfig(
                weight_construction="compact_c2_ramp",
                ramp_gamma=args.geometry_gamma,
                ramp_width=args.geometry_ramp_width,
            ),
            mismatch=MismatchGradientBlendConfig(),
            seam_c2=MismatchSeamC2BlendConfig(
                gamma=args.seam_gamma,
                ramp_width=args.seam_ramp_width,
                max_seams_per_axis=args.seam_max_per_axis,
                peak_relative_threshold=args.seam_peak_relative_threshold,
                profile_smoothing_steps=args.seam_profile_smoothing_steps,
                minimum_separation=args.seam_minimum_separation,
            ),
            weak_residual=WeakResidualReliabilityBlendConfig(
                gamma=args.weak_gamma,
                smoothing_steps=args.weak_smoothing_steps,
                smoothing_relaxation=args.weak_smoothing_relaxation,
                relative_floor=args.weak_relative_floor,
                eps=args.weak_eps,
            ),
            weak_sweep=args.weak_sweep,
            weak_sweep_gammas=tuple(args.weak_sweep_gammas),
            weak_sweep_relative_floors=tuple(args.weak_sweep_relative_floors),
            weak_sweep_smoothing_steps=tuple(args.weak_sweep_smoothing_steps),
        )
        summary = run_weak_residual_blend_comparison(request, logger=logger)
        logger.info("Summary written to %s", outdir / "summary.json")
        weak = summary["aggregate_metrics"]["weak_residual_reliability"]
        logger.info(
            "Weak reliability mean rel_sol=%.6f relative_change=%+.3f%%",
            weak["rel_sol_mean"],
            100.0 * weak["rel_sol_relative_change_vs_equal"],
        )


if __name__ == "__main__":
    CompareWeakResidualReliabilityBlendCLI().run()
