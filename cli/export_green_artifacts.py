from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence, cast

from rich.logging import RichHandler

from greenonet.green_artifacts import (
    EvalSplit,
    GreenArtifactRequest,
    SamplerMode,
    ScaleLength,
    export_green_artifacts,
)


class ExportGreenArtifactsCLI:
    """CLI for regenerating paper-oriented GreenNet artifacts from a checkpoint."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Export GreenNet checkpoint artifacts for paper figures."
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=True,
            help="Path to a GreenONet checkpoint.",
        )
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to the training config JSON or config_used.json.",
        )
        parser.add_argument(
            "--outdir",
            type=Path,
            required=True,
            help="Directory where artifacts will be written.",
        )
        parser.add_argument(
            "--coefficients",
            type=Path,
            default=None,
            help="Optional coefficient function file override.",
        )
        parser.add_argument(
            "--eval-seed",
            type=int,
            default=12345,
            help="Seed used before regenerating evaluation data.",
        )
        parser.add_argument(
            "--eval-split",
            choices=("train_like", "validation_like", "custom"),
            default="validation_like",
            help="Which config sampling settings to use by default.",
        )
        parser.add_argument(
            "--eval-samples-per-line",
            type=int,
            default=None,
            help="Optional evaluation sample count per axial line.",
        )
        parser.add_argument(
            "--eval-sampler-mode",
            choices=("forward", "backward"),
            default=None,
            help="Optional evaluation sampler mode override.",
        )
        parser.add_argument(
            "--eval-scale-length",
            type=float,
            nargs="+",
            default=None,
            help="Optional one value or two values for evaluation scale length.",
        )
        parser.add_argument(
            "--line-indices",
            type=int,
            nargs="*",
            default=None,
            help="Selected axial line indices. Defaults to first, middle, last.",
        )
        parser.add_argument(
            "--xi-fractions",
            type=float,
            nargs="+",
            default=[0.25, 0.5, 0.75],
            help="Fixed xi fractions to slice, mapped to nearest grid points.",
        )
        parser.add_argument(
            "--include-boundary-xi",
            action="store_true",
            help="Also include nearest interior xi points to the boundaries.",
        )
        parser.add_argument(
            "--theme",
            type=str,
            default="plotly_white",
            help="Plotly template name.",
        )
        parser.add_argument(
            "--save-generated-data",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save regenerated eval tensors and selected arrays as NPZ files.",
        )
        self.parser = parser

    @staticmethod
    def _parse_scale_length(values: Sequence[float] | None) -> ScaleLength | None:
        if values is None:
            return None
        if len(values) == 1:
            return float(values[0])
        if len(values) == 2:
            return (float(values[0]), float(values[1]))
        raise ValueError("--eval-scale-length accepts one value or two values.")

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("ExportGreenArtifacts")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            outdir / "export_green_artifacts.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        request = GreenArtifactRequest(
            checkpoint=args.checkpoint,
            config=args.config,
            outdir=args.outdir,
            coefficients=args.coefficients,
            eval_seed=args.eval_seed,
            eval_split=cast(EvalSplit, args.eval_split),
            eval_samples_per_line=args.eval_samples_per_line,
            eval_sampler_mode=cast(SamplerMode | None, args.eval_sampler_mode),
            eval_scale_length=self._parse_scale_length(args.eval_scale_length),
            line_indices=(
                None
                if args.line_indices is None
                else tuple(int(item) for item in args.line_indices)
            ),
            xi_fractions=tuple(float(item) for item in args.xi_fractions),
            include_boundary_xi=args.include_boundary_xi,
            theme=args.theme,
            save_generated_data=bool(args.save_generated_data),
        )
        logger = self._build_logger(request.outdir)
        summary = export_green_artifacts(request, logger=logger)
        logger.info(
            "Completed GreenNet artifact export (rel_green_valid=%s)",
            summary["rel_green_valid"],
        )


def main() -> None:
    ExportGreenArtifactsCLI().run()


if __name__ == "__main__":
    main()
