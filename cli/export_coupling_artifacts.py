from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from greenonet.coupling_artifact_runtime import (
    build_coupling_artifact_logger,
    export_coupling_artifact_request,
)
from greenonet.coupling_artifacts import CouplingArtifactRequest


class ExportCouplingArtifactsCLI:
    """CLI for exporting paper-facing CouplingNet artifacts."""

    @staticmethod
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be a positive integer")
        return parsed

    @staticmethod
    def _directional_color_quantile(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or parsed <= 0.5 or parsed > 1.0:
            raise argparse.ArgumentTypeError("value must be finite and in (0.5, 1.0]")
        return parsed

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Export CouplingNet checkpoint artifacts for paper figures."
        )
        parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to the training config JSON or config_used.json.",
        )
        parser.add_argument(
            "--coupling-checkpoint",
            type=Path,
            required=True,
            help="Path to a CouplingNet checkpoint.",
        )
        parser.add_argument(
            "--green-checkpoint",
            type=Path,
            required=True,
            help="Path to the GreenONet checkpoint used for reconstruction.",
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
            "--device",
            type=str,
            default=None,
            help=(
                "Optional torch device override, e.g. cpu, cuda, or cuda:0. "
                "Defaults to coupling_training.device in the config."
            ),
        )
        parser.add_argument(
            "--theme",
            type=str,
            default="plotly_white",
            help="Plotly template name.",
        )
        parser.add_argument(
            "--selected-samples",
            type=int,
            nargs="*",
            default=None,
            help=(
                "Selected test sample indices. If omitted, samples are selected "
                "from rel_sol quantiles: min, q25, q50, q75, max."
            ),
        )
        parser.add_argument(
            "--plot-workers",
            type=int,
            default=1,
            help="Reserved plot worker count recorded in metadata.",
        )
        parser.add_argument(
            "--save-generated-data",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Save selected samples, predictions, and diagnostics as NPZ files.",
        )
        parser.add_argument(
            "--coefficient-vector-max-points",
            type=self._positive_int,
            default=400,
            help=(
                "Maximum number of quiver arrows in complex-domain physical "
                "coefficient figures."
            ),
        )
        parser.add_argument(
            "--show-domain-boundary",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Overlay geometry-only boundary markers on complex-domain "
                "figures. Boundary points do not carry scalar field values."
            ),
        )
        parser.add_argument(
            "--visualization-mesh",
            type=Path,
            default=None,
            help=(
                "Optional precomputed complex-domain visualization mesh NPZ. "
                "When provided, scalar mesh figures are added."
            ),
        )
        parser.add_argument(
            "--directional-color-quantile",
            type=self._directional_color_quantile,
            default=None,
            help=(
                "Complex-only robust color quantile for phi/psi values and "
                "errors. Defaults to 0.99 in complex mode; use 1.0 for full range."
            ),
        )
        parser.add_argument(
            "--tangent-context",
            type=Path,
            default=None,
            help=(
                "Optional tangent response context sidecar override for complex "
                "artifacts."
            ),
        )
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        return build_coupling_artifact_logger(outdir)

    def run(self) -> None:
        args = self.parser.parse_args()
        request = CouplingArtifactRequest(
            config=args.config,
            coupling_checkpoint=args.coupling_checkpoint,
            green_checkpoint=args.green_checkpoint,
            outdir=args.outdir,
            coefficients=args.coefficients,
            device=args.device,
            theme=args.theme,
            selected_samples=(
                None
                if args.selected_samples is None
                else tuple(int(item) for item in args.selected_samples)
            ),
            plot_workers=args.plot_workers,
            save_generated_data=bool(args.save_generated_data),
            coefficient_vector_max_points=args.coefficient_vector_max_points,
            show_domain_boundary=bool(args.show_domain_boundary),
            visualization_mesh=args.visualization_mesh,
            directional_color_quantile=args.directional_color_quantile,
            tangent_context=args.tangent_context,
        )
        logger = self._build_logger(request.outdir)
        export_coupling_artifact_request(request, logger=logger)


def main() -> None:
    ExportCouplingArtifactsCLI().run()


if __name__ == "__main__":
    main()
