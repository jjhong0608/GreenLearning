from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_length_response_diagnostics import (
    ComplexLengthResponseDiagnosticRequest,
    run_complex_length_response_diagnostics,
)


class DiagnoseComplexLengthResponseCLI:
    """Checkpoint-backed diagnostic for complex segment-length amplification."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Diagnose where complex CouplingNet source errors acquire axial "
                "transition-line structure."
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
                "Output directory. Defaults to <coupling-checkpoint-parent>/"
                "length_response_diagnostics."
            ),
        )
        parser.add_argument("--geometry", type=Path, default=None)
        parser.add_argument("--test-path", type=Path, default=None)
        parser.add_argument("--coefficients", type=Path, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument(
            "--selected-samples",
            nargs="*",
            type=int,
            default=[47],
            help="Explicit sample indices; sample 47 is selected by default.",
        )
        parser.add_argument(
            "--include-rel-sol-quantiles",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Also select unique min/q25/q50/q75/max rel_sol samples.",
        )
        parser.add_argument(
            "--transition-coordinate",
            type=float,
            default=None,
            help="Optional inner-boundary transition coordinate override.",
        )
        parser.add_argument(
            "--transition-zone-radius",
            type=float,
            default=None,
            help="Optional cardinal-neighborhood radius override.",
        )
        parser.add_argument(
            "--cardinal-radius-grid-steps",
            type=int,
            default=2,
            help="Cardinal-neighborhood radius in grid steps when not overridden.",
        )
        parser.add_argument(
            "--equivalence-tolerance",
            type=float,
            default=1.0e-10,
            help="Tolerance for unit- and physical-integral equivalence.",
        )
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("DiagnoseComplexLengthResponse")
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
            outdir / "diagnose_complex_length_response.log",
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
            args.coupling_checkpoint.parent / "length_response_diagnostics"
        )
        request = ComplexLengthResponseDiagnosticRequest(
            config=args.config,
            coupling_checkpoint=args.coupling_checkpoint,
            green_checkpoint=args.green_checkpoint,
            outdir=outdir,
            geometry=args.geometry,
            test_path=args.test_path,
            coefficients=args.coefficients,
            device=args.device,
            theme=args.theme,
            selected_samples=tuple(args.selected_samples),
            include_rel_sol_quantiles=bool(args.include_rel_sol_quantiles),
            transition_coordinate=args.transition_coordinate,
            transition_zone_radius=args.transition_zone_radius,
            cardinal_radius_grid_steps=args.cardinal_radius_grid_steps,
            equivalence_tolerance=args.equivalence_tolerance,
        )
        logger = self._build_logger(outdir)
        logger.info(
            "Starting complex length-response diagnostic: config=%s checkpoint=%s",
            request.config,
            request.coupling_checkpoint,
        )
        summary = run_complex_length_response_diagnostics(request, logger=logger)
        logger.info(
            "Diagnostic outputs written to %s (samples=%s, figures=%s)",
            request.outdir,
            summary["num_samples"],
            summary["figure_count"],
        )


def main() -> None:
    DiagnoseComplexLengthResponseCLI().run()


if __name__ == "__main__":
    main()
