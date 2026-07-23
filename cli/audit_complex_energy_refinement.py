from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_energy_refinement import (
    ComplexEnergyRefinementRequest,
    audit_complex_energy_refinement,
)


class AuditComplexEnergyRefinementCLI:
    """Run a persistent-jump inverse-h audit across annulus refinements."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Verify that the canonical complex split energy assigns inverse-h "
                "growth to a persistent annulus interface jump."
            )
        )
        parser.add_argument("--geometries", type=Path, nargs="+", required=True)
        parser.add_argument("--outdir", type=Path, required=True)
        parser.add_argument("--jump-axis", choices=("x", "y"), default="y")
        parser.add_argument("--jump-coordinate", type=float)
        parser.add_argument("--exponent-min", type=float, default=-1.25)
        parser.add_argument("--exponent-max", type=float, default=-0.75)
        parser.add_argument(
            "--scaled-energy-relative-spread-max",
            type=float,
            default=0.35,
        )
        parser.add_argument(
            "--no-fail-on-violation",
            action="store_true",
        )
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("AuditComplexEnergyRefinement")
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
            outdir / "audit_complex_energy_refinement.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        request = ComplexEnergyRefinementRequest(
            geometries=tuple(args.geometries),
            outdir=args.outdir,
            jump_axis=args.jump_axis,
            jump_coordinate=args.jump_coordinate,
            exponent_min=args.exponent_min,
            exponent_max=args.exponent_max,
            scaled_energy_relative_spread_max=(args.scaled_energy_relative_spread_max),
            fail_on_violation=not args.no_fail_on_violation,
        )
        logger = self._build_logger(request.outdir)
        logger.info(
            "Auditing complex energy refinement with %d geometries.",
            len(request.geometries),
        )
        summary = audit_complex_energy_refinement(request, logger=logger)
        logger.info(
            "Outputs written to %s; acceptance=%s",
            request.outdir,
            summary["acceptance"],
        )


def main() -> None:
    AuditComplexEnergyRefinementCLI().run()


if __name__ == "__main__":
    main()
