from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_energy_nullspace import (
    ComplexEnergyNullspaceRequest,
    analyze_complex_energy_nullspace,
)


class AnalyzeComplexEnergyNullspaceCLI:
    """Analyze residual-energy coercivity on a complex valid-point graph."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Compare bulk and general connected-segment boundary-anchor "
                "null spaces for a complex geometry."
            )
        )
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--outdir", type=Path, required=True)
        parser.add_argument("--rank-tolerance", type=float, default=1.0e-10)
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("AnalyzeComplexEnergyNullspace")
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
            outdir / "analyze_complex_energy_nullspace.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        request = ComplexEnergyNullspaceRequest(
            geometry=args.geometry,
            outdir=args.outdir,
            rank_tolerance=args.rank_tolerance,
        )
        logger = self._build_logger(request.outdir)
        logger.info("Analyzing complex energy null space: %s", request.geometry)
        summary = analyze_complex_energy_nullspace(request, logger=logger)
        logger.info(
            "Outputs written to %s; conclusions=%s",
            request.outdir,
            summary["conclusions"],
        )


def main() -> None:
    AnalyzeComplexEnergyNullspaceCLI().run()


if __name__ == "__main__":
    main()
