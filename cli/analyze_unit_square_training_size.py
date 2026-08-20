from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.unit_square_training_size_analysis import (
    TrainingSizeAnalysisRequest,
    UnitSquareTrainingSizeAnalyzer,
)


DEFAULT_ROOT = Path("checkpoints/numerical_examples/unit_square_poisson")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the equal-step Unit-square Poisson training-source count study."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--saturation-tolerance", type=float, default=0.05)
    return parser


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("unit_square_training_size_analysis")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        omit_repeated_times=False,
    )
    formatter = logging.Formatter("%(funcName)s - %(message)s")
    rich_handler.setFormatter(formatter)
    rich_handler.setLevel(logging.INFO)
    logger.addHandler(rich_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s | %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def main() -> None:
    args = _build_parser().parse_args()
    root = args.root.resolve()
    outdir = (
        args.outdir.resolve()
        if args.outdir is not None
        else root / "training_size_analysis"
    )
    logger = _configure_logger(outdir / "analysis.log")
    request = TrainingSizeAnalysisRequest(
        root=root,
        outdir=outdir,
        saturation_tolerance=args.saturation_tolerance,
    )
    result = UnitSquareTrainingSizeAnalyzer(request, logger).analyze()
    logger.info("Report: %s", outdir / "analysis_report.md")
    logger.info(
        "Selected num_train=%d from %d completed runs",
        result.decision.recommended_num_train,
        len(result.runs),
    )


if __name__ == "__main__":
    main()
