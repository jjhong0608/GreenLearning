from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from rich.logging import RichHandler

from greenonet.complex_sources import (
    ComplexSourceGenerationConfig,
    ComplexSourceGenerator,
)


class MakeComplexSourcesCLI:
    """Generate deterministic source-only NPZ files without FEniCSx."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Generate fixed index-seeded GP sources for complex CouplingNet."
            )
        )
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--num-train", type=int, required=True)
        parser.add_argument("--num-valid", type=int, required=True)
        parser.add_argument("--lengthscale", type=float, default=0.2)
        parser.add_argument("--amplitude", type=float, default=1.0)
        parser.add_argument("--mean", type=float, default=0.0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument(
            "--validate",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
        self.parser = parser

    def parse_config(
        self,
        argv: Sequence[str] | None = None,
    ) -> ComplexSourceGenerationConfig:
        args = self.parser.parse_args(argv)
        return ComplexSourceGenerationConfig(
            geometry=args.geometry,
            out=args.out,
            num_train=args.num_train,
            num_valid=args.num_valid,
            lengthscale=args.lengthscale,
            amplitude=args.amplitude,
            mean=args.mean,
            seed=args.seed,
            overwrite=args.overwrite,
            validate=args.validate,
        )

    @staticmethod
    def _build_logger(out_dir: Path) -> logging.Logger:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("MakeComplexSources")
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
            out_dir / "make_complex_sources.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self, argv: Sequence[str] | None = None) -> dict[str, object]:
        config = self.parse_config(argv)
        logger = self._build_logger(config.out)
        logger.info("starting deterministic complex source generation")
        summary = ComplexSourceGenerator(config, logger=logger).run()
        logger.info("completed deterministic complex source generation")
        return summary


def main() -> None:
    MakeComplexSourcesCLI().run()


if __name__ == "__main__":
    main()
