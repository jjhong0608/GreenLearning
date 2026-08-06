from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_visualization_mesh import (
    ComplexVisualizationMeshConfig,
    ComplexVisualizationMeshGenerator,
)


class MakeComplexVisualizationMeshCLI:
    """Create a reusable conforming Gmsh cache for solution mesh figures."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Generate a complex-domain solution visualization mesh."
        )
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--gmsh-script", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--boundary-size-factor", type=float, default=3.0)
        parser.add_argument("--max-auxiliary-fraction", type=float, default=1.0e-3)
        parser.add_argument("--overwrite", action="store_true")
        self.parser = parser

    @staticmethod
    def _build_logger(out: Path) -> logging.Logger:
        out.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("MakeComplexVisualizationMesh")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(funcName)s - %(message)s")

        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(out.with_suffix(".log"), mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> Path:
        args = self.parser.parse_args()
        config = ComplexVisualizationMeshConfig(
            geometry=args.geometry,
            gmsh_script=args.gmsh_script,
            out=args.out,
            boundary_size_factor=args.boundary_size_factor,
            max_auxiliary_fraction=args.max_auxiliary_fraction,
            overwrite=bool(args.overwrite),
        )
        logger = self._build_logger(config.out)
        logger.info(
            "Generating visualization mesh (geometry=%s, gmsh_script=%s, "
            "boundary_size_factor=%s, max_auxiliary_fraction=%s)",
            config.geometry,
            config.gmsh_script,
            config.boundary_size_factor,
            config.max_auxiliary_fraction,
        )
        return ComplexVisualizationMeshGenerator(config, logger=logger).write()


def main() -> None:
    MakeComplexVisualizationMeshCLI().run()


if __name__ == "__main__":
    main()
