from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from rich.logging import RichHandler

from greenonet.fenicsx_samples.config import FenicsxSampleConfig
from greenonet.fenicsx_samples.generator import FenicsxSampleGenerator


class MakeFenicsxSamplesCLI:
    """Command-line surface for FEniCSx complex Coupling sample generation."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description="Generate complex Coupling sample NPZ files with FEniCSx."
        )
        parser.add_argument("--geometry", type=Path, required=True)
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--gmsh-script", type=Path, default=None)
        parser.add_argument("--msh", type=Path, default=None)
        parser.add_argument("--num-train", type=int, required=True)
        parser.add_argument("--num-valid", type=int, required=True)
        parser.add_argument("--num-test", type=int, required=True)
        parser.add_argument("--lengthscale", type=float, default=0.2)
        parser.add_argument("--amplitude", type=float, default=1.0)
        parser.add_argument("--mean", type=float, default=0.0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--solution-degree", type=int, default=2)
        parser.add_argument("--target-degree", type=int, default=1)
        parser.add_argument("--mesh-size", type=float, default=None)
        parser.add_argument("--num-workers", type=int, default=1)
        parser.add_argument(
            "--sample-seed-policy",
            choices=("sequential", "indexed"),
            default="sequential",
        )
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--skip-existing", action="store_true")
        parser.add_argument(
            "--embed-valid-points",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--require-valid-points-in-mesh",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument("--coefficients", type=Path, default=None)
        self.parser = parser

    def parse_config(self, argv: Sequence[str] | None = None) -> FenicsxSampleConfig:
        args = self.parser.parse_args(argv)
        if (args.gmsh_script is None) == (args.msh is None):
            self.parser.error("specify exactly one of --gmsh-script or --msh")
        embed_valid_points = (
            args.gmsh_script is not None
            if args.embed_valid_points is None
            else bool(args.embed_valid_points)
        )
        require_valid_points = (
            args.gmsh_script is not None
            if args.require_valid_points_in_mesh is None
            else bool(args.require_valid_points_in_mesh)
        )
        return FenicsxSampleConfig(
            geometry=args.geometry,
            out=args.out,
            gmsh_script=args.gmsh_script,
            msh=args.msh,
            num_train=int(args.num_train),
            num_valid=int(args.num_valid),
            num_test=int(args.num_test),
            lengthscale=float(args.lengthscale),
            amplitude=float(args.amplitude),
            mean=float(args.mean),
            seed=int(args.seed),
            solution_degree=int(args.solution_degree),
            target_degree=int(args.target_degree),
            mesh_size=args.mesh_size,
            embed_valid_points=embed_valid_points,
            require_valid_points_in_mesh=require_valid_points,
            coefficients=args.coefficients,
            num_workers=int(args.num_workers),
            sample_seed_policy=args.sample_seed_policy,
            overwrite=bool(args.overwrite),
            skip_existing=bool(args.skip_existing),
        )

    @staticmethod
    def _build_logger(out_dir: Path) -> logging.Logger:
        out_dir.mkdir(parents=True, exist_ok=True)
        build_logger = logging.getLogger("MakeFenicsxSamples")
        build_logger.handlers.clear()
        build_logger.propagate = False
        build_logger.setLevel(logging.INFO)
        logging.root.handlers.clear()

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            out_dir / "make_fenicsx_samples.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        build_logger.addHandler(rich_handler)
        build_logger.addHandler(file_handler)
        return build_logger

    def run(self, argv: Sequence[str] | None = None) -> dict[str, object]:
        config = self.parse_config(argv)
        build_logger = self._build_logger(config.out)
        build_logger.info("starting FEniCSx sample generation")
        summary = FenicsxSampleGenerator(config, logger=build_logger).run()
        build_logger.info("completed FEniCSx sample generation")
        return summary


def main() -> None:
    MakeFenicsxSamplesCLI().run()


if __name__ == "__main__":
    main()
