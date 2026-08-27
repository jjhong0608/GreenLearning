from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.geometry_k_connectivity_visualization import (
    GeometryKConnectivityRequest,
    GeometryKConnectivityVisualization,
    GeometryReachSpec,
)


DEFAULT_OUTDIR = Path("checkpoints/geometry_k_connectivity_visualization")


class VisualizeGeometryKConnectivityCLI:
    """Generate structural K=1 through K=4 reach figures for paper geometries."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Visualize geometry-only tangent reach from representative active "
                "points for K=1 through K=4."
            )
        )
        parser.add_argument(
            "--square-geometry",
            type=Path,
            default=Path("data/geometry/unit_square_h_1_128.npz"),
        )
        parser.add_argument(
            "--disk-geometry",
            type=Path,
            default=Path("data/geometry/disk_radius_05_1_128.npz"),
        )
        parser.add_argument(
            "--annulus-geometry",
            type=Path,
            default=Path("data/geometry/annulus_02_05_1_128.npz"),
        )
        parser.add_argument(
            "--pentagram-geometry",
            type=Path,
            default=Path("data/geometry/pentagram_r05_h00078125.npz"),
        )
        parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
        parser.add_argument("--max-k", type=self._positive_int, default=4)
        parser.add_argument("--global-threshold", type=float, default=0.99)
        parser.add_argument("--tail-quantile", type=float, default=0.05)
        parser.add_argument("--tail-threshold", type=float, default=0.99)
        parser.add_argument("--chunk-size", type=self._positive_int, default=256)
        parser.add_argument("--theme", type=str, default="plotly_white")
        self.parser = parser

    @staticmethod
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be positive")
        return parsed

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("VisualizeGeometryKConnectivity")
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
        rich_handler.setLevel(logging.INFO)
        logger.addHandler(rich_handler)

        file_handler = logging.FileHandler(
            outdir / "analysis.log",
            mode="w",
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"
            )
        )
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        return logger

    def run(self) -> None:
        args = self.parser.parse_args()
        outdir = args.outdir.resolve()
        request = GeometryKConnectivityRequest(
            geometries=(
                GeometryReachSpec(
                    slug="square",
                    label="Unit square",
                    path=args.square_geometry.resolve(),
                ),
                GeometryReachSpec(
                    slug="disk",
                    label="Disk",
                    path=args.disk_geometry.resolve(),
                ),
                GeometryReachSpec(
                    slug="annulus",
                    label="Annulus",
                    path=args.annulus_geometry.resolve(),
                ),
                GeometryReachSpec(
                    slug="pentagram",
                    label="Pentagram",
                    path=args.pentagram_geometry.resolve(),
                ),
            ),
            outdir=outdir,
            max_k=args.max_k,
            global_threshold=args.global_threshold,
            tail_quantile=args.tail_quantile,
            tail_threshold=args.tail_threshold,
            chunk_size=args.chunk_size,
            theme=args.theme,
        )
        logger = self._build_logger(outdir)
        summary = GeometryKConnectivityVisualization(request, logger).analyze()
        selected = {
            item["slug"]: item["selected_geometry_k"] for item in summary["geometries"]
        }
        logger.info("Selected geometry-only K values: %s", selected)


if __name__ == "__main__":
    VisualizeGeometryKConnectivityCLI().run()
