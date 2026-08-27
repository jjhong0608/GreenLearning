from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.complex_tangent_topology_analysis import (
    ComplexTangentTopologyAnalysis,
    TangentTopologyAnalysisRequest,
)


DEFAULT_GEOMETRY = Path("data/geometry/pentagram_r05_h00078125.npz")
DEFAULT_TRAINED_REPORT = Path(
    "checkpoints/pentagram/coupling8_11_k_comparison/analysis_report.md"
)
DEFAULT_FROZEN_AUDIT = Path(
    "checkpoints/pentagram/coupling9/tangent_subspace_k1_k4_audit"
)
DEFAULT_OUTDIR = Path("checkpoints/pentagram/tangent_topology_k_analysis")


class AnalyzeTangentTopologyCLI:
    """Build the Pentagram axial-topology and tangent-K evidence bundle."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Analyze axial-segment graph reach together with trained and frozen "
                "K=1 through K=4 Pentagram tangent evidence."
            )
        )
        parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
        parser.add_argument(
            "--trained-comparison-report",
            type=Path,
            default=DEFAULT_TRAINED_REPORT,
        )
        parser.add_argument(
            "--frozen-audit-dir",
            type=Path,
            default=DEFAULT_FROZEN_AUDIT,
        )
        parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
        parser.add_argument("--theme", type=str, default="plotly_white")
        parser.add_argument("--chunk-size", type=self._positive_int, default=256)
        self.parser = parser

    @staticmethod
    def _positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("value must be a positive integer")
        return parsed

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("AnalyzeTangentTopology")
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
        frozen_audit = args.frozen_audit_dir.resolve()
        logger = self._build_logger(outdir)
        request = TangentTopologyAnalysisRequest(
            geometry_path=args.geometry.resolve(),
            trained_comparison_report=args.trained_comparison_report.resolve(),
            frozen_metrics_path=(frozen_audit / "metrics" / "per_sample_k1_k4.csv"),
            selected_archive_path=(
                frozen_audit / "data" / "selected_k1_k4_tangent_subspace.npz"
            ),
            outdir=outdir,
            theme=args.theme,
            chunk_size=args.chunk_size,
        )
        summary = ComplexTangentTopologyAnalysis(request, logger).analyze()
        topology = summary["topology"]
        logger.info(
            "Saved report with %d points, point diameter %d, A diameter %d",
            topology["num_points"],
            topology["point_graph_diameter"],
            topology["a_graph_diameter"],
        )


if __name__ == "__main__":
    AnalyzeTangentTopologyCLI().run()
