from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rich.logging import RichHandler

from greenonet.coupling_artifacts import (
    CouplingArtifactRequest,
    export_coupling_artifacts,
)


class ExportCouplingArtifactsCLI:
    """CLI for exporting paper-facing CouplingNet artifacts."""

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
            help="Selected test sample indices. Defaults to the first max samples.",
        )
        parser.add_argument(
            "--max-samples",
            type=int,
            default=3,
            help="Number of leading samples to plot when selected samples are omitted.",
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
        self.parser = parser

    @staticmethod
    def _build_logger(outdir: Path) -> logging.Logger:
        outdir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("ExportCouplingArtifacts")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter("%(funcName)s - %(message)s")
        rich_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            omit_repeated_times=False,
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            outdir / "export_coupling_artifacts.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(rich_handler)
        logger.addHandler(file_handler)
        return logger

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
            max_samples=args.max_samples,
            plot_workers=args.plot_workers,
            save_generated_data=bool(args.save_generated_data),
        )
        logger = self._build_logger(request.outdir)
        summary = export_coupling_artifacts(request, logger=logger)
        logger.info(
            "Completed CouplingNet artifact export (selected_samples=%s)",
            summary["selected_samples"],
        )


def main() -> None:
    ExportCouplingArtifactsCLI().run()


if __name__ == "__main__":
    main()
