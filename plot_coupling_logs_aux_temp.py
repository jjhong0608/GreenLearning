from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from plot_coupling_logs import (
    VALUE_RE,
    _parse_float,
    make_fig_loss,
    make_fig_metric,
    parse_log,
    save_fig,
)


DEFAULT_LOG = Path(
    "checkpoints/Pure_Poisson/expreiments/raw_output_with_balance_loss/training.log"
)
DEFAULT_OUTDIR = DEFAULT_LOG.parent / "plots_temp_aux_losses"


def _effective_epoch_entries(
    entries: Iterable[Dict[str, float]],
) -> dict[float, Dict[str, float]]:
    aligned: dict[float, Dict[str, float]] = {}
    offset = 0.0
    last_raw: float | None = None
    last_effective = 0.0
    for entry in entries:
        raw_epoch = entry["raw_epoch"]
        if last_raw is not None and raw_epoch <= last_raw:
            offset = last_effective
        effective_epoch = raw_epoch + offset
        last_raw = raw_epoch
        last_effective = effective_epoch
        aligned[effective_epoch] = entry
    return aligned


def _parse_aux_entries(lines: Iterable[str]) -> list[Dict[str, float]]:
    pattern = re.compile(
        rf"epoch\s+(?P<epoch>\d+).*?\|\s*train\s+loss={VALUE_RE}"
        rf".*?\s+balance_loss=(?P<balance_loss_train>{VALUE_RE})"
        rf"\s+symmetric_boundary_loss="
        rf"(?P<symmetric_boundary_loss_train>{VALUE_RE})"
        rf"(?:.*?\|\s*val\s+loss={VALUE_RE}"
        rf".*?\s+balance_loss=(?P<balance_loss_val>{VALUE_RE})"
        rf"\s+symmetric_boundary_loss="
        rf"(?P<symmetric_boundary_loss_val>{VALUE_RE}))?",
        re.IGNORECASE,
    )
    entries: list[Dict[str, float]] = []
    for line in lines:
        match = pattern.search(line)
        if match is None:
            continue
        entries.append(
            {
                "raw_epoch": _parse_float(match.group("epoch")),
                "balance_loss_train": _parse_float(
                    match.group("balance_loss_train")
                ),
                "balance_loss_val": _parse_float(match.group("balance_loss_val")),
                "symmetric_boundary_loss_train": _parse_float(
                    match.group("symmetric_boundary_loss_train")
                ),
                "symmetric_boundary_loss_val": _parse_float(
                    match.group("symmetric_boundary_loss_val")
                ),
            }
        )
    return entries


def parse_log_with_aux(path: Path) -> Dict[str, List[float]]:
    metrics = parse_log(path)
    aux_by_epoch = _effective_epoch_entries(_parse_aux_entries(path.read_text().splitlines()))
    for key in (
        "balance_loss_train",
        "balance_loss_val",
        "symmetric_boundary_loss_train",
        "symmetric_boundary_loss_val",
    ):
        metrics[key] = []
    for epoch in metrics["epoch"]:
        aux = aux_by_epoch.get(epoch, {})
        metrics["balance_loss_train"].append(
            aux.get("balance_loss_train", float("nan"))
        )
        metrics["balance_loss_val"].append(aux.get("balance_loss_val", float("nan")))
        metrics["symmetric_boundary_loss_train"].append(
            aux.get("symmetric_boundary_loss_train", float("nan"))
        )
        metrics["symmetric_boundary_loss_val"].append(
            aux.get("symmetric_boundary_loss_val", float("nan"))
        )
    return metrics


class CouplingAuxTempPlotCLI:
    """One-off CouplingNet log plotter with balance-loss curves."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=(
                "Temporary CouplingNet log plotter for raw-output/balance-loss runs."
            )
        )
        parser.add_argument(
            "--logs",
            type=Path,
            nargs="+",
            default=[DEFAULT_LOG],
            help="Paths to training.log files.",
        )
        parser.add_argument(
            "--labels",
            type=str,
            nargs="*",
            default=None,
            help="Optional labels for each log.",
        )
        parser.add_argument(
            "--outdir",
            type=Path,
            default=DEFAULT_OUTDIR,
            help="Output directory for figures.",
        )
        parser.add_argument(
            "--font-family",
            type=str,
            default="Times New Roman",
            help="Font family for the plots.",
        )
        parser.add_argument(
            "--theme",
            type=str,
            default="plotly_white",
            help="Plotly template name.",
        )
        self.parser = parser

    def run(self) -> None:
        args = self.parser.parse_args()
        if args.labels and len(args.labels) != len(args.logs):
            raise ValueError("Number of labels must match number of logs.")

        series: List[Tuple[str, Dict[str, List[float]]]] = []
        for idx, log_path in enumerate(args.logs):
            metrics = parse_log_with_aux(log_path)
            if not metrics["epoch"]:
                print(f"Warning: no metrics parsed from {log_path}")
                continue
            label = args.labels[idx] if args.labels else log_path.parent.name
            series.append((label, metrics))
        if not series:
            raise RuntimeError("No valid log data parsed.")

        font = {"family": args.font_family, "size": 14}
        figures = {
            "loss": make_fig_loss(series, font, args.theme),
            "l2_consistency": make_fig_metric(
                series,
                metric_key="l2_cons",
                title="L2 Consistency",
                yaxis_title="L2 Consistency",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
            "energy_consistency": make_fig_metric(
                series,
                metric_key="energy_cons",
                title="Energy Consistency",
                yaxis_title="Energy Consistency",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
            "rel_flux": make_fig_metric(
                series,
                metric_key="rel_flux",
                title="Flux-Divergence Relative Error",
                yaxis_title="Relative Error",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
            "rel_sol": make_fig_metric(
                series,
                metric_key="rel_sol",
                title="Solution Relative Error",
                yaxis_title="Relative Error",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
            "balance_loss": make_fig_metric(
                series,
                metric_key="balance_loss",
                title="Balance Loss",
                yaxis_title="Balance Loss",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
            "symmetric_boundary_loss": make_fig_metric(
                series,
                metric_key="symmetric_boundary_loss",
                title="Symmetric Boundary Loss",
                yaxis_title="Symmetric Boundary Loss",
                log_scale=True,
                font=font,
                theme=args.theme,
            ),
        }
        for name, figure in figures.items():
            save_fig(figure, args.outdir / name)


def main() -> None:
    CouplingAuxTempPlotCLI().run()


if __name__ == "__main__":
    main()
