from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import plotly.graph_objs as go


def _parse_entries(lines: Iterable[str]) -> List[Dict[str, float]]:
    pattern_epoch = re.compile(
        r"Epoch\s+(?P<epoch>\d+).*?"
        r"train[^|]*?loss=(?P<loss_tr>[-+eE0-9\.]+)"
        r"[^|]*?cons=(?P<cons_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_tr>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_tr>[-+eE0-9\.]+)"
        r"(?:.*?\|\s*val[^|]*?cons=(?P<cons_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_val>[-+eE0-9\.]+))?",
        re.IGNORECASE,
    )
    pattern_lbfgs = re.compile(
        r"(?:Coupling\s+)?LBFGS\s+epoch\s+(?P<epoch>\d+).*?"
        r"last\s+loss:\s*(?P<loss_tr>[-+eE0-9\.]+)"
        r".*?cons=(?P<cons_tr>[-+eE0-9\.]+)"
        r".*?rel_flux=(?P<rflux_tr>[-+eE0-9\.]+)"
        r".*?rel_sol=(?P<rsol_tr>[-+eE0-9\.]+)"
        r"(?:.*?\|\s*val[^|]*?cons=(?P<cons_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_flux=(?P<rflux_val>[-+eE0-9\.]+)"
        r"[^|]*?rel_sol=(?P<rsol_val>[-+eE0-9\.]+))?",
        re.IGNORECASE,
    )
    entries: List[Dict[str, float]] = []

    for line in lines:
        match = pattern_epoch.search(line)
        if match:
            entries.append(
                {
                    "raw_epoch": float(match.group("epoch")),
                    "loss_train": float(match.group("loss_tr")),
                    "cons_train": float(match.group("cons_tr")),
                    "rel_flux_train": float(match.group("rflux_tr")),
                    "rel_sol_train": float(match.group("rsol_tr")),
                    "cons_val": float(match.group("cons_val"))
                    if match.group("cons_val")
                    else float("nan"),
                    "rel_flux_val": float(match.group("rflux_val"))
                    if match.group("rflux_val")
                    else float("nan"),
                    "rel_sol_val": float(match.group("rsol_val"))
                    if match.group("rsol_val")
                    else float("nan"),
                }
            )
            continue
        match = pattern_lbfgs.search(line)
        if match:
            entries.append(
                {
                    "raw_epoch": float(match.group("epoch")),
                    "loss_train": float(match.group("loss_tr")),
                    "cons_train": float(match.group("cons_tr")),
                    "rel_flux_train": float(match.group("rflux_tr")),
                    "rel_sol_train": float(match.group("rsol_tr")),
                    "cons_val": float(match.group("cons_val"))
                    if match.group("cons_val")
                    else float("nan"),
                    "rel_flux_val": float(match.group("rflux_val"))
                    if match.group("rflux_val")
                    else float("nan"),
                    "rel_sol_val": float(match.group("rsol_val"))
                    if match.group("rsol_val")
                    else float("nan"),
                }
            )
    return entries


def parse_log(path: Path) -> Dict[str, List[float]]:
    """Parse CouplingNet training.log and return epoch-aligned metrics."""
    entries = _parse_entries(path.read_text().splitlines())
    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "loss_train": [],
        "cons_train": [],
        "rel_flux_train": [],
        "rel_sol_train": [],
        "cons_val": [],
        "rel_flux_val": [],
        "rel_sol_val": [],
    }

    offset = 0.0
    last_raw = None
    last_effective = 0.0
    for entry in entries:
        raw_epoch = entry["raw_epoch"]
        if last_raw is not None and raw_epoch <= last_raw:
            offset = last_effective
        effective_epoch = raw_epoch + offset
        last_raw = raw_epoch
        last_effective = effective_epoch

        metrics["epoch"].append(effective_epoch)
        metrics["loss_train"].append(entry["loss_train"])
        metrics["cons_train"].append(entry["cons_train"])
        metrics["rel_flux_train"].append(entry["rel_flux_train"])
        metrics["rel_sol_train"].append(entry["rel_sol_train"])
        metrics["cons_val"].append(entry["cons_val"])
        metrics["rel_flux_val"].append(entry["rel_flux_val"])
        metrics["rel_sol_val"].append(entry["rel_sol_val"])
    return metrics


def _mask_nan(values: List[float], floor: float = 1e-16) -> List[float | None]:
    out: List[float | None] = []
    for v in values:
        if v != v:
            out.append(None)
        else:
            out.append(max(v, floor))
    return out


def _color_cycle() -> List[str]:
    return [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def make_fig_loss(
    series: List[Tuple[str, Dict[str, List[float]]]], font: Dict
) -> go.Figure:
    fig = go.Figure()
    colors = _color_cycle()
    for idx, (label, metrics) in enumerate(series):
        epochs = metrics["epoch"]
        color = colors[idx % len(colors)]
        for split, dash in (("train", "solid"), ("val", "dot")):
            key = f"cons_{split}"
            if key not in metrics:
                continue
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=_mask_nan(metrics[key]),
                    mode="lines",
                    name=f"{label} Consistency ({split})",
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
    fig.update_layout(
        title="Consistency Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",
        yaxis=dict(exponentformat="power"),
        template="plotly_white",
        font=font,
        legend=dict(
            x=1.0,
            y=0.9,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


def make_fig_error(
    series: List[Tuple[str, Dict[str, List[float]]]],
    metric_key: str,
    title: str,
    font: Dict,
) -> go.Figure:
    fig = go.Figure()
    colors = _color_cycle()
    for idx, (label, metrics) in enumerate(series):
        epochs = metrics["epoch"]
        color = colors[idx % len(colors)]
        for split, dash in (("train", "solid"), ("val", "dot")):
            key = f"{metric_key}_{split}"
            if key not in metrics:
                continue
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=_mask_nan(metrics[key]),
                    mode="lines",
                    name=f"{label} ({split})",
                    line=dict(color=color, dash=dash),
                    connectgaps=True,
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Error",
        yaxis_type="log",
        yaxis=dict(exponentformat="power"),
        template="plotly_white",
        font=font,
        legend=dict(
            x=1.0,
            y=0.9,
            xanchor="right",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    return fig


_warned_static = False


def save_fig(fig: go.Figure, base_path: Path) -> None:
    global _warned_static
    base_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(str(base_path.with_suffix(".png")))
        fig.write_image(str(base_path.with_suffix(".pdf")))
    except Exception:
        if not _warned_static:
            print(
                "Static export skipped (requires kaleido + Chrome); HTML saved instead."
            )
            _warned_static = True
    fig.write_html(str(base_path.with_suffix(".html")))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CouplingNet training logs with separate error figures."
    )
    parser.add_argument(
        "--logs",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to training.log files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional labels for each log (must match number of logs).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("plots"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        default="Times New Roman",
        help="Font family for the plots.",
    )
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.logs):
        raise ValueError("Number of labels must match number of logs.")

    series: List[Tuple[str, Dict[str, List[float]]]] = []
    for idx, log_path in enumerate(args.logs):
        metrics = parse_log(log_path)
        if not metrics.get("epoch"):
            print(f"Warning: no metrics parsed from {log_path}")
            continue
        label = args.labels[idx] if args.labels else log_path.stem.replace("_", " ")
        series.append((label, metrics))

    if not series:
        raise RuntimeError("No valid log data parsed.")

    font = {"family": args.font_family, "size": 14}

    fig_loss = make_fig_loss(series, font)
    fig_sol = make_fig_error(
        series, metric_key="rel_sol", title="Solution Error", font=font
    )
    fig_flux = make_fig_error(
        series, metric_key="rel_flux", title="Flux-Divergence Error", font=font
    )

    save_fig(fig_loss, args.outdir / "losses")
    save_fig(fig_sol, args.outdir / "errors_solution")
    save_fig(fig_flux, args.outdir / "errors_flux")


if __name__ == "__main__":
    main()
